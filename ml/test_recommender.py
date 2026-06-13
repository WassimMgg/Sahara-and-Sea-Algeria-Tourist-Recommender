"""
ml/test_recommender.py
======================
Lightweight tests for the from-scratch recommender algorithms.

Run with:
    python -m pytest ml/test_recommender.py   (from the project root)
    python ml/test_recommender.py             (standalone, from ml/ directory)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import unittest

from recommender import (
    MeanRecommender,
    BaselineModel,
    UserUserCF,
    ItemItemCF,
    MatrixFactorization,
    ALSRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    ALL_MODELS,
)
from train import ndcg_at_k

# ---------------------------------------------------------------------------
# Tiny synthetic dataset: 4 users, 5 items
# ---------------------------------------------------------------------------
_RATINGS = [
    (1, 1, 4.0), (1, 2, 5.0), (1, 3, 3.0),
    (2, 1, 3.0), (2, 2, 4.0), (2, 4, 5.0),
    (3, 2, 5.0), (3, 3, 4.0), (3, 5, 2.0),
    (4, 1, 2.0), (4, 4, 3.0), (4, 5, 4.0),
]
DF = pd.DataFrame(_RATINGS, columns=["user_id", "attraction_id", "rating"])

_ATTRACTIONS = pd.DataFrame([
    {"attraction_id": 1, "name": "Casbah",   "category": "Historic / UNESCO",
     "description": "Ottoman-era medina and citadel UNESCO heritage site"},
    {"attraction_id": 2, "name": "Tassili",  "category": "Nature / UNESCO",
     "description": "Sahara sandstone plateau with prehistoric rock art"},
    {"attraction_id": 3, "name": "Tipaza",   "category": "Roman Ruins",
     "description": "Ancient Roman ruins on the Mediterranean coast"},
    {"attraction_id": 4, "name": "Ghardaia", "category": "Desert Town",
     "description": "Mozabite fortified desert town in the M'zab valley"},
    {"attraction_id": 5, "name": "Oran",     "category": "Coastal City",
     "description": "Mediterranean port city with Spanish colonial history"},
])

_KNOWN = {1: 4.0, 2: 5.0}   # user has rated items 1 and 2


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _check_model(tc, model):
    """Every model must satisfy the shared contract."""
    model.fit(DF)

    # predict_from_ratings -> float in [1, 5]
    score = model.predict_from_ratings(3, _KNOWN)
    tc.assertIsInstance(score, float)
    tc.assertGreaterEqual(score, 1.0)
    tc.assertLessEqual(score, 5.0)

    # recommend -> list of (item_id, score) tuples, none already rated
    recs = model.recommend(_KNOWN, top_n=2)
    tc.assertEqual(len(recs), 2)
    for item_id, s in recs:
        tc.assertNotIn(item_id, _KNOWN)
        tc.assertIsInstance(s, float)


# ---------------------------------------------------------------------------
# 1. Standard interface — all models
# ---------------------------------------------------------------------------
class TestModelInterface(unittest.TestCase):
    def test_mean_recommender(self):      _check_model(self, MeanRecommender())
    def test_baseline(self):              _check_model(self, BaselineModel())
    def test_user_cf_cosine(self):        _check_model(self, UserUserCF("cosine"))
    def test_user_cf_pearson(self):       _check_model(self, UserUserCF("pearson"))
    def test_item_cf_cosine(self):        _check_model(self, ItemItemCF("cosine"))
    def test_item_cf_pearson(self):       _check_model(self, ItemItemCF("pearson"))
    def test_mf_sgd(self):               _check_model(self, MatrixFactorization())
    def test_als(self):                  _check_model(self, ALSRecommender())


# ---------------------------------------------------------------------------
# 2. Content-based recommender
# ---------------------------------------------------------------------------
class TestContentBased(unittest.TestCase):
    def setUp(self):
        self.model = ContentBasedRecommender().fit(DF, _ATTRACTIONS)

    def test_ready_after_fit_with_attractions(self):
        self.assertTrue(self.model._ready)

    def test_not_ready_without_attractions(self):
        m = ContentBasedRecommender().fit(DF, None)
        self.assertFalse(m._ready)

    def test_recommend_excludes_rated(self):
        recs = self.model.recommend(_KNOWN, top_n=3)
        self.assertEqual(len(recs), 3)
        for item_id, _ in recs:
            self.assertNotIn(item_id, _KNOWN)

    def test_cold_start_returns_global_mean(self):
        score = self.model.predict_from_ratings(3, {})
        self.assertAlmostEqual(score, float(DF["rating"].mean()))

    def test_score_in_range(self):
        score = self.model.predict_from_ratings(3, _KNOWN)
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 5.0)


# ---------------------------------------------------------------------------
# 3. ALS recommender
# ---------------------------------------------------------------------------
class TestALS(unittest.TestCase):
    def setUp(self):
        self.model = ALSRecommender(n_factors=4, n_iters=5).fit(DF)

    def test_predict_known_user(self):
        score = self.model.predict(1, 4)
        self.assertIsInstance(score, float)

    def test_fold_in_new_user(self):
        score = self.model.predict_from_ratings(4, _KNOWN)
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 5.0)

    def test_recommend_length(self):
        recs = self.model.recommend(_KNOWN, top_n=3)
        self.assertEqual(len(recs), 3)


# ---------------------------------------------------------------------------
# 4. Hybrid recommender
# ---------------------------------------------------------------------------
class TestHybrid(unittest.TestCase):
    def setUp(self):
        self.model = HybridRecommender(
            cf_cls=ItemItemCF, cf_kwargs={"similarity": "cosine"}, alpha=0.65
        ).fit(DF, _ATTRACTIONS)

    def test_recommend_excludes_rated(self):
        recs = self.model.recommend(_KNOWN, top_n=3)
        for item_id, _ in recs:
            self.assertNotIn(item_id, _KNOWN)

    def test_score_blended(self):
        score = self.model.predict_from_ratings(4, _KNOWN)
        cf_score = self.model.cf.predict_from_ratings(4, _KNOWN)
        cb_score = self.model.cb.predict_from_ratings(4, _KNOWN)
        expected = np.clip(0.65 * cf_score + 0.35 * cb_score, 1.0, 5.0)
        self.assertAlmostEqual(score, expected, places=5)


# ---------------------------------------------------------------------------
# 5. Diversity post-processing logic
# ---------------------------------------------------------------------------
class TestDiversity(unittest.TestCase):
    def _run_diversify(self, recs, top_n, max_per_type=2):
        """Mirror of services._diversify — tested independently."""
        counts = {}
        out = []
        for rec in recs:
            t = rec.get("place_type", "")
            if counts.get(t, 0) < max_per_type:
                out.append(rec)
                counts[t] = counts.get(t, 0) + 1
            if len(out) == top_n:
                break
        return out

    def test_caps_same_type(self):
        recs = [
            {"place_type": "Beach", "score": 4.9},
            {"place_type": "Beach", "score": 4.8},
            {"place_type": "Beach", "score": 4.7},
            {"place_type": "Desert", "score": 4.6},
            {"place_type": "Desert", "score": 4.5},
        ]
        out = self._run_diversify(recs, top_n=4)
        beach_count = sum(1 for r in out if r["place_type"] == "Beach")
        self.assertLessEqual(beach_count, 2)

    def test_returns_correct_count(self):
        recs = [{"place_type": "A"}] * 10
        out = self._run_diversify(recs, top_n=5, max_per_type=2)
        self.assertEqual(len(out), 2)  # capped by max_per_type before top_n


# ---------------------------------------------------------------------------
# 6. NDCG metric
# ---------------------------------------------------------------------------
class TestNDCG(unittest.TestCase):
    def test_perfect_ranking(self):
        recs = [1, 2, 3, 4, 5]
        liked = {1, 2, 3}
        self.assertAlmostEqual(ndcg_at_k(recs, liked, k=5), 1.0)

    def test_zero_hits(self):
        recs = [10, 11, 12]
        liked = {1, 2}
        self.assertAlmostEqual(ndcg_at_k(recs, liked), 0.0)

    def test_partial_hits_less_than_one(self):
        recs = [99, 1, 2, 3, 4]
        liked = {1, 2, 3}
        score = ndcg_at_k(recs, liked, k=5)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_liked(self):
        recs = [1, 2, 3]
        self.assertAlmostEqual(ndcg_at_k(recs, set()), 0.0)

    def test_rewards_early_hits(self):
        recs_early = [1, 99, 99, 99, 99]
        recs_late  = [99, 99, 99, 99, 1]
        liked = {1}
        self.assertGreater(
            ndcg_at_k(recs_early, liked, k=5),
            ndcg_at_k(recs_late,  liked, k=5),
        )


# ---------------------------------------------------------------------------
# 7. Registry completeness
# ---------------------------------------------------------------------------
class TestRegistry(unittest.TestCase):
    def test_all_models_have_required_meta(self):
        for key, (cls, kwargs, meta) in ALL_MODELS.items():
            for field in ("label", "family", "personalized", "blurb"):
                self.assertIn(field, meta, msg=f"'{key}' missing '{field}'")

    def test_all_models_instantiate(self):
        for key, (cls, kwargs, meta) in ALL_MODELS.items():
            try:
                model = cls(**kwargs)
            except Exception as e:
                self.fail(f"Could not instantiate '{key}': {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
