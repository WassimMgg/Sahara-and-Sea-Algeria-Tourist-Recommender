"""
api/services.py
===============
Bridge between the trained models (ml/models.pkl) and the web app.

The bundle holds EVERY trained algorithm. Which one is served is decided at
runtime by the "active_algorithm" RecommenderSetting row - switchable from
the admin panel (Recommender page) with no restart:

    get_active_key()            -> key of the algorithm currently served
    set_active_algorithm(key)   -> switch it (validates the key)
    available_algorithms()      -> metadata + CV metrics + which is active
    reload_models()             -> drop the cache (e.g. after retraining)

Recommendations are cached per-user (5-minute TTL) and invalidated when the
user submits a new rating.  The returned list is also diversity-filtered so
at most 2 items of the same place_type appear in any single recommendation set.
"""

import sys
import json
import pickle
import threading
from django.conf import settings
from django.core.cache import cache

from .models import Attraction, Rating, RecommenderSetting

# the pickled models reference classes defined in ml/recommender.py
sys.path.insert(0, str(settings.ML_DIR))

ACTIVE_KEY_SETTING = "active_algorithm"
_REC_TTL = 300   # seconds to keep cached recommendations
_MAX_PER_TYPE = 2  # diversity cap: max items of the same place_type

_bundle = None
_metrics = None
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# loading / cache
# --------------------------------------------------------------------------- #
def _load():
    global _bundle, _metrics
    if _bundle is None:
        with _lock:
            if _bundle is None:
                with open(settings.ML_DIR / "models.pkl", "rb") as fh:
                    _bundle = pickle.load(fh)
                with open(settings.ML_DIR / "metrics.json") as fh:
                    _metrics = json.load(fh)
    return _bundle, _metrics


def reload_models():
    """Forget the cached bundle (call after `python ml/train.py`)."""
    global _bundle, _metrics
    with _lock:
        _bundle, _metrics = None, None


# --------------------------------------------------------------------------- #
# algorithm selection
# --------------------------------------------------------------------------- #
def get_active_key():
    bundle, _ = _load()
    row = RecommenderSetting.objects.filter(key=ACTIVE_KEY_SETTING).first()
    if row and row.value in bundle["models"]:
        return row.value
    return bundle["default"]


def set_active_algorithm(key):
    """Switch the algorithm served to users. Returns the stored key."""
    bundle, _ = _load()
    if key not in bundle["models"]:
        raise ValueError(f"Unknown algorithm '{key}'. "
                         f"Choices: {', '.join(bundle['models'])}")
    RecommenderSetting.objects.update_or_create(
        key=ACTIVE_KEY_SETTING, defaults={"value": key})
    return key


def get_active_model():
    bundle, _ = _load()
    key = get_active_key()
    return bundle["models"][key], key


def available_algorithms():
    """List of dicts: key, label, family, blurb, metrics, active, default."""
    bundle, metrics = _load()
    active = get_active_key()
    out = []
    for key, res in metrics["results"].items():
        out.append({
            "key":            key,
            "label":          res["label"],
            "family":         res["family"],
            "personalized":   res["personalized"],
            "blurb":          res["blurb"],
            "rmse":           round(res["rmse"], 4),
            "mae":            round(res["mae"], 4),
            "precision_at_5": round(res["precision_at_5"], 3),
            "recall_at_5":    round(res["recall_at_5"], 3),
            "ndcg_at_5":      round(res.get("ndcg_at_5", 0.0), 3),
            "active":         key == active,
            "default":        key == metrics["default"],
        })
    return out


def training_info():
    _, metrics = _load()
    return {"n_ratings": metrics["n_ratings"], "n_users": metrics["n_users"],
            "n_items": metrics["n_items"], "folds": metrics["folds"],
            "trained_at": metrics["trained_at"]}


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def attraction_dict(att, score=None, reason=None):
    data = {
        "id": att.id, "name": att.name, "city": att.city, "region": att.region,
        "category": att.category, "place_type": att.place_type,
        "description": att.description, "image_url": att.image_url,
    }
    if score is not None:
        data["score"] = round(float(score), 2)
    if reason is not None:
        data["reason"] = reason
    return data


def popularity_ranking():
    """Average rating per attraction from historical Visitor ratings."""
    sums, counts = {}, {}
    for aid, r in Rating.objects.filter(visitor__isnull=False).values_list("attraction_id", "rating"):
        sums[aid] = sums.get(aid, 0.0) + r
        counts[aid] = counts.get(aid, 0) + 1
    avg = {aid: sums[aid] / counts[aid] for aid in sums}
    order = sorted(avg, key=lambda a: avg[a], reverse=True)
    return order, avg


def account_ratings(user):
    """{attraction_id: rating} for a logged-in account."""
    if not user or not user.is_authenticated:
        return {}
    return {aid: r for aid, r in
            Rating.objects.filter(account=user).values_list("attraction_id", "rating")}


def favourite_type(ratings_map):
    """The place_type the user rates highest (used for explanations)."""
    by_type = {}
    types = dict(Attraction.objects.values_list("id", "place_type"))
    for aid, r in ratings_map.items():
        by_type.setdefault(types.get(aid, ""), []).append(r)
    best, best_t = -1, None
    for t, scores in by_type.items():
        avg = sum(scores) / len(scores)
        if avg > best:
            best, best_t = avg, t
    return best_t


# --------------------------------------------------------------------------- #
# diversity post-processing
# --------------------------------------------------------------------------- #
def _diversify(recs, top_n):
    """Return at most top_n items with no more than _MAX_PER_TYPE of the same type."""
    counts = {}
    out = []
    for rec in recs:
        t = rec.get("place_type", "")
        if counts.get(t, 0) < _MAX_PER_TYPE:
            out.append(rec)
            counts[t] = counts.get(t, 0) + 1
        if len(out) == top_n:
            break
    return out


# --------------------------------------------------------------------------- #
# recommendation cache helpers
# --------------------------------------------------------------------------- #
def _rec_cache_key(user_id, top_n):
    return f"recs:u{user_id}:n{top_n}"


def invalidate_user_cache(user):
    """Bust all cached recommendation sets for this user."""
    if user and user.is_authenticated:
        for n in (5, 6, 10, 12):
            cache.delete(_rec_cache_key(user.id, n))


# --------------------------------------------------------------------------- #
# the recommendation entry point used by the API
# --------------------------------------------------------------------------- #
def recommend_for(user, top_n=5):
    if user and user.is_authenticated:
        key = _rec_cache_key(user.id, top_n)
        cached = cache.get(key)
        if cached is not None:
            return cached

    result = _compute_recommendations(user, top_n)

    if user and user.is_authenticated:
        cache.set(_rec_cache_key(user.id, top_n), result, _REC_TTL)
    return result


def _compute_recommendations(user, top_n=5):
    model, _ = get_active_model()
    ratings_map = account_ratings(user)
    attractions = {a.id: a for a in Attraction.objects.all()}

    if not ratings_map:
        # cold start -> non-personalized crowd favourites
        order, avg = popularity_ranking()
        recs = [attraction_dict(attractions[aid], avg[aid], "Popular with visitors")
                for aid in order if aid in attractions]
        return _diversify(recs, top_n)

    fav = favourite_type(ratings_map)
    model_name = getattr(model, "name", "").lower()
    out = []
    # request 3× candidates so diversity filtering still yields top_n results
    for aid, score in model.recommend(ratings_map, top_n=top_n * 3):
        aid = int(aid)
        if aid not in attractions:
            continue
        att = attractions[aid]
        if "content" in model_name:
            reason = "Similar to places you've enjoyed"
        elif "hybrid" in model_name:
            reason = (f"Matches your taste for {fav}"
                      if fav and att.place_type == fav
                      else "Personalised blend of taste and similarity")
        elif getattr(model, "item_score", None):  # non-personalized
            reason = "Popular with visitors"
        elif fav and att.place_type == fav:
            reason = f"Matches your taste for {fav}"
        else:
            reason = "Visitors with similar taste enjoyed this"
        out.append(attraction_dict(att, score, reason))
    return _diversify(out, top_n)


def place_types():
    return sorted(set(Attraction.objects.values_list("place_type", flat=True)))
