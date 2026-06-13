"""
ml/train.py
===========
'Personalized recommenders - evaluation' + model selection.

For EVERY algorithm in the registry we run 5-fold cross-validation and report:

  Error metrics (how close are predicted ratings to real ones - lower = better)
      RMSE  - root mean squared error (punishes big mistakes harder)
      MAE   - mean absolute error     (average size of a mistake)

  Ranking metrics (is the top-5 list actually good - higher = better)
      Precision@5  - share of the recommended 5 that the user really liked (>= 4*)
      Recall@5     - share of everything the user liked that made it into the 5
      NDCG@5       - Normalized Discounted Cumulative Gain; rewards top-ranked hits
                     more than lower-ranked ones (higher = better)

All models are then retrained on the FULL dataset and saved together to
ml/models.pkl. The DEFAULT model = the PERSONALIZED algorithm with the lowest
RMSE: non-personalized models can score deceptively well on RMSE while ranking
identically for every user, which defeats the point of a recommender.
The active algorithm can be changed at runtime from the admin panel.

Run:  python train.py
"""

import os
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from recommender import ALL_MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(HERE, "ratings_clean.csv")
ATTRACTIONS_PATH = os.path.join(HERE, "..", "data", "attractions.csv")
BUNDLE_PATH = os.path.join(HERE, "models.pkl")
METRICS_PATH = os.path.join(HERE, "metrics.json")

N_FOLDS = 5
SEED = 42
TOP_N = 5
LIKE_THRESHOLD = 4.0      # a test rating >= 4 counts as "the user liked it"


# --------------------------------------------------------------------------- #
# ranking metric helpers
# --------------------------------------------------------------------------- #
def ndcg_at_k(recs, liked, k=5):
    """Normalized Discounted Cumulative Gain at k.

    DCG = Σ 1/log2(rank+1) for each hit in the top-k recommendations.
    NDCG = DCG / ideal DCG (all liked items ranked first).
    """
    dcg = sum(1.0 / np.log2(rank + 2)
              for rank, item in enumerate(recs[:k])
              if item in liked)
    ideal_n = min(len(liked), k)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def _load_attractions():
    """Load the attractions CSV (for content-based models). Returns None if missing."""
    if not os.path.exists(ATTRACTIONS_PATH):
        print(f"  [warn] {ATTRACTIONS_PATH} not found — content-based models will skip TF-IDF")
        return None
    df = pd.read_csv(ATTRACTIONS_PATH, dtype=str).fillna("")
    df["attraction_id"] = pd.to_numeric(df["attraction_id"], errors="coerce")
    df = df.dropna(subset=["attraction_id"])
    df["attraction_id"] = df["attraction_id"].astype(int)
    return df


def evaluate(df, key, model_cls, kwargs, attractions_df=None):
    """5-fold CV -> RMSE, MAE, Precision@5, Recall@5, NDCG@5."""
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    folds = np.array_split(idx, N_FOLDS)

    sq, ab = [], []
    precisions, recalls, ndcgs = [], [], []

    for f in range(N_FOLDS):
        test_df = df.iloc[folds[f]]
        train_df = df.iloc[np.concatenate([folds[j] for j in range(N_FOLDS) if j != f])]
        model = model_cls(**kwargs).fit(train_df, attractions_df=attractions_df)

        # each user's TRAINING ratings (what the model may know about them)
        user_train = {}
        for u, i, r in zip(train_df["user_id"], train_df["attraction_id"], train_df["rating"]):
            user_train.setdefault(int(u), {})[int(i)] = float(r)

        # ---- error metrics ------------------------------------------------ #
        for u, i, r in zip(test_df["user_id"], test_df["attraction_id"], test_df["rating"]):
            pred = model.predict_from_ratings(int(i), user_train.get(int(u), {}))
            pred = float(np.clip(pred, 1.0, 5.0))
            sq.append((pred - r) ** 2)
            ab.append(abs(pred - r))

        # ---- ranking metrics ---------------------------------------------- #
        liked = {}
        for u, i, r in zip(test_df["user_id"], test_df["attraction_id"], test_df["rating"]):
            if r >= LIKE_THRESHOLD:
                liked.setdefault(int(u), set()).add(int(i))
        for u, liked_items in liked.items():
            known = user_train.get(u, {})
            recs = [i for i, _ in model.recommend(known, top_n=TOP_N)]
            if not recs:
                continue
            hits = len(set(recs) & liked_items)
            precisions.append(hits / len(recs))
            recalls.append(hits / len(liked_items))
            ndcgs.append(ndcg_at_k(recs, liked_items, k=TOP_N))

    return {
        "rmse":          float(np.sqrt(np.mean(sq))),
        "mae":           float(np.mean(ab)),
        "precision_at_5": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_5":    float(np.mean(recalls))    if recalls    else 0.0,
        "ndcg_at_5":      float(np.mean(ndcgs))      if ndcgs      else 0.0,
    }


def main():
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df)} clean ratings "
          f"({df['user_id'].nunique()} users x {df['attraction_id'].nunique()} attractions)\n")

    attractions_df = _load_attractions()
    if attractions_df is not None:
        print(f"Loaded {len(attractions_df)} attractions for content-based models\n")

    results = {}
    for key, (cls, kwargs, meta) in ALL_MODELS.items():
        m = evaluate(df, key, cls, kwargs, attractions_df=attractions_df)
        results[key] = {**m, **meta}
        print(f"  {meta['label']:<36} RMSE {m['rmse']:.4f}   MAE {m['mae']:.4f}   "
              f"P@5 {m['precision_at_5']:.3f}   R@5 {m['recall_at_5']:.3f}   "
              f"NDCG@5 {m['ndcg_at_5']:.3f}")

    # default = personalized model with the lowest RMSE
    personalized = {k: v for k, v in results.items() if v["personalized"]}
    default_key = min(personalized, key=lambda k: personalized[k]["rmse"])
    print(f"\nDefault (lowest-RMSE personalized) model: "
          f"{results[default_key]['label']}  (RMSE {results[default_key]['rmse']:.4f})")

    # retrain every model on the full dataset and bundle them
    bundle = {"models": {}, "default": default_key,
              "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key, (cls, kwargs, _) in ALL_MODELS.items():
        bundle["models"][key] = cls(**kwargs).fit(df, attractions_df=attractions_df)
    with open(BUNDLE_PATH, "wb") as fh:
        pickle.dump(bundle, fh)

    with open(METRICS_PATH, "w") as fh:
        json.dump({"results": results, "default": default_key,
                   "n_ratings": int(len(df)),
                   "n_users": int(df["user_id"].nunique()),
                   "n_items": int(df["attraction_id"].nunique()),
                   "folds": N_FOLDS, "trained_at": bundle["trained_at"]}, fh, indent=2)

    print(f"\nSaved {len(bundle['models'])} trained models -> {BUNDLE_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
