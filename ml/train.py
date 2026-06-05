"""
train.py
========
Step 3 of the project: "select the most efficient recommender system".

We compare the four models with 5-fold cross-validation and two standard
error metrics:

    RMSE - Root Mean Squared Error  (punishes big mistakes harder)
    MAE  - Mean Absolute Error      (average size of the mistake)

Lower is better for both. The model with the lowest RMSE is retrained on the
full dataset and saved to ml/model.pkl, ready for the Django app to load.

Run:  python train.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from recommender import ALL_MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(HERE, "ratings_clean.csv")
MODEL_PATH = os.path.join(HERE, "model.pkl")
METRICS_PATH = os.path.join(HERE, "metrics.json")

N_FOLDS = 5
SEED = 42


def predict_test(model, key, user_id, item_id, user_train_ratings):
    """Ask each model for a prediction using only training information."""
    if key in ("baseline", "mf"):
        return model.predict(user_id, item_id)
    # item_cf / user_cf predict from the user's known (training) ratings
    return model._predict_from_ratings(item_id, user_train_ratings)


def cross_validate(df, key, model_cls):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    folds = np.array_split(idx, N_FOLDS)

    sq_errors, abs_errors = [], []
    for f in range(N_FOLDS):
        test_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != f])
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = model_cls().fit(train_df)

        # each user's known ratings (from the training part of this fold)
        user_train = {}
        for u, i, r in zip(train_df["user_id"], train_df["attraction_id"], train_df["rating"]):
            user_train.setdefault(u, {})[i] = r

        for u, i, r in zip(test_df["user_id"], test_df["attraction_id"], test_df["rating"]):
            pred = predict_test(model, key, u, i, user_train.get(u, {}))
            sq_errors.append((pred - r) ** 2)
            abs_errors.append(abs(pred - r))

    rmse = float(np.sqrt(np.mean(sq_errors)))
    mae = float(np.mean(abs_errors))
    return rmse, mae


def main():
    df = pd.read_csv(CLEAN_PATH)
    print(f"Evaluating {len(ALL_MODELS)} models with {N_FOLDS}-fold "
          f"cross-validation on {len(df)} ratings.\n")

    results = {}
    for key, model_cls in ALL_MODELS.items():
        rmse, mae = cross_validate(df, key, model_cls)
        results[key] = {"name": model_cls.name, "rmse": rmse, "mae": mae}
        print(f"  {model_cls.name:32s}  RMSE = {rmse:.4f}   MAE = {mae:.4f}")

    # Lowest RMSE overall (may be the trivial baseline).
    best_overall = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nLowest RMSE overall:  {results[best_overall]['name']} "
          f"(RMSE = {results[best_overall]['rmse']:.4f})")

    # The app needs PERSONALIZED recommendations (requirement 4): the ranking has
    # to change depending on what each user rates. The pure baseline gives every
    # user the same ranking, so we choose the most efficient *personalized* model
    # to power the application.
    personalized = ["item_cf", "user_cf", "mf"]
    served_key = min(personalized, key=lambda k: results[k]["rmse"])
    served = results[served_key]
    if best_overall == served_key:
        print(f"Model used in the app: {served['name']} "
              f"(RMSE = {served['rmse']:.4f}) -- also the best overall.")
    else:
        print(f"Model used in the app: {served['name']} "
              f"(RMSE = {served['rmse']:.4f}) -- the best PERSONALIZED model.")
        print(f"  (The baseline's RMSE is marginally lower, but it recommends the\n"
              f"   same order to everyone, so it cannot personalize.)")

    # retrain the chosen model on ALL the data and save it for the backend
    served_model = ALL_MODELS[served_key]().fit(df)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"key": served_key, "name": served["name"], "model": served_model}, fh)

    with open(METRICS_PATH, "w") as fh:
        json.dump({"results": results, "best_overall": best_overall,
                   "served": served_key, "n_folds": N_FOLDS,
                   "n_ratings": int(len(df))}, fh, indent=2)

    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
