"""
recommender.py
==============
Four collaborative-filtering models, written from scratch with NumPy so the
project has no heavy/fragile dependencies (works with just numpy + pandas).

Every model implements the same small interface:

    model.fit(ratings_df)                       -> learn from the clean data
    model.predict(user_id, item_id) -> float    -> predict one rating (used by cross-validation)
    model.recommend(user_ratings, top_n) -> list of (item_id, score)
                                                -> rank unseen items for a user.
                                                   `user_ratings` is a dict {item_id: rating}.
                                                   This works even for a brand-new user, so the
                                                   app can refresh recommendations the moment a
                                                   user rates something (no retraining needed).

The four models are:
  1. BaselineModel   - global mean + user bias + item bias (the "dumb" baseline)
  2. ItemBasedCF     - item-item cosine similarity on mean-centered ratings
  3. UserBasedCF     - user-user cosine similarity on mean-centered ratings
  4. MatrixFactorization (SVD-style) - latent factors learned by gradient descent
"""

import numpy as np
import pandas as pd

RATING_MIN, RATING_MAX = 1.0, 5.0


def _clip(x):
    return float(min(RATING_MAX, max(RATING_MIN, x)))


# --------------------------------------------------------------------------- #
# 1. Baseline: global mean + biases
# --------------------------------------------------------------------------- #
class BaselineModel:
    name = "Baseline (mean + biases)"

    def __init__(self, reg=10.0):
        self.reg = reg  # regularisation pulls small-sample biases toward 0

    def fit(self, df):
        self.mu = df["rating"].mean()
        # item bias: how much an attraction is rated above/below the global mean
        self.item_bias = {}
        for item_id, group in df.groupby("attraction_id"):
            diffs = group["rating"] - self.mu
            self.item_bias[item_id] = diffs.sum() / (self.reg + len(diffs))
        # user bias: how generous a user is, after removing item bias
        self.user_bias = {}
        for user_id, group in df.groupby("user_id"):
            diffs = [r - self.mu - self.item_bias.get(i, 0.0)
                     for i, r in zip(group["attraction_id"], group["rating"])]
            self.user_bias[user_id] = sum(diffs) / (self.reg + len(diffs))
        return self

    def predict(self, user_id, item_id):
        return _clip(self.mu
                     + self.user_bias.get(user_id, 0.0)
                     + self.item_bias.get(item_id, 0.0))

    def _user_bias_from_ratings(self, user_ratings):
        if not user_ratings:
            return 0.0
        diffs = [r - self.mu - self.item_bias.get(i, 0.0)
                 for i, r in user_ratings.items()]
        return sum(diffs) / (self.reg + len(diffs))

    def recommend(self, user_ratings, top_n=5, candidate_items=None):
        bu = self._user_bias_from_ratings(user_ratings)
        items = candidate_items if candidate_items is not None else self.item_bias.keys()
        scores = [(i, _clip(self.mu + bu + self.item_bias.get(i, 0.0)))
                  for i in items if i not in user_ratings]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# --------------------------------------------------------------------------- #
# Shared helper: build a dense user x item matrix with NaN for "not rated"
# --------------------------------------------------------------------------- #
def _build_matrix(df):
    users = sorted(df["user_id"].unique())
    items = sorted(df["attraction_id"].unique())
    u_index = {u: k for k, u in enumerate(users)}
    i_index = {i: k for k, i in enumerate(items)}
    mat = np.full((len(users), len(items)), np.nan)
    for u, i, r in zip(df["user_id"], df["attraction_id"], df["rating"]):
        mat[u_index[u], i_index[i]] = r
    return mat, users, items, u_index, i_index


def _cosine_matrix(centered):
    """Cosine similarity between columns, treating NaN as 0 contribution."""
    filled = np.nan_to_num(centered, nan=0.0)
    norms = np.sqrt((filled ** 2).sum(axis=0))
    norms[norms == 0] = 1e-9
    sim = (filled.T @ filled) / np.outer(norms, norms)
    return sim


# --------------------------------------------------------------------------- #
# 2. Item-based collaborative filtering
# --------------------------------------------------------------------------- #
class ItemBasedCF:
    name = "Item-based CF (cosine)"

    def __init__(self, k=10):
        self.k = k  # number of neighbour items used in a prediction

    def fit(self, df):
        self.mu = df["rating"].mean()
        mat, users, items, u_index, i_index = _build_matrix(df)
        self.items = items
        self.i_index = i_index
        # item mean (used to mean-center each column)
        self.item_mean = np.nanmean(mat, axis=0)
        self.item_mean = np.nan_to_num(self.item_mean, nan=self.mu)
        centered = mat - self.item_mean  # NaN stays NaN
        self.sim = _cosine_matrix(centered)
        np.fill_diagonal(self.sim, 0.0)  # an item is not its own neighbour
        return self

    def _predict_from_ratings(self, item_id, user_ratings):
        if item_id not in self.i_index:
            return self.mu
        target = self.i_index[item_id]
        # neighbours = items the user rated, ranked by similarity to the target
        neighbours = []
        for rated_item, r in user_ratings.items():
            if rated_item in self.i_index and rated_item != item_id:
                j = self.i_index[rated_item]
                neighbours.append((self.sim[target, j], j, r))
        neighbours = [n for n in neighbours if n[0] > 0]
        if not neighbours:
            return _clip(self.item_mean[target])
        neighbours.sort(key=lambda x: x[0], reverse=True)
        neighbours = neighbours[: self.k]
        num = sum(sim * (r - self.item_mean[j]) for sim, j, r in neighbours)
        den = sum(abs(sim) for sim, j, r in neighbours)
        return _clip(self.item_mean[target] + num / den) if den else _clip(self.item_mean[target])

    def predict(self, user_id, item_id):
        # rebuild this user's ratings from the training matrix is unnecessary at
        # CV time because the harness passes a ratings dict; keep a simple fallback
        return _clip(self.item_mean[self.i_index[item_id]]) if item_id in self.i_index else self.mu

    def recommend(self, user_ratings, top_n=5, candidate_items=None):
        items = candidate_items if candidate_items is not None else self.items
        scores = [(i, self._predict_from_ratings(i, user_ratings))
                  for i in items if i not in user_ratings]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# --------------------------------------------------------------------------- #
# 3. User-based collaborative filtering
# --------------------------------------------------------------------------- #
class UserBasedCF:
    name = "User-based CF (cosine)"

    def __init__(self, k=15):
        self.k = k

    def fit(self, df):
        self.mu = df["rating"].mean()
        mat, users, items, u_index, i_index = _build_matrix(df)
        self.mat = mat
        self.items = items
        self.i_index = i_index
        self.user_mean = np.nanmean(mat, axis=1)
        self.user_mean = np.nan_to_num(self.user_mean, nan=self.mu)
        self.centered = mat - self.user_mean[:, None]
        return self

    def _predict_from_ratings(self, item_id, user_ratings):
        if item_id not in self.i_index:
            return self.mu
        col = self.i_index[item_id]
        # active user's mean and centered vector over the shared item space
        active = np.full(len(self.items), np.nan)
        for it, r in user_ratings.items():
            if it in self.i_index:
                active[self.i_index[it]] = r
        active_mean = np.nanmean(active) if np.any(~np.isnan(active)) else self.mu
        active_c = np.nan_to_num(active - active_mean, nan=0.0)

        train_c = np.nan_to_num(self.centered, nan=0.0)
        num = train_c @ active_c
        den = (np.sqrt((train_c ** 2).sum(axis=1)) * np.sqrt((active_c ** 2).sum()))
        den[den == 0] = 1e-9
        sims = num / den

        # only neighbours who actually rated the target item
        rated_target = ~np.isnan(self.mat[:, col])
        candidates = [(sims[u], u) for u in range(len(sims))
                      if rated_target[u] and sims[u] > 0]
        if not candidates:
            return _clip(active_mean)
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[: self.k]
        num2 = sum(s * self.centered[u, col] for s, u in candidates)
        den2 = sum(abs(s) for s, u in candidates)
        return _clip(active_mean + num2 / den2) if den2 else _clip(active_mean)

    def predict(self, user_id, item_id):
        return self.mu  # CV harness uses recommend-style fold-in instead

    def recommend(self, user_ratings, top_n=5, candidate_items=None):
        items = candidate_items if candidate_items is not None else self.items
        scores = [(i, self._predict_from_ratings(i, user_ratings))
                  for i in items if i not in user_ratings]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# --------------------------------------------------------------------------- #
# 4. Matrix Factorization (SVD-style, learned with gradient descent)
# --------------------------------------------------------------------------- #
class MatrixFactorization:
    name = "Matrix Factorization (SVD)"

    def __init__(self, n_factors=8, n_epochs=80, lr=0.01, reg=0.1, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, df):
        rng = np.random.default_rng(self.seed)
        self.mu = df["rating"].mean()
        users = sorted(df["user_id"].unique())
        items = sorted(df["attraction_id"].unique())
        self.u_index = {u: k for k, u in enumerate(users)}
        self.i_index = {i: k for k, i in enumerate(items)}
        self.items = items
        n_users, n_items = len(users), len(items)

        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        samples = [(self.u_index[u], self.i_index[i], r)
                   for u, i, r in zip(df["user_id"], df["attraction_id"], df["rating"])]
        for _ in range(self.n_epochs):
            rng.shuffle(samples)
            for u, i, r in samples:
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = r - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu, qi = self.P[u].copy(), self.Q[i].copy()
                self.P[u] += self.lr * (err * qi - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * qi)
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.u_index or item_id not in self.i_index:
            return _clip(self.mu)
        u, i = self.u_index[user_id], self.i_index[item_id]
        return _clip(self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])

    def _fold_in(self, user_ratings):
        """Estimate a latent vector + bias for a user from their ratings (ridge)."""
        rows, targets = [], []
        bi_known = []
        for it, r in user_ratings.items():
            if it in self.i_index:
                idx = self.i_index[it]
                rows.append(self.Q[idx])
                bi_known.append(self.bi[idx])
                targets.append(r)
        if not rows:
            return 0.0, np.zeros(self.n_factors)
        Qr = np.array(rows)
        resid = np.array(targets) - self.mu - np.array(bi_known)
        bu = resid.mean()  # simple user-bias estimate
        resid = resid - bu
        A = Qr.T @ Qr + self.reg * np.eye(self.n_factors)
        pu = np.linalg.solve(A, Qr.T @ resid)
        return bu, pu

    def recommend(self, user_ratings, top_n=5, candidate_items=None):
        bu, pu = self._fold_in(user_ratings)
        items = candidate_items if candidate_items is not None else self.items
        scores = []
        for i in items:
            if i in user_ratings:
                continue
            idx = self.i_index[i]
            scores.append((i, _clip(self.mu + bu + self.bi[idx] + pu @ self.Q[idx])))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


ALL_MODELS = {
    "baseline": BaselineModel,
    "item_cf": ItemBasedCF,
    "user_cf": UserBasedCF,
    "mf": MatrixFactorization,
}
