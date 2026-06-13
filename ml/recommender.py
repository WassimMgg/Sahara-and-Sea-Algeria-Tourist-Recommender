"""
ml/recommender.py
=================
Every recommender the course covers, implemented from scratch with NumPy.

Course mapping
--------------
1. NON-PERSONALIZED  (recommendations based on mean calculations)
     * MeanRecommender      - damped item-mean ranking ("crowd favourites")
     * BaselineModel        - global mean + user bias + item bias
2. SIMILARITY CALCULATIONS
     * cosine similarity            (raw vectors, missing = no contribution)
     * Pearson correlation          (mean-centered, on co-rated entries only)
   -> both available for the two collaborative-filtering families below.
3. PERSONALIZED - USER-USER and ITEM-ITEM collaborative filtering
     * UserUserCF(similarity="cosine" | "pearson")
     * ItemItemCF(similarity="cosine" | "pearson")
4. (bonus) MatrixFactorization - SGD-trained latent factors

Every model implements the same interface:
    fit(df)                                   df: user_id, attraction_id, rating
    predict(user_id, item_id)                 known training user
    predict_from_ratings(item_id, ratings)    NEW user: {item_id: rating}
    recommend(ratings, top_n=5)               -> [(item_id, score), ...]
"""

import re
import numpy as np
import pandas as pd

K_NEIGHBOURS = 20          # neighbourhood size for the CF models
DAMPING = 5                # damped-mean strength for the mean recommender
MIN_CORATED = 2            # min co-rated entries for a Pearson similarity


# ============================================================================ #
#  shared helpers
# ============================================================================ #
def build_matrix(df):
    """Pivot the ratings into a users x items matrix (NaN = not rated)."""
    users = np.sort(df["user_id"].unique())
    items = np.sort(df["attraction_id"].unique())
    u_idx = {u: k for k, u in enumerate(users)}
    i_idx = {i: k for k, i in enumerate(items)}
    M = np.full((len(users), len(items)), np.nan)
    for u, i, r in zip(df["user_id"], df["attraction_id"], df["rating"]):
        M[u_idx[u], i_idx[i]] = r
    return M, users, items, u_idx, i_idx


def cosine_columns(M):
    """Cosine similarity between COLUMNS of M; NaN contributes nothing."""
    F = np.nan_to_num(M, nan=0.0)
    norms = np.linalg.norm(F, axis=0)
    norms[norms == 0] = 1e-12
    S = (F.T @ F) / np.outer(norms, norms)
    np.fill_diagonal(S, 0.0)
    return S


def pearson_columns(M):
    """
    Pearson correlation between COLUMNS of M computed only on co-rated rows.
    Columns are centered by their own mean over the co-rated subset.
    """
    n = M.shape[1]
    S = np.zeros((n, n))
    mask = ~np.isnan(M)
    for a in range(n):
        for b in range(a + 1, n):
            co = mask[:, a] & mask[:, b]
            if co.sum() < MIN_CORATED:
                continue
            x, y = M[co, a], M[co, b]
            xc, yc = x - x.mean(), y - y.mean()
            denom = np.linalg.norm(xc) * np.linalg.norm(yc)
            if denom < 1e-12:
                continue
            S[a, b] = S[b, a] = float(xc @ yc / denom)
    return S


def similarity_matrix(M, kind):
    """kind: 'cosine' (over columns of M) or 'pearson' (co-rated, centered)."""
    return cosine_columns(M) if kind == "cosine" else pearson_columns(M)


# ============================================================================ #
#  1a. NON-PERSONALIZED - damped item means
# ============================================================================ #
class MeanRecommender:
    """
    'Recommendations based on mean calculations'.

    Score(item) = (sum_of_ratings + DAMPING * global_mean) / (count + DAMPING)
    The damping keeps an item with one lucky 5* from beating an item with
    fifty 4.5*s. Ranking is THE SAME for every user (non-personalized).
    """
    name = "Mean ratings (non-personalized)"

    def fit(self, df, attractions_df=None):
        self.global_mean = float(df["rating"].mean())
        g = df.groupby("attraction_id")["rating"].agg(["sum", "count"])
        self.item_score = {
            int(i): (row["sum"] + DAMPING * self.global_mean) / (row["count"] + DAMPING)
            for i, row in g.iterrows()
        }
        self.items = np.array(sorted(self.item_score))
        return self

    def predict(self, user_id, item_id):
        return self.item_score.get(int(item_id), self.global_mean)

    def predict_from_ratings(self, item_id, ratings):
        return self.predict(None, item_id)

    def recommend(self, ratings, top_n=5, candidate_items=None):
        cands = candidate_items if candidate_items is not None else self.items
        scored = [(int(i), self.item_score.get(int(i), self.global_mean))
                  for i in cands if int(i) not in ratings]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ============================================================================ #
#  1b. NON-PERSONALIZED - bias model (global mean + user/item bias)
# ============================================================================ #
class BaselineModel:
    """r̂(u,i) = μ + b_u + b_i   (biases = damped deviations from the mean)."""
    name = "Bias baseline (μ + user bias + item bias)"

    def fit(self, df, attractions_df=None, damping=10):
        self.mu = float(df["rating"].mean())
        ig = df.groupby("attraction_id")["rating"]
        self.b_i = {int(i): (s - c * self.mu) / (c + damping)
                    for i, s, c in zip(ig.sum().index, ig.sum(), ig.count())}
        # user bias measured against mu + b_i
        df = df.copy()
        df["resid"] = df["rating"] - df["attraction_id"].map(self.b_i).fillna(0) - self.mu
        ug = df.groupby("user_id")["resid"]
        self.b_u = {int(u): s / (c + damping)
                    for u, s, c in zip(ug.sum().index, ug.sum(), ug.count())}
        self.items = np.array(sorted(self.b_i))
        return self

    def predict(self, user_id, item_id):
        return self.mu + self.b_u.get(int(user_id), 0.0) + self.b_i.get(int(item_id), 0.0)

    def predict_from_ratings(self, item_id, ratings):
        if ratings:
            known = [self.mu + self.b_i.get(int(i), 0.0) for i in ratings]
            b_u = float(np.mean([r for r in ratings.values()]) - np.mean(known))
        else:
            b_u = 0.0
        return self.mu + b_u + self.b_i.get(int(item_id), 0.0)

    def recommend(self, ratings, top_n=5, candidate_items=None):
        cands = candidate_items if candidate_items is not None else self.items
        scored = [(int(i), self.predict_from_ratings(i, ratings))
                  for i in cands if int(i) not in ratings]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ============================================================================ #
#  3a. PERSONALIZED - ITEM-ITEM collaborative filtering
# ============================================================================ #
class ItemItemCF:
    """
    'People who liked the places you liked also liked ...'

    Similarity between ITEMS (columns of the matrix), using cosine or Pearson.
    Prediction = similarity-weighted average of the user's OWN ratings of the
    k most similar items.
    """

    def __init__(self, similarity="cosine"):
        self.similarity = similarity
        self.name = f"Item-item CF ({similarity})"

    def fit(self, df, attractions_df=None):
        M, self.users, self.items, self.u_idx, self.i_idx = build_matrix(df)
        self.item_mean = np.nanmean(M, axis=0)
        # adjusted-cosine variant: center each USER's row first for cosine
        if self.similarity == "cosine":
            centered = M - np.nanmean(M, axis=1, keepdims=True)
            self.S = cosine_columns(centered)
        else:
            self.S = pearson_columns(M)
        self.M = M
        self.global_mean = float(np.nanmean(M))
        return self

    # ---- core: predict an item from a dict of the user's ratings ---------- #
    def predict_from_ratings(self, item_id, ratings):
        j = self.i_idx.get(int(item_id))
        if j is None or not ratings:
            return self.global_mean
        sims, vals = [], []
        for i, r in ratings.items():
            k = self.i_idx.get(int(i))
            if k is None:
                continue
            s = self.S[j, k]
            if s > 0:
                sims.append(s)
                vals.append(r - self.item_mean[k])
        if not sims:
            return float(self.item_mean[j])
        sims, vals = np.array(sims), np.array(vals)
        if len(sims) > K_NEIGHBOURS:
            top = np.argsort(sims)[-K_NEIGHBOURS:]
            sims, vals = sims[top], vals[top]
        return float(self.item_mean[j] + (sims @ vals) / sims.sum())

    def predict(self, user_id, item_id):
        u = self.u_idx.get(int(user_id))
        if u is None:
            return self.global_mean
        row = self.M[u]
        ratings = {int(self.items[k]): row[k] for k in range(len(self.items))
                   if not np.isnan(row[k])}
        return self.predict_from_ratings(item_id, ratings)

    def recommend(self, ratings, top_n=5, candidate_items=None):
        cands = candidate_items if candidate_items is not None else self.items
        scored = [(int(i), self.predict_from_ratings(i, ratings))
                  for i in cands if int(i) not in ratings]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ============================================================================ #
#  3b. PERSONALIZED - USER-USER collaborative filtering
# ============================================================================ #
class UserUserCF:
    """
    'Visitors with taste like yours liked ...'

    Similarity between USERS (rows). Prediction = the user's mean plus the
    similarity-weighted average of the k most similar users' (mean-centered)
    ratings of the target item.
    """

    def __init__(self, similarity="cosine"):
        self.similarity = similarity
        self.name = f"User-user CF ({similarity})"

    def fit(self, df, attractions_df=None):
        M, self.users, self.items, self.u_idx, self.i_idx = build_matrix(df)
        self.M = M
        self.user_mean = np.nanmean(M, axis=1)
        self.item_mean = np.nanmean(M, axis=0)
        self.global_mean = float(np.nanmean(M))
        self.centered = np.nan_to_num(M - self.user_mean[:, None], nan=0.0)
        # similarity between users = columns of the TRANSPOSED matrix
        if self.similarity == "cosine":
            self.S = cosine_columns(self.centered.T)
        else:
            self.S = pearson_columns(M.T)
        return self

    # ---- similarity of a NEW ratings-dict against all training users ------ #
    def _sims_for_new_user(self, ratings):
        v = np.zeros(len(self.items))
        mask = np.zeros(len(self.items), dtype=bool)
        for i, r in ratings.items():
            k = self.i_idx.get(int(i))
            if k is not None:
                v[k], mask[k] = r, True
        if not mask.any():
            return None, None
        u_mean = v[mask].mean()
        vc = np.where(mask, v - u_mean, 0.0)
        if self.similarity == "cosine":
            norms = np.linalg.norm(self.centered, axis=1) * (np.linalg.norm(vc) + 1e-12)
            norms[norms == 0] = 1e-12
            sims = (self.centered @ vc) / norms
        else:
            sims = np.zeros(len(self.users))
            for uu in range(len(self.users)):
                co = mask & ~np.isnan(self.M[uu])
                if co.sum() < MIN_CORATED:
                    continue
                x, y = v[co], self.M[uu, co]
                xc, yc = x - x.mean(), y - y.mean()
                denom = np.linalg.norm(xc) * np.linalg.norm(yc)
                if denom > 1e-12:
                    sims[uu] = float(xc @ yc / denom)
        return sims, u_mean

    def predict_from_ratings(self, item_id, ratings):
        j = self.i_idx.get(int(item_id))
        if j is None:
            return self.global_mean
        sims, u_mean = self._sims_for_new_user(ratings)
        if sims is None:
            return float(self.item_mean[j])
        rated_j = ~np.isnan(self.M[:, j])
        sims = np.where(rated_j, sims, 0.0)
        pos = sims > 0
        if not pos.any():
            return float(self.item_mean[j])
        idx = np.argsort(sims)[-K_NEIGHBOURS:]
        idx = idx[sims[idx] > 0]
        num = float(sims[idx] @ (self.M[idx, j] - self.user_mean[idx]))
        den = float(np.abs(sims[idx]).sum())
        return float(u_mean + num / den) if den > 0 else float(self.item_mean[j])

    def predict(self, user_id, item_id):
        u = self.u_idx.get(int(user_id))
        if u is None:
            return self.global_mean
        row = self.M[u]
        ratings = {int(self.items[k]): row[k] for k in range(len(self.items))
                   if not np.isnan(row[k])}
        return self.predict_from_ratings(item_id, ratings)

    def recommend(self, ratings, top_n=5, candidate_items=None):
        cands = candidate_items if candidate_items is not None else self.items
        scored = [(int(i), self.predict_from_ratings(i, ratings))
                  for i in cands if int(i) not in ratings]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ============================================================================ #
#  4. BONUS - matrix factorization (SGD)
# ============================================================================ #
class MatrixFactorization:
    """r̂(u,i) = μ + b_u + b_i + p_u · q_i, trained with SGD."""
    name = "Matrix factorization (SGD)"

    def __init__(self, n_factors=8, n_epochs=60, lr=0.01, reg=0.05, seed=42):
        self.f, self.epochs, self.lr, self.reg, self.seed = n_factors, n_epochs, lr, reg, seed

    def fit(self, df, attractions_df=None):
        rng = np.random.default_rng(self.seed)
        self.users = np.sort(df["user_id"].unique())
        self.items = np.sort(df["attraction_id"].unique())
        self.u_idx = {u: k for k, u in enumerate(self.users)}
        self.i_idx = {i: k for k, i in enumerate(self.items)}
        nU, nI = len(self.users), len(self.items)
        self.mu = float(df["rating"].mean())
        self.bu, self.bi = np.zeros(nU), np.zeros(nI)
        self.P = rng.normal(0, .1, (nU, self.f))
        self.Q = rng.normal(0, .1, (nI, self.f))
        trip = list(zip(df["user_id"].map(self.u_idx), df["attraction_id"].map(self.i_idx),
                        df["rating"].astype(float)))
        for _ in range(self.epochs):
            rng.shuffle(trip)
            for u, i, r in trip:
                e = r - (self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])
                self.bu[u] += self.lr * (e - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (e * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (e * pu - self.reg * self.Q[i])
        return self

    def predict(self, user_id, item_id):
        u, i = self.u_idx.get(int(user_id)), self.i_idx.get(int(item_id))
        if u is None or i is None:
            return self.mu
        return float(self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])

    def _fold_in(self, ratings):
        """Ridge-regression a latent vector for a NEW user from their ratings."""
        idx, y = [], []
        for i, r in ratings.items():
            k = self.i_idx.get(int(i))
            if k is not None:
                idx.append(k)
                y.append(r - self.mu - self.bi[k])
        if not idx:
            return np.zeros(self.f), 0.0
        Q = self.Q[idx]
        y = np.array(y)
        b_u = float(y.mean()) * 0.5
        A = Q.T @ Q + self.reg * 10 * np.eye(self.f)
        p_u = np.linalg.solve(A, Q.T @ (y - b_u))
        return p_u, b_u

    def predict_from_ratings(self, item_id, ratings):
        k = self.i_idx.get(int(item_id))
        if k is None:
            return self.mu
        p_u, b_u = self._fold_in(ratings)
        return float(self.mu + b_u + self.bi[k] + p_u @ self.Q[k])

    def recommend(self, ratings, top_n=5, candidate_items=None):
        cands = candidate_items if candidate_items is not None else self.items
        p_u, b_u = self._fold_in(ratings)
        scored = []
        for i in cands:
            if int(i) in ratings:
                continue
            k = self.i_idx[int(i)]
            scored.append((int(i), float(self.mu + b_u + self.bi[k] + p_u @ self.Q[k])))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ============================================================================ #
#  5. Content-based filtering (TF-IDF, from scratch)
# ============================================================================ #
class ContentBasedRecommender:
    """
    TF-IDF content-based filtering.

    Text document per attraction = name + category + description.
    Similarity between items = cosine of their TF-IDF vectors.
    score(unrated_item) = Σ sim(item, rated_i) * rating_i  /  Σ |sim|
    The result is on the same scale as user ratings, so it blends
    directly with CF scores.
    """
    name = "Content-based (TF-IDF)"

    _STOPWORDS = frozenset({
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
        "is", "its", "with", "that", "this", "as", "by", "from", "was", "are",
        "be", "it", "also", "has", "have", "been", "were", "their", "they",
        "into", "more", "which", "not", "but", "so", "if", "one", "two",
    })

    def fit(self, df, attractions_df=None):
        self.global_mean = float(df["rating"].mean())
        self.items = np.array(sorted(df["attraction_id"].unique()))
        self._ready = False
        if attractions_df is not None and not attractions_df.empty:
            self._build_tfidf(attractions_df)
            self._ready = True
        return self

    def _tokenize(self, text):
        return [w for w in re.findall(r"\b[a-z]{2,}\b", text.lower())
                if w not in self._STOPWORDS]

    def _build_tfidf(self, attractions_df):
        item_ids, docs = [], []
        for _, row in attractions_df.iterrows():
            item_ids.append(int(row["attraction_id"]))
            text = (f"{row.get('name', '')} {row.get('category', '')} "
                    f"{row.get('description', '')}")
            docs.append(self._tokenize(text))

        vocab = sorted({w for doc in docs for w in doc})
        vocab_idx = {w: k for k, w in enumerate(vocab)}
        nI, nT = len(item_ids), len(vocab)

        tf = np.zeros((nI, nT))
        for idx, tokens in enumerate(docs):
            if not tokens:
                continue
            for w in tokens:
                tf[idx, vocab_idx[w]] += 1
            tf[idx] /= len(tokens)

        df_cnt = (tf > 0).sum(axis=0)
        idf = np.log((nI + 1) / (df_cnt + 1)) + 1.0
        tfidf = tf * idf

        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        tfidf /= norms

        self._item_ids = item_ids
        self._item_idx = {iid: k for k, iid in enumerate(item_ids)}
        self._tfidf = tfidf

    def _cosine(self, i, j):
        ii = self._item_idx.get(int(i))
        jj = self._item_idx.get(int(j))
        if ii is None or jj is None:
            return 0.0
        return float(self._tfidf[ii] @ self._tfidf[jj])

    def predict_from_ratings(self, item_id, ratings):
        if not self._ready or not ratings:
            return self.global_mean
        sims = np.array([self._cosine(item_id, i) for i in ratings])
        vals = np.array(list(ratings.values()), dtype=float)
        denom = np.abs(sims).sum()
        if denom < 1e-12:
            return self.global_mean
        return float(np.clip(sims @ vals / denom, 1.0, 5.0))

    def predict(self, user_id, item_id):
        return self.global_mean

    def recommend(self, ratings, top_n=5, candidate_items=None):
        cands = candidate_items if candidate_items is not None else self.items
        scored = [(int(i), self.predict_from_ratings(i, ratings))
                  for i in cands if int(i) not in ratings]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ============================================================================ #
#  6. Matrix Factorization via Alternating Least Squares (ALS)
# ============================================================================ #
class ALSRecommender:
    """
    Matrix Factorization via Alternating Least Squares.

    More numerically stable than SGD on sparse data: each sub-problem is a
    ridge-regression with a closed-form solution, so no learning rate needed.
    r̂(u,i) = μ + p_u · q_i.  New users are folded in with ridge regression.
    """
    name = "Matrix factorization (ALS)"

    def __init__(self, n_factors=8, n_iters=20, reg=0.1, seed=42):
        self.f, self.n_iters, self.reg, self.seed = n_factors, n_iters, reg, seed

    def fit(self, df, attractions_df=None):
        rng = np.random.default_rng(self.seed)
        self.users = np.sort(df["user_id"].unique())
        self.items = np.sort(df["attraction_id"].unique())
        self.u_idx = {u: k for k, u in enumerate(self.users)}
        self.i_idx = {i: k for k, i in enumerate(self.items)}
        nU, nI = len(self.users), len(self.items)
        self.mu = float(df["rating"].mean())

        R = np.zeros((nU, nI))
        mask = np.zeros((nU, nI), dtype=bool)
        for u, i, r in zip(df["user_id"].map(self.u_idx),
                           df["attraction_id"].map(self.i_idx),
                           df["rating"].astype(float)):
            R[int(u), int(i)] = r - self.mu
            mask[int(u), int(i)] = True

        P = rng.normal(0, 0.1, (nU, self.f))
        Q = rng.normal(0, 0.1, (nI, self.f))
        reg_I = self.reg * np.eye(self.f)

        for _ in range(self.n_iters):
            for u in range(nU):
                idx = np.where(mask[u])[0]
                if len(idx) == 0:
                    continue
                Qi = Q[idx]
                P[u] = np.linalg.solve(Qi.T @ Qi + reg_I, Qi.T @ R[u, idx])
            for i in range(nI):
                idx = np.where(mask[:, i])[0]
                if len(idx) == 0:
                    continue
                Pu = P[idx]
                Q[i] = np.linalg.solve(Pu.T @ Pu + reg_I, Pu.T @ R[idx, i])

        self.P, self.Q = P, Q
        return self

    def predict(self, user_id, item_id):
        u = self.u_idx.get(int(user_id))
        i = self.i_idx.get(int(item_id))
        if u is None or i is None:
            return self.mu
        return float(self.mu + self.P[u] @ self.Q[i])

    def _fold_in(self, ratings):
        idx, y = [], []
        for i, r in ratings.items():
            k = self.i_idx.get(int(i))
            if k is not None:
                idx.append(k)
                y.append(r - self.mu)
        if not idx:
            return np.zeros(self.f)
        Qi = self.Q[idx]
        y = np.array(y)
        return np.linalg.solve(Qi.T @ Qi + self.reg * np.eye(self.f), Qi.T @ y)

    def predict_from_ratings(self, item_id, ratings):
        k = self.i_idx.get(int(item_id))
        if k is None:
            return self.mu
        return float(np.clip(self.mu + self._fold_in(ratings) @ self.Q[k], 1.0, 5.0))

    def recommend(self, ratings, top_n=5, candidate_items=None):
        cands = candidate_items if candidate_items is not None else self.items
        p_u = self._fold_in(ratings)
        scored = [(int(i), float(self.mu + p_u @ self.Q[self.i_idx[int(i)]]))
                  for i in cands
                  if int(i) not in ratings and int(i) in self.i_idx]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ============================================================================ #
#  7. Hybrid recommender (CF + content-based, weighted blend)
# ============================================================================ #
class HybridRecommender:
    """
    Weighted blend of a CF model and the content-based model.

    score = alpha * CF_score + (1-alpha) * CB_score

    Both constituent models produce scores on the [1,5] rating scale, so no
    normalisation is needed before blending.  The CF component handles taste
    matching from neighbour behaviour; the CB component adds a signal from
    attraction text similarity, helping when the neighbour graph is sparse.
    """

    def __init__(self, cf_cls, cf_kwargs=None, alpha=0.65):
        self.cf_cls = cf_cls
        self.cf_kwargs = cf_kwargs or {}
        self.alpha = alpha
        _cf_tmp = cf_cls(**self.cf_kwargs)
        self.name = f"Hybrid ({_cf_tmp.name} + content, α={alpha})"

    def fit(self, df, attractions_df=None):
        self.cf = self.cf_cls(**self.cf_kwargs).fit(df)
        self.cb = ContentBasedRecommender().fit(df, attractions_df)
        self.items = self.cf.items
        self.i_idx = getattr(self.cf, "i_idx", {})
        self.global_mean = float(getattr(self.cf, "global_mean",
                                         df["rating"].mean()))
        return self

    def predict_from_ratings(self, item_id, ratings):
        cf = self.cf.predict_from_ratings(item_id, ratings)
        cb = self.cb.predict_from_ratings(item_id, ratings)
        return float(np.clip(self.alpha * cf + (1 - self.alpha) * cb, 1.0, 5.0))

    def predict(self, user_id, item_id):
        return self.cf.predict(user_id, item_id)

    def recommend(self, ratings, top_n=5, candidate_items=None):
        cands = candidate_items if candidate_items is not None else self.items
        scored = [(int(i), self.predict_from_ratings(i, ratings))
                  for i in cands if int(i) not in ratings]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ============================================================================ #
#  registry - every algorithm the app can serve
# ============================================================================ #
ALL_MODELS = {
    "pop_mean":        (MeanRecommender, {},
                        {"label": "Mean ratings",          "family": "Non-personalized",
                         "personalized": False,
                         "blurb": "Ranks places by their (damped) average rating - the same list for everyone."}),
    "baseline":        (BaselineModel, {},
                        {"label": "Bias baseline",          "family": "Non-personalized",
                         "personalized": False,
                         "blurb": "Global mean + user bias + item bias. Strong reference point for the error metrics."}),
    "user_cf_cosine":  (UserUserCF, {"similarity": "cosine"},
                        {"label": "User-user CF · cosine",  "family": "User-user CF",
                         "personalized": True,
                         "blurb": "Finds visitors whose tastes are similar to yours (cosine on mean-centered ratings) and recommends what they loved."}),
    "user_cf_pearson": (UserUserCF, {"similarity": "pearson"},
                        {"label": "User-user CF · Pearson", "family": "User-user CF",
                         "personalized": True,
                         "blurb": "Same idea, but neighbours are matched with Pearson correlation on co-rated places."}),
    "item_cf_cosine":  (ItemItemCF, {"similarity": "cosine"},
                        {"label": "Item-item CF · cosine",  "family": "Item-item CF",
                         "personalized": True,
                         "blurb": "Scores a place by how similar it is to the places YOU already rated highly (adjusted cosine)."}),
    "item_cf_pearson": (ItemItemCF, {"similarity": "pearson"},
                        {"label": "Item-item CF · Pearson", "family": "Item-item CF",
                         "personalized": True,
                         "blurb": "Item-item neighbours matched with Pearson correlation on co-rated entries."}),
    "mf_sgd":          (MatrixFactorization, {},
                        {"label": "Matrix factorization (SGD)", "family": "Latent factors",
                         "personalized": True,
                         "blurb": "Learns hidden taste dimensions with SGD; new users are folded in with ridge regression."}),
    "mf_als":          (ALSRecommender, {},
                        {"label": "Matrix factorization (ALS)", "family": "Latent factors",
                         "personalized": True,
                         "blurb": "ALS solves for user/item factors via ridge regression, more numerically stable than SGD on sparse data."}),
    "content_based":   (ContentBasedRecommender, {},
                        {"label": "Content-based (TF-IDF)",    "family": "Content-based",
                         "personalized": True,
                         "blurb": "Scores each place by its text similarity (name, category, description) to the places you rated highly."}),
    "hybrid":          (HybridRecommender,
                        {"cf_cls": ItemItemCF, "cf_kwargs": {"similarity": "cosine"}, "alpha": 0.65},
                        {"label": "Hybrid (Item-item + content)", "family": "Hybrid",
                         "personalized": True,
                         "blurb": "Blends item-item CF (65%) with TF-IDF content similarity (35%) for robust recommendations even with few ratings."}),
}
