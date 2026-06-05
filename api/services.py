"""
api/services.py
===============
Bridge between the trained model (ml/model.pkl) and the web app.

The model is loaded once and cached. All ratings now come from the DATABASE:
  * historical Visitor ratings  -> used for the "popular" cold-start ranking
  * the logged-in account's own ratings -> folded into the model for personal recs
"""

import sys
import pickle
import threading
from django.conf import settings

from .models import Attraction, Rating

# the pickled model references classes in ml/recommender.py
sys.path.insert(0, str(settings.ML_DIR))

_model = None
_model_name = None
_lock = threading.Lock()


def _load_model():
    global _model, _model_name
    if _model is None:
        with _lock:
            if _model is None:
                with open(settings.ML_DIR / "model.pkl", "rb") as fh:
                    payload = pickle.load(fh)
                _model = payload["model"]
                _model_name = payload["name"]
    return _model, _model_name


def model_name():
    return _load_model()[1]


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
    """The place_type of the user's highest-rated attractions (for explanations)."""
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


def recommend_for(user, top_n=5):
    model, _ = _load_model()
    ratings_map = account_ratings(user)
    attractions = {a.id: a for a in Attraction.objects.all()}

    if not ratings_map:
        # cold start -> crowd favourites
        order, avg = popularity_ranking()
        return [attraction_dict(attractions[aid], avg[aid], "Popular with visitors")
                for aid in order[:top_n] if aid in attractions]

    fav = favourite_type(ratings_map)
    out = []
    for aid, score in model.recommend(ratings_map, top_n=top_n):
        aid = int(aid)
        if aid not in attractions:
            continue
        att = attractions[aid]
        if fav and att.place_type == fav:
            reason = f"Matches your taste for {fav}"
        else:
            reason = "Visitors with similar taste enjoyed this"
        out.append(attraction_dict(att, score, reason))
    return out


def place_types():
    return sorted(set(Attraction.objects.values_list("place_type", flat=True)))
