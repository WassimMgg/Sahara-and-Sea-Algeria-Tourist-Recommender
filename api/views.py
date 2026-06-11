"""
api/views.py
============
Two kinds of views:
  * PAGE views  -> render HTML templates (home, places, search, model, about, auth)
  * API  views  -> return JSON used by the page JavaScript
"""

import json

from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import ensure_csrf_cookie

from .models import Attraction, Rating
from . import services


# ============================== PAGE VIEWS ============================== #
@ensure_csrf_cookie
def home(request):
    recs = services.recommend_for(request.user, top_n=6)
    attractions = list(Attraction.objects.all())

    def by_id(aid):
        return next((a for a in attractions if a.id == aid), attractions[0] if attractions else None)

    def wide(url, width=1600):
        if not url:
            return url
        return url.replace("width=1000", f"width={width}")

    # group attractions by place type for the "explore by type" gallery
    grouped = {}
    for a in attractions:
        grouped.setdefault(a.place_type, []).append(a)
    categories = [
        {"type": t, "count": len(items), "image_url": items[0].image_url}
        for t, items in sorted(grouped.items())
    ]

    hero_main = by_id(2)            # Tassili n'Ajjer (desert)
    hero_small = [by_id(1), by_id(5)]  # Casbah (city), Tipaza (coast)

    stats = {
        "attractions": len(attractions),
        "types": len(grouped),
        "regions": len({a.region for a in attractions}),
        "ratings": Rating.objects.filter(visitor__isnull=False).count(),
    }

    context = {
        "recommendations": recs,
        "categories": categories,
        "hero_main": hero_main,
        "hero_main_image": wide(hero_main.image_url) if hero_main else "",
        "hero_small": hero_small,
        "stats": stats,
    }
    return render(request, "home.html", context)


@ensure_csrf_cookie
def places(request):
    return render(request, "places.html", {"types": services.place_types()})


def search(request):
    return render(request, "search.html", {"types": services.place_types()})


def used_model(request):
    algos = services.available_algorithms()
    active = next(a for a in algos if a["active"])
    return render(request, "model.html", {
        "algorithms": sorted(algos, key=lambda a: a["rmse"]),
        "active": active,
        "info": services.training_info(),
    })


def about(request):
    return render(request, "about.html")


# ============================== AUTH VIEWS ============================== #
def signup_view(request):
    if request.user.is_authenticated:
        return redirect("home")
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")
        confirm = request.POST.get("confirm", "")
        if not username or not password:
            messages.error(request, "Username and password are required.")
        elif password != confirm:
            messages.error(request, "Passwords do not match.")
        elif len(password) < 6:
            messages.error(request, "Password must be at least 6 characters.")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "That username is already taken.")
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            login(request, user)
            return redirect("home")
    return render(request, "signup.html")


def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(request.GET.get("next") or "home")
        messages.error(request, "Invalid username or password.")
    return render(request, "login.html")


@require_POST
def logout_view(request):
    logout(request)
    return redirect("home")


# =============================== API VIEWS ============================== #
@require_GET
def api_attractions(request):
    q = request.GET.get("q", "").strip().lower()
    place_type = request.GET.get("type", "").strip()
    qs = Attraction.objects.all()
    if place_type and place_type.lower() != "all":
        qs = qs.filter(place_type=place_type)
    items = []
    for att in qs:
        if q and q not in (att.name + att.city + att.region + att.description).lower():
            continue
        items.append(services.attraction_dict(att))
    return JsonResponse({"attractions": items, "types": services.place_types()})


@require_GET
def api_recommendations(request):
    top_n = int(request.GET.get("n", 6))
    return JsonResponse({"recommendations": services.recommend_for(request.user, top_n=top_n)})


@require_GET
def api_my_ratings(request):
    return JsonResponse({"ratings": services.account_ratings(request.user)})


@require_POST
def api_rate(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "login_required"}, status=401)
    try:
        payload = json.loads(request.body.decode("utf-8"))
        attraction_id = int(payload["attraction_id"])
        rating = float(payload["rating"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return HttpResponseBadRequest("Expected JSON: attraction_id, rating")
    if not (1 <= rating <= 5):
        return HttpResponseBadRequest("rating must be between 1 and 5")
    if not Attraction.objects.filter(id=attraction_id).exists():
        return HttpResponseBadRequest("unknown attraction")

    Rating.objects.update_or_create(
        account=request.user, attraction_id=attraction_id,
        defaults={"rating": rating},
    )
    recs = services.recommend_for(request.user, top_n=int(payload.get("n", 6)))
    return JsonResponse({"status": "ok", "recommendations": recs})


