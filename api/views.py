"""
api/views.py
============
Two kinds of views:
  * PAGE views  -> render HTML templates (home, places, search, model, about, auth)
  * API  views  -> return JSON used by the page JavaScript
"""

import json
from pathlib import Path

from django.conf import settings
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
    return render(request, "home.html", {"recommendations": recs})


@ensure_csrf_cookie
def places(request):
    return render(request, "places.html", {"types": services.place_types()})


def search(request):
    return render(request, "search.html", {"types": services.place_types()})


def used_model(request):
    metrics_path = Path(settings.ML_DIR) / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    rows = []
    if metrics.get("results"):
        rows = sorted(metrics["results"].items(), key=lambda kv: kv[1]["rmse"])
    return render(request, "model.html", {
        "model_name": services.model_name(),
        "metrics": metrics,
        "rows": rows,
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
