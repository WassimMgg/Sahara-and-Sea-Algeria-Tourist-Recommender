"""
panel/views.py
==============
The hand-built admin panel (served at /admin/). No django.contrib.admin —
every page here is a plain staff-only Django view rendering the templates
in frontend/templates/panel/.

Sections:
  * auth            - branded staff login / logout
  * dashboard       - KPI cards + charts + activity feed
  * attractions     - searchable/filterable CRUD + CSV export
  * visitors        - searchable/filterable CRUD + CSV export
  * ratings         - filterable list, inline star editing, CRUD, CSV export
  * users           - account management (create, edit, quick toggles, delete)
  * recommender     - inspect metrics and switch the serving algorithm
"""

import csv
import json
import re
from functools import wraps

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from api import services
from api.models import Attraction, Rating, Visitor

from .forms import (AttractionForm, RatingForm, UserCreateForm, UserEditForm,
                    VisitorForm)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def staff_required(view):
    """Allow active staff only; everyone else is sent to the panel login."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated and request.user.is_active and request.user.is_staff:
            return view(request, *args, **kwargs)
        return redirect(f"{reverse('panel:login')}?next={request.path}")
    return wrapper


def paginate(request, queryset, per_page):
    page = Paginator(queryset, per_page).get_page(request.GET.get("page"))
    page.elided = page.paginator.get_elided_page_range(
        page.number, on_each_side=2, on_ends=1)
    params = request.GET.copy()
    params.pop("page", None)
    return page, params.urlencode()


def export_csv(queryset, model, filename):
    fields = [f.name for f in model._meta.fields]
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = f"attachment; filename={filename}.csv"
    writer = csv.writer(resp)
    writer.writerow(fields)
    for obj in queryset:
        writer.writerow([getattr(obj, f) for f in fields])
    return resp


def thumb_url(image_url, width=160):
    if not image_url:
        return ""
    if "width=" in image_url:
        return re.sub(r"width=\d+", f"width={width}", image_url)
    sep = "&" if "?" in image_url else "?"
    return f"{image_url}{sep}width={width}"


def star_distribution(qs):
    """{1..5: count} with float ratings rounded to the nearest star."""
    dist = {s: 0 for s in range(1, 6)}
    for (r,) in qs.values_list("rating"):
        dist[min(5, max(1, round(r)))] += 1
    return dist


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
def login_view(request):
    if request.user.is_authenticated and request.user.is_staff:
        return redirect("panel:dashboard")
    error = ""
    if request.method == "POST":
        user = authenticate(request,
                            username=request.POST.get("username", "").strip(),
                            password=request.POST.get("password", ""))
        if user is None:
            error = "Invalid username or password."
        elif not user.is_staff:
            error = "This account has no staff access to the panel."
        else:
            login(request, user)
            return redirect(request.GET.get("next") or "panel:dashboard")
    return render(request, "panel/login.html", {"error": error})


@require_POST
def logout_view(request):
    logout(request)
    return redirect("panel:login")


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
@staff_required
def dashboard(request):
    attractions = list(
        Attraction.objects.annotate(n=Count("ratings"), avg=Avg("ratings__rating")))
    rated = [a for a in attractions if a.n]
    top_rated = sorted(rated, key=lambda a: (a.avg or 0, a.n), reverse=True)[:5]
    most_rated = sorted(attractions, key=lambda a: a.n, reverse=True)[:5]

    dist = star_distribution(Rating.objects.all())
    by_type = (Attraction.objects.values("place_type")
               .annotate(avg=Avg("ratings__rating"), n=Count("ratings"))
               .order_by("-avg"))
    type_rows = [(t["place_type"], round(t["avg"], 2) if t["avg"] else 0, t["n"])
                 for t in by_type]

    overall = Rating.objects.aggregate(avg=Avg("rating"))["avg"]
    algos = services.available_algorithms()
    active_algo = next(a for a in algos if a["active"])

    context = {
        "nav": "dashboard",
        "stat_attractions": len(attractions),
        "stat_visitors": Visitor.objects.count(),
        "stat_accounts": User.objects.count(),
        "stat_ratings": Rating.objects.count(),
        "stat_ratings_app": Rating.objects.filter(account__isnull=False).count(),
        "stat_ratings_hist": Rating.objects.filter(visitor__isnull=False).count(),
        "stat_avg": round(overall, 2) if overall else 0,
        "active_algo": active_algo,
        "top_rated": [(a, round(a.avg, 2)) for a in top_rated],
        "most_rated": most_rated,
        "recent_ratings": (Rating.objects.filter(account__isnull=False)
                           .select_related("account", "attraction")
                           .order_by("-id")[:8]),
        "chart_dist": json.dumps({
            "labels": [f"{s} star" for s in range(1, 6)],
            "counts": [dist[s] for s in range(1, 6)],
        }),
        "chart_types": json.dumps({
            "labels": [t[0] for t in type_rows],
            "avgs": [t[1] for t in type_rows],
            "counts": [t[2] for t in type_rows],
        }),
    }
    return render(request, "panel/dashboard.html", context)


# --------------------------------------------------------------------------- #
# attractions
# --------------------------------------------------------------------------- #
def filtered_attractions(request):
    qs = Attraction.objects.annotate(n=Count("ratings"), avg=Avg("ratings__rating"))
    q = request.GET.get("q", "").strip()
    if q:
        qs = qs.filter(Q(name__icontains=q) | Q(city__icontains=q) |
                       Q(region__icontains=q) | Q(description__icontains=q))
    if request.GET.get("region"):
        qs = qs.filter(region=request.GET["region"])
    if request.GET.get("type"):
        qs = qs.filter(place_type=request.GET["type"])
    sort = request.GET.get("sort", "id")
    allowed = {"id", "name", "city", "-n", "-avg"}
    return qs.order_by(sort if sort in allowed else "id"), q, sort


@staff_required
def attraction_list(request):
    qs, q, sort = filtered_attractions(request)
    if request.GET.get("export") == "csv":
        return export_csv(qs, Attraction, "attractions")
    page, querystring = paginate(request, qs, 12)
    for a in page:
        a.thumb = thumb_url(a.image_url)
    context = {
        "nav": "attractions", "page": page, "querystring": querystring,
        "q": q, "sort": sort,
        "regions": sorted(set(Attraction.objects.values_list("region", flat=True))),
        "types": services.place_types(),
        "sel_region": request.GET.get("region", ""),
        "sel_type": request.GET.get("type", ""),
        "total": qs.count(),
    }
    return render(request, "panel/attraction_list.html", context)


@staff_required
def attraction_form(request, pk=None):
    obj = get_object_or_404(Attraction, pk=pk) if pk else None
    if request.method == "POST":
        form = AttractionForm(request.POST, instance=obj)
        if form.is_valid():
            att = form.save()
            messages.success(request, f"Attraction “{att.name}” saved.")
            return redirect("panel:attractions")
    else:
        form = AttractionForm(instance=obj)

    breakdown = []
    recent = []
    if obj:
        dist = star_distribution(obj.ratings.all())
        total = sum(dist.values())
        breakdown = [(s, dist[s], round(100 * dist[s] / total) if total else 0)
                     for s in (5, 4, 3, 2, 1)]
        recent = (obj.ratings.select_related("account", "visitor")
                  .order_by("-id")[:6])
        obj.thumb = thumb_url(obj.image_url, 480)
        stats = obj.ratings.aggregate(n=Count("id"), avg=Avg("rating"))
        obj.n, obj.avg = stats["n"], round(stats["avg"], 2) if stats["avg"] else None

    context = {
        "nav": "attractions", "form": form, "obj": obj,
        "breakdown": breakdown, "recent": recent,
        "types": services.place_types(),
    }
    return render(request, "panel/attraction_form.html", context)


@staff_required
@require_POST
def attraction_delete(request, pk):
    obj = get_object_or_404(Attraction, pk=pk)
    name = obj.name
    obj.delete()
    messages.success(request, f"Attraction “{name}” and its ratings were deleted.")
    return redirect("panel:attractions")


# --------------------------------------------------------------------------- #
# visitors
# --------------------------------------------------------------------------- #
def filtered_visitors(request):
    qs = Visitor.objects.annotate(n=Count("ratings"))
    q = request.GET.get("q", "").strip()
    if q:
        qs = qs.filter(Q(name__icontains=q) | Q(home_country__icontains=q))
    if request.GET.get("gender"):
        qs = qs.filter(gender=request.GET["gender"])
    if request.GET.get("country"):
        qs = qs.filter(home_country=request.GET["country"])
    return qs.order_by("id"), q


@staff_required
def visitor_list(request):
    qs, q = filtered_visitors(request)
    if request.GET.get("export") == "csv":
        return export_csv(qs, Visitor, "visitors")
    page, querystring = paginate(request, qs, 20)
    context = {
        "nav": "visitors", "page": page, "querystring": querystring, "q": q,
        "genders": sorted({g for g in Visitor.objects.values_list("gender", flat=True) if g}),
        "countries": sorted({c for c in Visitor.objects.values_list("home_country", flat=True) if c}),
        "sel_gender": request.GET.get("gender", ""),
        "sel_country": request.GET.get("country", ""),
        "total": qs.count(),
    }
    return render(request, "panel/visitor_list.html", context)


@staff_required
def visitor_form(request, pk=None):
    obj = get_object_or_404(Visitor, pk=pk) if pk else None
    if request.method == "POST":
        form = VisitorForm(request.POST, instance=obj)
        if form.is_valid():
            visitor = form.save()
            messages.success(request, f"Visitor “{visitor.name}” saved.")
            return redirect("panel:visitors")
    else:
        form = VisitorForm(instance=obj)
    recent = obj.ratings.select_related("attraction").order_by("-id")[:8] if obj else []
    return render(request, "panel/visitor_form.html",
                  {"nav": "visitors", "form": form, "obj": obj, "recent": recent})


@staff_required
@require_POST
def visitor_delete(request, pk):
    obj = get_object_or_404(Visitor, pk=pk)
    name = obj.name
    obj.delete()
    messages.success(request, f"Visitor “{name}” and their ratings were deleted.")
    return redirect("panel:visitors")


# --------------------------------------------------------------------------- #
# ratings
# --------------------------------------------------------------------------- #
def filtered_ratings(request):
    qs = Rating.objects.select_related("attraction", "visitor", "account")
    q = request.GET.get("q", "").strip()
    if q:
        qs = qs.filter(Q(attraction__name__icontains=q) |
                       Q(visitor__name__icontains=q) |
                       Q(account__username__icontains=q))
    source = request.GET.get("source", "")
    if source == "app":
        qs = qs.filter(account__isnull=False)
    elif source == "hist":
        qs = qs.filter(visitor__isnull=False)
    stars = request.GET.get("stars", "")
    if stars.isdigit():
        s = int(stars)
        qs = qs.filter(rating__gte=s - 0.5, rating__lt=s + 0.5)
    if request.GET.get("type"):
        qs = qs.filter(attraction__place_type=request.GET["type"])
    return qs.order_by("-id"), q


@staff_required
def rating_list(request):
    qs, q = filtered_ratings(request)
    if request.GET.get("export") == "csv":
        return export_csv(qs, Rating, "ratings")
    page, querystring = paginate(request, qs, 25)
    context = {
        "nav": "ratings", "page": page, "querystring": querystring, "q": q,
        "types": services.place_types(),
        "sel_source": request.GET.get("source", ""),
        "sel_stars": request.GET.get("stars", ""),
        "sel_type": request.GET.get("type", ""),
        "total": qs.count(),
    }
    return render(request, "panel/rating_list.html", context)


@staff_required
def rating_form(request, pk=None):
    obj = get_object_or_404(Rating, pk=pk) if pk else None
    if request.method == "POST":
        form = RatingForm(request.POST, instance=obj)
        if form.is_valid():
            form.save()
            messages.success(request, "Rating saved.")
            return redirect("panel:ratings")
    else:
        form = RatingForm(instance=obj)
    return render(request, "panel/rating_form.html",
                  {"nav": "ratings", "form": form, "obj": obj})


@staff_required
@require_POST
def rating_delete(request, pk):
    get_object_or_404(Rating, pk=pk).delete()
    messages.success(request, "Rating deleted.")
    return redirect("panel:ratings")


@staff_required
@require_POST
def rating_inline(request, pk):
    """Inline star edit from the list page (fetch + JSON)."""
    obj = get_object_or_404(Rating, pk=pk)
    try:
        value = float(json.loads(request.body)["rating"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return JsonResponse({"error": "bad payload"}, status=400)
    if not 1 <= value <= 5:
        return JsonResponse({"error": "rating must be 1-5"}, status=400)
    obj.rating = value
    obj.save(update_fields=["rating"])
    return JsonResponse({"ok": True, "rating": value})


# --------------------------------------------------------------------------- #
# users
# --------------------------------------------------------------------------- #
@staff_required
def user_list(request):
    qs = User.objects.annotate(n=Count("ratings"))
    q = request.GET.get("q", "").strip()
    if q:
        qs = qs.filter(Q(username__icontains=q) | Q(email__icontains=q))
    role = request.GET.get("role", "")
    if role == "staff":
        qs = qs.filter(is_staff=True)
    elif role == "member":
        qs = qs.filter(is_staff=False)
    qs = qs.order_by("-date_joined")
    if request.GET.get("export") == "csv":
        return export_csv(qs, User, "users")
    page, querystring = paginate(request, qs, 20)
    context = {
        "nav": "users", "page": page, "querystring": querystring, "q": q,
        "sel_role": role, "total": qs.count(),
    }
    return render(request, "panel/user_list.html", context)


@staff_required
def user_form(request, pk=None):
    obj = get_object_or_404(User, pk=pk) if pk else None
    form_class = UserEditForm if obj else UserCreateForm
    if request.method == "POST":
        form = form_class(request.POST, instance=obj)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"Account “{user.username}” saved.")
            return redirect("panel:users")
    else:
        form = form_class(instance=obj)
    recent = obj.ratings.select_related("attraction").order_by("-id")[:8] if obj else []
    return render(request, "panel/user_form.html",
                  {"nav": "users", "form": form, "obj": obj, "recent": recent})


@staff_required
@require_POST
def user_toggle(request, pk):
    """Quick toggles from the list: active / staff."""
    obj = get_object_or_404(User, pk=pk)
    field = request.POST.get("field")
    if field not in ("is_active", "is_staff"):
        return redirect("panel:users")
    if obj == request.user:
        messages.error(request, "You cannot change your own access flags.")
        return redirect("panel:users")
    setattr(obj, field, not getattr(obj, field))
    obj.save(update_fields=[field])
    label = "active" if field == "is_active" else "staff"
    state = "now" if getattr(obj, field) else "no longer"
    messages.success(request, f"“{obj.username}” is {state} {label}.")
    return redirect("panel:users")


@staff_required
@require_POST
def user_delete(request, pk):
    obj = get_object_or_404(User, pk=pk)
    if obj == request.user:
        messages.error(request, "You cannot delete your own account.")
        return redirect("panel:users")
    name = obj.username
    obj.delete()
    messages.success(request, f"Account “{name}” and its ratings were deleted.")
    return redirect("panel:users")


# --------------------------------------------------------------------------- #
# recommender engine
# --------------------------------------------------------------------------- #
@staff_required
def recommender(request):
    if request.method == "POST":
        if request.POST.get("action") == "reload":
            services.reload_models()
            messages.success(request, "Model bundle reloaded from disk.")
        else:
            key = request.POST.get("algorithm", "")
            try:
                services.set_active_algorithm(key)
                label = next(a["label"] for a in services.available_algorithms()
                             if a["key"] == key)
                messages.success(request, f"Active recommender switched to: {label}.")
            except (ValueError, StopIteration) as exc:
                messages.error(request, str(exc))
        return redirect("panel:recommender")

    algos = services.available_algorithms()
    best_rmse = min(a["rmse"] for a in algos)
    for a in algos:
        a["meter"] = round(100 * best_rmse / a["rmse"]) if a["rmse"] else 0
    context = {
        "nav": "recommender",
        "algorithms": algos,
        "active": next(a for a in algos if a["active"]),
        "best_rmse": best_rmse,
        "info": services.training_info(),
    }
    return render(request, "panel/recommender.html", context)
