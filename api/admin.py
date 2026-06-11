"""
api/admin.py
============
Customised admin panel for the project.

UI:
  * branded theme (see templates/admin/base_site.html + static/admin_custom/admin.css)
  * a project DASHBOARD on the index page: stat cards, ratings-distribution chart,
    average score by place type, top/most-rated lists, recent activity, quick actions

Functionality:
  * Attraction: image thumbnails, rating count + average, quick-edit of place_type,
    a per-attraction rating breakdown + recent ratings on the edit page, CSV export
  * Visitor / Rating: filters, search, CSV export; ratings are quick-editable
  * Users / Groups management
"""

import csv
import re
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin, GroupAdmin
from django.contrib.auth.models import User, Group
from django.db.models import Avg, Count
from django.http import HttpResponse
from django.utils.html import format_html, format_html_join, mark_safe

from .models import Attraction, Visitor, Rating


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def thumb_url(image_url, width=120):
    if not image_url:
        return ""
    if "width=" in image_url:
        return re.sub(r"width=\d+", f"width={width}", image_url)
    sep = "&" if "?" in image_url else "?"
    return f"{image_url}{sep}width={width}"


@admin.action(description="Export selected to CSV")
def export_as_csv(modeladmin, request, queryset):
    """Generic CSV export usable on any model admin."""
    fields = [f.name for f in modeladmin.model._meta.fields]
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = (
        f"attachment; filename={modeladmin.model._meta.model_name}_export.csv")
    writer = csv.writer(resp)
    writer.writerow(fields)
    for obj in queryset:
        writer.writerow([getattr(obj, f) for f in fields])
    return resp


def star_bar(count, total, label):
    """One row of the CSS bar chart used in breakdowns/dashboard."""
    pct = round(100 * count / total) if total else 0
    return format_html(
        '<div class="bar-row"><span class="bar-lbl">{}</span>'
        '<span class="bar-track"><span class="bar-fill" style="width:{}%"></span></span>'
        '<span class="bar-val">{}</span></div>', label, pct, count)


def dashboard_stats():
    attractions = (Attraction.objects
                   .annotate(n=Count("ratings"), avg=Avg("ratings__rating")))
    rated = [a for a in attractions if a.n]
    top_rated = sorted(rated, key=lambda a: (a.avg or 0), reverse=True)[:5]
    most_rated = sorted(attractions, key=lambda a: a.n, reverse=True)[:5]

    # rating distribution (rounded to nearest star)
    dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for (r,) in Rating.objects.values_list("rating"):
        b = min(5, max(1, round(r)))
        dist[b] += 1
    dist_total = sum(dist.values()) or 1

    # average score by place type
    by_type = (Attraction.objects.values("place_type")
               .annotate(avg=Avg("ratings__rating"), n=Count("ratings"))
               .order_by("-avg"))

    return {
        "stat_attractions": Attraction.objects.count(),
        "stat_visitors": Visitor.objects.count(),
        "stat_accounts": User.objects.count(),
        "stat_ratings_total": Rating.objects.count(),
        "stat_ratings_app": Rating.objects.filter(account__isnull=False).count(),
        "stat_ratings_hist": Rating.objects.filter(visitor__isnull=False).count(),
        "dist_rows": [(f"{s}\u2605", dist[s], dist_total) for s in (5, 4, 3, 2, 1)],
        "type_rows": [(t["place_type"], round(t["avg"], 2) if t["avg"] else 0, t["n"])
                      for t in by_type],
        "top_rated": [(a.name, round(a.avg, 2), a.n) for a in top_rated],
        "most_rated": [(a.name, a.n) for a in most_rated],
        "recent_app_ratings": list(
            Rating.objects.filter(account__isnull=False)
            .select_related("account", "attraction").order_by("-id")[:8]),
    }


# --------------------------------------------------------------------------- #
# custom admin site
# --------------------------------------------------------------------------- #
class RecAdminSite(admin.AdminSite):
    site_header = "Sahara & Sea - Administration"
    site_title = "Sahara & Sea Admin"
    index_title = "Project administration"
    index_template = "admin/dashboard_index.html"

    def get_urls(self):
        from django.urls import path
        return [
            path("recommender/", self.admin_view(self.recommender_view),
                 name="recommender"),
        ] + super().get_urls()

    def recommender_view(self, request):
        """Admin page to inspect, compare and SWITCH the serving algorithm."""
        from django.shortcuts import render, redirect
        from django.contrib import messages
        from . import services

        if request.method == "POST":
            action = request.POST.get("action", "switch")
            if action == "reload":
                services.reload_models()
                messages.success(request, "Model bundle reloaded from disk.")
            else:
                key = request.POST.get("algorithm", "")
                try:
                    services.set_active_algorithm(key)
                    label = next(a["label"] for a in services.available_algorithms()
                                 if a["key"] == key)
                    messages.success(
                        request, f"Active recommender switched to: {label}.")
                except (ValueError, StopIteration) as exc:
                    messages.error(request, str(exc))
            return redirect("admin:recommender")

        algos = services.available_algorithms()
        best_rmse = min(a["rmse"] for a in algos)
        context = {
            **self.each_context(request),
            "title": "Recommender engine",
            "algorithms": algos,
            "active": next(a for a in algos if a["active"]),
            "best_rmse": best_rmse,
            "info": services.training_info(),
        }
        return render(request, "admin/recommender.html", context)

    def index(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context.update(dashboard_stats())
        return super().index(request, extra_context)


admin_site = RecAdminSite(name="recadmin")


# --------------------------------------------------------------------------- #
# inlines
# --------------------------------------------------------------------------- #
class RatingInline(admin.TabularInline):
    model = Rating
    extra = 0
    max_num = 0                       # display only, no add rows
    can_delete = True
    fields = ("rater_display", "rating", "visit_date")
    readonly_fields = ("rater_display",)
    verbose_name_plural = "Ratings for this attraction"

    @admin.display(description="Rated by")
    def rater_display(self, obj):
        if obj.account_id:
            return f"{obj.account.username} (app)"
        return obj.visitor.name if obj.visitor else "-"


# --------------------------------------------------------------------------- #
# model admins
# --------------------------------------------------------------------------- #
@admin.register(Attraction, site=admin_site)
class AttractionAdmin(admin.ModelAdmin):
    list_display = ("id", "preview", "name", "city", "region", "place_type",
                    "num_ratings", "avg_badge")
    list_display_links = ("name",)
    list_editable = ("place_type",)
    list_filter = ("region", "place_type")
    search_fields = ("name", "city", "region", "description")
    ordering = ("id",)
    list_per_page = 25
    save_on_top = True
    actions = [export_as_csv]
    inlines = [RatingInline]
    readonly_fields = ("big_preview", "rating_breakdown")
    fieldsets = (
        ("Basic info", {"fields": ("id", "name", "city", "region",
                                   ("category", "place_type"), "description")}),
        ("Image", {"fields": ("image_url", "big_preview")}),
        ("Ratings", {"fields": ("rating_breakdown",)}),
    )

    def get_queryset(self, request):
        return (super().get_queryset(request)
                .annotate(_n=Count("ratings"), _avg=Avg("ratings__rating")))

    @admin.display(description="")
    def preview(self, obj):
        return format_html('<img src="{}" style="height:38px;width:60px;'
                           'object-fit:cover;border-radius:4px" loading="lazy">',
                           thumb_url(obj.image_url))

    @admin.display(description="Image")
    def big_preview(self, obj):
        return format_html('<img src="{}" style="max-width:380px;border-radius:10px">',
                           thumb_url(obj.image_url, 420))

    @admin.display(description="Ratings", ordering="_n")
    def num_ratings(self, obj):
        return obj._n

    @admin.display(description="Avg", ordering="_avg")
    def avg_badge(self, obj):
        if obj._avg is None:
            return "-"
        avg = round(obj._avg, 2)
        color = "#1f6f6b" if avg >= 4 else ("#d99a2b" if avg >= 3 else "#a8472a")
        return format_html('<span style="background:{};color:#fff;padding:2px 8px;'
                           'border-radius:999px;font-weight:600">{} \u2605</span>', color, avg)

    @admin.display(description="Rating breakdown")
    def rating_breakdown(self, obj):
        if not obj.pk:
            return "Save first to see ratings."
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for (r,) in obj.ratings.values_list("rating"):
            dist[min(5, max(1, round(r)))] += 1
        total = sum(dist.values())
        if not total:
            return "No ratings yet."
        bars = format_html_join("", "{}", (
            (star_bar(dist[s], total, f"{s}\u2605"),) for s in (5, 4, 3, 2, 1)))
        return mark_safe(f'<div class="admin-bars">{bars}</div>')


@admin.register(Visitor, site=admin_site)
class VisitorAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "age", "gender", "home_country", "num_ratings")
    list_filter = ("gender", "home_country")
    search_fields = ("name", "home_country")
    ordering = ("id",)
    list_per_page = 25
    actions = [export_as_csv]
    inlines = [RatingInline]

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_n=Count("ratings"))

    @admin.display(description="Ratings", ordering="_n")
    def num_ratings(self, obj):
        return obj._n


@admin.register(Rating, site=admin_site)
class RatingAdmin(admin.ModelAdmin):
    list_display = ("id", "rater", "source", "attraction", "rating", "visit_date")
    list_editable = ("rating",)
    list_filter = ("rating", ("account", admin.EmptyFieldListFilter), "attraction__place_type")
    search_fields = ("attraction__name", "visitor__name", "account__username")
    autocomplete_fields = ("attraction",)
    ordering = ("-id",)
    list_per_page = 30
    actions = [export_as_csv]

    def get_queryset(self, request):
        return (super().get_queryset(request)
                .select_related("attraction", "visitor", "account"))

    @admin.display(description="Rated by")
    def rater(self, obj):
        if obj.account:
            return obj.account.username
        return obj.visitor.name if obj.visitor else "-"

    @admin.display(description="Source")
    def source(self, obj):
        return "App user" if obj.account_id else "Historical"


admin_site.register(User, UserAdmin)
admin_site.register(Group, GroupAdmin)
