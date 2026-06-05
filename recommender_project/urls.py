from django.urls import path
from api import views
from api.admin import admin_site

urlpatterns = [
    # admin panel
    path("admin/", admin_site.urls),

    # pages
    path("", views.home, name="home"),
    path("places/", views.places, name="places"),
    path("search/", views.search, name="search"),
    path("model/", views.used_model, name="used_model"),
    path("about/", views.about, name="about"),
    # auth
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),
    # api
    path("api/attractions/", views.api_attractions),
    path("api/recommendations/", views.api_recommendations),
    path("api/my-ratings/", views.api_my_ratings),
    path("api/rate/", views.api_rate),
]
