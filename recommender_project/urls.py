from django.urls import include, path
from api import views

urlpatterns = [
    # hand-built admin panel
    path("admin/", include("panel.urls", namespace="panel")),

    # pages
    path("",        views.home,       name="home"),
    path("places/", views.places,     name="places"),
    path("search/", views.search,     name="search"),
    path("model/",  views.used_model, name="used_model"),
    path("about/",  views.about,      name="about"),

    # auth
    path("login/",  views.login_view,  name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),

    # api — authoritative routes live in api/urls.py
    path("api/", include("api.urls")),
]
