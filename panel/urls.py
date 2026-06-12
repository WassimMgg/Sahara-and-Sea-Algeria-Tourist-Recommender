from django.urls import path

from . import views

app_name = "panel"

urlpatterns = [
    # auth
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # dashboard
    path("", views.dashboard, name="dashboard"),

    # attractions
    path("attractions/", views.attraction_list, name="attractions"),
    path("attractions/add/", views.attraction_form, name="attraction_add"),
    path("attractions/<int:pk>/", views.attraction_form, name="attraction_edit"),
    path("attractions/<int:pk>/delete/", views.attraction_delete, name="attraction_delete"),

    # visitors
    path("visitors/", views.visitor_list, name="visitors"),
    path("visitors/add/", views.visitor_form, name="visitor_add"),
    path("visitors/<int:pk>/", views.visitor_form, name="visitor_edit"),
    path("visitors/<int:pk>/delete/", views.visitor_delete, name="visitor_delete"),

    # ratings
    path("ratings/", views.rating_list, name="ratings"),
    path("ratings/add/", views.rating_form, name="rating_add"),
    path("ratings/<int:pk>/", views.rating_form, name="rating_edit"),
    path("ratings/<int:pk>/delete/", views.rating_delete, name="rating_delete"),
    path("ratings/<int:pk>/inline/", views.rating_inline, name="rating_inline"),

    # users
    path("users/", views.user_list, name="users"),
    path("users/add/", views.user_form, name="user_add"),
    path("users/<int:pk>/", views.user_form, name="user_edit"),
    path("users/<int:pk>/toggle/", views.user_toggle, name="user_toggle"),
    path("users/<int:pk>/delete/", views.user_delete, name="user_delete"),

    # recommender engine
    path("recommender/", views.recommender, name="recommender"),
]
