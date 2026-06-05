from django.urls import path
from . import views

urlpatterns = [
    path("attractions/", views.attractions),
    path("users/", views.users),
    path("new-user/", views.new_user),
    path("ratings/", views.ratings),
    path("recommendations/", views.recommendations),
    path("rate/", views.rate),
    path("model-info/", views.model_info),
]
