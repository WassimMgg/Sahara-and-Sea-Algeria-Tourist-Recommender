from django.urls import path
from . import views

urlpatterns = [
    path("attractions/",    views.api_attractions),
    path("recommendations/", views.api_recommendations),
    path("my-ratings/",     views.api_my_ratings),
    path("rate/",           views.api_rate),
]
