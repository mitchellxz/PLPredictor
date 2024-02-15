from django.urls import path
from . import views

urlpatterns = [
    path("index", views.index, name="index"),
    path("", views.home, name="home"),
    path("display_columns", views.display_columns, name='display_columns'),
    path("results", views.results, name="results"),
]