from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register, name="register"),
    path("login/", auth_views.LoginView.as_view(), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("my-details/", views.profile, name="profile"),
    path("check/", views.predict, name="predict"),
    path("check/<int:pk>/", views.result, name="result"),
    path("history/", views.history, name="history"),
    path("history/<int:pk>/delete/", views.delete_prediction, name="delete_prediction"),
    path("model/", views.model_info, name="model_info"),
]
