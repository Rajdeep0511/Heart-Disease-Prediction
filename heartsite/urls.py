from django.urls import include, path

urlpatterns = [path("", include("patients.urls"))]
