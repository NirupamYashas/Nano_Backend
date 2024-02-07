from django.urls import path
from . import views

urlpatterns = [
    path('single-input', views.singleinputDatapost),
    path('file-input', views.fileinputDatapost)
]