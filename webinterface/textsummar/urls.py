from django.contrib import admin
from django.urls import path,include
from textsummar import views

urlpatterns = [
    path('',views.index,name="index"),
    path('generate',views.generate,name="generate"),
]