from django.urls import path

from . import views

urlpatterns = [
    path('card_9cls', views.card_9cls, name='card_9cls'),
]