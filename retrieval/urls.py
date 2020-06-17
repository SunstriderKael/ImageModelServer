from django.urls import path

from . import views

urlpatterns = [
    path('add_feature', views.add_feature, name='add_feature'),
    path('delete_feature', views.delete_feature, name='delete_feature'),
    path('retrieval', views.retrieval, name='retrieval'),
    path('global_retrieval', views.global_retrieval, name='global_retrieval'),
]