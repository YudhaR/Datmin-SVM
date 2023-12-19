from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('test/', views.test, name='test'),
    path('upload-tweet/', views.upload_tweet, name='upload_tweet'),
    path('upload-test/', views.upload_test, name='upload_test'),
    path('list-tweet/', views.list_tweet, name='list_tweet'),
    path('list-clean/', views.list_clean, name='list_clean'),
    path('list-training/', views.list_training, name='list_training'),
    path('list-testing/', views.list_testing, name='list_testing'),
    path('delete/list-tweet/<int:idt>/', views.delete_tweet, name='delete_tweet'),
    path('prepro/', views.prepro, name='prepro'),
    path('label/', views.label, name='label'),
    path('latih/', views.latih, name='latih'),
]