from django.urls import path

from . import views

app_name = 'statVisualiser'

urlpatterns = [
    path('', views.index, name='index'),
]
