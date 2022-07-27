from django.urls import path

from . import views

app_name = 'statVisualiser'

urlpatterns = [
    path('', views.index, name='index'),
    path('distributions', views.distributions, name='distributions'),
    path('largeNumbers', views.large_numbers, name='largeNumbers'),
    path('wheelSpin', views.generating_samples, name="wheelSpin"),


    path('about', views.about, name='about')
]
