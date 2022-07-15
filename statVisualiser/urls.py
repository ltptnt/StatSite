from django.urls import path

from . import views

app_name = 'statVisualiser'

urlpatterns = [
    path('', views.index, name='index'),
    path('distributions', views.distributions, name='distributions'),


    path('about', views.about, name='about')
]
