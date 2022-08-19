from django.urls import path

from . import views

app_name = 'statVisualiser'

urlpatterns = [
    path('', views.index, name='index'),
    path('distributions', views.distributions, name='distributions'),
    path('large_numbers', views.large_numbers, name='large_numbers'),
    path('generating_samples', views.generating_samples, name="generating_samples"),
]
