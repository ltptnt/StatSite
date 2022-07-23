from django.core.validators import *
from django.db import models


class Distribution(models.Model):
    name = models.CharField(max_length=50)
    continuous = models.BooleanField(blank=False)
    required_variable_count = models.IntegerField(blank=False, default=0)
    required_variable_names = models.JSONField(blank=True)

    def __str__(self) -> str:
        return self.name


"""
information required for large numbers:
min, max trials
step size, default is 5
binomial information
"""
class NormalApproximation(models.Model):
    min_trials = models.IntegerField(default=10)
    max_trials = models.IntegerField(default=100)
    step = models.IntegerField(default=10)
    probability = models.FloatField(default=0.5)


class PoissonApproximation(models.Model):
    min_trials = models.IntegerField(default=10)
    max_trials = models.IntegerField(default=100)
    step = models.IntegerField(default=10)
    mean = models.FloatField(default=4)
