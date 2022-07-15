from django.core.validators import *
from django.db import models


class Distribution(models.Model):
    name = models.CharField(max_length=50)
    continuous = models.BooleanField(blank=False)
    supported_region_min = models.BigIntegerField(blank=True, default=-9223372036854775808)
    supported_region_max = models.BigIntegerField(blank=True, default=9223372036854775807)
    required_variable_count = models.IntegerField(blank=False, default=0)
    required_variable_names = models.JSONField(blank=True)

    def __str__(self) -> str:
        return self.name
