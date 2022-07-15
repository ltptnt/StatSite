from django.core.validators import *
from django.db import models


class Distribution(models.Model):

    def get_empty(self) -> dict:
        return dict()

    name = models.CharField(max_length=50)
    continuous = models.BooleanField(blank=False)
    supportedRegionMin = models.BigIntegerField(blank=True, default=9223372036854775807)
    supportedRegionMax = models.BigIntegerField(blank=True, default=-9223372036854775808)
    requiredVariableCount = models.IntegerField(blank=False, default=0)
    requiredVariableNames = models.JSONField(blank=True, default=get_empty)

    def __str__(self) -> str:
        return self.name
