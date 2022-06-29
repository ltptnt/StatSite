from django.core.validators import *
from django.db import models


class Distributions(models.Model):
    name = models.CharField(max_length=20)
    supported_region = models.CharField(max_length=20)
    param = models.CharField(max_length=100)

    def __str__(self):
        return self.name

