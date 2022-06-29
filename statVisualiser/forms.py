from django import forms
from .models import Distributions


class Picker(forms.Form):
    Equation = forms.ModelChoiceField(queryset=Distributions.objects.all(), required=True)
    Min = forms.IntegerField()
    Max = forms.IntegerField()

