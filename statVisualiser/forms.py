from django import forms
from .models import Distribution


class Picker(forms.Form):
    Equation = forms.ModelChoiceField(queryset=Distribution.objects.all(), required=True)
    Min = forms.IntegerField()
    Max = forms.IntegerField()

