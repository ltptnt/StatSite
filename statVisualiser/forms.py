from django import forms
from .models import Distribution
import json
from django.core.exceptions import ValidationError


class DistributionSelect(forms.Form):
    Type = forms.ModelChoiceField(label='Distribution', queryset=Distribution.objects.all(), required=True, blank=False, empty_label='')
    Rate = forms.FloatField(label='Rate', required=False)
    Min = forms.FloatField(label='Minimum Value', required=False)
    Max = forms.FloatField(label='Maximum Value', required=False)
    Mean = forms.FloatField(label='Mean', required=False)
    Sd = forms.FloatField(label='Standard Deviation', required=False)
    Probability = forms.FloatField(label='Probability', min_value=0, max_value=1, required=False)
    Trials = forms.IntegerField(label='Trials', min_value=0, required=False)


    def get_data(self) -> dict[str, float] | dict[str, int]:
        data = dict()
        dist = self.cleaned_data.get("Type")
        stored_info = Distribution.objects.get(name=dist)
        required_variables = json.loads(json.dumps(stored_info.required_variable_names))

        for variable in required_variables:
            if self.cleaned_data.get(variable) == None:
                raise ValidationError('Please input a value for ' + variable)
            data[variable] = self.cleaned_data.get(variable)

        return data
