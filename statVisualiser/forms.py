from django import forms
from .models import Distribution
from django.core.exceptions import ValidationError
import json


class DistributionPicker(forms.Form):
    Type = forms.ModelChoiceField(queryset=Distribution.objects.all(), required=True)
    Rate = forms.DecimalField(required=False)
    Min = forms.DecimalField(required=False)
    Max = forms.DecimalField(required=False)
    Mean = forms.DecimalField(required=False)
    Sd = forms.DecimalField(required=False)
    Probability = forms.DecimalField(min_value=0, max_value=1, required=False)
    Trials = forms.IntegerField(min_value=0, required=False)

    def get_data(self) -> int:
        data = dict()

        dist = self.cleaned_data.get("Type")
        stored_info = Distribution.objects.get(name=dist)
        print(stored_info.required_variable_names)
        required_variables = json.loads(json.dumps(stored_info.required_variable_names))

        for variable in required_variables:
            if variable == "Min":
                if stored_info.supported_region_min < self.cleaned_data.get("Min") < stored_info.supported_region_max:
                    data["Min"] = self.cleaned_data.get("Min")
                else:
                    raise ValidationError("Your Minimium value is not within the supported range!")

            if variable == "Max":
                if stored_info.supported_region_max > self.cleaned_data.get("Max") > stored_info.supported_region_min:
                    data["Max"] = self.cleaned_data.get("Max")
                else:
                    raise ValidationError("Your Maximium value is not within the supported range!")

            else:
                data[variable] = self.data.get(variable)

        return data
