from django import forms

import statVisualiser.models
from .models import Distribution
import json


class SampleDist(forms.Form):
    Type = forms.ModelChoiceField(label='Distribution', queryset=Distribution.objects.all(), required=False,
                                  empty_label='')
    Rate = forms.FloatField(label='Rate', required=False)
    Min = forms.FloatField(label='Minimum Value', required=False)
    Max = forms.FloatField(label='Maximum Value', required=False)
    Mean = forms.FloatField(label='Mean', required=False)
    Sd = forms.FloatField(label='Standard Deviation', required=False)
    Probability = forms.FloatField(label='Probability', min_value=0, max_value=1, required=False)
    Trials = forms.IntegerField(label='Trials', min_value=0, required=False)

    def get_data(self) -> dict[str, float] | dict[str, int] | None:
        data = dict()
        try:
            dist = self.cleaned_data.get("Type")
            stored_info = Distribution.objects.get(name=dist)
        except statVisualiser.models.Distribution.DoesNotExist:
            return None

        required_variables = json.loads(json.dumps(stored_info.required_variable_names))

        for variable in required_variables:
            if self.cleaned_data.get(variable) is None:
                data[variable] = ''
            else:
                data[variable] = self.cleaned_data.get(variable)

        return data


class DistributionSelect(SampleDist):
    Outputs = (
        ("pdf", "Probability Density Function"),
        ("cdf", "Cumulative Distribution Function"),
    )

    G_Min = forms.FloatField(label='Domain Minimum (Optional)', initial=None, required=False)
    G_Max = forms.FloatField(label='Domain Maximum (Optional)', initial=None, required=False)
    Output = forms.MultipleChoiceField(label='', widget=forms.CheckboxSelectMultiple,
                                       choices=Outputs, required=False)


class NormalApproximation(forms.Form):
    min_trials = forms.IntegerField(initial=10, required=False, label="Minimum Trials (Default: 10)")
    max_trials = forms.IntegerField(initial=100, required=False, label="Maximum Trials (Default: 100)")
    step = forms.IntegerField(initial=10, required=False, label="Step (Default: 10)")
    probability = forms.FloatField(initial=0.5, min_value=0, max_value=1, required=False,
                                   label="Probability (Default: 0.5)")


class PoissonApproximation(forms.Form):
    min_trials = forms.IntegerField(initial=10, required=False, label="Minimum Trials (Default: 10)")
    max_trials = forms.IntegerField(initial=100, required=False, label="Maximum Trials (Default: 100)")
    step = forms.IntegerField(initial=10, required=False, label="Step (Default: 10)")
    mean = forms.FloatField(initial=4, required=False, label="Mean (Default: 4)")


class DatasetParams(forms.Form):
    n_trials = forms.IntegerField(label="Number of Trials", initial=None, min_value=1, max_value=10000, required=False)
    std_error = forms.FloatField(label="Standard Error Term (optional)", initial=0, required=False)


class Download(forms.Form):
    download = forms.BooleanField(label="Download dataset", initial=False, required=False)
    convolution = forms.BooleanField(label="Plot the Convolution?", initial=False, required=False)


class ConvolutionOptions(forms.Form):
    choices = (
        ("prob", "Probability Functions"),
    )
    type = (
        ("sum", "Sum"),
        ("product", "Product"),

    )
    Output = forms.BooleanField(label='Plot Convolution',
                                initial=False, required=False)
    Type = forms.ChoiceField(choices=type)
