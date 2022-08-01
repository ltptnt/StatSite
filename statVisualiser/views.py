import numpy as np
from django.http import HttpResponse
from django.template import loader
from .forms import *
from .util.distributions import *
from .util import largenumbers as ln
import csv


def index(request):
    template = loader.get_template('statVisualiser/index.html')
    context = {}
    return HttpResponse(template.render(context, request))


def distributions(request):
    dist_one_select = DistributionSelect(auto_id=True, prefix='picker1')
    dist_two_select = DistributionSelect(auto_id=True, prefix='picker2')
    dist_table = Distribution.objects.all().values()
    template = loader.get_template('statVisualiser/distributions.html')

    context = {
        'dist': dist_table,
        'picker1': dist_one_select,
        'picker2': dist_two_select,
        'graph': None
    }

    if request.method == "POST":
        pick_one = DistributionSelect(request.POST, prefix='picker1', label_suffix='')
        pick_two = DistributionSelect(request.POST, prefix='picker2', label_suffix='')

        if pick_one.is_valid() and pick_two.is_valid():
            # Figure form: choose whether pdf or cdf, with the choice append a title to a list.
            # This list will be the subplot_titles argument in make_subplots
            # Titles each plot dynamically
            # Form also requires for the pdf, cdf:
            # min, max values to plot over.
            # If they want to make more than one graph, requires a geometry argument

            a = dist_selector(pick_one)
            b = dist_selector(pick_two)
            fig = make_subplots(rows=2, cols=2, subplot_titles=[str(a), str(b), str(a), str(b)])
            a.graph_pdf(0, 10, fig=fig, geom=(1, 2), titles=True)
            b.graph_pdf(0, 10, fig=fig, geom=(2, 1), titles=True)
            a.graph_cdf(0, 10, fig=fig, geom=(1, 1))
            b.graph_cdf(0, 10, fig=fig, geom=(2, 2))
            context['graph'] = fig.to_html(full_html=False, default_height=500, default_width=700)

    return HttpResponse(template.render(context, request))


def dist_selector(picker: DistributionSelect) -> Variable | None:
    data = picker.get_data()
    print(type(data))
    print(data)
    match str(picker.cleaned_data.get('Type')):
        case 'Exponential':
            return Exponential(data.get('Rate', 0))
        case 'Uniform':
            return Uniform(data.get('Min', 0), data.get('Max', 0))
        case 'Normal':
            return Normal(data.get('Mean', 0), data.get('Sd', 0))
        case 'Poisson':
            return Poisson(data.get('Rate', 0))
        case 'Bernoulli':
            return Bernoulli(data.get('Probability', 0))
        case 'Binomial':
            return Binomial(data.get('Trials', 0), data.get('Probability', 0))
        case _:
            return None


def large_numbers(request):
    normal_approx = NormalApproximation(request.POST, prefix='normal')
    poi_approx = PoissonApproximation(request.POST, prefix='poisson')
    template = loader.get_template("statVisualiser/largeNumbers.html")
    context = {
        'normal': normal_approx,
        'normal_graph': None,
        'poisson': poi_approx,
        'poi_graph': None
    }
    if poi_approx.is_valid():
        p_mean = poi_approx.cleaned_data.get('mean')
        p_min = poi_approx.cleaned_data.get('min_trials')
        p_max = poi_approx.cleaned_data.get('max_trials')
        p_step = poi_approx.cleaned_data.get('step')
        poi_graph = ln.binomial_poi_approx(p_min, p_max, p_mean, steps=p_step)
        context['poi_graph'] = poi_graph.to_html(full_html=False, default_height=500, default_width=700)

    if normal_approx.is_valid():
        b_min = normal_approx.cleaned_data.get('min_trials')
        b_max = normal_approx.cleaned_data.get('max_trials')
        b_step = normal_approx.cleaned_data.get('step')
        b_prob = normal_approx.cleaned_data.get('probability')
        norm_graph = ln.binomial_normal(b_min, b_max, b_prob, steps=b_step)
        context['normal_graph'] = norm_graph.to_html(full_html=False, default_height=500, default_width=700)

    return HttpResponse(template.render(context, request))


"""
Plan for generating samples:
lets users generate sample from a distribution they choose
they can overlay the pdf with the sample histogram
See an approximation of a product of random variables
"""


def generating_samples(request):
    dist_one_select = DistributionSelect(auto_id=True, prefix='picker1')
    dist_two_select = DistributionSelect(auto_id=True, prefix='picker2')
    d1 = DatasetParams(auto_id=True, prefix='data1')
    d2 = DatasetParams(auto_id=True, prefix='data2')
    dist_table = Distribution.objects.all().values()
    template = loader.get_template('statVisualiser/generatingSamples.html')

    context = {
        'dist': dist_table,
        'picker1': dist_one_select,
        'picker2': dist_two_select,
        'data1': d1,
        'data2': d2,
        'graph1': None,
        'graph2': None,
        'graph3': None
    }

    # fig =
    # Overlaying the distribution the sample is from.
    # minim, maxim = a.get_region()
    # a.graph_pdf(minim, maxim, fig=fig)

    if request.method == "POST":
        print("posted")
        pick_one = DistributionSelect(request.POST, prefix='picker1', label_suffix='')
        pick_two = DistributionSelect(request.POST, prefix='picker2', label_suffix='')
        data1 = DatasetParams(request.POST, prefix='data1', label_suffix='')
        data2 = DatasetParams(request.POST, prefix='data2', label_suffix='')
        download_data = True

        print(data1.is_valid(), data2.is_valid())
        # New plan: integrate over each interval to find the expected value for interval in the partition
        if pick_one.is_valid() and data1.is_valid():
            var1 = dist_selector(pick_one)
            dataset1 = var1.generate_dataset(data1.cleaned_data.get("n_trials"), data1.cleaned_data.get("std_error"))
            fig1 = dataset_plots(var1, dataset1)
            context['graph1'] = fig1.to_html(full_html=False, default_height=700, default_width=700)

            if download_data:
                response = HttpResponse(content_type='text/csv')
                download = csv.writer(response)
                download.writerows([[i] for i in dataset1])
                response['Content-Disposition'] = 'download; filename="file.csv"'
                writer = csv.writer(response)
                writer.writerow(dataset1)
                return response

        if pick_two.is_valid() and data2.is_valid():
            var2 = dist_selector(pick_two)
            dataset2 = var2.generate_dataset(data2.cleaned_data.get("n_trials"), data2.cleaned_data.get("std_error"))
            fig2 = dataset_plots(var2, dataset2)
            context['graph2'] = fig2.to_html(full_html=False, default_height=700, default_width=700)

        if data1.is_valid() and data2.is_valid() and data1.cleaned_data.get("convolution") and data2.cleaned_data.get("convolution"):
            print("yes")
            conv_fig = graph_density_product(dataset1, dataset2)
            context['graph3'] = conv_fig.to_html(full_html=False, default_height=700, default_width=700)
        else:
            print(data1.data.get("convolution"), data2.data.get("convolution"))

        #Add the downloadable data



    return HttpResponse(template.render(context, request))




def about(request):
    template = loader.get_template('statVisualiser/about.html')
    context = {}
    return HttpResponse(template.render(context, request))
