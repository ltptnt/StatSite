import numpy as np
from django.http import HttpResponse
from django.template import loader
from .forms import *
from .util.distributions import *
from .util import largenumbers as ln
from django.contrib import messages
import csv
from statsite.settings import BASE_DIR


def index(request):
    template = loader.get_template('statVisualiser/index.html')
    context = {}
    return HttpResponse(template.render(context, request))


def distributions(request):
    dist_one_select = DistributionSelect(auto_id=True, prefix='picker1', label_suffix='')
    dist_two_select = DistributionSelect(auto_id=True, prefix='picker2', label_suffix='')
    dist_table = Distribution.objects.all().values()
    template = loader.get_template('statVisualiser/distributions.html')

    context = {
        'dist': dist_table,
        'picker1': dist_one_select,
        'picker2': dist_two_select,
        'graph': None
    }

    if request.method == "POST" and 'submit' in request.POST:
        dist_one_select = DistributionSelect(request.POST, auto_id=True, prefix='picker1', label_suffix='')
        dist_two_select = DistributionSelect(request.POST, auto_id=True, prefix='picker2', label_suffix='')
        context['picker1'] = dist_one_select
        context['picker2'] = dist_two_select

        if dist_one_select.is_valid() and dist_two_select.is_valid():
            messages_text = []

            data1 = dist_one_select.get_data()
            data2 = dist_two_select.get_data()

            if data1 is None:
                messages_text.append("ERROR: Please select a distribution!")
            else:
                for key, value in data1.items():
                    if value == '':
                        messages_text.append("ERROR: No Value for {key_name} in Distribution 1!".format(key_name=key))

                if len(dist_two_select.cleaned_data.get('Output')) == 0:
                    messages.warning(request, "WARN: You did not select a PDF or CDF for distribution 1.",
                                     extra_tags='alert')

                if data2 is not None:
                    for key, value in data2.items():
                        if value == '':
                            messages_text.append(
                                "ERROR: No Value for {key_name} in Distribution 2!".format(key_name=key))

                    if len(dist_two_select.cleaned_data.get('Output')) == 0:
                        messages.warning(request, "WARN: You did not select a PDF or CDF for distribution 2.",
                                         extra_tags='alert')

            if len(messages_text) == 0:
                a = dist_selector(dist_one_select)
                b = dist_selector(dist_two_select)
                graph_count = len(dist_one_select.cleaned_data.get('Output')) + len(
                    dist_two_select.cleaned_data.get('Output'))

                if graph_count == 1:
                    fig = make_subplots(rows=1, cols=1, subplot_titles= '')
                else:
                    fig = make_subplots(rows=round((graph_count + 1) / 2), cols=2, subplot_titles=[' ', ' ', ' ', ' '])
                count = 1
                titles = []
                for value in dist_one_select.cleaned_data.get('Output'):
                    if str(value) == 'pdf':
                        print("1")
                        a.graph_pdf(0, 10, fig=fig, geom=(int((count + 1) / 2), 2 - (count % 2)), titles=True)
                        titles.append("PDF of " + str(a))
                    elif str(value) == 'cdf':
                        print("2")
                        a.graph_cdf(0, 10, fig=fig, geom=(int((count + 1) / 2), 2 - (count % 2)), titles=True)
                        titles.append("CDF of " + str(a))
                    count += 1


                for values in dist_two_select.cleaned_data.get('Output'):
                    if str(values) == 'pdf':
                        b.graph_pdf(0, 10, fig=fig, geom=(round((count + 1) / 2), 2 - (count % 2)), titles=True)
                        titles.append("PDF of " + str(b))
                    elif str(values) == 'cdf':
                        b.graph_cdf(0, 10, fig=fig, geom=(round((count + 1) / 2), 2 - (count % 2)), titles=True)
                        titles.append("CDF of " + str(b))
                    count += 1

                count = 0
                for label in titles:
                    fig.layout.annotations[count].update(text=str(label))
                    count += 1

                context['graph'] = fig.to_html(full_html=False, default_height=750, default_width=1000,
                                               div_id='graph')

            else:
                for text in messages_text:
                    messages.error(request, text, extra_tags='alert')

            # Figure form: choose whether pdf or cdf, with the choice append a title to a list.
            # This list will be the subplot_titles argument in make_subplots
            # Titles each plot dynamically
            # Form also requires for the pdf, cdf:
            # min, max values to plot over.
            # If they want to make more than one graph, requires a geometry argument
    return HttpResponse(template.render(context, request))


def dist_selector(picker: DistributionSelect) -> Variable | None:
    data = picker.get_data()
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
Stretch goal for samples:
Implement a user input to add a custom dataset to be turned into a histogram, maybe with a desired proposal pdf?
e.g. they can choose to have an exp(1) pdf layered over their data.
"""


def generating_samples(request):
    dist_one_select = DistributionSelect(auto_id=True, prefix='picker1')
    dist_two_select = DistributionSelect(auto_id=True, prefix='picker2')
    download = Download(auto_id=True, prefix='download')
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
        'download': download,
        'graph1': None,
        'graph2': None,
        'graph3': None
    }

    if request.method == "POST":
        pick_one = DistributionSelect(request.POST, prefix='picker1', label_suffix='')
        pick_two = DistributionSelect(request.POST, prefix='picker2', label_suffix='')
        data1 = DatasetParams(request.POST, prefix='data1', label_suffix='')
        data2 = DatasetParams(request.POST, prefix='data2', label_suffix='')
        download_data = Download(request.POST, prefix='download', label_suffix='')

        print(pick_two.is_valid(), pick_one.is_valid())
        print(data1.is_valid(), data2.is_valid())
        dataset1 = []
        dataset2 = []
        var1 = None
        var2 = None

        if pick_one.is_valid() and data1.is_valid():
            var1 = dist_selector(pick_one)
            dataset1 = var1.generate_dataset(data1.cleaned_data.get("n_trials"), data1.cleaned_data.get("std_error"))
            fig1 = dataset_plots(var1, dataset1)
            context['graph1'] = fig1.to_html(full_html=False, default_height=700, default_width=700)

        if pick_two.is_valid() and data2.is_valid():
            var2 = dist_selector(pick_two)
            dataset2 = var2.generate_dataset(data2.cleaned_data.get("n_trials"), data2.cleaned_data.get("std_error"))
            fig2 = dataset_plots(var2, dataset2)
            context['graph2'] = fig2.to_html(full_html=False, default_height=700, default_width=700)

        if download_data.is_valid() and download_data.cleaned_data.get("convolution"):
            while len(dataset1) < len(dataset2):
                dataset1.append([None])
            while len(dataset1) > len(dataset2):
                dataset2.append([None])
            conv_fig = graph_density_product(dataset1, dataset2)
            context['graph3'] = conv_fig.to_html(full_html=False, default_height=700, default_width=700)

        if download_data.is_valid() and download_data.cleaned_data.get("download"):
            response = HttpResponse(content_type='text/csv',
                                    headers={'Content-Disposition': 'attachment; filename="sample_dataset.csv"'},)
            title1 = str(var1) if var1 is not None else ""
            title2 = str(var2) if var2 is not None else ""
            title = [title1, title2]
            while len(dataset1) < len(dataset2):
                dataset1.append([None])
            while len(dataset1) > len(dataset2):
                dataset2.append([None])

            big_data = list(zip(dataset1, dataset2))
            download = csv.writer(response)
            download.writerow(title)
            download.writerows([trial[0], trial[1]] for trial in big_data)
            print("download return ")
            return response
    print("render return")
    return HttpResponse(template.render(context, request))



def about(request):
    template = loader.get_template('statVisualiser/about.html')
    context = {}
    return HttpResponse(template.render(context, request))
