from django.http import HttpResponse
from django.template import loader
from .forms import *
from .util.distributions import *
from .util import largenumbers as ln
from django.contrib import messages
import csv


def index(request):
    template = loader.get_template('statVisualiser/index.html')
    context = {}
    return HttpResponse(template.render(context, request))


def distributions(request) -> HttpResponse:
    # Setup context for html file
    dist1s = DistributionSelect(auto_id=True, prefix='picker1', label_suffix='')
    dist2s = DistributionSelect(auto_id=True, prefix='picker2', label_suffix='')
    dist_table = Distribution.objects.all().values()
    convolution = ConvolutionOptions(auto_id=True, prefix="convol", label_suffix="")
    template = loader.get_template('statVisualiser/distributions.html')
    context = {
        'dist': dist_table,
        'picker1': dist1s,
        'picker2': dist2s,
        'convol': convolution,
        'graph': None,
        'conv_graph': None,
        'supported': None,
    }

    if request.method == "POST" and 'submit' in request.POST:
        # get results of forms if POST from server
        dist1s = DistributionSelect(request.POST, auto_id=True, prefix='picker1', label_suffix='')
        dist2s = DistributionSelect(request.POST, auto_id=True, prefix='picker2', label_suffix='')
        convolution = ConvolutionOptions(request.POST, auto_id=True, prefix="convol", label_suffix="")
        context['picker1'] = dist1s  # sets the context to these values. This prevents the data from being lost
        context['picker2'] = dist2s  # after refreshing the page
        messages_text = []  # Array of messages that are sent to the user

        # if both distributions have valid inputs. This should almost always be true.
        if dist1s.is_valid() and dist2s.is_valid():
            data1 = dist1s.get_data()
            data2 = dist2s.get_data()
            a = None
            b = None
            # if there is no data in form 1
            if data1 is None:
                messages_text.append("ERROR: Please select a distribution!")
            else:
                # add messages to the end for all the values that are not present but should be
                for key, value in data1.items():
                    if value == '':
                        messages_text.append("ERROR: No Value for {key_name} in Distribution 1!".format(key_name=key))

                # if a distribution has not been selected warn the user
                # the program will still return a graph, but it will most likely be empty
                if len(dist1s.cleaned_data.get('Output')) == 0:
                    messages_text.append("ERROR: You did not select a PDF or CDF for distribution 1.")

                # the second form is allowed to be none
                if data2 is not None:
                    for key, value in data2.items():
                        if value == '':
                            messages_text.append(
                                "ERROR: No Value for {key_name} in Distribution 2!".format(key_name=key))

                    if len(dist2s.cleaned_data.get('Output')) == 0:
                        messages.warning(request, "WARN: You did not select a PDF or CDF for distribution 2.",
                                         extra_tags='alert')

            if len(messages_text) == 0:
                a = dist_selector(dist1s)
                b = dist_selector(dist2s)

                graph_count = len(dist1s.cleaned_data.get('Output')) + len(
                    dist2s.cleaned_data.get('Output'))

                if graph_count == 1:
                    fig = make_subplots(rows=1, cols=1, subplot_titles=[' '])
                else:
                    fig = make_subplots(rows=int((graph_count + 1) / 2), cols=2, subplot_titles=[' ', ' ', ' ', ' '])

                g_min1 = dist1s.cleaned_data.get('G_Min') if \
                    dist1s.cleaned_data.get('G_Min') is not None else a.get_region()[0]
                g_max1 = dist1s.cleaned_data.get('G_Max') if \
                    dist1s.cleaned_data.get('G_Min') is not None else a.get_region()[1]

                if b is not None:
                    g_min2 = dist2s.cleaned_data.get('G_Min') if \
                        dist2s.cleaned_data.get('G_Min') is not None else b.get_region()[0]
                    g_max2 = dist2s.cleaned_data.get('G_Max') if \
                        dist2s.cleaned_data.get('G_Max') is not None else b.get_region()[1]

                count = 1
                titles = []
                for value in dist1s.cleaned_data.get('Output'):
                    if str(value) == 'pdf':
                        a.graph_pdf(g_min1, g_max1, fig=fig, geom=(int((count + 1) / 2), 2 - (count % 2)))
                        title = "PDF of " + str(a) if a.continuous else "PMF of " + str(a)
                        titles.append(title)
                    elif str(value) == 'cdf':
                        a.graph_cdf(g_min1, g_max1, fig=fig, geom=(int((count + 1) / 2), 2 - (count % 2)))
                        titles.append("CDF of " + str(a))
                    count += 1

                for values in dist2s.cleaned_data.get('Output'):
                    if str(values) == 'pdf':
                        b.graph_pdf(g_min2, g_max2, fig=fig, geom=(int((count + 1) / 2), 2 - (count % 2)))
                        title = "PDF of " + str(b) if a.continuous else "PMF of " + str(b)
                        titles.append(title)
                    elif str(values) == 'cdf':
                        b.graph_cdf(g_min2, g_max2, fig=fig, geom=(int((count + 1) / 2), 2 - (count % 2)))
                        titles.append("CDF of " + str(b))
                    count += 1

                count = 0
                for label in titles:
                    fig.layout.annotations[count].update(text=str(label))
                    count += 1

                context['graph'] = fig.to_html(full_html=False,
                                               div_id='graph')

                # Figure form: choose whether pdf or cdf, with the choice append a title to a list.
                # This list will be the subplot_titles argument in make_subplots
                # Titles each plot dynamically
                # Form also requires for the pdf, cdf:
                # min, max values to plot over.
                # If they want to make more than one graph, requires a geometry argument
                if convolution.is_valid() and convolution.get_context():
                    fig2 = None
                    fig3 = None
                    if convolution.cleaned_data["Output"] and a is not None and b is not None:
                        fig2 = make_subplots(rows=1, cols=2, specs=[[{'type': 'xy'}, {'type': 'surface'}]],
                                             subplot_titles=["CDF of Convolution", "PDF of Convolution"])
                        convolution_pdf(a, b, fig=fig2, geom=(1, 2))
                        convolution_cdf(a, b, fig=fig2, geom=(1, 1))

                        match convolution.cleaned_data["Type"]:
                            case "sum":
                                fig3 = two_var_3d(a, b, conv_type="sum")
                            case "product":
                                fig3 = two_var_3d(a, b)

                        if fig2 is not None:
                            context['conv_graph'] = fig2.to_html(full_html=False, div_id='conv_graph')

                        if fig3 is not None:
                            context['supported'] = fig3.to_html(full_html=False, div_id='supported')
            else:
                for text in messages_text:
                    messages.error(request, text, extra_tags='alert')



    return HttpResponse(template.render(context, request))


def dist_selector(picker: DistributionSelect | SampleDist) -> Variable | None:
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
    n_approx = NormalApproximation(request.POST, prefix='normal')
    p_approx = PoissonApproximation(request.POST, prefix='poisson')
    template = loader.get_template("statVisualiser/largeNumbers.html")
    context = {
        'normal': n_approx,
        'normal_graph': None,
        'poisson': p_approx,
        'poi_graph': None
    }

    if request.method == "POST":
        if p_approx.is_valid():
            p_mean = p_approx.cleaned_data.get('mean') if \
                p_approx.cleaned_data.get('mean') is not None else 4
            p_min = p_approx.cleaned_data.get('min_trials') if \
                p_approx.cleaned_data.get('min_trials') is not None else 10
            p_max = p_approx.cleaned_data.get('max_trials') if \
                p_approx.cleaned_data.get('max_trials') is not None else 10
            p_step = p_approx.cleaned_data.get('step') if \
                p_approx.cleaned_data.get('step') is not None else 10
            poi_graph = ln.binomial_poi_approx(p_min, p_max, p_mean, steps=p_step)
            context['poi_graph'] = poi_graph.to_html(full_html=False)

        if n_approx.is_valid():
            b_min = n_approx.cleaned_data.get('min_trials') if \
                n_approx.cleaned_data.get('min_trials') is not None else 10
            b_max = n_approx.cleaned_data.get('max_trials') if \
                n_approx.cleaned_data.get('max_trials') is not None else 100
            b_step = n_approx.cleaned_data.get('step') if \
                n_approx.cleaned_data.get('step') is not None else 10
            b_prob = n_approx.cleaned_data.get('probability') if \
                n_approx.cleaned_data.get('probability') is not None else 0.5
            norm_graph = ln.binomial_normal(b_min, b_max, b_prob, steps=b_step)
            context['normal_graph'] = norm_graph.to_html(full_html=False)

    return HttpResponse(template.render(context, request))


"""
Stretch goal for samples:
Implement a user input to add a custom dataset to be turned into a histogram, maybe with a desired proposal pdf?
e.g. they can choose to have an exp(1) pdf layered over their data.
"""


def generating_samples(request):
    dist_one_select = SampleDist(auto_id=True, prefix='picker1')
    dist_two_select = SampleDist(auto_id=True, prefix='picker2')
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
        pick_one = SampleDist(request.POST, prefix='picker1', label_suffix='')
        pick_two = SampleDist(request.POST, prefix='picker2', label_suffix='')
        data1 = DatasetParams(request.POST, prefix='data1', label_suffix='')
        data2 = DatasetParams(request.POST, prefix='data2', label_suffix='')
        download_data = Download(request.POST, prefix='download', label_suffix='')

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
                                    headers={'Content-Disposition': 'attachment; filename="sample_dataset.csv"'}, )
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
            return response
    return HttpResponse(template.render(context, request))
