import csv

from django.contrib import messages
from django.http import HttpResponse
from django.template import loader

from .forms import *
from .util import largenumbers as ln
from .util.distributions import *


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
        convolution = ConvolutionOptions(request.POST, auto_id=True, prefix="convol",
                                         label_suffix='')
        context[
            'picker1'] = dist1s  # sets the context to these values. This prevents the data from being lost
        context['picker2'] = dist2s  # after refreshing the page
        context['convol'] = convolution
        messages_text = []  # Array of messages that are sent to the user

        # if both distributions have valid inputs. This should almost always be true.
        if dist1s.is_valid() and dist2s.is_valid():
            data1 = dist1s.get_data()
            data2 = dist2s.get_data()
            # if there is no data in form 1
            if data1 is None:
                messages_text.append("ERROR: Please select a distribution!")
            else:
                # add messages to the end for all the values that are not present but should be
                required_vars = ''
                for key, value in data1.items():
                    if value == '':
                        required_vars += key + ', '
                if required_vars != '':
                    messages_text.append(
                        "ERROR: No Value/s for {var} in Distribution 1!".format(var=required_vars))

                # Ensure that if the domain is entered it is a valid range i.e. Min <= Max
                if dist1s.cleaned_data.get('G_Min') is not None \
                        and dist1s.cleaned_data.get('G_Min') >= dist1s.cleaned_data.get('G_Max'):
                    messages_text.append("ERROR: Invalid Domain for Distribution 1!")

                # if a distribution has not been selected warn the user
                # the program will still return a graph, but it will most likely be empty
                if len(dist1s.cleaned_data.get('Output')) == 0:
                    messages_text.append(
                        "ERROR: You did not select a PDF or CDF for distribution 1.")

                # the second form is allowed to be none
                if data2 is not None:
                    required_vars = ''
                    for key, value in data2.items():
                        if value == '':
                            required_vars += key + ', '
                    if required_vars != '':
                        messages_text.append(
                            "ERROR: No Value/s for {var} in Distribution 2!".format(
                                var=required_vars))

                    # Ensure that if the domain is entered it is a valid range i.e. Min <= Max
                    if dist2s.cleaned_data.get('G_Min') is not None \
                            and dist2s.cleaned_data.get('G_Min') >= dist2s.cleaned_data.get(
                        'G_Max'):
                        messages_text.append("ERROR: Invalid Domain for Distribution 2!")

                    # Warn the user if there are values selected but no output selected
                    if len(dist2s.cleaned_data.get('Output')) == 0:
                        messages.warning(request,
                                         "WARN: You did not select a PDF or CDF for distribution 2.",
                                         extra_tags='alert')

            if len(messages_text) == 0:
                a = dist_selector(dist1s)
                b = dist_selector(dist2s)

                graph_count = len(dist1s.cleaned_data.get('Output')) + len(
                    dist2s.cleaned_data.get('Output'))

                if graph_count == 1:
                    fig = make_subplots(rows=1, cols=1, subplot_titles=[' '])
                else:
                    fig = make_subplots(rows=int((graph_count + 1) / 2), cols=2,
                                        subplot_titles=[' ', ' ', ' ', ' '])

                g_min1 = dist1s.cleaned_data.get('G_Min') if \
                    dist1s.cleaned_data.get('G_Min') is not None else a.get_region()[0]
                g_max1 = dist1s.cleaned_data.get('G_Max') if \
                    dist1s.cleaned_data.get('G_Min') is not None else a.get_region()[1]

                count = 1
                titles = []

                for value in dist1s.cleaned_data.get('Output'):
                    if str(value) == 'pdf':
                        a.graph_pdf(g_min1, g_max1, fig=fig,
                                    geom=(int((count + 1) / 2), 2 - (count % 2)))
                        title = "PDF of " + str(a) if a.continuous else "PMF of " + str(a)
                        titles.append(title)
                    elif str(value) == 'cdf':
                        a.graph_cdf(g_min1, g_max1, fig=fig,
                                    geom=(int((count + 1) / 2), 2 - (count % 2)))
                        titles.append("CDF of " + str(a))
                    count += 1

                if b is not None:
                    g_min2 = dist2s.cleaned_data.get('G_Min') if \
                        dist2s.cleaned_data.get('G_Min') is not None else b.get_region()[0]
                    g_max2 = dist2s.cleaned_data.get('G_Max') if \
                        dist2s.cleaned_data.get('G_Max') is not None else b.get_region()[1]

                    for values in dist2s.cleaned_data.get('Output'):
                        if str(values) == 'pdf':
                            b.graph_pdf(g_min2, g_max2, fig=fig,
                                        geom=(int((count + 1) / 2), 2 - (count % 2)))
                            title = "PDF of " + str(b) if a.continuous else "PMF of " + str(b)
                            titles.append(title)
                        elif str(values) == 'cdf':
                            b.graph_cdf(g_min2, g_max2, fig=fig,
                                        geom=(int((count + 1) / 2), 2 - (count % 2)))
                            titles.append("CDF of " + str(b))
                        count += 1

                count = 0
                for label in titles:
                    fig.layout.annotations[count].update(text=str(label))
                    count += 1

                context['graph'] = fig.to_html(full_html=False, div_id='graph')

                # Figure form: choose whether pdf or cdf, with the choice append a title to a list.
                # This list will be the subplot_titles argument in make_subplots
                # Titles each plot dynamically
                # Form also requires for the pdf, cdf:
                # min, max values to plot over.
                # If they want to make more than one graph, requires a geometry argument
                if convolution.is_valid() and convolution.get_context():
                    if convolution.cleaned_data["Output"] and a is not None and b is not None:
                        fig2 = make_subplots(rows=1, cols=2,
                                             specs=[[{'type': 'xy'}, {'type': 'surface'}]],
                                             subplot_titles=["CDF of Convolution",
                                                             "PDF of Convolution"])
                        fig3 = None
                        convolution_pdf(a, b, fig=fig2, geom=(1, 2))
                        convolution_cdf(a, b, fig=fig2, geom=(1, 1))

                        match convolution.cleaned_data["Type"]:
                            case "sum":
                                fig3 = two_var_3d(a, b, conv_type="sum")
                            case "product":
                                fig3 = two_var_3d(a, b)

                        if fig2 is not None:
                            context['conv_graph'] = fig2.to_html(full_html=False,
                                                                 div_id='conv_graph')

                        if fig3 is not None:
                            context['supported'] = fig3.to_html(full_html=False, div_id='supported')
                    elif convolution.cleaned_data["Output"] and b is None:
                        messages.warning(request,
                                         "WARN: Cannot plot convolution, no data in Distribution 2.",
                                         extra_tags='alert')
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
    n_approx = NormalApproximation(request.POST, prefix='normal', label_suffix='')
    p_approx = PoissonApproximation(request.POST, prefix='poisson', label_suffix='')
    template = loader.get_template("statVisualiser/large_numbers.html")
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


def generating_samples(request):
    dist_one_select = SampleDist(auto_id=True, prefix='picker1', label_suffix='')
    dist_two_select = SampleDist(auto_id=True, prefix='picker2', label_suffix='')
    download = Download(auto_id=True, prefix='download')
    d1 = DatasetParams(auto_id=True, prefix='data1')
    d2 = DatasetParams(auto_id=True, prefix='data2')
    dist_table = Distribution.objects.all().values()
    template = loader.get_template('statVisualiser/generating_samples.html')

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
        pick_one = SampleDist(request.POST, auto_id=True, prefix='picker1', label_suffix='')
        pick_two = SampleDist(request.POST, auto_id=True, prefix='picker2', label_suffix='')
        data1 = DatasetParams(request.POST, auto_id=True, prefix='data1', label_suffix='')
        data2 = DatasetParams(request.POST, auto_id=True, prefix='data2', label_suffix='')
        download_data = Download(request.POST, auto_id=True, prefix='download', label_suffix='')
        context['picker1'] = pick_one
        context['picker2'] = pick_two
        context['data1'] = data1
        context['data2'] = data2

        messages_text = []  # Array of messages that are sent to the user
        dataset1 = []
        dataset2 = []
        var1 = None
        var2 = None

        if pick_one.is_valid() and pick_two.is_valid() and data1.is_valid() and data2.is_valid() and download_data.is_valid():
            if data1.cleaned_data.get("n_trials") is None:
                messages_text.append("ERROR: Please Specify the Number of Trials")
            if pick_one.get_data() is None:
                messages_text.append("ERROR: Please Select Distribution 1")
            else:
                # add messages to the end for all the values that are not present but should be
                required_vars = ''
                for key, value in pick_one.get_data().items():
                    if value == '':
                        required_vars += key + ', '
                if required_vars != '':
                    messages_text.append(
                        "ERROR: No Value/s for {var} in Distribution 1!".format(var=required_vars))

            if pick_two.get_data() is not None:
                required_vars = ''
                for key, value in pick_two.get_data().items():
                    if value == '':
                        required_vars += key + ', '
                if required_vars != '':
                    messages_text.append(
                        "ERROR: No Value/s for {var} in Distribution 2!".format(var=required_vars))

            if download_data.cleaned_data.get("convolution"):
                if pick_two.get_data() is None:
                    messages_text.append("ERROR: Please Select Distribution 2")
                elif data2.cleaned_data.get("n_trials") is None:
                    messages_text.append("ERROR: Please Specify the Number of Trials")

        if len(messages_text) != 0:
            for text in messages_text:
                messages.error(request, text, extra_tags='alert')
        else:
            var1 = dist_selector(pick_one)
            dataset1 = var1.generate_dataset(data1.cleaned_data.get("n_trials"),
                                             data1.cleaned_data.get("std_error"))
            fig1 = dataset_plots(var1, dataset1)
            context['graph1'] = fig1.to_html(full_html=False)

            if pick_two.get_data() is not None:
                var2 = dist_selector(pick_two)
                dataset2 = var2.generate_dataset(data2.cleaned_data.get("n_trials"),
                                                 data2.cleaned_data.get("std_error"))
                fig2 = dataset_plots(var2, dataset2)
                context['graph2'] = fig2.to_html(full_html=False)

            if download_data.is_valid() and download_data.cleaned_data.get("convolution"):
                while len(dataset1) < len(dataset2):
                    dataset1.append("")
                while len(dataset1) > len(dataset2):
                    dataset2.append("")
                conv_fig = graph_density_product(dataset1, dataset2)
                context['graph3'] = conv_fig.to_html(full_html=False)

            if download_data.cleaned_data.get("download"):
                response = HttpResponse(content_type='text/csv',
                                        headers={
                                            'Content-Disposition': 'attachment; filename="sample_dataset.csv"'}, )
                title1 = str(var1) if var1 is not None else ""
                title2 = str(var2) if var2 is not None else ""
                title = title1
                download = csv.writer(response)
                if download_data.cleaned_data.get("convolution") and var2 is not None:
                    title = [title1, title2, "Convolution"]
                    if len(dataset1) != len(dataset2):
                        mindex = min(len(dataset1), len(dataset2))
                        dataset1 = dataset1[:mindex]
                        dataset2 = dataset2[:mindex]
                    dataset_conv = []
                    for i in range(len(dataset1)):
                        dataset_conv.append(dataset1[i] * dataset2[i])
                    bigger_data = list(zip(dataset1, dataset2, dataset_conv))
                    download.writerow(title)
                    download.writerows([trial[0], trial[1], trial[2]] for trial in bigger_data)
                elif var2 is not None:
                    title = [title1, title2]
                    while len(dataset1) < len(dataset2):
                        dataset1.append("")
                    while len(dataset1) > len(dataset2):
                        dataset2.append("")
                    big_data = list(zip(dataset1, dataset2))
                    download.writerow(title)
                    download.writerows([trial[0], trial[1]] for trial in big_data)
                else:
                    download.writerow([title])
                    download.writerows([i] for i in dataset1)
                return response
    return HttpResponse(template.render(context, request))
