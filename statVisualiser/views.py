from django.http import HttpResponse
from django.template import loader
from .forms import *
from .util.distributions import *
from .util import largenumbers as ln


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
            #Figure form: choose whether pdf or cdf, with the choice append a title to a list.
            #This list will be the subplot_titles argument in make_subplots
            #Titles each plot dynamically
            #Form also requires for the pdf, cdf:
            #min, max values to plot over.
            #If they want to make more than one graph, requires a geometry argument
            fig = make_subplots(rows=2, cols=2)

            a = dist_selector(pick_one)
            b = dist_selector(pick_two)
            a.graph_pdf(0, 10, fig=fig, geom=(1, 2), titles=True)
            b.graph_pdf(0, 10, fig=fig, geom=(2, 1), titles=True)
            a.graph_cdf(0, 10, fig=fig, geom=(1, 1))
            b.graph_cdf(0, 10, fig=fig, geom=(2, 2))
            fig.update_layout(subplot_titles=[str(a), str(b), str(a), str(b)])
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
    if normal_approx.is_valid() and poi_approx.is_valid():
        p_mean = poi_approx.cleaned_data.get('mean')
        p_min = poi_approx.cleaned_data.get('min_trials')
        p_max = poi_approx.cleaned_data.get('max_trials')
        p_step = poi_approx.cleaned_data.get('step')

        b_min = normal_approx.cleaned_data.get('min_trials')
        b_max = normal_approx.cleaned_data.get('max_trials')
        b_step = normal_approx.cleaned_data.get('step')
        b_prob = normal_approx.cleaned_data.get('probability')

        poi_graph = ln.binomial_poi_approx(p_min, p_max, p_mean, steps=p_step)
        norm_graph = ln.binomial_normal(b_min, b_max, b_prob, steps=b_step)
        context['normal_graph'] = norm_graph.to_html(full_html=False, default_height=500, default_width=700)
        context['poi_graph'] = poi_graph.to_html(full_html=False, default_height=500, default_width=700)
    return HttpResponse(template.render(context, request))


def wheel_spin(request):
    pass



def about(request):
    template = loader.get_template('statVisualiser/about.html')
    context = {}
    return HttpResponse(template.render(context, request))

