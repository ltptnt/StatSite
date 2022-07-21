from django.core.exceptions import ValidationError
from django.http import HttpResponse
from django.template import loader
from .forms import *
from .util.distributions import *
from plotly.io import to_html

from .util.distributions import Exponential, Uniform, Normal, Poisson, Bernoulli, Binomial


def index(request):
    template = loader.get_template('statVisualiser/index.html')
    context = {}
    return HttpResponse(template.render(context, request))


def distributions(request):
    dist_one_select = DistributionSelect(prefix='picker1')
    dist_two_select = DistributionSelect(prefix='picker2')
    dist_table = Distribution.objects.all().values()
    template = loader.get_template('statVisualiser/distributions.html')

    context = {
        'dist': dist_table,
        'picker1': dist_one_select,
        'picker2': dist_two_select,
        'graph': None
    }

    if request.method == "POST":
        pick_one = DistributionSelect(request.POST, prefix='picker1')
        pick_two = DistributionSelect(request.POST, prefix='picker2')

        if pick_one.is_valid() and pick_two.is_valid():
            fig = make_subplots(rows=2, cols=2)
            a = dist_selector(pick_one)
            b = dist_selector(pick_two)
            a.graph_pdf(0, 10, fig=fig, geom=(1, 2), titles=True)
            b.graph_pdf(0, 10, fig=fig, geom=(2, 1), titles=True)
            a.graph_cdf(0, 10, fig=fig, geom=(1, 1))
            b.graph_cdf(0, 10, fig=fig, geom=(2, 2))
            context['graph'] = fig.to_html(full_html=False, default_height=500, default_width=700)

    return HttpResponse(template.render(context, request))


def dist_selector(picker: DistributionSelect) -> Exponential | Uniform | Normal | Poisson | Bernoulli | Binomial | None:
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
# Need Large numbers then wheel spin here. Ordered as seen on website navigation bar to avoid confusion


def about(request):
    template = loader.get_template('statVisualiser/about.html')
    context = {}
    return HttpResponse(template.render(context, request))







"""
def index(request):
    graph = ""
    if request.method == "POST":
        form = Picker(request.POST)
        if form.is_valid():
            data_id = int(str(form.cleaned_data.get("Data")))
            equation_name = form.cleaned_data.get("Equation")
            data = Data.objects.get(id=data_id).data
            model = StatModel.objects.get(name=equation_name).id
            graph = makeGraph(model, data)
    models = StatModel.objects.all().values()
    data = Data.objects.all().values()
    data_picker = Picker()
    template = loader.get_template('statVisualiser/index.html')
    context = {
        'models': models,
        'data': data,
        'picker': data_picker,
        'graph' : graph,
    }
    return HttpResponse(template.render(context, request))


def Exponential(request):
    pass



def makeGraph(model, data):
    data = data.split(",")
    if model == 1:
        x = np.linspace(-5, 5, 100)
        y = x ** 2
    if model == 0:
        x = np.linspace(-5, 5, 100)
        y = x

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    fig = plt.figure()
    plt.plot(x, y, 'r')

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

"""
