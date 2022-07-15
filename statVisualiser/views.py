from django.http import HttpResponse
from django.template import loader
from .models import Distribution
from .forms import Picker
from io import StringIO


def index(request):
    data_picker = Picker()
    dist = Distribution.objects.all().values()
    template = loader.get_template('statVisualiser/index.html')
    context = {
        'dist': dist,
        'picker': data_picker,
    }
    return HttpResponse(template.render(context, request))


def distributions(request):
    template = loader.get_template('statVisualiser/distributions.html')
    context = {}
    return HttpResponse(template.render(context, request))


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
