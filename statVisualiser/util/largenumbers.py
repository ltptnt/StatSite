#Comment out the from .\ line if testing this package alone
from .\
     import distributions as dt
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import time
def binomial_normal(min_trials: int, max_trials: int, prob: float, steps=10):
    fig = make_subplots()
    titles = []
    for trials in range(min_trials, max_trials+1, steps):
        binom = dt.Binomial(trials, prob)
        converge = dt.Normal(binom.trials * binom.prob, np.sqrt(binom.trials * binom.prob * (1 - binom.prob)))
        minim, maxim = binom.get_region()
        brute_binomial(int(minim), maxim, binom, fig=fig)
        converge.graph_pdf(minim, maxim, fig=fig)
        titles.append("Normal Approximation to " + str(binom))
    fig.data[0].visible = True
    fig.update_layout(yaxis_title="Probability X=x")
    increment = []
    for i in range(0, len(fig.data), 2):
        step = dict(method="update",
                    args=[{"visible": [False] * len(fig.data)},
                          {"title": titles[int(i / 2)]}],
                    label=round(i/2) * steps + min_trials
                    )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i + 1] = True  # Toggle i'th trace to "visible"
        increment.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Trials: "},
        pad={"t": 50},
        steps=increment
    )]

    fig.update_layout(
        sliders=sliders
    )
    for i in fig.data:
        i.visible = False
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.update_layout(title=titles[0])
    return fig


def brute_binomial(minim: float, maxim: float, binom: dt.Binomial, fig=None):
    x1 = [i for i in range(int(minim), int(maxim) + 1)]
    pmf = [binom.pdf(i) for i in x1]
    trace = go.Bar(x=x1, y=pmf, name="PMF of " + str(binom))

    if fig is None:
        fig = make_subplots()
    fig.add_trace(trace)

    return fig

def binomial_poi_approx(min_trials: int, max_trials: int, mean: int, steps=10):
    fig = make_subplots()
    converge = dt.Poisson(mean)
    minim, maxim = converge.get_region()
    converge.graph_pdf(0, maxim, fig=fig)

    for trials in range(min_trials, max_trials+1, steps):
        prob = mean/trials
        binom = dt.Binomial(trials, prob)
        brute_binomial(0, maxim, binom, fig=fig)


    fig.update_layout(xaxis_title="Probability X=x", title="Poisson approximation with a mean of " + str(mean))
    increment = []
    for i in range(1, len(fig.data)):
        step = dict(method="update",
                    args=[{"visible": [False] * len(fig.data)}],
                    label=i * steps
                    )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][0] = True
        increment.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Trials: "},
        pad={"t": 50},
        steps=increment
    )]

    fig.update_layout(
        sliders=sliders
    )
    for i in fig.data:
        i.visible = False
    fig.data[0].visible = True
    fig.data[1].visible = True
    return fig


def main():
    #start = time.time()
    #fig = binomial_poi_approx(1, 1000, 4, steps=10)
    #end=time.time()
    #print(end - start)
    #fig = binomial_normal(10, 100,0.5)
    #fig.show()
    pass


if __name__ == '__main__':
    main()
