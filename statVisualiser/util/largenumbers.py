import distributions as dt
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.express as px

"""
Used to generate a frame of the large numbers animation. var must be Binomial. converge can be Normal or Poisson
"""


def binomial_normal(min_trials: int, max_trials: int, prob: float, steps=10):
    fig = make_subplots()
    titles = []
    for trials in range(min_trials, max_trials+1, steps):
        binom = dt.Binomial(trials, prob)
        converge = dt.Normal(binom.trials * binom.prob, np.sqrt(binom.trials * binom.prob * (1 - binom.prob)))
        minim, maxim = binom.get_region()
        binom.graph_pdf(0, maxim, fig=fig)
        converge.graph_pdf(0, maxim, fig=fig)
        titles.append("Normal Approximation to " + str(binom))
    fig.data[0].visible = True
    fig.update_layout(xaxis_title="Probability X=x")
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
    return fig


def binomial_poi_approx(min_trials: int, max_trials: int, mean: int, steps=10):
    fig = make_subplots()
    for trials in range(min_trials, max_trials + 1, steps):
        prob = mean/trials
        binom = dt.Binomial(trials, prob)
        converge = dt.Poisson(mean)
        minim, maxim = binom.get_region()
        binom.graph_pdf(0, maxim, fig=fig)
    fig.data[0].visible = True
    fig.update_layout(xaxis_title="Probability X=x", title="Binomial approximation with a mean of " + str(mean))
    increment = []
    for i in range(0, len(fig.data), 2):
        step = dict(method="update",
                    args=[{"visible": [False] * len(fig.data)}],
                    label=round(i / 2) * steps + min_trials
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
    return fig


def main():
    fig = binomial_poi_approx(10, 100, 5)
    #fig = binomial_normal(10, 100,0.5)
    fig.show()
    # anim = approx_anim(fig)
    # writer = an.FFMpegWriter(fps=60)
    # anim.save("bingo.mp4") #(r"C:\Users\61435\Desktop\Animations", writer=writer)


if __name__ == '__main__':
    main()
