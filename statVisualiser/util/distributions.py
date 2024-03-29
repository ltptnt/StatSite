import math
import random as rd
import time
from abc import ABC, abstractmethod

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import scipy.special as sp
import scipy.stats as st
from plotly.subplots import make_subplots

DP = 3
"""
Abstract class representing the basic methods a distribution needs
"""


class Variable(ABC):
    @property
    @abstractmethod
    def continuous(self):
        pass

    @property
    @abstractmethod
    def cache(self) -> dict:
        pass

    @abstractmethod
    def get_region(self):
        pass

    @abstractmethod
    def pdf(self, x: float) -> float:
        pass

    def cdf(self, x: float):
        if x not in self.cache.keys():
            if x == 0:
                self.cache[x] = self.pdf(x)
            else:
                self.cache[x] = self.pdf(x) + self.cdf(x - 1)
        return self.cache.get(x)

    def prob_between(self, x: float, y: float):
        return np.abs(self.cdf(x) - self.cdf(y))

    """
    Param:
    x: between 0 and 1, representing the probability that X is less than or equal to an unknown x 
    
    returns:
    the value of the distribution that corresponds to the cdf value of the input x.
    """

    @abstractmethod
    def inverse_cdf(self, x: float):
        pass

    def trial(self) -> float:
        return self.inverse_cdf(rd.random())

    @abstractmethod
    def __str__(self):
        pass

    # Additional functionality, specifying what tile to add the pdf to in the case of multiple graphs
    """
    Notes for functionality:
    If just inputting min and max, you get a figure with the axes.
    However, if you want to add it to an existing figure (e.g a collection of graphs), \n
    input the figure object and specify its geometry.
    - accepts plotly kwargs
    - Continuous variables use a scatter plot
    - Discrete variables use a bar plot
    Requires the figure to have a gridspec. If triple digit specifications are entered, a new gridspec is made \n
    and the graph is plotted to the requested square.
    """

    def graph_pdf(self, minim: float, maxim: float, fig=None, geom=None, titles=False, **kwargs):
        if self.continuous:
            x1 = np.linspace(minim, maxim, 10 ** 5)
            pdf = [self.pdf(i) for i in x1]
            trace = go.Scatter(x=x1, y=pdf, name="PDF of " + str(self), **kwargs)
        else:
            x1 = [i for i in range(int(minim), int(maxim) + 1)]
            pmf = [self.pdf(i) for i in x1]
            trace = go.Bar(x=x1, y=pmf, name="PMF of " + str(self), **kwargs)

        if fig is None:
            fig = make_subplots()
            fig.add_trace(trace)
        else:
            if geom is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=geom[0], col=geom[1])

        if titles:
            fig.update_layout(title=str(self))

        return fig

    def graph_cdf(self, minim: float, maxim: float, fig=None, geom=None, titles=False, **kwargs):
        if self.continuous:
            x1 = np.linspace(minim, maxim, 10 ** 5)
            cdf = [self.cdf(i) for i in x1]
            trace = go.Scatter(x=x1, y=cdf, name="CDF of " + str(self), **kwargs)
        else:
            bot, top = self.get_region()
            cdf = [self.pdf(bot)]
            x1 = [i for i in range(bot, int(maxim) + 1)]
            for i in range(1, len(x1)):
                cdf.append(cdf[i - 1] + self.pdf(i))
            trace = go.Bar(x=x1, y=cdf, name="CDF of " + str(self), **kwargs)
        if fig is None:
            fig = make_subplots()
            if geom is None:
                fig.add_trace(trace)
        if geom is not None:
            fig.add_trace(trace, row=geom[0], col=geom[1])
        if titles:
            fig.update_layout(title="CDF of " + str(self))

        return fig

    """
    Standard error lets user choose the standard error in the sample size, it is disabled by default.
    """

    def generate_dataset(self, trials: int, std_error=0):
        if std_error == 0:
            data = [self.trial() for _ in range(trials)]
        else:
            error = Normal(0, std_error)
            data = [self.trial() + error.trial() for _ in range(trials)]
        return data


class Exponential(Variable):
    continuous = True
    cache = {}

    def __init__(self, rate: float):
        self.rate = rate

    def get_region(self):
        return 0, 10 / self.rate

    def pdf(self, x: float):
        return 0 if x < 0 else self.rate * np.e ** (-self.rate * x)

    def cdf(self, x: float) -> float:
        return 0 if x < 0 else 1 - np.e ** (-self.rate * x)

    def inverse_cdf(self, x: float):
        return -np.log(1 - x) / self.rate

    def __str__(self):
        return "Exp({0})".format(round(self.rate, DP))


class Uniform(Variable):
    continuous = True
    cache = {}

    def __init__(self, minim, maxim):
        self.max = maxim
        self.min = minim

    def get_region(self):
        return self.min - 1, self.max + 1

    def pdf(self, x: float) -> float:
        return 1 / (self.max - self.min) if self.min <= x <= self.max else 0

    def cdf(self, x: float) -> float:
        return (x - self.min) / (self.max - self.min) if self.min <= x <= self.max else 0

    def inverse_cdf(self, x: float) -> float:
        return x * (self.max - self.min) + self.min if 0 <= x <= 1 else 0

    def trial(self) -> float:
        return rd.uniform(self.min, self.max)

    def __str__(self):
        return "U[{}, {}]".format(self.min, self.max)


class Normal(Variable):
    continuous = True
    cache = {}

    def __init__(self, mean: float, deviation: float):
        self.mean = mean
        self.sd = deviation
        self.cache = {"pdf_const": 1 / (self.sd * np.sqrt(2 * np.pi))}

    def get_region(self):
        return -4 * self.sd + self.mean, 4 * self.sd + self.mean

    def pdf(self, x: float) -> float:
        return self.cache.get("pdf_const") * math.e ** (-1 / 2 * ((x - self.mean) / self.sd) ** 2)

    def cdf(self, x: int):
        x = (x - self.mean) / self.sd
        return sp.ndtr(x)

    def inverse_cdf(self, x: float):
        return st.norm.ppf(x, loc=self.mean, scale=self.sd)

    def trial(self) -> float:
        return st.norm.rvs(loc=self.mean, scale=self.sd)

    def __str__(self):
        return "N({0},{1}\u00b2)".format(round(self.mean, 3), round(self.sd, DP))


class Poisson(Variable):
    continuous = False
    cache = {}

    def __init__(self, rate: float):
        self.rate = rate

    def get_region(self):
        return 0, self.rate * 5 + 1

    def pdf(self, x: int):
        if x % 1 != 0 or x < 0:
            return 0
        else:
            a = -self.rate + x * math.log(self.rate) - math.log(math.factorial(x))
            return np.e ** a

    def inverse_cdf(self, x: float):
        val = 0
        next_val = self.cdf(val + 1)

        if x < self.cdf(val):
            return val

        if self.cdf(val) <= x < next_val:
            return val + 1
        val += 1
        while True:
            next_val = self.cdf(val + 1)
            if np.abs(next_val - 1) < 0.001:
                next_val = 1.01
            if x < next_val:
                return val + 1
            val += 1

            if val > 10 ** 5:
                break

    def __str__(self):
        return "Poi({0})".format(int(self.rate))


class Bernoulli(Variable):
    continuous = False
    cache = {}

    def __init__(self, prob):
        self.prob = prob

    def get_region(self):
        return 0, 1

    def inverse_cdf(self, x: float):
        if x <= self.prob:
            return 1
        return 0

    def pdf(self, x: int):
        match x:
            case 0:
                return 1 - self.prob
            case 1:
                return self.prob
            case _:
                return 0

    def cdf(self, x: int):
        return x if x in [0, 1] else 0

    def __str__(self):
        return "Ber({0})".format(self.prob)


class Binomial(Variable):
    continuous = False
    cache = {}

    def __init__(self, trials: int, prob: float):
        self.prob = prob
        self.trials = trials
        try:
            self.cache = {
                "log_p": math.log(self.prob),
                "log_q": math.log(1 - self.prob)
            }
        except ValueError:
            raise ValueError("Invalid input for binomial probability")

    def get_region(self):
        return 0, self.trials

    def inverse_cdf(self, x: float):
        if x == 1:
            return self.trials
        val = 0

        if x < self.cdf(0):
            return val

        if self.cdf(0) <= x < self.cdf(val + 1):
            return val + 1
        val += 1
        "Loop depth here could be a problem"
        while True:
            if val + 1 == self.trials:
                return self.trials

            next_val = self.cdf(val + 1)

            if x < next_val:
                return val + 1
            val += 1

    def trial(self):
        event = Bernoulli(self.prob)
        return sum([event.trial() for _ in range(0, self.trials)])

    def pdf(self, x: int):
        if not 0 <= x <= self.trials:
            return 0
        a = math.log(math.comb(self.trials, x)) + x * self.cache.get("log_p") \
            + (self.trials - x) * self.cache.get("log_q")
        return np.e ** a

    def graph_pdf(self, minim: float, maxim: float, fig=None, geom=None, titles=False, **kwargs):
        if self.trials > 100:
            criteria = self.trials * self.prob
            criteria2 = self.trials * (1 - self.prob)
            # Use of binomial and poisson approximation to improve efficiency
            match criteria < 5 or criteria2 < 5:
                case True:
                    criteria = min(criteria, criteria2)
                    approximator = Poisson(criteria).graph_pdf(minim, maxim, fig=fig, geom=geom,
                                                               titles=titles,
                                                               **kwargs)
                    approximator.update_layout(title="PDF of " + str(self))
                    return approximator
                case False:
                    if not criteria > 5 and not criteria2 > 5:
                        return super().graph_pdf(minim, maxim, fig=fig, geom=geom, titles=titles,
                                                 **kwargs)
                    approximator = Normal(criteria, np.sqrt(criteria * (
                            1 - self.prob)))  # .graph_pdf(minim, maxim, fig=fig, geom=geom, titles=titles, **kwargs)
                    x1 = [i + 0.5 for i in range(int(minim), int(maxim) + 1)]
                    pmf = [approximator.pdf(i) for i in x1]
                    trace = go.Bar(x=x1, y=pmf, name="PMF of " + str(self), **kwargs)

                    if fig is None:
                        fig = make_subplots()
                        fig.add_trace(trace)
                    else:
                        if geom is None:
                            fig.add_trace(trace)
                        else:
                            fig.add_trace(trace, row=geom[0], col=geom[1])

                    if titles:
                        fig.update_layout(title=str(self))

                    return fig

        return super().graph_pdf(minim, maxim, fig=fig, geom=geom, titles=titles, **kwargs)

    def graph_cdf(self, minim: float, maxim: float, fig=None, geom=None, titles=False, **kwargs):
        if self.trials < 100:
            return super().graph_cdf(minim, maxim, fig=fig, geom=geom, titles=titles, **kwargs)
        criteria = self.trials * self.prob
        criteria2 = self.trials * (1 - self.prob)
        # Use of binomial and poisson approximation to improve efficiency
        match criteria < 5 or criteria2 < 5:
            case True:
                approximator = Poisson(criteria).graph_cdf(minim, maxim, fig=fig, geom=geom,
                                                           titles=titles,
                                                           **kwargs)
                approximator.update_layout(title="CDF of " + str(self))
                return approximator
            case False:
                if not criteria > 5 and not criteria2 > 5:
                    return super().graph_cdf(minim, maxim, fig=fig, geom=geom, titles=titles,
                                             **kwargs)
                approximator = Normal(criteria, np.sqrt(criteria * (1 - self.prob)))
                x1 = [i for i in range(int(minim), int(maxim) + 1)]
                cdf = [approximator.cdf(i) for i in x1]
                trace = go.Bar(x=x1, y=cdf, name="CDF of " + str(self), **kwargs)

                if fig is None:
                    fig = make_subplots()
                    fig.add_trace(trace)
                else:
                    if geom is None:
                        fig.add_trace(trace)
                    else:
                        fig.add_trace(trace, row=geom[0], col=geom[1])

                if titles:
                    fig.update_layout(title=str(self))

                return fig

    def __str__(self):
        return "Bin({0}, {1})".format(self.trials, round(self.prob, DP))


"""
Note this function is only applicable for independent variables collected in pairs i.e var1_i belongs to \n
 the same event as var2_i."""


def graph_density_product(data1, data2):
    fig = px.density_heatmap(x=data1, y=data2,
                             histnorm="probability", marginal_x='histogram', marginal_y='histogram',
                             title="Joint density heatmap of the input datasets")

    return fig


def two_var_3d(var1, var2, conv_type="product"):
    var1_trials = [var1.trial() for _ in range(10 ** 5)]
    var2_trials = [var2.trial() for _ in range(10 ** 5)]
    match conv_type:
        case "product":
            title_text = "Product of "
            convolution_trials = [var1_trials[i] * var2_trials[i] for i in range(len(var1_trials))]
        case "sum":
            title_text = "Sum of "
            convolution_trials = [var1_trials[i] + var2_trials[i] for i in range(len(var1_trials))]
        case _:
            raise TypeError("Only product and sum are supported")
    fig = px.scatter_3d(x=var1_trials, y=var2_trials, z=convolution_trials)
    fig.update_layout(
        title="Simulated supported region of the convolution of " + str(var1) + " and " + str(var2),
        scene=dict(xaxis_title=str(var1),
                   yaxis_title=str(var2),
                   zaxis_title=title_text + str(var1) + " and " + str(var2)))
    return fig


def convolution_pdf(var1, var2, fig=None, geom=None):
    min1, max1 = var1.get_region()
    min2, max2 = var2.get_region()
    pdf_pmf, dens = ("PMF", "Mass") if not var1.continuous and not var2.continuous else (
    "PDF", "Density")

    if var1.continuous:
        var1_region = np.linspace(max(min1, -100), min(max1, 100), 10 ** 3)
    else:
        var1_region = [i for i in range(int(min1), int(max1) + 1)]

    if var2.continuous:
        var2_region = np.linspace(max(min2, -100), min(max2, 100), 10 ** 3)
    else:
        var2_region = [i for i in range(int(min2), int(max2 + 1))]

    var1_pdf = [var1.pdf(i) for i in var1_region]
    var2_pdf = [var2.pdf(i) for i in var2_region]
    conv_pdf = [[] for _ in var1_region]
    for i in range(len(var1_region)):
        for y in var2_pdf:
            conv_pdf[i].append(var1_pdf[i] * y)
    trace = go.Surface(y=var1_region, x=var2_region, z=conv_pdf)
    if fig is None:
        fig = go.Figure()
        fig.add_trace(trace)
        fig.update_layout(dict(
            title="Joint" + pdf_pmf + " of the convolution of " + str(var1) + " and " + str(var2),
            coloraxis_colorbar=dict(title="Probability" + dens),
            scene=dict(
                xaxis_title="X ~ " + str(var2),
                yaxis_title="Y ~ " + str(var1),
                zaxis_title="Probability " + dens + " of X*Y")), )

    else:
        if geom is None:
            fig.add_trace(trace)
        else:
            row, col = geom
            fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(title_text="X ~ " + str(var2), row=row, col=col)
            fig.update_yaxes(title_text="Y ~ " + str(var1), row=row, col=col)

    return fig


def convolution_cdf(var1, var2, conv_type="product", fig=None, geom=None):
    var1_trials = [var1.trial() for _ in range(10 ** 5)]
    var2_trials = [var2.trial() for _ in range(10 ** 5)]
    match conv_type:
        case "product":
            convolution_trials = [var1_trials[i] * var2_trials[i] for i in range(len(var1_trials))]
        case "sum":
            convolution_trials = [var1_trials[i] + var2_trials[i] for i in range(len(var1_trials))]
        case _:
            raise ValueError("sum or product only champ")

    cdf_points = np.linspace(min(convolution_trials), max(convolution_trials), 1000)
    # continuous_offset = [i - (max(convolution_trials)-min(convolution_trials))/1000 for i in cdf_points]
    probability = []
    for point in cdf_points:
        probability.append(len([i for i in convolution_trials if i <= point]) / 10 ** 5)

    if fig is None:
        fig = px.scatter(x=cdf_points, y=probability)
        fig.update_layout(title="CDF of the convolution of " + str(var1) + " and " + str(var2),
                          xaxis_title="x",
                          yaxis_title="Probability X=x")
    else:

        trace = go.Scatter(x=cdf_points, y=probability)
        # title =

        if geom is None:
            geom = 1, 1
        row, col = geom
        fig.add_trace(trace, row=row, col=col)
        fig.update_yaxes(title_text="Probability X=x", row=row, col=col)
    return fig


"""
Takes an input variable and a dataset of 1 var.
Returns a histogram of the data
and a histogram of the probability density and the prob density of the sample 
"""


def dataset_plots(var: Variable, data: []) -> go.Figure:
    minim2, maxim2 = var.get_region()
    fig2 = make_subplots(rows=2, cols=1, subplot_titles=["Histogram of generated sample",
                                                         "Probability Density of Sample"])
    fig2.add_trace(
        go.Histogram(x=data, name="Sample data"),
        row=1, col=1)
    bins = int(maxim2 - minim2) * 10
    if not var.continuous:
        bins = max(data) + 1
    fig2.add_trace(go.Histogram(x=data, histnorm='probability density',
                                name="Probability density of the sample data"), row=2, col=1)
    fig2.update_yaxes(title_text="Probability density of sample X", row=2, col=1)
    fig2.update_yaxes(title_text="Number of X in each interval", row=1, col=1)
    fig2.update_layout(title_text="Plots of the generated dataset")
    var.graph_pdf(min(data), max(data), fig=fig2, geom=(2, 1))
    return fig2


def main():
    a = Binomial(10000, 0.5)
    c, d = a.get_region()
    start = time.time()
    b = a.graph_pdf(c, d)
    end = time.time()
    print(end - start)
    b.show()
    start = time.time()
    b = a.graph_cdf(c, d)
    end = time.time()
    print(end - start)
    b.show()

    # dataset_plots(a, a.generate_dataset(10000)).show()


# 1000 trials null
# 100 trials

if __name__ == '__main__':
    main()
