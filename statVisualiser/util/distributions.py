import math
import random as rd
from abc import ABC, abstractmethod

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import scipy.stats as st
DP = 3
"""
Abstract class representing the basic methods a distribution needs
"""
class Variable(ABC):
    @property
    @abstractmethod
    def continuous(self):
        pass

    @abstractmethod
    def get_region(self):
        pass

    @abstractmethod
    def pdf(self, x: float) -> float:
        pass

    def cdf(self, x: float):
        pmf = [self.pdf(i) for i in range(0, int(x))]
        return sum(pmf)

    def prob_between(self, x: float, y: float):
        return self.cdf(y)-self.cdf(x)

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

    def graph_cdf(self, minim: float, maxim: float, fig=None, geom=None, titles=False,  **kwargs):
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


class Exponential(Variable):
    continuous = True

    def __init__(self, rate: float):
        self.rate = rate

    def get_region(self):
        return 0, 10/self.rate

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

    def __init__(self, minim, maxim):
        self.max = maxim
        self.min = minim

    def get_region(self):
        return self.min, self.max

    def pdf(self, x: float) -> float:
        return 1/(self.max-self.min) if self.min <= x <= self.max else 0

    def cdf(self, x: float) -> float:
        return (x-self.min)/(self.max-self.min) if self.min <= x <= self.max else 0

    def inverse_cdf(self, x: float) -> float:
        return x*(self.max-self.min) + self.min if self.min <= x <= self.max else 0

    def __str__(self):
        return "U[{}, {}]".format(self.min, self.max)


class Normal(Variable):
    continuous = True

    def __init__(self, mean: float, deviation: float):
        self.mean = mean
        self.sd = deviation
        self.cache = {"pdf_const": 1/(self.sd*np.sqrt(2*np.pi))}

    def get_region(self):
        return -4*self.sd+self.mean, 4*self.sd+self.mean

    def pdf(self, x: float) -> float:
        return self.cache.get("pdf_const") * math.e**(-1/2*((x-self.mean)/self.sd)**2)

    def cdf(self, x: int):
        x = (x-self.mean)/self.sd
        return st.norm.cdf(x, loc=0, scale=1)

    def inverse_cdf(self, x: float):
        return st.norm.ppf(x, loc=self.mean, scale=self.sd)

    def trial(self) -> float:
        return st.norm.rvs(loc=self.mean, scale=self.sd)

    def __str__(self):
        return "N({0},{1}\u00b2)".format(round(self.mean, 3), round(self.sd, DP))


class Poisson(Variable):
    continuous = False

    def __init__(self, rate: float):
        self.rate = rate

    def get_region(self):
        return 0, self.rate * 5+1

    def pdf(self, x: int):
        if x % 1 != 0 or x < 0:
            return 0
        else:
            a = -self.rate + x * math.log(self.rate) - math.log(math.factorial(x))
            return np.e ** a

    def inverse_cdf(self, x: float):
        val = 0
        found = False
        next = self.cdf(val + 1)
        current = self.cdf(val)

        if current <= x < self.cdf(val + 1):
            return val
        val += 1
        "Loop depth here could be a problem"
        while not found:
            current = next
            next = self.cdf(val + 1)
            if current <= x < next:
                return val
            val += 1

    def __str__(self):
        return "Poi({0})".format(int(self.rate))


class Bernoulli(Variable):
    continuous = False

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
                return 1-self.prob
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

    def __init__(self, trials: int, prob: float):
        self.prob = prob
        self.trials = trials
        self.cache = {
            "log_p":  math.log(self.prob),
            "log_q": math.log(1 - self.prob)
        }

    def get_region(self):
        return 0, self.trials

    def inverse_cdf(self, x: float):
        if x == 1:
            return self.trials
        val = 0
        found = False
        next = self.cdf(val + 1)
        current = self.cdf(val)
        print(current, next)
        if current <= x < self.cdf(val + 1):
            print("prankt")
            return val
        val += 1
        "Loop depth here could be a problem"
        while not found:
            current = next
            next = self.cdf(val + 1)
            if current <= x < next:
                return val
            val += 1

    def trial(self):
        event = Bernoulli(self.prob)
        return sum([event.trial() for i in range(0, self.trials)])

    def pdf(self, x: int):
        if not 0 <= x <= self.trials:
            return 0
        a = math.log(math.comb(self.trials, x)) + x * self.cache.get("log_p") \
            + (self.trials - x) * self.cache.get("log_q")
        return np.e ** a

    def __str__(self):
        return "Bin({0}, {1})".format(self.trials, round(self.prob,DP))

#This might be useless
def graph_density_sum(var1, var2):
    #Generating the supported region through inverse-transform processes
    var1_trials = [var1.trial() for _ in range(10**5)]
    var2_trials = [var2.trial() for _ in range(10**5)]
    convolution_trials = [var1_trials[i]*var2_trials[i] for i in range(len(var1_trials))]
    fig = px.density_heatmap(x=var1_trials, y=var2_trials, z=convolution_trials, histfunc="count",
                             marginal_x="histogram", marginal_y="histogram",
                             )
    return fig


def two_var_3d(var1, var2, fig=None, geom=None, type="product", **kwargs):
    var1_trials = [var1.trial() for i in range(10 ** 5)]
    var2_trials = [var2.trial() for i in range(10 ** 5)]
    match type:
        case "product":
            title_text = "Product of "
            convolution_trials = [var1_trials[i] * var2_trials[i] for i in range(len(var1_trials))]
        case "sum":
            title_text = "Sum of "
            convolution_trials = [var1_trials[i] + var2_trials[i] for i in range(len(var1_trials))]
        case _:
            raise TypeError("Only product and sum are supported")
    if fig is None:
        fig = px.scatter_3d(x=var1_trials, y=var2_trials, z=convolution_trials)
    else:
        trace = px.scatter_3d(x=var1_trials, y=var2_trials, z=convolution_trials)
        if geom is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=geom[0], col=geom[1])
    fig.update_layout(title="Simulated supported region of the convolution, ",
                      scene=dict(xaxis_title=str(var1),
                                 yaxis_title=str(var2),
                                 zaxis_title=title_text + str(var1) + " and " + str(var2)))
    return fig

def convolution_pdf(var1, var2):
    min1, max1 = var1.get_region()
    min2, max2 = var2.get_region()
    if var1.continuous:
        var1_region = np.linspace(max(min1, -100), min(max1, 100), 10**3)
    else:
        var1_region = [i for i in range(min1, max1+1)]

    if var2.continuous:
        var2_region = np.linspace(max(min2, -100), min(max2, 100), 10**3)
    else:
        var2_region = [i for i in range(int(min2), int(max2+1))]

    var1_pdf = [var1.pdf(i) for i in var1_region]
    var2_pdf = [var2.pdf(i) for i in var2_region]
    conv_pdf = [[] for i in var1_region]
    for i in range(len(var1_region)):
        for y in var2_pdf:
            conv_pdf[i].append(var1_pdf[i]*y)
    fig = go.Figure(data=go.Surface(y=var1_region, x=var2_region, z=conv_pdf))
    fig.update_layout(
        title="PDF of the convolution of " + str(var1) + " and " + str(var2),
        coloraxis_colorbar=dict(title="Density"),
        scene=dict(
            xaxis_title="X ~ " + str(var2),
            yaxis_title="Y ~ " + str(var1),
            zaxis_title="Probability density of X*Y"),
    )
    return fig


def convolution_cdf(var1, var2):
    var1_trials = [var1.trial() for i in range(10 ** 5)]
    var2_trials = [var2.trial() for i in range(10 ** 5)]
    convolution_trials = [var1_trials[i] * var2_trials[i] for i in range(len(var1_trials))]
    cdf_points = np.linspace(min(convolution_trials), max(convolution_trials), 1000)
    probability = []
    for point in cdf_points:
        probability.append(len([i for i in convolution_trials if i <= point])/10**5)
    fig = px.scatter(x=cdf_points, y=probability)
    fig.update_layout(title="CDF of the convolution of " + str(var1) + " and " + str(var2),
                      xaxis_title="x",
                      yaxis_title="Probability X=x")
    return fig
    #fig = px.scatter_3d(x=var1_trials, y=var2_trials, z=convolution_trials)


def main():
    fig = make_subplots(rows=2, cols=2)
    a = Exponential(2)
    b = Normal(3, 1)
    two_var_3d(a, b).show()
    convolution_pdf(a,b).show()
    convolution_cdf(a,b).show()

if __name__ == '__main__':
    main()
