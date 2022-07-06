import math
import random as rd
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
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

    @abstractmethod
    def trial(self) -> float:
        pass
    @abstractmethod
    def __str__(self):
        pass

# Additional functionality, specifying what tile to add the pdf to in the case of multiple graphs
    """
    Notes for functionality:
    If just inputting min and max, you get a figure with the axes.
    However, if you want to add it to an existing figure (e.g a collection of graphs), \n
    input the figure object and specify its geometry.
    - accepts matplotlib kwargs.
    - Continuous variables use a pyplot plot
    - Discrete variables use a bar plot
    Requires the figure to have a gridspec. If triple digit specifications are entered, a new gridspec is made \n
    and the graph is plotted to the requested square.
    """
    def graph_pdf(self, minim: float, maxim: float, fig=None, geometry=None, titles=False, **kwargs):
        if geometry is None:
            geometry = 111
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(geometry)
        if titles:
            ax.set_xlabel(str(self))
            ax.set_ylabel("Probability X=x")
        if self.continuous:
            x1 = np.linspace(minim, maxim, 10 ** 5)
            pdf = [self.pdf(i) for i in x1]
            ax.plot(x1, pdf, **kwargs)
        else:
            x1 = [i for i in range(int(minim), int(maxim) + 1)]
            pmf = [self.pdf(i) for i in x1]
            ax.bar(x1, pmf, width=1, **kwargs)
        return fig

    def graph_cdf(self, minim: float, maxim: float, fig=None, geometry=None, titles=False,  **kwargs):
        if fig is None:
            fig = plt.figure()
        if geometry is None:
            geometry = 111
        ax = fig.add_subplot(geometry)
        if titles:
            ax.set_xlabel(str(self))
            ax.set_ylabel("Probability X \u2264 x ")
        if self.continuous:
            x1 = np.linspace(minim, maxim, 10 ** 5)
            cdf = [self.cdf(i) for i in x1]
            ax.plot(x1, cdf, **kwargs)
        else:
            bot, top = self.get_region()
            cdf = [self.pdf(bot)]
            x1 = [i for i in range(bot, int(maxim)+1)]
            for i in range(1, len(x1)):
                cdf.append(cdf[i-1]+self.pdf(i))
            ax.bar(x1, cdf, width=1, **kwargs)
        return fig


class Exponential(Variable):
    continuous = True
    def __init__(self, rate: float):
        self.rate = rate

    def get_region(self):
       return 0, 10**6

    def pdf(self, x: float):
        return 0 if x <= 0 else self.rate * np.e ** (-self.rate * x)

    def cdf(self, x: float) -> float:
        return 0 if x <= 0 else 1 - np.e ** (-self.rate * x)

    def trial(self) -> float:
        return -np.log(-rd.random() + 1) / self.rate

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

    def cdf(self, x: float)-> float:
        return (x-self.min)/(self.max-self.min) if self.min <= x <= self.max else 0

    def trial(self) -> float:
        return rd.uniform(self.min, self.max)

    def __str__(self):
        return "U[{}, {}]".format(self.min, self.max)


class Normal(Variable):
    continuous = True

    def __init__(self, mean: float, deviation: float):
        self.mean = mean
        self.sd = deviation

    def get_region(self):
        return 10**6, 10**6

    def pdf(self, x: float) -> float:
        return 1/(self.sd*np.sqrt(2*np.pi)) * math.e**(-1/2*((x-self.mean)/self.sd)**2)

    def cdf(self, x: int):
        x = (x-self.mean)/self.sd
        return st.norm.cdf(x, loc=0, scale=1)

    def trial(self) -> float:
        return st.norm.rvs(loc=self.mean, scale=self.sd)

    def __str__(self):
        return "N({0},{1}\u00b2)".format(round(self.mean, 3), round(self.sd, DP))


class Poisson(Variable):
    continuous = False

    def __init__(self, rate: float):
        self.rate = rate

    def get_region(self):
        return 0, self.rate * 10**5

    def pdf(self, x: int):
        if x % 1 != 0 or x < 0:
            return 0
        else:
            return np.e ** (-self.rate) * self.rate ** x / math.factorial(x)

    def trial(self) -> int:
        found = False
        trial = rd.random()
        val = 0
        next = self.cdf(val+1)
        current = self.cdf(val)
        old = 0
        if current <= trial < self.cdf(val+1):
            return val
        val += 1
        "Loop depth here could be a problem"
        while not found:
            old = current
            current = next
            next = self.cdf(val+1)
            if old <= trial < next:
                return val
            val += 1

    def __str__(self):
        return "Poi({0})".format(round(self.rate, DP))


class Bernoulli(Variable):
    continuous = False

    def __init__(self, prob):
        self.prob = prob

    def get_region(self):
        return 0, 1

    def trial(self):
        if rd.random() <= self.prob:
            return 1
        return 0

    def pdf(self, x: int):
        return self.prob ** x * (1 - self.prob) ** (1 - x)

    def cdf(self, x: int):
        return x

    def __str__(self):
        return "Ber({0})".format(self.prob)


class Binomial(Variable):
    continuous = False

    def __init__(self, trials: int, prob: float):
        self.prob = prob
        self.trials = trials

    def get_region(self):
        return 0, self.trials

    def trial(self):
        event = Bernoulli(self.prob)
        return sum([event.trial() for i in range(0, self.trials)])

    def pdf(self, x: int):
        return math.comb(self.trials, x) * self.prob ** x * (1 - self.prob) ** (self.trials-x)

    def __str__(self):
        return "Bin({0}, {1})".format(self.trials, round(self.prob,DP))


def graph_supported_region(var1, var2):
    #Generating the supported region through inverse-transform processes
    var1_trials = [var1.trial() for _ in range(10**5)]
    var2_trials = [var2.trial() for _ in range(10**5)]
    convolution_trials = [var1_trials[i]*var2_trials[i] for i in range(len(var1_trials))]

    #Creating the figures used
    fig1 = plt.figure(figsize=(8, 8))
    fig1.suptitle("Density plot and cross of the two variables.")
    fig, axs = plt.subplots(2, sharex="col")
    fig.suptitle("Heaps of data")
    #Edits the ratio of histograms to scatter, the number of graphs etc
    gs = fig1.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Settings to set the scope of the plot
    var1_lim, var1_max = min(var1_trials), max(var1_trials)
    var2_lim, var2_max = min(var2_trials), max(var2_trials)

    # Adding the plots
    # The marginal dist vs the convolution for each variable
    axs[0].scatter(convolution_trials, var1_trials)
    axs[0].set_ylabel(str(var1))
    axs[1].scatter(convolution_trials, var2_trials)
    axs[1].set_ylabel(str(var2))
    axs[1].set_xlabel("Convolution of {0}".format("{0} x {1}".format(str(var1),str(var2))))

    # The density of each variable alongside the convolution
    ax = fig1.add_subplot(gs[1, 0])
    ax_histx = fig1.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig1.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.set_xlim(var1_lim, var1_max + 1)
    ax.set_ylim(var2_lim, var2_max + 1)

    if var1.continuous:
        ax_histx.hist(var1_trials, density=True, bins=10**3)
    else:
        ax_histx.hist(var1_trials, density=True)
    if var2.continuous:
        ax_histy.hist(var2_trials, density=True, orientation="horizontal", bins=10**3)
    else:
        ax_histy.hist(var2_trials, density=True, orientation="horizontal")

    ax.scatter(var1_trials, var2_trials)

    #Setting the limits
    axs[0].set_xlim(min(convolution_trials)-1, max(convolution_trials) + 1)
    axs[0].set_ylim(var1_lim, var1_max + 1)
    axs[1].set_ylim(var2_lim, var2_max + 1)

    fig.show()
    fig1.show()
    return fig, fig1


def convolution3d(var1,var2):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    var1_trials = [var1.trial() for i in range(10 ** 5)]
    var2_trials = [var2.trial() for i in range(10 ** 5)]
    convolution_trials = [var1_trials[i] * var2_trials[i] for i in range(len(var1_trials))]
    ax.scatter3D(var1_trials, var2_trials, convolution_trials)
    ax.set_xlabel(str(var1))
    ax.set_ylabel(str(var2))
    ax.set_zlabel("Convolution of {} and {}".format(str(var1), str(var2)))
    return fig


def main():
   # a = Normal(0, 1)
    #b = Exponential(1/10)
   # fig, fig1 = graph_supported_region(a, b)
   # fig.show()
    #fig1.show()
    a = Uniform(1, 2)
    b = Uniform(2, 4)
    a.graph_pdf(0, 3,titles=True).show()
    b.graph_cdf(0, 5, titles=True).show()
    graph_supported_region(a, b)

if __name__ == '__main__':
    main()
