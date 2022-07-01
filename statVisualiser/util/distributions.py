import math
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod, abstractproperty

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

    def cdf(self, x: int):
        pmf = [self.pdf(i) for i in range(0, x)]
        return sum(pmf)

    @abstractmethod
    def trial(self) -> float:
        pass
    @abstractmethod
    def __str__(self):
        pass

    def graph_pdf(self, min: float, max: float, **kwargs):
        if self.continuous:
            x1 = np.linspace(min, max, 10 ** 5)
            pdf = [self.pdf(i) for i in x1]
            return plt.plot(x1, pdf, **kwargs)
        else:
            x1 = [i for i in range(min, max)]
            pmf = [self.pdf(i) for i in x1]
            return plt.scatter(x1, pmf, **kwargs)

    def graph_cdf(self, min: float, max: float, **kwargs):
        if self.continuous:
            x1 = np.linspace(min, max, 10 ** 5)
            cdf = [self.cdf(i) for i in x1]
            return plt.plot(x1, cdf, **kwargs)
        else:
            x1 = [i for i in range(min, max + 1)]
            cdf = [self.cdf(i) for i in x1]
            return plt.scatter(x1, cdf, **kwargs)


class Exponential(Variable):
    continuous = True
    def __init__(self, rate: float):
        self.rate = rate

    def get_region(self):
        return "(0,∞)"

    def pdf(self, x: float):
        return 0 if x <= 0 else self.rate * np.e ** (-self.rate * x)

    def cdf(self, x: float) -> float:
        return 0 if x <= 0 else 1 - np.e ** (-self.rate * x)

    def trial(self) -> float:
        return -np.log(-rd.random() + 1) / self.rate

    def __str__(self):
        return "Exp({0})".format(self.rate)


class Poisson(Variable):
    continuous = False

    def __init__(self, rate: float):
        self.rate = rate

    def get_region(self):
        return "(0,1,..,∞) for x in the integers"

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
        return "Poi({0})".format(self.rate)


class Bernoulli(Variable):
    continuous = False
    def __init__(self, prob):
        self.prob = prob

    def get_region(self):
        return "{0,1}"

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
        return "{0,...,{0}}".format(self.trials)

    def trial(self):
        event = Bernoulli(self.prob)
        return sum([event.trial() for i in range(0, self.trials)])

    def pdf(self, x: int):
        return math.comb(self.trials, x) * self.prob ** x * (1 - self.prob) ** (self.trials-x)

    def __str__(self):
        return "Bin({0}, {1})".format(self.trials,self.prob)


def graph_supported_region(var1, var2):
    #Generating the supported region through inverse-transform processes
    var1_trials = [var1.trial() for i in range(10**5)]
    var2_trials = [var2.trial() for i in range(10**5)]
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
    fig.show()
    # The density of each variable alongside the convolution
    ax = fig1.add_subplot(gs[1, 0])
    ax_histx = fig1.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig1.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.set_xlim(var1_lim - 1, var1_max + 1)
    ax.set_ylim(var2_lim - 1, var2_max + 1)

    ax.scatter(var1_trials, var2_trials)
    ax_histx.hist(var1_trials, density=True)
    ax_histy.hist(var2_trials, density=True, orientation="horizontal")

    #Setting the limits
    axs[0].set_xlim(min(convolution_trials)-1, max(convolution_trials) + 1)
    axs[0].set_ylim(var1_lim - 1, var1_max + 1)
    axs[1].set_ylim(var2_lim - 1, var2_max + 1)


    fig1.show()
    return fig, fig1

def main():
    a = Exponential(2)
    b = Binomial(4, 0.5)
    graph_supported_region(a,b)


if __name__ == '__main__':
    main()
