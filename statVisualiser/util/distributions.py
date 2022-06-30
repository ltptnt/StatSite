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
    def __init__(self, rate: float):
        self.rate = rate
        continuous = True

    def get_region(self):
        return "(0,∞)"

    def pdf(self, x: float):
        return 0 if x <= 0 else self.rate * np.e ** (-self.rate * x)

    def cdf(self, x: float) -> float:
        return 0 if x <= 0 else 1 - np.e ** (-self.rate * x)

    def trial(self) -> float:
        return -np.log(-rd.random() + 1) / self.rate


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
        old=0
        if old <= current < next:
            return val
        val += 1
        "Loop depth here could be a problem"
        while not found:
            old=current
            current=next
            next=self.cdf(val+1)
            if old <= current < next:
                return val
            val += 1


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

class Binomial(Variable):
    continuous = False
    def __init__(self,trials,prob):
        self.prob = prob
        self.trials=prob

    def trial(self):
        event = Bernoulli(self.prob)
        return sum([event.trial() for i in range(0,self.trials)])

    def pdf(self, x):
        return math.comb(self.trials, x) * (self.rate) ** x * (1 - self.rate) ** (self.trials-x)

class Convolution(object):
    def __init__(self, var1:Variable, var2:Variable):
        self.var1=var1
        self.var2=var2
        continuous = var1.continuous or var2.continuous

    def graph_supported_region(self):
        var1_trials = [self.var1.trial() for i in range(10**5)]
        var2_trials = [self.var2.trial() for i in range(10**5)]
        convolution_trials=[var1_trials[i]*var2_trials[i] for i in var1_trials]

        fig, axes = plt.subplot(2)
        fig.suptitle("work")
        axes[0].scatter(convolution_trials,var1_trials)
        axes[1].scatter(convolution_trials,var2_trials)
        print(fig)


def main():
    a = Poisson(2)
    b = Bernoulli(0.5)
    c = Convolution(a,b)
    c.graph_supported_region()


if __name__ == '__main__':
    main()