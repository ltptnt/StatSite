import random as rd
import numpy as np
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

    @abstractmethod
    def cdf(self, x: float) -> float:
        pass

    @abstractmethod
    def trial(self) -> float:
        pass


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
    def __init__(self, rate: float):
        self.rate = rate
        continuous = False

    def get_region(self):
        return "(0,1,..,∞) for x in the integers"

    def pdf(self, x: int):
        if x % 1 != 0 or x < 0:
            return 0
        else:
            return np.e ** (-self.rate) * self.rate ** x / np.factorial(x)

    def cdf(self, x: int):
        pmf = [self.pdf(i) for i in range(0, x + 1)]
        return sum(pmf)

    def trial(self) -> int:
        found = False
        trial = rd.random()
        val=0
        if 0<=cdf(val)<cdf(val+1):
            return val
        val+=1
        while(not found):
            if cdf(val-1)<=cdf(val)<cdf(val+1):
                return val
            val+=1

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

