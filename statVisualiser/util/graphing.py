import matplotlib.pyplot as plt
import numpy as np
import distributions as dist


def graph_pdf(min: float, max: float, variable: dist.Variable, **kwargs):
    if variable.continuous:
        x1 = np.linspace(min, max, 10 ** 5)
        pdf = [variable.pdf(i) for i in x1]
        return plt.plot(x1, pdf, **kwargs)
    else:
        x1 = [i for i in range(min, max)]
        pmf = [variable.pdf(i) for i in x1]
        return plt.scatter(x1, pmf, **kwargs)

def graph_cdf(min: float, max: float, variable: dist.Variable, **kwargs):
    if variable.continuous:
        x1 = np.linspace(min, max, 10 ** 5)
        cdf = [variable.cdf(i) for i in x1]
        return plt.plot(x1, cdf, **kwargs)
    else:
        x1 = [i for i in range(min, max+1)]
        cdf = [variable.cdf(i) for i in x1]
        return plt.scatter(x1, cdf, **kwargs)


a = dist.Exponential(1)

graph_pdf(0, 15, a)
