import distributions as dt
import numpy as np
import matplotlib.pyplot as plt


"""
Used to generate a frame of the large numbers animation. var must be Binomial. converge can be Normal or Poisson
"""
def binomial_norm_approx(binom: dt.Binomial):
    fig = plt.figure()
    converge = dt.Normal(binom.trials * binom.prob, np.sqrt(binom.trials * binom.prob * (1 - binom.prob)))
    minim, maxim = binom.get_region()
    #ax = fig.add_subplot()
    fig = binom.graph_pdf(fig, minim, maxim)

    x1 = np.linspace(minim, maxim, 10 ** 5)
    pdf = [converge.pdf(i) for i in x1]
    fig.axes[0].plot(x1, pdf, color='r')
    fig.axes[0].legend([str(converge), str(binom)], loc='upper left')
    fig.axes[0].set_title("PMF of {} overlayed with the normal approximation".format(str(binom)))
    fig.axes[0].set_xlabel("x")
    fig.axes[0].set_ylabel("Probability X=x")
    fig.show()
    return fig


def binomial_poi_approx(binom: dt.Binomial):
    fig = plt.figure()
    poi = dt.Poisson(binom.trials * binom.prob)
    minim, maxim = binom.get_region()
    # ax = fig.add_subplot()
    fig = binom.graph_pdf(fig, minim, maxim)

    x1 = [i for i in range(minim,maxim+1)]
    pdf = [poi.pdf(i) for i in x1]
    fig.axes[0].scatter(x1, pdf, color='r')
    fig.axes[0].legend([str(poi), str(binom)], loc='upper left')
    fig.axes[0].set_title("PMF of {} overlayed with the Poisson approximation".format(str(binom)))
    fig.axes[0].set_xlabel("x")
    fig.axes[0].set_ylabel("Probability X=x")
    fig.show()
    return fig

def main():
    a = dt.Binomial(100, 0.05)
    fig = binomial_poi_approx(a)
    #ax.hist(c[0], bins=[i for i in range(0, a.trials+1)])
    #d = b.graph_pdf(0, a.trials)
    #print(d)
    #plt.plot(d[0], d[1])

    #c = graph_overlay(a, b)
    #c.show()

if __name__ == '__main__':
    main()
