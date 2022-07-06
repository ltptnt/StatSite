import distributions as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as an
import ffmpeg as ff


"""
Used to generate a frame of the large numbers animation. var must be Binomial. converge can be Normal or Poisson
"""


def binomial_norm_frame(trials: int):
    binom = dt.Binomial(trials, 0.5)
    converge = dt.Normal(binom.trials * binom.prob, np.sqrt(binom.trials * binom.prob * (1 - binom.prob)))
    minim, maxim = binom.get_region()
    #ax = fig.add_subplot()
    ax = binom.graph_pdf(minim, maxim).axes[0]
    #converge.graph_pdf(minim, maxim, fig)

    x1 = np.linspace(minim, maxim, 10 ** 5)
    pdf = [converge.pdf(i) for i in x1]
    ax.plot(x1, pdf, color='r')
    ax.legend([str(converge), str(binom)], loc='upper left')
    ax.set_title("PMF of {} overlayed with the normal approximation".format(str(binom)))
    ax.set_xlabel("x")
    ax.set_ylabel("Probability X=x")

    return ax.lines


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


def approx_anim(fig):
    animation = an.FuncAnimation(fig, binomial_norm_frame, frames=[i for i in range(10, 100, 10)], interval=10, blit=True, repeat=True )
    return animation


def main():
    fig = plt.figure()
    fig = dt.Binomial(10,0.5).graph_pdf(0, 10, titles=True)
    fig.show()
    fig = dt.Normal(0, 1).graph_pdf(-3, 3)
    fig.show()

    #anim = approx_anim(fig)
    #writer = an.FFMpegWriter(fps=60)
    #anim.save("bingo.mp4") #(r"C:\Users\61435\Desktop\Animations", writer=writer)


if __name__ == '__main__':
    main()
