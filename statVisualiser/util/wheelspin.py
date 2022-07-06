import distributions as dist
import numpy as np
import matplotlib.pyplot as plt


def wheel_spin(bet: float, variable: dist.Variable) -> float:
    return bet * 2 * variable.trial()


def main() -> list:
    variable = dist.Bernoulli(0.75)
    bank = 100
    seq = [bank]
    iterations = 100
    for n in range(1, iterations):
        bet = 2/3*bank
        bank -= bet
        bank += (wheel_spin(bet, variable))
        seq.append(bank)
    plt.scatter([i for i in range(iterations)], seq)
    plt.xlabel("value")
    plt.ylabel("name")
    plt.title("simulation")
    plt.show()
    return seq


if __name__ == '__main__':
    main()