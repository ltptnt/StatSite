import distributions

def wheel_spin(bet: int, variable: Variable) -> float:
    return bet * 2 * variable.trial()

def main() -> list:
    prob = 0.6
    bank = 100
    seq = [bank]
    iterations = 10
    for n in range(1, iterations):
        bet = bank / 2
        bank -= bet
        bank += (wheel_spin(bet, prob))
        seq.append(bank)
    plt.scatter([i for i in range(iterations)], seq)
    plt.xlabel("value")
    plt.ylabel("name")
    plt.title("simulation")
    return seq

if __name__ == '__main__':
    main()