"""AI-generated style starter file with intentional bugs for homework."""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import special


def energy_levels(hbar=1.0, omega=1.0, n_levels=5):
    return np.array([hbar * omega * n for n in range(n_levels)])  # BUG 1


def harmonic_state(n, x):
    herm = special.hermite_poly(n, x)  # BUG 3
    norm = 1.0 / np.sqrt(math.factorial(n))  # BUG 4
    return norm * np.exp(-x**2 / 2) * herm


def main():
    x = np.linspace(-5, 5, 600)
    states = np.array([harmonic_state(n, x) for n in range(5)])
    states = states[1:]  # BUG 2

    print("First five E_n:", energy_levels())
    plt.figure(figsize=(8, 5))
    for n, psi in enumerate(states):
        plt.plot(x, psi, label=f"n={n}")  # BUG 5
    plt.title("Buggy AI oscillator output")
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
