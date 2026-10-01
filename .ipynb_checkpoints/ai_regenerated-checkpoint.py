"""One-dimensional quantum harmonic oscillator: energy levels and eigenfunctions.

Formulas (Griffiths, Introduction to Quantum Mechanics, 3rd ed., ch. 2):
    E_n      = (n + 1/2) hbar omega                                    [Eq. 2.62]
    psi_n(x) = (m omega / (pi hbar))^(1/4) (2^n n!)^(-1/2) H_n(xi) exp(-xi^2 / 2),
               xi = sqrt(m omega / hbar) x                              [Eq. 2.86]
With hbar = m = omega = 1 this reduces to
    psi_n(x) = (2^n n! sqrt(pi))^(-1/2) H_n(x) exp(-x^2 / 2).

H_n are the *physicists'* Hermite polynomials (H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, ...),
orthogonal with weight exp(-x^2). These are the ones Griffiths uses (Table 2.1), and they
are what scipy.special.eval_hermite computes. (scipy.special.eval_hermitenorm gives the
probabilists' He_n with weight exp(-x^2/2), which would need a different normalization.)
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_hermite  # physicists' Hermite polynomial H_n(x)


def energy_levels(hbar=1.0, omega=1.0, n_levels=5):
    """Return E_n = (n + 1/2) hbar omega for n = 0 .. n_levels - 1."""
    n = np.arange(n_levels, dtype=float)
    return (n + 0.5) * hbar * omega


def harmonic_state(n, x):
    """Normalized eigenfunction psi_n(x) in units hbar = m = omega = 1."""
    x = np.asarray(x, dtype=float)
    norm = 1.0 / math.sqrt(2.0**n * math.factorial(n) * math.sqrt(math.pi))
    return norm * eval_hermite(n, x) * np.exp(-x**2 / 2.0)


def main():
    n_states = 5
    energies = energy_levels(hbar=1.0, omega=1.0, n_levels=n_states)
    print("First five energy levels (units of hbar*omega):")
    for n, E in enumerate(energies):
        print(f"  E_{n} = {E:.1f}")

    x = np.linspace(-6.0, 6.0, 1201)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in range(n_states):
        ax.plot(x, harmonic_state(n, x), label=f"n={n}")
    ax.set_xlabel(r"$x$  (units of $\sqrt{\hbar / m\omega}$)")
    ax.set_ylabel(r"Wavefunction $\psi_n(x)$")
    ax.set_title("Quantum harmonic oscillator eigenfunctions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
