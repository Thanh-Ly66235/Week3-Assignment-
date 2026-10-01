"""Quantum harmonic oscillator: first five energy levels and wavefunctions.

Corrected version of ``ai_generated_oscillator.py`` (PHYS 7440, Week 03).
Units: hbar = m = omega = 1, so the oscillator length sqrt(hbar / m omega) = 1.

References: Griffiths, Introduction to Quantum Mechanics, 3rd ed.
    E_n     = (n + 1/2) hbar omega                                              Eq. 2.62
    psi_n   = (m omega / pi hbar)^(1/4) (2^n n!)^(-1/2) H_n(xi) exp(-xi^2 / 2)   Eq. 2.86
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import special


def energy_levels(hbar=1.0, omega=1.0, n_levels=5):
    """Return E_n = hbar*omega*(n + 1/2) for n = 0 .. n_levels-1.

    FIX 1 (wrong physics formula): the original returned hbar*omega*n, which has no
    zero-point energy and gives E_0 = 0 in violation of the uncertainty principle.
    """
    return hbar * omega * (np.arange(n_levels) + 0.5)


def harmonic_state(n, x):
    """Return the normalized wavefunction psi_n(x) on the array x (hbar = m = omega = 1).

    FIX 3 (hallucinated API): ``scipy.special.hermite_poly`` does not exist. The physicists'
    Hermite polynomial H_n(x) is ``scipy.special.eval_hermite``. Note that the interpreter's
    suggestion, ``hermitenorm``, is the *probabilists'* polynomial He_n and would be wrong.

    FIX 4 (incorrect normalization): the prefactor must be (2^n n! sqrt(pi))^(-1/2), not
    (n!)^(-1/2), so that the integral of |psi_n|^2 over all x equals 1.
    """
    herm = special.eval_hermite(n, x)
    norm = 1.0 / np.sqrt(2.0**n * math.factorial(n) * np.sqrt(np.pi))
    return norm * np.exp(-x**2 / 2.0) * herm


def main():
    x = np.linspace(-5, 5, 600)

    # FIX 2 (off-by-one): the original sliced ``states = states[1:]`` and threw away the
    # ground state, leaving four curves whose legend labels were then all shifted by one.
    states = np.array([harmonic_state(n, x) for n in range(5)])

    print("First five E_n:", energy_levels())

    plt.figure(figsize=(8, 5))
    for n, psi in enumerate(states):
        plt.plot(x, psi, label=f"n={n}")

    # FIX 5 (wrong output quantity): the curves are the wavefunctions psi_n(x), which go
    # negative, so the axis must not be labelled "Probability density" (that is |psi|^2 >= 0).
    plt.title("Wavefunctions of the quantum harmonic oscillator")
    plt.xlabel(r"$x\ \left(\sqrt{\hbar / m\omega}\right)$")
    plt.ylabel(r"$\psi_n(x)$")
    plt.axhline(0, color="k", linewidth=0.6, alpha=0.5)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
