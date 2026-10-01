"""Physics-based pytest suite for the quantum harmonic oscillator script (PHYS 7440, Week 03).

Each test checks a physical property of the harmonic oscillator that any correct
implementation must satisfy, independent of how the code is written. Together they
would have caught all five bugs in ``ai_generated_oscillator.py`` before it was run:

    Bug 1  missing zero-point energy          -> test_ground_state_has_zero_point_energy
    Bug 2  ground state dropped (states[1:])   -> test_first_five_states_present_with_correct_node_count
    Bug 3  hallucinated scipy API              -> every test that calls harmonic_state (crashes)
    Bug 4  wrong normalization constant        -> test_wavefunctions_are_unit_normalized
    Bug 5  psi plotted but labelled as density -> test_probability_density_is_non_negative
                                                  + test_plot_label_matches_plotted_quantity
    (all)  is psi_n really the n-th eigenfunction, and does E_n match it?
                                               -> test_wavefunction_satisfies_schrodinger_equation

Units: hbar = m = omega = 1, so the length scale sqrt(hbar / m omega) = 1 and
psi_n(x) = (2^n n! sqrt(pi))^(-1/2) H_n(x) exp(-x^2 / 2)        [Griffiths 3rd ed. Eq. 2.86]
E_n = (n + 1/2) hbar omega                                       [Griffiths 3rd ed. Eq. 2.62]

Select the module under test with the environment variable QHO_MODULE
(default: ai_generated_oscillator). Example:

    QHO_MODULE=ai_generated_oscillator_fixed pytest -v test_quantum_oscillator.py
"""
import importlib
import os

import matplotlib
import numpy as np
import pytest
from scipy.integrate import simpson

matplotlib.use("Agg")  # never open a window during tests
import matplotlib.pyplot as plt  # noqa: E402

MODULE_NAME = os.environ.get("QHO_MODULE", "ai_generated_oscillator")
qho = importlib.import_module(MODULE_NAME)

X = np.linspace(-8.0, 8.0, 4001)  # wide enough that psi_4 has decayed to ~1e-13 at the edges
N_STATES = 5


def _states():
    """Return an array of shape (5, len(X)) holding psi_0 .. psi_4 evaluated on X."""
    return np.array([qho.harmonic_state(n, X) for n in range(N_STATES)])


def _count_nodes(psi, floor=1e-6):
    """Count interior sign changes of psi, ignoring the tails where psi is numerically zero."""
    core = psi[np.abs(psi) > floor * np.abs(psi).max()]
    return int(np.sum(np.diff(np.sign(core)) != 0))


# --------------------------------------------------------------------------------------
# Test 1 — Bug 1
# --------------------------------------------------------------------------------------
def test_ground_state_has_zero_point_energy():
    """Physical property: the ground state energy is E_0 = (1/2) hbar omega, never zero.

    Why: the uncertainty principle forbids a particle from sitting at rest at the bottom
    of the well (x = 0 and p = 0 simultaneously), so the lowest allowed energy is strictly
    positive. Griffiths 3rd ed. Eq. 2.62 gives E_n = (n + 1/2) hbar omega. A formula that
    omits the +1/2 returns E_0 = 0, which is unphysical even though the level *spacing* is
    still correct - exactly the kind of error that looks fine at a glance.
    """
    E = np.asarray(qho.energy_levels(hbar=1.0, omega=1.0, n_levels=N_STATES), dtype=float)
    assert E.shape == (N_STATES,), f"expected {N_STATES} energy levels, got shape {E.shape}"
    assert E[0] == pytest.approx(0.5), f"E_0 should be 0.5 hbar*omega (zero-point energy), got {E[0]}"
    assert np.allclose(E, np.arange(N_STATES) + 0.5), f"E_n should be n + 1/2, got {E}"


# --------------------------------------------------------------------------------------
# Test 2 — Bug 1 (independent check of the spectrum's structure)
# --------------------------------------------------------------------------------------
def test_energy_levels_are_equally_spaced_by_hbar_omega():
    """Physical property: consecutive harmonic-oscillator levels differ by exactly hbar omega,
    and the whole spectrum scales linearly with both hbar and omega.

    Why: the equal spacing is the defining feature of the oscillator spectrum (it is what
    makes the ladder operators work, Griffiths Sec. 2.3.1). Checking the spacing separately
    from E_0 distinguishes 'wrong offset' from 'wrong formula altogether', and checking the
    scaling catches implementations that hard-code hbar = omega = 1.
    """
    E = np.asarray(qho.energy_levels(hbar=1.0, omega=1.0, n_levels=N_STATES), dtype=float)
    assert np.allclose(np.diff(E), 1.0), f"level spacing should be hbar*omega = 1, got {np.diff(E)}"

    E_scaled = np.asarray(qho.energy_levels(hbar=2.0, omega=3.0, n_levels=N_STATES), dtype=float)
    assert np.allclose(E_scaled, 6.0 * (np.arange(N_STATES) + 0.5)), (
        "energy_levels must scale as hbar*omega*(n + 1/2) for general hbar, omega"
    )


# --------------------------------------------------------------------------------------
# Test 3 — Bug 4
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", range(N_STATES))
def test_wavefunctions_are_unit_normalized(n):
    """Physical property: the integral of |psi_n|^2 over all space equals 1 for every n.

    Why: |psi|^2 is a probability density, so the total probability of finding the particle
    *somewhere* must be exactly 1 (Griffiths Eq. 1.20). The correct prefactor is
    (2^n n! sqrt(pi))^(-1/2) (Eq. 2.86 with hbar = m = omega = 1). Dropping the 2^n and
    sqrt(pi) factors gives integrals of 1.77, 3.54, 7.09, 14.2, 28.4 for n = 0..4 - an error
    that grows exponentially with n, so it is tested for each state separately.
    """
    psi = np.asarray(qho.harmonic_state(n, X), dtype=float)
    total_probability = simpson(psi**2, x=X)
    assert total_probability == pytest.approx(1.0, abs=1e-6), (
        f"integral of |psi_{n}|^2 dx = {total_probability:.6f}, expected 1"
    )


# --------------------------------------------------------------------------------------
# Test 4 — Bug 2 (and the legend off-by-one that comes with it)
# --------------------------------------------------------------------------------------
def test_first_five_states_present_with_correct_node_count():
    """Physical property: the set of 'first five states' must contain psi_0 .. psi_4, and the
    n-th eigenfunction has exactly n nodes (interior zero crossings).

    Why: the node theorem for 1-D bound states (Griffiths Sec. 2.3, Fig. 2.7) says the ground
    state has no nodes and each successive level gains one. So a node count is a fingerprint
    of the quantum number that does not depend on labels or array indices. If the ground
    state is sliced away (states[1:]) or the labels are shifted by one, the state in position
    k has k + 1 nodes instead of k and this test fails. It also checks the ground state is
    strictly positive everywhere, which only psi_0 satisfies.
    """
    states = _states()
    assert states.shape[0] == N_STATES, f"expected {N_STATES} states, got {states.shape[0]}"

    nodes = [_count_nodes(psi) for psi in states]
    assert nodes == list(range(N_STATES)), (
        f"state k must have k nodes; got node counts {nodes} for k = 0..{N_STATES - 1}"
    )
    assert np.all(states[0] > 0), "the ground state psi_0 must have no sign change"

    # parity check as a second, independent fingerprint: psi_n(-x) = (-1)^n psi_n(x)
    for n, psi in enumerate(states):
        assert np.allclose(psi[::-1], (-1) ** n * psi, atol=1e-12), f"psi_{n} has the wrong parity"


# --------------------------------------------------------------------------------------
# Test 5 — Bug 5
# --------------------------------------------------------------------------------------
def test_probability_density_is_non_negative():
    """Physical property: a probability density can never be negative, so whatever is
    presented as 'probability density' must satisfy |psi_n|^2 >= 0 at every x.

    Why: psi_n itself is real and changes sign n times, dipping to about -0.64 for n = 1.
    Plotting psi under a 'Probability density' axis label therefore shows negative
    probabilities. This test computes the density explicitly and also confirms that the raw
    wavefunction for n >= 1 does go negative - i.e. that psi and |psi|^2 are genuinely
    different quantities, so the plot must commit to one of them.
    """
    states = _states()
    density = states**2
    assert np.all(density >= 0.0), "|psi|^2 must be non-negative everywhere"
    assert np.all(density.max(axis=1) > 0.0), "each density must be non-trivial"

    for n in range(1, N_STATES):
        assert states[n].min() < 0.0, (
            f"psi_{n} should take negative values; if it does not, it is not a wavefunction"
        )


# --------------------------------------------------------------------------------------
# Test 6 — Bug 5 (checks the figure the script actually produces)
# --------------------------------------------------------------------------------------
def _identify_state(x, y):
    """Return (n, kind) for a plotted curve, or (None, reason) if it is not an oscillator state.

    kind is 'psi' if y matches a*psi_n(x) + b, 'density' if y matches a*|psi_n(x)|^2 + b with
    a > 0. The vertical offset b and scale a are fitted by least squares, so a curve drawn
    at height E_n inside the potential well, or scaled for visibility, is still recognised.
    The identification is against the exact analytic states on the script's own x grid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.ptp(y) < 1e-12:
        return None, "flat line"
    ones = np.ones_like(x)
    for n in range(N_STATES):
        psi_ref = np.asarray(qho.harmonic_state(n, x), dtype=float)
        for kind, ref in (("psi", psi_ref), ("density", psi_ref**2)):
            A = np.column_stack([ref, ones])
            (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
            if kind == "density" and a <= 0:
                continue
            if np.allclose(y, a * ref + b, atol=1e-6 * np.ptp(y)):
                return n, kind
    return None, "no match to psi_0..psi_4 or their densities"


def test_plot_label_matches_plotted_quantity(monkeypatch):
    """Physical property: the figure must show the first five stationary states, each legend
    label must carry the true quantum number of its curve, and the y-axis label must name
    the quantity actually drawn - a probability density is non-negative, a wavefunction is not.

    Why: this is the output-quantity bug in its natural habitat - the figure. main() is run
    with plt.show() suppressed and the live Axes is inspected. Each plotted curve is
    identified by least-squares comparison against the analytic psi_0..psi_4 and |psi_n|^2
    (allowing a vertical offset and a scale factor, so the textbook presentation of states
    drawn at height E_n inside V(x) is accepted, as are extra reference lines such as the
    potential). The test is therefore agnostic about *how* the author plots, and fails only
    on genuine mislabelling: a legend 'n=k' on a curve that is really psi_{k+1} (Bug 2's
    off-by-one), or an axis that says 'probability density' over curves that go negative
    (Bug 5).
    """
    if not hasattr(qho, "main"):
        pytest.skip("module has no main(); nothing is plotted")

    captured = {}
    monkeypatch.setattr(plt, "show", lambda *a, **k: captured.setdefault("fig", plt.gcf()))
    plt.close("all")
    qho.main()
    fig = captured.get("fig", plt.gcf())
    ax = fig.axes[0]

    identified = []   # (n, kind, line) for every curve that is an oscillator state
    for ln in ax.get_lines():
        if len(ln.get_xdata()) <= 10:
            continue
        n, kind = _identify_state(ln.get_xdata(), ln.get_ydata())
        if n is not None:
            identified.append((n, kind, ln))
    assert sorted(n for n, _, _ in identified) == list(range(N_STATES)), (
        f"expected the states n=0..{N_STATES - 1} to be plotted exactly once each; "
        f"found n = {sorted(n for n, _, _ in identified)}"
    )

    # legend label must state the curve's true quantum number (Bug 2's legend off-by-one)
    for n, kind, ln in identified:
        label = ln.get_label()
        digits = [int(ch) for ch in label if ch.isdigit()]
        assert digits, f"legend label {label!r} does not state the quantum number"
        assert digits[0] == n, f"legend says {label!r} but the curve is {kind} of the n={n} state"

    # y-axis label must agree with what is drawn (Bug 5)
    ylabel = ax.get_ylabel().lower()
    says_density = "density" in ylabel or "probab" in ylabel or "|" in ylabel or "^2" in ylabel
    kinds = {kind for _, kind, _ in identified}
    if says_density:
        assert kinds == {"density"}, (
            f"y-axis says {ax.get_ylabel()!r} but the curves are the wavefunctions psi_n, which go negative"
        )
    elif "psi" in ylabel or "wavefunction" in ylabel or "ψ" in ylabel:
        assert kinds == {"psi"}, (
            f"y-axis says {ax.get_ylabel()!r} but the curves are |psi_n|^2"
        )
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Test 7 — closes the gap the other tests leave: is psi_n actually an eigenfunction?
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", range(N_STATES))
def test_wavefunction_satisfies_schrodinger_equation(n):
    """Physical property: psi_n is an eigenfunction of the oscillator Hamiltonian
    H = -(1/2) d^2/dx^2 + (1/2) x^2 (hbar = m = omega = 1) with eigenvalue n + 1/2, and that
    eigenvalue agrees with what energy_levels() reports.

    Why: every other test checks a *property* of the oscillator states - energy offset,
    normalization, nodes, parity, sign. A function could satisfy all of them and still be
    the wrong function (a Gaussian of the wrong width, or the probabilists' He_n with a
    compensating normalization). Applying the Hamiltonian is the definitive check. Two
    quantities are computed with a second-order finite difference on the test grid:
    the expectation value <psi|H|psi>, which must equal n + 1/2, and the residual
    ||(H - E) psi||^2, i.e. the energy *variance*, which vanishes only for a true eigenstate
    (Griffiths 3rd ed. Eq. 2.45 for H, Eq. 2.62 for E_n, Sec. 3.2.2 on determinate states).
    This is also the only test that couples energy_levels() to harmonic_state(), so it would
    notice if the two functions silently assumed different values of hbar, m or omega.
    """
    psi = np.asarray(qho.harmonic_state(n, X), dtype=float)
    dx = X[1] - X[0]
    d2psi = np.zeros_like(psi)
    d2psi[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dx**2   # O(dx^2) central difference
    H_psi = -0.5 * d2psi + 0.5 * X**2 * psi

    E_measured = simpson(psi * H_psi, x=X)                  # <psi|H|psi>, psi is unit-normalized
    variance = simpson((H_psi - E_measured * psi) ** 2, x=X)  # <(H - <H>)^2> = 0 for an eigenstate
    E_reported = float(np.asarray(qho.energy_levels(n_levels=N_STATES), dtype=float)[n])

    assert E_measured == pytest.approx(n + 0.5, abs=1e-4), (
        f"<psi_{n}|H|psi_{n}> = {E_measured:.6f}, expected {n + 0.5}"
    )
    assert variance < 1e-6, (
        f"psi_{n} is not an eigenstate: ||(H - E)psi||^2 = {variance:.2e} (should be ~0)"
    )
    assert E_reported == pytest.approx(E_measured, abs=1e-4), (
        f"energy_levels()[{n}] = {E_reported} disagrees with <H> = {E_measured:.6f} of harmonic_state({n})"
    )
