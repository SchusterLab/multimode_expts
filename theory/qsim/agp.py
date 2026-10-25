"""Closed-system two-time correlations and AGP norms of the central-boson model.

For a probe O diagonal in the Fock basis of one fixed-N sector of dimension D,
with eigenstates |E_i> of H (``theory.qsim.central_boson``),

    C(t) = Tr[O(t) O] / D = sum_ij w_ij cos(2 pi (E_i - E_j) t),   w_ij = |<E_i|O|E_j>|^2 / D,

and the regularized AGP norm, for a parameter g (the coupling) and cutoff mu,

    g^2 ||A_mu||^2 = g^2 sum_ij w_ij Delta_ij^2 / (Delta_ij^2 + mu^2)^2,   Delta_ij = E_i - E_j.

The 2 pi factors cancel in the AGP norm, so it is the same in cyclic or
angular units. This is the infinite-time target: no decay and no finite window.
The measured counterpart is ``fitting.qsim.agp.finite_time_agp``; the same
kernel applied to :func:`two_time_correlation` on the measured grid gives the
closed-system value of the finite window.

Energies, couplings and cutoffs are cyclic MHz, times are us. From jonginn's
10-23 notebook (``agp_storage7_mu1_to10_t1_theory_ENTIERLYCODEX.ipynb``),
generalized to any probe mode, on 2026-10-24.
"""
import numpy as np

PROBES = ("number", "number_squared", "kerr")


def probe_values(fock_basis, mode_index, probe="number"):
    """-> the diagonal of the probe in ``fock_basis`` (states, modes).

    ``probe``: ``number`` n, ``number_squared`` n^2, ``kerr`` n (n - 1) / 2,
    all of mode ``mode_index`` (0 is M1).
    """
    number = np.asarray(fock_basis, dtype=float)[:, mode_index]
    if probe == "number":
        return number
    if probe == "number_squared":
        return number**2
    if probe == "kerr":
        return number * (number - 1) / 2
    raise ValueError(f"probe must be one of {PROBES}")


def spectral_weights(model, values):
    """-> (differences_MHz, weights), both (D, D), for the diagonal probe ``values``.

    ``model`` is a ``fixed_n_hamiltonian`` result. The weights sum to the
    sector mean of values^2, which is C(0).
    """
    states = np.asarray(model.states)
    values = np.asarray(values, dtype=float)
    matrix = states.conj().T @ (values[:, None] * states)
    weights = np.abs(matrix)**2 / len(values)
    energies = np.asarray(model.energies_MHz)
    return energies[:, None] - energies[None, :], weights


def two_time_correlation(differences_MHz, weights, time_us):
    """-> C(t) at ``time_us``, closed system."""
    phases = 2 * np.pi * np.asarray(time_us, dtype=float)[:, None] * np.ravel(differences_MHz)[None, :]
    return np.cos(phases) @ np.ravel(weights)


def spectral_agp(differences_MHz, weights, cutoff_mhz, coupling_mhz):
    """-> g^2 ||A_mu||^2 (dimensionless) for each cutoff in ``cutoff_mhz``."""
    cutoffs = np.atleast_1d(np.asarray(cutoff_mhz, dtype=float))
    if np.any(cutoffs <= 0):
        raise ValueError("cutoffs must be positive")
    squared = np.ravel(differences_MHz)**2
    filters = squared[None, :] / (squared[None, :] + cutoffs[:, None]**2)**2
    values = coupling_mhz**2 * filters @ np.ravel(weights)
    return values if np.ndim(cutoff_mhz) else float(values[0])
