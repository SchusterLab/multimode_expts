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


def split_dc(differences_MHz, weights, tolerance_mhz=1e-12):
    """-> (dc, mask): the weight of the zero-frequency (diagonal and degenerate) terms, and their mask.

    ``dc`` is the infinite-time average of the closed-system C(t), the part that
    the AGP filter removes only over an infinite window.
    """
    mask = np.isclose(np.asarray(differences_MHz), 0, atol=tolerance_mhz)
    return float(np.asarray(weights)[mask].sum()), mask


def damped_correlation(differences_MHz, weights, time_us, rate_per_us=0.):
    """-> sum_ij w_ij cos(2 pi Delta_ij t) e^{-rate_ij t}: a toy open system.

    ``rate_per_us`` is a scalar or one rate per term (the shape of ``weights``).
    Not a Lindblad model: every term only decays, at the rate given.
    """
    rates = np.broadcast_to(np.asarray(rate_per_us, dtype=float), np.shape(weights))
    times = np.asarray(time_us, dtype=float)[:, None]
    phases = 2 * np.pi * times * np.ravel(differences_MHz)[None, :]
    return (np.cos(phases) * np.exp(-times * np.ravel(rates)[None, :])) @ np.ravel(weights)


def damped_agp(differences_MHz, weights, cutoff_mhz, coupling_mhz, rate_per_us=0.):
    """-> the AGP kernel integral from 0 to infinity of ``damped_correlation`` (dimensionless).

    The trace of a system that decays to 0, zero-padded to infinity. With
    s = mu + rate - i omega, the term integral is Re[(1/s - mu/s^2) / (2 mu)].
    ``rate_per_us=0`` gives :func:`spectral_agp`; a decaying zero-frequency term
    contributes w rate / (2 mu (mu + rate)^2), which the closed system does not have.
    """
    rates = np.ravel(np.broadcast_to(np.asarray(rate_per_us, dtype=float), np.shape(weights)))
    mu = 2 * np.pi * np.atleast_1d(np.asarray(cutoff_mhz, dtype=float))[:, None]
    s = mu + rates[None, :] - 2j * np.pi * np.ravel(differences_MHz)[None, :]
    values = (2 * np.pi * coupling_mhz)**2 * np.real((1 / s - mu / s**2) / (2 * mu)) @ np.ravel(weights)
    return values if np.ndim(cutoff_mhz) else float(values[0])


def window_agp(differences_MHz, weights, cutoff_mhz, coupling_mhz, window_us, rate_per_us=0.):
    """-> the AGP kernel integral from 0 to ``window_us`` of ``damped_correlation``, exact (no grid).

    With s = mu + rate - i omega, the term integral is
    Re[(1 - e^{-sT}) / s - mu (1 - e^{-sT} (1 + sT)) / s^2] / (2 mu). Compare with
    the same kernel on a sampled grid (``fitting.qsim.agp.agp_kernel_weights``)
    to see the sampling error alone; ``window_us`` -> infinity gives :func:`damped_agp`.
    """
    rates = np.ravel(np.broadcast_to(np.asarray(rate_per_us, dtype=float), np.shape(weights)))
    mu = 2 * np.pi * np.atleast_1d(np.asarray(cutoff_mhz, dtype=float))[:, None]
    s = mu + rates[None, :] - 2j * np.pi * np.ravel(differences_MHz)[None, :]
    tail = np.exp(-s * window_us)
    terms = ((1 - tail) / s - mu * (1 - tail * (1 + s * window_us)) / s**2) / (2 * mu)
    values = (2 * np.pi * coupling_mhz)**2 * np.real(terms) @ np.ravel(weights)
    return values if np.ndim(cutoff_mhz) else float(values[0])


def spectral_agp(differences_MHz, weights, cutoff_mhz, coupling_mhz):
    """-> g^2 ||A_mu||^2 (dimensionless) for each cutoff in ``cutoff_mhz``."""
    cutoffs = np.atleast_1d(np.asarray(cutoff_mhz, dtype=float))
    if np.any(cutoffs <= 0):
        raise ValueError("cutoffs must be positive")
    squared = np.ravel(differences_MHz)**2
    filters = squared[None, :] / (squared[None, :] + cutoffs[:, None]**2)**2
    values = coupling_mhz**2 * filters @ np.ravel(weights)
    return values if np.ndim(cutoff_mhz) else float(values[0])


# General operators, the AGP tensor and spectral-function helpers (2026-11-19).
# ``states`` are eigenstates as columns (``np.linalg.eigh``); weights are divided by D, as above.

def energy_differences(energies_MHz):
    """-> (D, D) array of E_i - E_j."""
    energies = np.asarray(energies_MHz, dtype=float)
    return energies[:, None] - energies[None, :]


def operator_weights(states, operator):
    """-> (D, D) weights |<E_i|O|E_j>|^2 / D for any operator matrix O in the basis of ``states``."""
    states = np.asarray(states)
    matrix = states.conj().T @ np.asarray(operator) @ states
    return np.abs(matrix)**2 / len(states)


def cross_weights(states, operators):
    """-> (n, n, D, D) array Re[<E_i|A_a|E_j><E_j|A_b|E_i>] / D for Hermitian operators A_a.

    The diagonal blocks ``[a, a]`` are :func:`operator_weights` of A_a.
    """
    states = np.asarray(states)
    matrices = [states.conj().T @ np.asarray(op) @ states for op in operators]
    out = np.empty((len(matrices), len(matrices)) + matrices[0].shape)
    for a, A in enumerate(matrices):
        for b, B in enumerate(matrices):
            out[a, b] = np.real(A * B.conj()) / len(states)   # <j|B|i> = conj(B_ij) for Hermitian B
    return out


def regularized_qgt(differences_MHz, cross, cutoff_mhz):
    """-> (n, n) regularized metric g_ab = sum_ij w^ab_ij Delta_ij^2 / (Delta_ij^2 + mu^2)^2.

    ``cross`` is a :func:`cross_weights` result. With one operator and coupling 1 this is
    :func:`spectral_agp`. Its diagonal entries are the AGP norms (fidelity susceptibilities)
    of the directions a.
    """
    squared = np.asarray(differences_MHz)**2
    kernel = squared / (squared + cutoff_mhz**2)**2
    return np.einsum("abij,ij->ab", np.asarray(cross), kernel)


def qgt_directions(qgt):
    """-> dict(min_value, max_value, min_angle, max_angle) of a 2 x 2 metric.

    Angles in radians, modulo pi, of the unit direction (cos phi, sin phi) in the parameter
    plane of the two operators; the AGP norm of that direction is the eigenvalue.
    """
    values, vectors = np.linalg.eigh(np.asarray(qgt, dtype=float))
    angles = np.mod(np.arctan2(vectors[1], vectors[0]), np.pi)
    return dict(min_value=values[0], max_value=values[1], min_angle=angles[0], max_angle=angles[1])


def positive_sticks(differences_MHz, weights, tolerance_mhz=1e-9):
    """-> (frequencies, weights, dc): the sticks at Delta > tolerance, and the weight at |Delta| <= tolerance.

    The spectrum is mirrored, so dc + 2 * weights.sum() is C(0) (the sum rule).
    """
    f = np.ravel(differences_MHz)
    w = np.ravel(weights)
    keep = f > tolerance_mhz
    return f[keep], w[keep], float(w[np.abs(f) <= tolerance_mhz].sum())


def binned_density(frequencies_MHz, weights, edges_MHz):
    """-> the stick weight per MHz in each bin of ``edges_MHz``."""
    edges = np.asarray(edges_MHz, dtype=float)
    return np.histogram(frequencies_MHz, edges, weights=weights)[0] / np.diff(edges)


def broadened_spectral_function(frequencies_MHz, weights, omega_MHz, width_MHz):
    """-> sum_k w_k G(omega - f_k), with G a normalized Gaussian of standard deviation ``width_MHz``.

    The smoothing of Kim and Polkovnikov (arXiv:2308.09745, Eq. 11). Pass the sticks to include
    (for their Phi(omega): all pairs i != j, both signs).
    """
    f = np.ravel(np.asarray(frequencies_MHz, dtype=float))
    w = np.ravel(np.asarray(weights, dtype=float))
    out = np.empty(np.size(omega_MHz))
    norm = 1 / (width_MHz * np.sqrt(2 * np.pi))
    for k, omega in enumerate(np.ravel(omega_MHz)):
        out[k] = norm * np.exp(-0.5 * ((omega - f) / width_MHz)**2) @ w
    return out.reshape(np.shape(omega_MHz))
