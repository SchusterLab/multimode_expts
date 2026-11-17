"""Closed-system predictions for the exact disorder table used in a scan."""
import numpy as np

from theory.qsim.agp import (probe_values, spectral_agp, spectral_weights,
                             two_time_correlation, window_agp)
from theory.qsim.central_boson import fixed_n_hamiltonian


def ensemble_prediction(photon_number, detunings_mhz, couplings_mhz, kerr_mhz,
                        mode_index, time_us, mu_khz, probe="number"):
    """Average existing model predictions over the supplied pulse detunings.

    Rows are realizations and columns are storages, in the acquisition order.
    Pulse detunings are passed directly to ``fixed_n_hamiltonian``, which
    already implements their onsite-energy sign. Cutoffs are cyclic kHz.
    ``agp_theory_window`` is the continuous-time finite-window integral;
    ``agp_theory`` is the infinite-time spectral sum.
    """
    detunings_mhz = np.asarray(detunings_mhz, dtype=float)
    couplings_mhz = np.asarray(couplings_mhz, dtype=float)
    times = np.asarray(time_us, dtype=float)
    cutoffs_mhz = np.asarray(mu_khz, dtype=float).reshape(-1) / 1000
    coupling_mhz = float(np.mean(np.abs(couplings_mhz)))
    correlation = np.zeros(times.shape)
    finite_window = np.zeros(cutoffs_mhz.shape)
    spectral = np.zeros(cutoffs_mhz.shape)
    for detunings in detunings_mhz:
        model = fixed_n_hamiltonian(photon_number, len(couplings_mhz) + 1,
                                    detunings, couplings_mhz, kerr_mhz)
        values = probe_values(model.fock_basis, mode_index, probe)
        differences, weights = spectral_weights(model, values)
        correlation += two_time_correlation(differences, weights, times)
        finite_window += window_agp(differences, weights, cutoffs_mhz,
                                    coupling_mhz, times[-1])
        spectral += spectral_agp(differences, weights, cutoffs_mhz, coupling_mhz)
    realization_count = len(detunings_mhz)
    return dict(C1_theory=correlation / realization_count,
                agp_theory_window=finite_window / realization_count,
                agp_theory=spectral / realization_count,
                theory_kerr_mhz=np.asarray(kerr_mhz))


def metric_probe_values(occupations, swap_stors, directions):
    """Return Fock-diagonal derivatives for (K, pulse-detuning) coordinates.

    The pulse detuning has onsite energy -delta, so its derivative is -n.
    The Kerr derivative is n_M1 (n_M1 - 1) / 2. Columns follow directions.
    """
    mode_order = [0, *swap_stors]
    return np.column_stack([
        probe_values(occupations, mode_order.index(direction),
                     "kerr" if direction == 0 else "number") * (1 if direction == 0 else -1)
        for direction in directions])


def metric_parameter_projection(directions, swap_stors, disorder_patterns):
    """Return each realization's Jacobian from (K,W) to measured local axes."""
    patterns = np.asarray(disorder_patterns, dtype=float)
    projection = np.zeros((len(patterns), len(directions), 2))
    for axis, direction in enumerate(directions):
        if direction == 0:
            projection[:, axis, 0] = 1
        else:
            projection[:, axis, 1] = patterns[:, list(swap_stors).index(direction)]
    coordinate_available = np.array([0 in directions,
                                      all(storage in directions for storage in swap_stors)])
    return projection, np.logical_and.outer(coordinate_available, coordinate_available)


def _metric_spectral_sums(model, values, time_us, cutoffs_mhz):
    """Use signed cross spectral weights in the existing AGP spectral sums."""
    states = np.asarray(model.states)
    matrix_elements = np.einsum("im,ia,in->amn", states.conj(), values, states)
    differences = model.energies_MHz[:, None] - model.energies_MHz[None, :]
    axis_count = values.shape[-1]
    correlation = np.empty((len(time_us), axis_count, axis_count))
    spectral = np.empty((len(cutoffs_mhz), axis_count, axis_count))
    window = np.empty_like(spectral)
    # A 1-kHz reference converts the existing MHz spectral sum to kHz^-2.
    metric_reference_mhz = 1 / 1000
    for first in range(axis_count):
        for second in range(first, axis_count):
            weights = np.real(matrix_elements[first] * matrix_elements[second].T) / len(states)
            trace = two_time_correlation(differences, weights, time_us)
            infinite = spectral_agp(differences, weights, cutoffs_mhz, metric_reference_mhz)
            finite = window_agp(differences, weights, cutoffs_mhz, metric_reference_mhz, time_us[-1])
            correlation[:, first, second] = correlation[:, second, first] = trace
            spectral[:, first, second] = spectral[:, second, first] = infinite
            window[:, first, second] = window[:, second, first] = finite
    return correlation, spectral, window


def ensemble_metric_prediction(photon_number, detunings_mhz, couplings_mhz, kerr_mhz,
                               directions, swap_stors, disorder_patterns, time_us, mu_khz):
    """Predict local and ensemble K/W metrics, without experimental corrections.

    Local coordinates are (K, pulse detunings), all in cyclic kHz; physical
    metric units are kHz^-2. Project each realization with its saved pattern
    before averaging. The window prediction is an analytic finite-time
    integral; the unqualified prediction is the infinite-time spectral sum.
    """
    patterns = np.asarray(disorder_patterns, dtype=float)
    directions, swap_stors = list(directions), list(swap_stors)
    projection, available = metric_parameter_projection(directions, swap_stors, patterns)
    times = np.asarray(time_us, dtype=float)
    cutoffs_mhz = np.atleast_1d(np.asarray(mu_khz, dtype=float)) / 1000
    local_totals, target_totals = None, None
    for realization, detunings in enumerate(np.asarray(detunings_mhz, dtype=float)):
        model = fixed_n_hamiltonian(photon_number, len(swap_stors) + 1,
                                    detunings, couplings_mhz, kerr_mhz)
        values = metric_probe_values(model.fock_basis, swap_stors, directions)
        local = _metric_spectral_sums(model, values, times, cutoffs_mhz)
        jacobian = projection[realization]
        target = [np.einsum("pa,upq,qb->uab", jacobian, array, jacobian)
                  for array in local]
        if local_totals is None:
            local_totals, target_totals = list(local), target
        else:
            for index in range(len(local)):
                local_totals[index] += local[index]
                target_totals[index] += target[index]
    correlation_local, spectral_local, window_local = [array / len(patterns) for array in local_totals]
    correlation, spectral, window = [array / len(patterns) for array in target_totals]
    for array in (correlation, spectral, window):
        array[:, ~available] = np.nan
    coupling_squared = (1000 * np.mean(np.abs(couplings_mhz)))**2
    return dict(C1_theory=np.diagonal(correlation_local, axis1=-2, axis2=-1).T,
                C_metric_theory=correlation, C_metric_local_theory=correlation_local,
                agp_metric_theory=spectral, agp_metric_theory_window=window,
                agp_metric_local_theory=spectral_local, agp_metric_local_theory_window=window_local,
                agp_theory=np.diagonal(spectral_local, axis1=-2, axis2=-1).T * coupling_squared,
                agp_theory_window=np.diagonal(window_local, axis1=-2, axis2=-1).T * coupling_squared,
                theory_kerr_mhz=np.asarray(kerr_mhz))
