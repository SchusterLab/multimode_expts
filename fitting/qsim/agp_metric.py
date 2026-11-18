"""Measured AGP metrics from heralded, sequential disorder-ensemble shots.

Every estimate uses measured labels, their readout calibration, and the existing
time kernel. Hamiltonian predictions never enter the reconstruction. Physical
metric coordinates are cyclic kHz; the saved metric therefore has units kHz^-2.
"""
from math import comb
import warnings

import numpy as np

from fitting.qsim.agp import agp_kernel_weights, subtract_background
from theory.qsim.agp_ensemble import metric_parameter_projection, metric_probe_values


def validate_metric_coverage(occupations, swap_stors, ro_stors, directions,
                             calibration_shots=None):
    """Require the initial states and output modes needed by every metric axis."""
    mode_order = [0, *swap_stors]
    photon_number = int(np.sum(occupations[0]))
    unique_states = np.unique(occupations, axis=0)
    for direction in directions:
        minimum = 2 if direction == 0 else 1
        expected = (comb(photon_number - minimum + len(mode_order) - 1,
                         len(mode_order) - 1) if photon_number >= minimum else 0)
        measured = np.count_nonzero(unique_states[:, mode_order.index(direction)] >= minimum)
        if direction not in ro_stors or measured != expected:
            raise ValueError(f"Metric direction {direction} needs its output and complete initial-state sector")
    if calibration_shots is not None and (not np.isfinite(calibration_shots).all()
                                           or np.any(calibration_shots < 1)):
        raise ValueError("Calibration row counts must be positive for uncertainty sampling")


def _score_lookup(directions, ro_stors, confusion_matrices):
    """Map each measured label to its number or Kerr probe, with the device sign."""
    numbers = np.arange(4, dtype=float)
    lookup = np.array([numbers * (numbers - 1) / 2 if direction == 0 else -numbers
                       for direction in directions])
    if confusion_matrices is not None:
        for axis, direction in enumerate(directions):
            matrix = confusion_matrices[:, :, ro_stors.index(direction)]
            lookup[axis] = np.linalg.solve(matrix, lookup[axis])
    return lookup


def _grouped_readout_labels(labels, directions, ro_stors):
    """Encode (initial state, time, label) once for fast weighted histograms."""
    state_count, time_count = labels.shape[:2]
    row_offsets = 4 * np.arange(state_count * time_count).reshape(state_count, time_count, 1)
    return [np.asarray(labels[..., ro_stors.index(direction)] + row_offsets,
                       dtype=np.int64).reshape(-1) for direction in directions]


def _label_histogram(grouped_labels, shot_weights, coefficient=None):
    """Reduce shots to four bins; coefficients retain realization-specific weights."""
    if coefficient is not None:
        shot_weights = shot_weights * coefficient[:, None, :]
    state_count, time_count = shot_weights.shape[:2]
    return np.bincount(grouped_labels, weights=shot_weights.reshape(-1),
                        minlength=state_count * time_count * 4).reshape(state_count, time_count, 4)


def _correlation_statistics(grouped_labels, accepted, initial_values, projection,
                             directions, available, multiplicities=None,
                             representative_storage=None):
    """Keep exact herald denominators while reducing all label-linear statistics.

    Local initial probes are independent of realization, so four-bin output
    histograms suffice. K/W cross and W/W entries additionally use the saved
    pattern factors before summing shots. This avoids a state*time*R*P*P array.
    """
    shot_weights = accepted if multiplicities is None else accepted * multiplicities
    counts = shot_weights.sum(axis=-1)
    normalized_weights = np.full(accepted.shape, np.nan)
    np.divide(shot_weights, counts[..., None], out=normalized_weights,
              where=counts[..., None] > 0)
    local_histograms = np.stack([_label_histogram(labels, normalized_weights)
                                 for labels in grouped_labels], axis=2)
    target_histograms = np.zeros((accepted.shape[1], 2, 2, len(directions), 4))
    kerr_available, disorder_available = bool(available[0, 0]), bool(available[1, 1])
    kerr_axis = directions.index(0) if kerr_available else None
    if kerr_available:
        initial_kerr = initial_values[:, kerr_axis]
        target_histograms[:, 0, 0, kerr_axis] = (
            initial_kerr[:, None, None] * local_histograms[:, :, kerr_axis]).sum(axis=0)
    if disorder_available:
        if representative_storage is None:
            initial_disorder = initial_values @ projection[:, :, 1].T
        else:
            representative_axis = directions.index(representative_storage)
            storage_count = len(directions) - int(0 in directions)
            # Permutation symmetry replaces the sum of initial storage legs.
            # The final W leg below still uses every saved pattern coefficient.
            initial_disorder = (storage_count * initial_values[:, representative_axis, None]
                                * projection[:, representative_axis, 1])
        for axis, direction in enumerate(directions):
            if direction == 0:
                histogram = _label_histogram(grouped_labels[axis], normalized_weights,
                                               initial_disorder)
                target_histograms[:, 1, 0, axis] = histogram.sum(axis=0)
            else:
                pattern = projection[:, axis, 1]
                histogram = _label_histogram(grouped_labels[axis], normalized_weights,
                                               initial_disorder * pattern)
                target_histograms[:, 1, 1, axis] = histogram.sum(axis=0)
                if kerr_available:
                    histogram = _label_histogram(grouped_labels[axis], normalized_weights,
                                                   initial_kerr[:, None] * pattern)
                    target_histograms[:, 0, 1, axis] = histogram.sum(axis=0)
    return local_histograms, target_histograms


def _evaluate_correlations(statistics, lookup, initial_values, dimension):
    """Convert label histograms to calibrated local and K/W correlations."""
    local_histograms, target_histograms = statistics
    row_means = (local_histograms * lookup).sum(axis=-1)
    direction_count = initial_values.shape[1]
    local = np.empty((row_means.shape[1], direction_count, direction_count))
    for initial_direction in range(direction_count):
        for final_direction in range(direction_count):
            local[:, initial_direction, final_direction] = (
                initial_values[:, initial_direction, None]
                * row_means[:, :, final_direction]).sum(axis=0) / dimension
    target = (target_histograms * lookup).sum(axis=(-2, -1)) / dimension
    return ((local + local.swapaxes(-1, -2)) / 2,
            (target + target.swapaxes(-1, -2)) / 2)


def _integrate_correlations(kernel_weights, correlations):
    """Convert the rad/us kernel to derivatives with respect to cyclic kHz."""
    integrals = np.empty((len(kernel_weights), *correlations.shape[1:]))
    for cutoff_index, weights in enumerate(kernel_weights):
        integrals[cutoff_index] = (weights[:, None, None] * correlations).sum(axis=0)
    return (2 * np.pi / 1000)**2 * integrals


def _sample_calibration(confusion_matrices, calibration_shots, random_generator):
    """Draw each prepared-state row once, shared by all experimental rows/times."""
    if confusion_matrices is None or calibration_shots is None:
        return confusion_matrices
    sampled = np.empty_like(confusion_matrices)
    for step in range(confusion_matrices.shape[-1]):
        for prepared in range(4):
            count = int(calibration_shots[prepared, step])
            sampled[prepared, :, step] = random_generator.multinomial(
                count, confusion_matrices[prepared, :, step]) / count
    return sampled


def _bootstrap_metric_samples(grouped_labels, accepted, initial_values, projection, dimension,
                              directions, ro_stors, available, kernel_weights, confusion_matrices,
                              calibration_shots, samples, seed, representative_storage=None):
    """Resample whole realizations and the shared calibration, without refitting."""
    random_generator = np.random.default_rng(seed)
    realization_count = accepted.shape[2]
    local_samples, target_samples, diagonal_samples = [], [], []
    if realization_count < 2 or samples == 0:
        return local_samples, target_samples, diagonal_samples
    changing_calibration = confusion_matrices is not None and calibration_shots is not None
    lookup = _score_lookup(directions, ro_stors, confusion_matrices)
    for _ in range(samples):
        indices = random_generator.integers(realization_count, size=realization_count)
        multiplicities = np.bincount(indices, minlength=realization_count)
        if changing_calibration:
            calibration = _sample_calibration(confusion_matrices, calibration_shots,
                                                random_generator)
            try:
                lookup = _score_lookup(directions, ro_stors, calibration)
            except np.linalg.LinAlgError:
                continue
        statistics = _correlation_statistics(grouped_labels, accepted, initial_values,
                                              projection, directions, available, multiplicities,
                                              representative_storage)
        local, target = _evaluate_correlations(statistics, lookup, initial_values, dimension)
        if np.isfinite(local).all():
            local_samples.append(_integrate_correlations(kernel_weights, local))
            target_samples.append(_integrate_correlations(kernel_weights, target))
            diagonal_samples.append(np.diagonal(local, axis1=-2, axis2=-1).T)
    return local_samples, target_samples, diagonal_samples


def _sample_error(samples, shape):
    """Keep unknown sampling errors as NaN instead of reporting zero precision."""
    return (np.std(samples, axis=0, ddof=1) if len(samples) > 1
            else np.full(shape, np.nan))


def _eigensystem(metric):
    """Diagonalize complete finite matrices, preserving negative eigenvalues."""
    eigenvalues = np.full(metric.shape[:-1], np.nan)
    eigenvectors = np.full(metric.shape, np.nan)
    finite = np.isfinite(metric).all(axis=(-2, -1))
    if np.any(finite):
        eigenvalues[finite], eigenvectors[finite] = np.linalg.eigh(metric[finite])
    return eigenvalues, eigenvectors


def background_subtracted_metric(time_us, correlations, mu_khz, coupling_mhz, start_us):
    """Fit aggregate K/W backgrounds with the existing exponential estimator.

    Original correlations are untouched. Signed amplitudes are allowed;
    invalid decays retain the original component with a false fit-valid flag.
    None disables subtraction. The saved background is only what was removed.
    Parameter covariance is saved, but corrected-metric errors are not inferred.
    """
    original = np.asarray(correlations, dtype=float)
    subtracted = original.copy()
    background = np.zeros_like(original)
    amplitude, rate = np.full((2, 2), np.nan), np.full((2, 2), np.nan)
    covariance, valid = np.full((2, 2, 2, 2), np.nan), np.zeros((2, 2), dtype=bool)
    if start_us is not None:
        for first, second, label in [(0, 0, "KK"), (0, 1, "KW"), (1, 1, "WW")]:
            try:
                trace, fit = subtract_background(time_us, original[:, first, second], start_us=start_us)
            except (RuntimeError, ValueError) as error:
                warnings.warn(f"{label} background fit failed; using original trace: {error}", stacklevel=2)
                continue
            amplitude[first, second] = amplitude[second, first] = fit["amplitude"]
            rate[first, second] = rate[second, first] = fit["rate_per_us"]
            covariance[first, second] = covariance[second, first] = fit["covariance"]
            valid[first, second] = valid[second, first] = (
                np.isfinite([fit["amplitude"], fit["rate_per_us"]]).all()
                and fit["rate_per_us"] > 0 and np.isfinite(trace).all())
            if valid[first, second]:
                background[:, first, second] = background[:, second, first] = original[:, first, second] - trace
                subtracted[:, first, second] = subtracted[:, second, first] = trace
            else:
                warnings.warn(f"{label} background fit is not a finite decay; using original trace",
                              stacklevel=2)
    weights = np.array([agp_kernel_weights(time_us, cutoff / 1000)
                        for cutoff in np.atleast_1d(mu_khz)])
    metric = _integrate_correlations(weights, subtracted)
    eigenvalues, eigenvectors = _eigensystem(metric)
    return dict(C_metric_background=background, C_metric_background_subtracted=subtracted,
                C_metric_background_amplitude=amplitude, C_metric_background_rate_per_us=rate,
                C_metric_background_covariance=covariance, C_metric_background_fit_valid=valid,
                C_metric_background_enabled=np.asarray(start_us is not None),
                C_metric_background_start_us=np.asarray(np.nan if start_us is None else start_us),
                agp_metric_background_subtracted=metric,
                agp_metric_background_subtracted_scaled=metric * (1000 * coupling_mhz)**2,
                agp_metric_background_subtracted_eigenvalues=eigenvalues,
                agp_metric_background_subtracted_eigenvectors=eigenvectors)


def ensemble_agp_metric(time_us, labels, accepted, occupations, swap_stors, ro_stors,
                        directions, disorder_patterns, mu_khz, coupling_mhz,
                        confusion_matrices=None, calibration_shots=None,
                        bootstrap_samples=300, seed=0):
    """Return local and K/W metrics from labels (state,time,realization,mode).

    ``accepted`` is one common mode mask with axes (state,time,realization).
    Disorder patterns retain the original normalized vectors even at W=0.
    Every axis uses the same kernel and measured time window. ``agp`` retains
    the existing dimensionless coupling-squared convention; ``agp_metric``
    and ``agp_metric_local`` use cyclic-kHz coordinates (units kHz^-2).
    Errors jointly include finite-ensemble/shot variation and calibration
    sampling when its counts are supplied; these sources are not separated.
    Requested/valid bootstrap counts expose draws lost to singular sampled
    calibrations or empty herald selections; errors use the valid draws.

    One selected storage with complete final readout uses storage permutation
    symmetry: S*(-xi_s*n_s) initially, retaining all final storage contributions.
    """
    labels, accepted = np.asarray(labels), np.asarray(accepted, dtype=bool)
    occupations = np.asarray(occupations, dtype=float)
    directions, ro_stors, swap_stors = list(directions), list(ro_stors), list(swap_stors)
    if calibration_shots is not None:
        calibration_shots = np.asarray(calibration_shots)
    validate_metric_coverage(occupations, swap_stors, ro_stors, directions, calibration_shots)
    selected_storages = [storage for storage in swap_stors if storage in directions]
    representative_storage = (selected_storages[0] if len(selected_storages) == 1
                              and len(swap_stors) > 1
                              and all(storage in ro_stors for storage in swap_stors) else None)
    output_directions = directions.copy()
    if representative_storage is not None:
        output_directions += [storage for storage in swap_stors if storage not in directions]
    patterns = np.asarray(disorder_patterns, dtype=float)
    projection, available = metric_parameter_projection(output_directions, swap_stors, patterns)
    values = metric_probe_values(occupations, swap_stors, directions)
    grouped_labels = _grouped_readout_labels(labels, output_directions, ro_stors)
    dimension = comb(int(occupations[0].sum()) + len(swap_stors), len(swap_stors))
    cutoffs = np.atleast_1d(np.asarray(mu_khz, dtype=float))
    kernel_weights = np.array([agp_kernel_weights(time_us, cutoff / 1000) for cutoff in cutoffs])
    statistics = _correlation_statistics(grouped_labels, accepted, values, projection,
                                          output_directions, available,
                                          representative_storage=representative_storage)
    raw_local, raw_target = _evaluate_correlations(
        statistics, _score_lookup(output_directions, ro_stors, None), values, dimension)
    local, target = raw_local, raw_target
    if confusion_matrices is not None:
        confusion_matrices = np.asarray(confusion_matrices, dtype=float)
        local, target = _evaluate_correlations(
            statistics, _score_lookup(output_directions, ro_stors, confusion_matrices), values,
            dimension)
    local_metric = _integrate_correlations(kernel_weights, local)
    metric = _integrate_correlations(kernel_weights, target)
    samples = _bootstrap_metric_samples(
        grouped_labels, accepted, values, projection, dimension, output_directions, ro_stors,
        available, kernel_weights, confusion_matrices, calibration_shots, bootstrap_samples, seed,
        representative_storage)
    local_error = _sample_error(samples[0], local_metric.shape)
    metric_error = _sample_error(samples[1], metric.shape)
    metric[:, ~available] = np.nan
    metric_error[:, ~available] = np.nan
    target[:, ~available] = np.nan
    coupling_squared = (1000 * coupling_mhz)**2
    eigenvalues, eigenvectors = _eigensystem(metric)
    result = dict(C1=np.diagonal(raw_local, axis1=-2, axis2=-1).T,
                  C1_err=_sample_error(samples[2], (len(directions), len(time_us))),
                  C_metric_local=local, C_metric=target,
                  agp_mu=cutoffs, agp=np.diagonal(local_metric, axis1=-2, axis2=-1).T * coupling_squared,
                  agp_err=np.diagonal(local_error, axis1=-2, axis2=-1).T * coupling_squared,
                  agp_metric_local=local_metric, agp_metric_local_err=local_error,
                  agp_metric=metric, agp_metric_err=metric_error,
                  agp_metric_scaled=metric * coupling_squared,
                  agp_metric_scaled_err=metric_error * coupling_squared,
                  agp_metric_available=available, agp_metric_axes=np.asarray(["K", "W"], dtype="S"),
                  agp_metric_local_axes=np.asarray(directions, dtype=int),
                  agp_metric_units=np.asarray("kHz^-2", dtype="S"),
                  agp_metric_eigenvalues=eigenvalues, agp_metric_eigenvectors=eigenvectors,
                  shot_counts=accepted.sum(axis=-1),
                  bootstrap_valid_samples=np.asarray(len(samples[0])),
                  bootstrap_requested_samples=np.asarray(bootstrap_samples),
                  calibration_uncertainty_included=np.asarray(
                      confusion_matrices is not None and calibration_shots is not None))
    if confusion_matrices is not None:
        result["C1_corrected"] = np.diagonal(local, axis1=-2, axis2=-1).T
    if representative_storage is not None:
        result["agp_metric_representative_storage"] = np.asarray(representative_storage)
    return result
