"""AGP kernel estimates from single-shot disorder ensembles.

The realization axis is shared by every initial state and time. Sampling
errors resample that axis jointly; readout-calibration errors remain shared.
There is no Hamiltonian model, decay fit, or tail completion in this estimator.
"""
import numpy as np

from fitting.qsim.agp import agp_kernel_weights, calibration_error_factors


def _correlation(scores, accepted, initial_weights, multiplicities=None):
    """Average accepted shots within each state, then apply its trace weight."""
    shot_weights = accepted if multiplicities is None else accepted * multiplicities
    shot_counts = shot_weights.sum(axis=-1)
    row_sums = (scores * shot_weights).sum(axis=-1)
    row_means = np.full(row_sums.shape, np.nan)
    np.divide(row_sums, shot_counts, out=row_means, where=shot_counts > 0)
    return initial_weights @ row_means


def _sampling_covariance(scores, accepted, initial_weights, samples, seed):
    """Bootstrap whole realizations, including their shared time dependence."""
    _, time_count, realization_count = scores.shape
    correlations = []
    random_generator = np.random.default_rng(seed)
    if realization_count > 1:
        for _ in range(samples):
            indices = random_generator.integers(realization_count, size=realization_count)
            multiplicities = np.bincount(indices, minlength=realization_count)
            correlation = _correlation(scores, accepted, initial_weights, multiplicities)
            # A small ensemble can resample only rejected shots at one point.
            if np.all(np.isfinite(correlation)):
                correlations.append(correlation)
    covariance = np.full((time_count, time_count), np.nan)
    if len(correlations) > 1:
        covariance = np.cov(np.asarray(correlations), rowvar=False, ddof=1)
    return covariance, len(correlations)


def _calibration_factors(labels, accepted, initial_weights, measured_values,
                         confusion_matrix, calibration_shots):
    """Propagate one common confusion matrix through all states and times."""
    counts = accepted.sum(axis=-1)
    probabilities = np.full((*counts.shape, 4), np.nan)
    for number in range(4):
        totals = ((labels == number) & accepted).sum(axis=-1)
        np.divide(totals, counts, out=probabilities[..., number], where=counts > 0)
    corrected = np.linalg.solve(confusion_matrix.T, probabilities.reshape(-1, 4).T).T
    factors = calibration_error_factors(
        corrected.reshape(probabilities.shape), confusion_matrix, calibration_shots,
        measured_values)
    return np.einsum("s,sta->ta", initial_weights, factors)


def ensemble_agp(time_us, labels, accepted, initial_weights, mu_khz, coupling_mhz,
                 measured_values, confusion_matrix=None, calibration_shots=None,
                 bootstrap_samples=300, seed=0):
    """Return the measured correlation and dimensionless g^2 ||A_mu||^2.

    ``labels`` and ``accepted`` have shape (state, time, realization).
    ``initial_weights`` includes the fixed-sector 1/D normalization, and
    ``measured_values`` contains the probe values for labels 0, 1, 2, 3.
    Cutoffs are cyclic kHz; coupling is cyclic MHz and times are us.

    ``C1_err`` and ``agp_err`` include joint-realization bootstrap variation
    (shot noise plus finite-ensemble variation) and shared calibration error.
    These two sampling contributions cannot be separated from one shot per
    realization. Fewer than two valid bootstrap draws give NaN sampling errors.
    """
    labels = np.asarray(labels, dtype=int)
    accepted = np.asarray(accepted, dtype=bool)
    initial_weights = np.asarray(initial_weights, dtype=float)
    measured_values = np.asarray(measured_values, dtype=float)
    cutoffs = np.asarray(mu_khz, dtype=float).reshape(-1)
    raw_scores = measured_values[labels]
    correlation = _correlation(raw_scores, accepted, initial_weights)
    result = dict(C1=correlation, agp_mu=cutoffs, shot_counts=accepted.sum(axis=-1))

    scores = raw_scores
    if confusion_matrix is not None:
        confusion_matrix = np.asarray(confusion_matrix, dtype=float)
        corrected_values = np.linalg.solve(confusion_matrix, measured_values)
        scores = corrected_values[labels]
        correlation = _correlation(scores, accepted, initial_weights)
        result["C1_corrected"] = correlation

    sampling_covariance, valid_samples = _sampling_covariance(
        scores, accepted, initial_weights, bootstrap_samples, seed)
    calibration_covariance = np.zeros(sampling_covariance.shape)
    if confusion_matrix is not None and calibration_shots is not None:
        factors = _calibration_factors(labels, accepted, initial_weights, measured_values,
                                       confusion_matrix, calibration_shots)
        calibration_covariance = factors @ factors.T
        result["C1_calibration_factors"] = factors
    covariance = sampling_covariance + calibration_covariance
    kernel = (2 * np.pi * coupling_mhz)**2 * np.asarray([
        agp_kernel_weights(time_us, cutoff / 1000) for cutoff in cutoffs])
    sampling_variance = np.einsum("mt,tu,mu->m", kernel, sampling_covariance, kernel)
    calibration_variance = np.einsum("mt,tu,mu->m", kernel, calibration_covariance, kernel)
    result.update(C1_covariance=covariance, C1_err=np.sqrt(np.maximum(np.diag(covariance), 0)),
                  C1_sampling_err=np.sqrt(np.maximum(np.diag(sampling_covariance), 0)),
                  C1_calibration_err=np.sqrt(np.maximum(np.diag(calibration_covariance), 0)),
                  agp=kernel @ correlation,
                  agp_err=np.sqrt(np.maximum(sampling_variance + calibration_variance, 0)),
                  agp_sampling_err=np.sqrt(np.maximum(sampling_variance, 0)),
                  agp_calibration_err=np.sqrt(np.maximum(calibration_variance, 0)),
                  bootstrap_valid_samples=np.asarray(valid_samples))
    return result
