"""Benchmarks 3 and 4 on real data (spec 5.3, 5.4).

A :class:`RealSpectrum` is one measured spectrum with its model (loaded from the registry by
``experiments.qsim.pole_data.load_spectra``). Benchmark 3 needs no model: fit residual,
pole weights, the Michaille-Pique ratio d and the stability of P(r < r0) under re-merging.
Benchmark 4 compares with the model, within a model-error floor: validation only, do not
tune on it. Pure numerics.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fitting.qsim.poles.matching import alias_distance, level_tolerances, match_poles
from fitting.qsim.poles.pole_fit import normalize_to_initial_return, poles_from
from fitting.qsim.poles.statistics import small_gap_ratio_fraction

#: The rebuilt theory differs from the recorded one by up to 0.3 kHz (docs/STATUS.md).
MODEL_ERROR_MHz = 0.3e-3


@dataclass(frozen=True)
class RealSpectrum:
    """One measured spectrum (diagonal rows) and its model's distinct levels."""
    label: str
    data_set: str
    complete: bool
    time_us: np.ndarray
    A: np.ndarray
    levels_MHz: np.ndarray
    multiplicities: np.ndarray
    row_weights: np.ndarray

    @property
    def dt_us(self):
        return float(self.time_us[1] - self.time_us[0])

    @property
    def bin_MHz(self):
        return 1 / (len(self.time_us) * self.dt_us)


def fit_residual(fit, spectrum):
    """-> |a - sum_lambda c z^n| / |a|, a = A / A(0), over all rows."""
    a = normalize_to_initial_return(spectrum.A)
    z = poles_from(fit.frequencies_MHz, fit.decays_per_us, spectrum.dt_us)
    model = fit.amplitudes @ z[:, None] ** np.arange(len(spectrum.time_us))
    return float(np.linalg.norm(a - model) / np.linalg.norm(a))


def barycenter_merge(frequencies_MHz, weights, resolution_MHz):
    """-> poles after merging, closest pair first, every pair closer than the resolution
    into its weight barycenter (Michaille and Pique): what a coarser resolution would see."""
    frequencies, weights = list(frequencies_MHz), list(np.abs(weights) + 1e-12)
    while len(frequencies) > 1:
        order = np.argsort(frequencies)
        gaps = np.diff(np.asarray(frequencies)[order])
        k = int(np.argmin(gaps))
        if gaps[k] >= resolution_MHz:
            break
        i, j = order[k], order[k + 1]
        total = weights[i] + weights[j]
        frequencies[i] = (weights[i] * frequencies[i] + weights[j] * frequencies[j]) / total
        weights[i] = total
        del frequencies[j], weights[j]
    return np.sort(frequencies)


def multiplet_weights(fit, spectrum):
    """-> (per model level, the summed weight of the poles nearest to it within half a bin;
    the number of poles farther than half a bin from every level)."""
    distance = alias_distance(fit.frequencies_MHz, spectrum.levels_MHz, 1 / spectrum.dt_us)
    nearest = np.argmin(np.abs(distance), axis=1)
    near = np.abs(distance[np.arange(len(nearest)), nearest]) <= 0.5 * spectrum.bin_MHz
    return np.bincount(nearest[near], weights=fit.weights[near], minlength=len(spectrum.levels_MHz)), int(np.sum(~near))


def run_self_consistency_bench(spectra, fitters, lambda_eff_bins=1.0, r0=0.25):
    """Benchmark 3: per spectrum and fitter, the checks that need no model. -> DataFrame.

    ``d = lambda_eff / Delta'`` with Delta' the mean spacing of the found poles (flag above
    0.25, spec 5.3). Complete basis: weights per multiplet against the multiplicities. Partial:
    the poles heavier than 1.2. ``I_at_1.25`` and ``I_at_1.5``: P(r < r0) after re-merging at
    1.25 and 1.5 lambda_eff; a steep change says the data set is at its resolution limit.
    """
    rows = []
    for spectrum in spectra:
        for name, (fit, settings) in fitters.items():
            f = fit(spectrum.A, spectrum.time_us, settings)
            lam = lambda_eff_bins * spectrum.bin_MHz
            row = dict(data_set=spectrum.data_set, spectrum=spectrum.label, fitter=name, poles=len(f.frequencies_MHz),
                       residual=fit_residual(f, spectrum), weight_sum=f.weights.sum(),
                       d=lam / np.mean(np.diff(np.sort(f.frequencies_MHz))),
                       I=small_gap_ratio_fraction([f.frequencies_MHz], r0),
                       **{f"I_at_{x}": small_gap_ratio_fraction([barycenter_merge(f.frequencies_MHz, f.weights, x * lam)], r0)
                          for x in (1.25, 1.5)})
            if spectrum.complete:
                sums, far = multiplet_weights(f, spectrum)
                row.update(multiplet_error=np.mean(np.abs(sums - spectrum.multiplicities)), far_poles=far)
            else:
                row.update(heavy_poles=int(np.sum(f.weights > 1.2)))
            rows.append(row)
    return pd.DataFrame(rows)


def run_model_bench(spectra, fitters, model_error_MHz=MODEL_ERROR_MHz, r0=0.25, seed=0):
    """Benchmark 4: per spectrum and fitter, the match to the model. -> DataFrame.

    Each level is matched within max(min(0.25 bin, 1/4 of its nearest separation), the model
    error); ``random`` is the mean match of as many random poles over the model's span.
    ``I_model`` is the model's own P(r < r0). Validation only: do not tune on it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for spectrum in spectra:
        levels = spectrum.levels_MHz
        tolerance = np.maximum(level_tolerances(levels, spectrum.bin_MHz), model_error_MHz)
        true_weights = spectrum.row_weights.sum(axis=0)
        for name, (fit, settings) in fitters.items():
            f = fit(spectrum.A, spectrum.time_us, settings)
            match = match_poles(f.frequencies_MHz, levels, tolerance, 1 / spectrum.dt_us)
            random = np.mean([len(match_poles(rng.uniform(levels.min(), levels.max(), len(f.frequencies_MHz)),
                                              levels, tolerance, 1 / spectrum.dt_us).level_index) for _ in range(50)])
            rows.append(dict(
                data_set=spectrum.data_set, spectrum=spectrum.label, fitter=name, levels=len(levels),
                poles=len(f.frequencies_MHz), matched=len(match.level_index), random=random,
                median_error_kHz=1e3 * np.median(np.abs(match.errors_MHz)) if len(match.errors_MHz) else np.nan,
                median_weight_error=np.median(np.abs(f.weights[match.pole_index] - true_weights[match.level_index]))
                if len(match.pole_index) else np.nan,
                I=small_gap_ratio_fraction([f.frequencies_MHz], r0), I_model=small_gap_ratio_fraction([levels], r0)))
    return pd.DataFrame(rows)
