"""Benchmarks 1 and 2 on synthetic data (spec 5.1, 5.2).

A benchmark fits every case with every fitter and scores each fit against the
truth (:func:`score_fit`). The benchmarks know only a fitter's ``fit`` and its
settings; the summaries of the scores are in :mod:`fitting.qsim.poles.bench_summaries`.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from fitting.qsim.poles import fft_peaks, joint_pencil, per_row_reconciled
from fitting.qsim.poles.matching import level_tolerances, match_poles, resolved_levels
from fitting.qsim.poles.synthetic import Hardware, Nonideal, synthetic_returns

#: The fitters of phase 1, by the letters of spec section 4.
FITTERS = {
    "A": (per_row_reconciled.fit, per_row_reconciled.CAMPAIGN_SETTINGS),
    "B": (joint_pencil.fit, joint_pencil.JointPencilSettings()),
    "E": (fft_peaks.fit, fft_peaks.FFTPeakSettings()),
}


@dataclass(frozen=True)
class CaseScore:
    """One fitter on one case: what it found, and how that compares with the truth."""
    fitter: str
    kerr_over_g: float
    disorder_over_g: float
    draw: int
    nonideal: Nonideal
    conditioning: float
    levels_MHz: np.ndarray
    true_weights: np.ndarray
    found_MHz: np.ndarray
    found_weights: np.ndarray
    matched_errors_bins: np.ndarray
    matched_weight_errors: np.ndarray
    level_matched: np.ndarray
    level_resolved: np.ndarray
    false_pole_count: int
    rank: int
    seconds: float


@dataclass(frozen=True)
class BenchResult:
    """All the scores of one benchmark run, and what it ran with."""
    name: str
    hardware: Hardware
    fitter_settings: dict
    match_tolerance_bins: float
    scores: tuple = field(repr=False)


def score_fit(name, fit, case, match_tolerance_bins, seconds):
    """-> the CaseScore of a PoleFit on a SyntheticCase."""
    bin_MHz = case.hardware.bin_MHz
    truth = case.truth
    true_weights = truth.row_weights.sum(axis=0)
    match = match_poles(fit.frequencies_MHz, truth.levels_MHz,
                        level_tolerances(truth.levels_MHz, bin_MHz, match_tolerance_bins), 1 / case.hardware.dt_us)
    level_matched = np.isin(np.arange(len(truth.levels_MHz)), match.level_index)
    return CaseScore(
        name, case.point.kerr_over_g, case.point.disorder_over_g, case.point.draw, case.nonideal, truth.conditioning,
        truth.levels_MHz, true_weights, fit.frequencies_MHz, fit.weights,
        match.errors_MHz / bin_MHz, fit.weights[match.pole_index] - true_weights[match.level_index],
        level_matched, resolved_levels(match, truth.levels_MHz), len(match.false_poles),
        fit.rank, seconds)


def fit_and_score(name, fitter, case, match_tolerance_bins):
    """-> the CaseScore of one fitter (fit, settings) on one case, timed."""
    fit, settings = fitter
    start = time.perf_counter()
    result = fit(case.A, case.time_us, settings)
    return score_fit(name, result, case, match_tolerance_bins, time.perf_counter() - start)


def run_ideal_bench(points, hardware=Hardware(), fitters=FITTERS, match_tolerance_bins=0.25):
    """Benchmark 1: every fitter on noise-free returns at every phase-diagram point."""
    cases = [synthetic_returns(point, hardware) for point in points]
    scores = [fit_and_score(name, fitter, case, match_tolerance_bins)
              for case in cases for name, fitter in fitters.items()]
    return BenchResult(f"ideal_{hardware.samples}", hardware, settings_of(fitters), match_tolerance_bins, tuple(scores))


def run_nonideal_bench(points, conditions, seeds, hardware=Hardware(), fitters=FITTERS,
                       match_tolerance_bins=0.25, name=None):
    """Benchmark 2: every fitter at every point, condition (a Nonideal) and noise seed.
    ``name`` labels the result (and its HDF5 group); default ``nonideal_<samples>``."""
    scores = []
    for condition in conditions:
        for seed in range(seeds):
            nonideal = condition.model_copy(update=dict(seed=seed))
            for point in points:
                case = synthetic_returns(point, hardware, nonideal)
                scores += [fit_and_score(name, fitter, case, match_tolerance_bins)
                           for name, fitter in fitters.items()]
    return BenchResult(name or f"nonideal_{hardware.samples}", hardware, settings_of(fitters), match_tolerance_bins, tuple(scores))


def settings_of(fitters):
    """-> {fitter: its settings as a dict}, the provenance of a run."""
    return {name: settings.model_dump() for name, (_, settings) in fitters.items()}


def ideal_pass(score, frequency_tolerance_bins, weight_tolerance):
    """-> benchmark 1's verdict: every level matched, no false pole, errors within tolerance."""
    return bool(np.all(score.level_matched) and score.false_pole_count == 0
                and np.max(np.abs(score.matched_errors_bins)) <= frequency_tolerance_bins
                and np.max(np.abs(score.matched_weight_errors)) <= weight_tolerance)
