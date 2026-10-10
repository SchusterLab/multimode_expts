"""What the benchmark scores say (spec 5.2): tables, resolution, and the statistic.

Every function takes one :class:`fitting.qsim.poles.benchmarks.BenchResult` or a list of
them (for example one per window length) and returns a pandas DataFrame. The condition
of a case is its window length ``samples`` and its non-idealities.
"""
import numpy as np
import pandas as pd

from fitting.qsim.poles.matching import nearest_separation_bins
from fitting.qsim.poles.statistics import small_gap_ratio_fraction

NONIDEAL = ["snr", "decay_per_us", "offset_sigma_MHz"]
CONDITION = ["samples", *NONIDEAL]
POINT = ["kerr_over_g", "disorder_over_g"]


def scores_of(results):
    """-> (condition dict, score) for every score of one result or a list of results."""
    for result in results if isinstance(results, (list, tuple)) else [results]:
        for s in result.scores:
            yield dict(samples=result.hardware.samples, **{k: getattr(s.nonideal, k) for k in NONIDEAL}), s


def scores_table(results):
    """-> one row per fit: the case, the counts, the errors and the time."""
    return pd.DataFrame([dict(
        fitter=s.fitter, kerr_over_g=s.kerr_over_g, disorder_over_g=s.disorder_over_g, draw=s.draw,
        conditioning=s.conditioning, **condition, seed=s.nonideal.seed,
        levels=len(s.levels_MHz), found=len(s.found_MHz), matched=int(np.sum(s.level_matched)),
        resolved=int(np.sum(s.level_resolved)), false_poles=s.false_pole_count, rank=s.rank,
        max_error_bins=np.max(np.abs(s.matched_errors_bins), initial=0.),
        mean_error_bins=np.mean(s.matched_errors_bins) if len(s.matched_errors_bins) else np.nan,
        max_weight_error=np.max(np.abs(s.matched_weight_errors), initial=0.),
        seconds=s.seconds) for condition, s in scores_of(results)])


def resolution_curve(results, unit_MHz, edges=np.r_[0:2.01:0.125, np.inf]):
    """-> P(level resolved) against its nearest-neighbour separation in units of ``unit_MHz``
    (one bin of the measured grid, so that window lengths compare), per fitter and condition."""
    rows = []
    for condition, s in scores_of(results):
        separation = nearest_separation_bins(s.levels_MHz, unit_MHz)
        rows += [dict(fitter=s.fitter, **condition, separation=sep, resolved=resolved)
                 for sep, resolved in zip(separation, s.level_resolved)]
    table = pd.DataFrame(rows)
    table["bin"] = pd.cut(table.separation, edges, right=False)
    return (table.groupby(["fitter", *CONDITION, "bin"], observed=True)
            .resolved.agg(probability="mean", count="size").reset_index())


def lambda_eff(curve):
    """-> per fitter and condition, the separation resolved in half the cases:
    the left edge of the first separation bin with P >= 0.5 (NaN if none)."""
    def first_half(group):
        above = group[group.probability >= 0.5]
        return above.bin.iloc[0].left if len(above) else np.nan
    return curve.groupby(["fitter", *CONDITION]).apply(first_half, include_groups=False).rename("lambda_eff")


def small_gap_table(results, r0=0.25):
    """-> I(r0) of the found poles and of the true levels, pooled over draws and seeds,
    per fitter, condition and phase-diagram point."""
    groups = {}
    for condition, s in scores_of(results):
        key = (s.fitter, *condition.values(), s.kerr_over_g, s.disorder_over_g)
        groups.setdefault(key, []).append(s)
    return pd.DataFrame([dict(zip(["fitter", *CONDITION, *POINT], key),
                              found=small_gap_ratio_fraction([s.found_MHz for s in group], r0),
                              true=small_gap_ratio_fraction([s.levels_MHz for s in group], r0),
                              cases=len(group))
                         for key, group in groups.items()])
