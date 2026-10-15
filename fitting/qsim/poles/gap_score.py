"""The gap score: found poles against true levels by the gaps, not the absolute positions.

With free row offsets a fit fixes a level only to about the offsets' prior (0.4 kHz absolute
on the August sets), while the level statistics need only the gaps; so the score of benchmarks
2-4 (absolute positions within 0.3-0.57 kHz) counts near misses as false poles
(docs/log/2026-09-28_pole-finding-diagnostics.md, fitter F). Here:

1. The found poles are shifted by the one common shift the data cannot fix: E -> E + s with
   every row offset delta_b -> delta_b - s gives the same rows. Where the true offsets are known
   (synthetic data) the shift is ``gauge_shift``; else it is searched (``common_shift``), which
   can lock onto a wrong alignment when many levels are missed.
2. They are matched to the levels in order (``ordered_match``): one-to-one, monotone, a pole
   within half the level's nearest gap.
3. Each adjacent gap is found if both its ends are matched to adjacent poles (no pole between);
   its error is scored against its Cramér-Rao error (``fitting.qsim.poles.design.gap_errors``).
4. Small gaps (under half the mean gap, the ones P(r < r0) counts) found, false poles, and
   P(r < r0) of the found poles against that of the levels.

Plan: ``docs/qsim/pole_finding_explore.md``, T0. Pure numerics.
"""
from dataclasses import dataclass, field

import numpy as np

from fitting.qsim.poles.matching import alias_distance
from fitting.qsim.poles.statistics import small_gap_ratio_fraction

#: A gap is resolvable if it is at least this many of its Cramér-Rao errors (as ``design``).
MIN_GAP_SNR = 4.


@dataclass(frozen=True)
class GapScore:
    """One fit against one truth. ``level_pole[k]`` is the found pole matched to level k (-1: none).

    Gaps are those of the sorted levels; ``found_gaps_MHz`` is NaN where the gap is not found.
    ``gap_bounds_MHz`` maps a bound's name (e.g. "complex", "real") to each gap's error bound.
    """
    levels_MHz: np.ndarray
    found_MHz: np.ndarray
    shift_MHz: float
    level_pole: np.ndarray
    true_gaps_MHz: np.ndarray
    found_gaps_MHz: np.ndarray
    small: np.ndarray
    gap_bounds_MHz: dict = field(default_factory=dict)

    @property
    def gap_found(self):
        return ~np.isnan(self.found_gaps_MHz)

    @property
    def gap_errors_MHz(self):
        return self.found_gaps_MHz - self.true_gaps_MHz

    @property
    def false_poles(self):
        return int(len(self.found_MHz) - np.sum(self.level_pole >= 0))

    def resolvable(self, bound):
        """-> per gap: at least ``MIN_GAP_SNR`` of its bound."""
        return self.true_gaps_MHz >= MIN_GAP_SNR * self.gap_bounds_MHz[bound]

    def gap_z(self, bound):
        """-> per gap: error over its bound (NaN where not found)."""
        return self.gap_errors_MHz / self.gap_bounds_MHz[bound]

    def summary(self, r0=0.25):
        """-> dict of counts: levels, matched, false poles; gaps and small gaps found; per bound
        the resolvable ones, found of those, and the median |error| / bound of the found gaps;
        and P(r < r0) of this case alone (pool over cases with ``pooled_small_gap_fraction``)."""
        found, small = self.gap_found, self.small
        row = dict(levels=len(self.levels_MHz), poles=len(self.found_MHz), matched=int(np.sum(self.level_pole >= 0)),
                   false_poles=self.false_poles, gaps=len(small), gaps_found=int(found.sum()),
                   small_gaps=int(small.sum()), small_found=int(np.sum(small & found)),
                   shift_kHz=1e3 * self.shift_MHz)
        for bound in self.gap_bounds_MHz:
            resolvable = self.resolvable(bound)
            z = np.abs(self.gap_z(bound)[found])
            row |= {f"resolvable_{bound}": int(resolvable.sum()),
                    f"found_of_resolvable_{bound}": int(np.sum(resolvable & found)),
                    f"small_resolvable_{bound}": int(np.sum(small & resolvable)),
                    f"median_abs_z_{bound}": float(np.median(z)) if len(z) else np.nan}
        row |= dict(P_true=small_gap_ratio_fraction([self.levels_MHz], r0),
                    P_found=small_gap_ratio_fraction([self.found_MHz], r0))
        return row


def half_nearest_gap(levels_MHz):
    """-> per sorted level, half the distance to its nearest neighbour (the match tolerance)."""
    gaps = np.diff(levels_MHz)
    return 0.5 * np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])


def unwrap_to(found_MHz, levels_MHz, sampling_MHz):
    """-> the found frequencies on the alias branch nearest the levels' centre."""
    centre = 0.5 * (np.min(levels_MHz) + np.max(levels_MHz))
    return centre + alias_distance(found_MHz, [centre], sampling_MHz)[:, 0]


def capped_cost(found_MHz, levels_MHz, tolerances_MHz):
    """-> sum over levels of min(|nearest pole - E|, tol) / tol: 0 for a perfect match, 1 per miss."""
    if len(found_MHz) == 0:
        return float(len(levels_MHz))
    distance = np.min(np.abs(np.subtract.outer(found_MHz, levels_MHz)), axis=0)
    return float(np.sum(np.minimum(distance, tolerances_MHz) / tolerances_MHz))


def common_shift(found_MHz, levels_MHz, tolerances_MHz, max_shift_MHz, step_MHz):
    """-> the shift s (|s| <= max) minimizing ``capped_cost(found + s)``, on a grid of ``step``."""
    shifts = np.arange(-max_shift_MHz, max_shift_MHz + step_MHz / 2, step_MHz)
    costs = [capped_cost(found_MHz + s, levels_MHz, tolerances_MHz) for s in shifts]
    return float(shifts[np.argmin(costs)])


def ordered_match(found_MHz, levels_MHz, tolerances_MHz):
    """-> ``level_pole``: the monotone one-to-one matching (both sorted) with the most pairs
    within tolerance, and among those the least sum |error| / tol (dynamic programming)."""
    F, L = len(found_MHz), len(levels_MHz)
    distance = np.abs(np.subtract.outer(found_MHz, levels_MHz)) / tolerances_MHz[None, :]
    # value[i, j]: best (pairs, -cost) on the first i poles and j levels, as pairs - cost / (L + 1)
    value = np.zeros((F + 1, L + 1))
    move = np.zeros((F + 1, L + 1), dtype=int)          # 0: skip pole, 1: skip level, 2: pair
    for i in range(F + 1):
        for j in range(L + 1):
            options = [(value[i - 1, j], 0) if i else (-np.inf, 0), (value[i, j - 1], 1) if j else (-np.inf, 1)]
            if i and j and distance[i - 1, j - 1] <= 1:
                options.append((value[i - 1, j - 1] + 1 - distance[i - 1, j - 1] / (L + 1), 2))
            if i or j:
                value[i, j], move[i, j] = max(options, key=lambda option: option[0])
    level_pole = np.full(L, -1)
    i, j = F, L
    while i and j:
        if move[i, j] == 2:
            level_pole[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif move[i, j] == 0:
            i -= 1
        else:
            j -= 1
    return level_pole


def gauge_shift(found_offsets_MHz, true_offsets_MHz):
    """-> s = mean(found offsets) - mean(true offsets): row b sees a level at E + delta_b, so the
    found levels + s are on the true levels' gauge (rows weighed equally). None offsets: 0."""
    found = 0. if found_offsets_MHz is None else float(np.mean(found_offsets_MHz))
    return found - float(np.mean(true_offsets_MHz))


def score_gaps(found_MHz, levels_MHz, sampling_MHz, gap_bounds_MHz=None, shift_MHz=None, max_shift_MHz=2e-3):
    """-> the GapScore of found frequencies against the distinct levels (any order).

    ``gap_bounds_MHz``: {name: bound per gap of the sorted levels}; ``shift_MHz``: the common
    shift (``gauge_shift``), None to search it within ``max_shift_MHz`` (about 2-4 offset priors).
    """
    levels_MHz = np.sort(np.asarray(levels_MHz, dtype=float))
    found_MHz = np.sort(unwrap_to(np.asarray(found_MHz, dtype=float), levels_MHz, sampling_MHz))
    tolerances = half_nearest_gap(levels_MHz)
    shift = shift_MHz if shift_MHz is not None else common_shift(found_MHz, levels_MHz, tolerances, max_shift_MHz,
                                                                  np.min(tolerances) / 4)
    found_MHz = found_MHz + shift
    level_pole = ordered_match(found_MHz, levels_MHz, tolerances)
    true_gaps = np.diff(levels_MHz)
    left, right = level_pole[:-1], level_pole[1:]
    found = (left >= 0) & (right == left + 1)
    found_gaps = np.where(found, found_MHz[right] - found_MHz[left], np.nan)
    return GapScore(levels_MHz, found_MHz, shift, level_pole, true_gaps, found_gaps,
                    true_gaps < 0.5 * np.mean(true_gaps), dict(gap_bounds_MHz or {}))


def pooled_small_gap_fraction(scores, r0=0.25):
    """-> (P(r < r0) of the levels, of the found poles), pooled over the GapScores."""
    return (small_gap_ratio_fraction([s.levels_MHz for s in scores], r0),
            small_gap_ratio_fraction([s.found_MHz for s in scores], r0))

