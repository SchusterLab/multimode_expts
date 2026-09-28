"""The small-gap-ratio statistic (spec 7).

``I(r0) = P(r < r0)`` of the adjacent gap ratios ``r_n = min(s_n, s_n+1) / max(s_n, s_n+1)``
of the bulk levels, pooled over realizations. The gap ratio needs no unfolding.
Merging removes small gaps (I goes down, more repelling); false poles add
them (I goes up, more Poisson).

Pure numerics.
"""
import numpy as np

from fitting.qsim.mbr_disorder import adjacent_gap_ratios, trim_count


def bulk_gap_ratios(levels_MHz, edge_fraction=0.10):
    """-> r_n of the levels with ceil(edge_fraction x count) cut at each edge; empty if too few."""
    levels_MHz = np.asarray(levels_MHz, dtype=float)
    try:
        return adjacent_gap_ratios(levels_MHz, trim_count(edge_fraction, len(levels_MHz)))
    except ValueError:
        return np.array([])


def small_gap_ratio_fraction(level_sets_MHz, r0=0.25, edge_fraction=0.10):
    """-> I(r0) = #{r < r0} / #{r}, pooled over the level sets; NaN if none."""
    ratios = np.concatenate([bulk_gap_ratios(levels, edge_fraction) for levels in level_sets_MHz])
    return float(np.mean(ratios < r0)) if len(ratios) else np.nan
