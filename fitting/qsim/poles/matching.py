"""Found poles against true levels (spec 5, "Matching").

The same method as :func:`fitting.qsim.mbr_disorder.match_levels` (one-to-one
Hungarian assignment within a tolerance), with two additions: the distance is
taken on the alias circle, and the true levels are the *distinct* levels
(:func:`fitting.qsim.poles.synthetic.distinct_levels`), so a degenerate
multiplet is one level with its multiplicity.
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Match:
    """``pole_index[k]`` is matched to ``level_index[k]`` with signed error ``errors_MHz[k]``."""
    pole_index: np.ndarray
    level_index: np.ndarray
    errors_MHz: np.ndarray
    missed_levels: np.ndarray
    false_poles: np.ndarray
    tolerance_MHz: np.ndarray


def alias_distance(found_MHz, levels_MHz, sampling_MHz):
    """-> d_{ij} = ((f_i - E_j + F/2) mod F) - F/2, signed, for sampling frequency F."""
    difference = np.subtract.outer(np.asarray(found_MHz), np.asarray(levels_MHz))
    return (difference + sampling_MHz / 2) % sampling_MHz - sampling_MHz / 2


def match_poles(found_MHz, levels_MHz, tolerance_MHz, sampling_MHz):
    """-> the one-to-one Match minimizing the total |error| over pairs within the tolerance.

    ``tolerance_MHz``: one number, or one per level (:func:`level_tolerances`).
    """
    distance = alias_distance(found_MHz, levels_MHz, sampling_MHz)
    tolerance_MHz = np.broadcast_to(np.asarray(tolerance_MHz, dtype=float), (len(levels_MHz),))
    inside = np.abs(distance) <= tolerance_MHz[None, :]
    cost = np.where(inside, np.abs(distance), 1e6 * (np.max(tolerance_MHz, initial=0.) + 1))
    poles, levels = linear_sum_assignment(cost) if distance.size else ([], [])
    keep = inside[poles, levels] if distance.size else []
    poles, levels = np.asarray(poles)[keep].astype(int), np.asarray(levels)[keep].astype(int)
    return Match(poles, levels, distance[poles, levels],
                 np.setdiff1d(np.arange(len(levels_MHz)), levels),
                 np.setdiff1d(np.arange(len(found_MHz)), poles), tolerance_MHz)


def level_tolerances(levels_MHz, bin_MHz, tolerance_bins=0.25, separation_fraction=0.25):
    """-> per level, min(tolerance_bins, separation_fraction x nearest separation) x bin.

    In a dense spectrum a fixed tolerance near the mean spacing lets any poles spread over
    the span match many levels by chance (random poles then "resolve" 10-20% of the
    levels closer than a bin; with the separation cap, a few percent).
    """
    return np.minimum(tolerance_bins, separation_fraction * nearest_separation_bins(levels_MHz, bin_MHz)) * bin_MHz


def nearest_separation_bins(levels_MHz, bin_MHz):
    """-> each level's distance to its nearest neighbour, in FFT bins."""
    order = np.argsort(levels_MHz)
    gaps = np.diff(np.asarray(levels_MHz)[order]) / bin_MHz
    separation = np.empty(len(order))
    separation[order] = np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])
    return separation


def resolved_levels(match, levels_MHz):
    """-> for each level: matched, and its nearest neighbour level also matched.

    A merged pair gives one matched level and one missed, so neither counts as
    resolved; this is what the resolution probability of benchmark 2 counts.
    """
    matched = np.isin(np.arange(len(levels_MHz)), match.level_index)
    order = np.argsort(levels_MHz)
    gaps = np.diff(levels_MHz[order])
    neighbour = np.empty(len(order), dtype=int)
    neighbour[order] = np.where(np.r_[np.inf, gaps] < np.r_[gaps, np.inf],
                                np.r_[order[:1], order[:-1]], np.r_[order[1:], order[-1:]])
    return matched & matched[neighbour]


def display_match(match, found_MHz, levels_MHz, bin_MHz, title="", ax=None):
    """Found poles (top) against true levels (bottom), matches joined. -> ax."""
    ax = ax or plt.subplots(figsize=(10, 2.2))[1]
    for pole, level in zip(match.pole_index, match.level_index):
        ax.plot([found_MHz[pole] / bin_MHz, levels_MHz[level] / bin_MHz], [1, 0], color="0.6", lw=0.8)
    ax.plot(np.asarray(found_MHz) / bin_MHz, np.ones(len(found_MHz)), "|", ms=14, color="C0", label="found")
    ax.plot(np.asarray(levels_MHz) / bin_MHz, np.zeros(len(levels_MHz)), "|", ms=14, color="C1", label="true")
    ax.plot(np.asarray(found_MHz)[match.false_poles] / bin_MHz, np.ones(len(match.false_poles)), "x", color="C3",
            label="false pole")
    ax.plot(np.asarray(levels_MHz)[match.missed_levels] / bin_MHz, np.zeros(len(match.missed_levels)), "x",
            color="C3", label="missed")
    ax.set(yticks=[0, 1], yticklabels=["true", "found"], ylim=(-0.5, 1.5), xlabel="frequency (FFT bins)",
           title=title or f"{len(match.level_index)} of {len(levels_MHz)} levels matched, "
                          f"{len(match.false_poles)} false poles")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    return ax
