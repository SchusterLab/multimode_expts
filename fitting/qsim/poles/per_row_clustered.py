"""Fitter D: a pencil per row, the rows' poles clustered once (spec 4.3).

Per row, the same pencil and rank rule as B on that row alone; then all rows' poles are
clustered by frequency, at most one pole per row per cluster; a cluster is one pole at the
weighted mean of its members. A row's own frequency offset then moves only that row's poles,
and the cluster absorbs it (B's shared-pole model splits the level instead). Amplitudes as B.

Pure numerics.
"""
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from fitting.qsim.poles.joint_pencil import shift_invariance_poles, stacked_hankel_pair
from fitting.qsim.poles.pole_fit import (check_shape, frequencies_and_decays, least_squares_amplitudes,
                                         normalize_to_initial_return, pole_fit, poles_from, sample_time)
from fitting.qsim.poles.rank import RankRule, choose_rank


class PerRowClusteredSettings(BaseModel):
    """The free parameters of fitter D."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: How many poles each row has (spec 4.6).
    rank_rule: RankRule = RankRule()
    #: Pencil length as a fraction of the samples (as B).
    pencil_fraction: Annotated[float, Field(gt=0., lt=1.)] = 2 / 3
    #: Row poles closer than this (FFT bins) join one cluster. Larger absorbs larger row
    #: offsets; smaller keeps close levels apart.
    merge_tolerance_bins: Annotated[float, Field(gt=0.)] = 0.5
    #: A cluster needs this many rows. 1 keeps a level seen in one row only.
    minimum_rows: Annotated[int, Field(ge=1)] = 1


def fit(A, time_us, settings=PerRowClusteredSettings(), row_groups=None):
    """-> the PoleFit of rows ``A`` on ``time_us``. ``row_groups`` unused."""
    dt_us = sample_time(time_us)
    a = normalize_to_initial_return(check_shape(A, time_us))
    frequencies, decays, weights, rows = row_poles(a, dt_us, settings)
    clusters = cluster_row_poles(frequencies, rows, settings.merge_tolerance_bins / (a.shape[1] * dt_us))
    kept = [c for c in clusters if len(c) >= settings.minimum_rows]
    poles = np.array([np.exp(np.average(np.log(poles_from(frequencies[c], decays[c], dt_us)), weights=weights[c]))
                      for c in kept])
    return pole_fit(poles, a, dt_us, len(poles))


def row_poles(a, dt_us, settings):
    """-> every row's poles: (frequencies, decays, |amplitudes|, row index), concatenated."""
    out = [[], [], [], []]
    for row, trace in enumerate(a):
        H0, H1 = stacked_hankel_pair(trace[None, :], round(settings.pencil_fraction * len(trace)))
        U, s, Vh = np.linalg.svd(H0, full_matrices=False)
        poles = shift_invariance_poles(U, s, Vh, H1, choose_rank(s, H0.shape[0], settings.rank_rule))
        frequencies, decays = frequencies_and_decays(poles, dt_us)
        amplitudes = np.abs(least_squares_amplitudes(trace[None, :], poles)[0])
        for values, new in zip(out, (frequencies, decays, amplitudes, np.full(len(poles), row))):
            values.extend(new)
    return tuple(np.asarray(values) for values in out)


def cluster_row_poles(frequencies_MHz, rows, tolerance_MHz):
    """-> clusters (index arrays): frequency-sorted poles join the current cluster while the
    gap to its last member is within the tolerance and its row is not in it yet."""
    clusters, current = [], []
    for index in np.argsort(frequencies_MHz):
        if current and (frequencies_MHz[index] - frequencies_MHz[current[-1]] > tolerance_MHz
                        or rows[index] in rows[current]):
            clusters.append(np.array(current))
            current = []
        current.append(index)
    return clusters + [np.array(current)] if current else clusters
