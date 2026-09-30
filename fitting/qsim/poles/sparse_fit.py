"""Fitter T3: a convex sparse fit on a fine grid, global for fixed row offsets.

Plan ``docs/qsim/pole_finding_explore.md``, T3. With the row offsets delta_b fixed, the model of
F (``pursuit``) is linear in the amplitudes once the poles sit on a grid:

    y_b(t) = a_b(t) exp(+2 pi i delta_b t) = sum_{j,d} x_{b,j,d} v_{j,d}(t),
    v_{j,d}(t) = exp((-gamma_d - 2 pi i E_j) t),   x >= 0 real,

E_j a fine grid (``grid_per_bin`` points per FFT bin, only within the start poles' span plus a
margin), gamma_d one or a few decays (default: the start's median alone). In whitened units (u = x |v| / s_b,
rows over their noise s_b) the fit is

    min_u  1/2 sum_b |y_b / s_b - V^ u_b|^2 + lambda sum_{j,d} |u_{:,j,d}|_2,   u >= 0,

a non-negative group lasso, a group being one grid point across all rows (the levels are shared
by the rows). Convex, so it does not depend on a start; only the offsets come from outside.

Why the group norm and not the l1 norm: with u >= 0 the l1 norm sum_{b,j} u is linear, and for
diagonal rows it is almost fixed by the data (sum_lambda c_{b,lambda} = a_b(0) = 1): it shrinks
every row's total, not the number of grid points. The group norm is what counts grid points
shared by the rows: at u = 0 a group enters iff sum_b max(Re <v^, r_b> / s_b, 0)^2 > lambda^2,
the chi^2 gain of one real non-negative pole in all rows (F's candidate gain). So
lambda^2 = ``lambda_chi2`` is in F's units (``add_chi2``). ``penalty="l1"`` for comparison: a
row's entry enters on its own row's gain.

The solver: cvxpy (Clarabel) on a working set of grid points, grown by the optimality test on the
whole grid (a working-set method, exact at its end): solve on the set; per grid point the
statistic |max(Re V^H r, 0)|_2 over rows (l1: the max over rows), for all points at once by a
zero-padded FFT (the grid is uniform in E); add the ``working_add`` worst points over lambda,
drop the zero ones; stop when none is over lambda (1 + ``kkt_tolerance``). The whole grid in one
cvxpy problem is too slow (a 2000-point grid, 10 rows: 6 min); a set of a few hundred points
takes seconds. The set starts at the grid points nearest the start's poles. On the set the
dictionary is compressed exactly by a QR (|M u - y|^2 = |R u - Q^T y|^2 + const).

Then: runs of adjacent non-zero grid points are clustered into one pole each (weighted mean E
and gamma), split where the weight dips; then close poles are merged and weak ones dropped by
chi^2 (``merge_chi2``; the lasso splits one level into two spikes a few grid points apart). The
offsets may alternate with the sparse fit (``alternations``: per row group the offset that best
fits the grid poles, with the prior; off by default: it did not help, see the log of
2026-09-29, T3). Optionally F's ``refine`` and ``prune`` of the final poles (``refine``; slow).
F's add/drop loop from this start: ``pursuit.fit(A, time_us, settings, start=<this fit>)``.
Time: about 10 s for 10 rows, 70-115 s for 35 rows (the fixed set; C's start not included).

Pure numerics.
"""
from dataclasses import dataclass
from typing import Annotated, Literal

import cvxpy as cp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import minimize_scalar

from fitting.qsim.poles import joint_refined, pursuit
from fitting.qsim.poles.joint_refined import remove_offsets
from fitting.qsim.poles.noise import row_noise
from fitting.qsim.poles.pole_fit import (PoleFit, check_shape, normalize_to_initial_return, sample_time,
                                         wrap_frequency)


class SparseFitSettings(BaseModel):
    """The free parameters of fitter T3."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The fit that gives the offsets, the decay scale and the band (C), if no start is given.
    start: joint_refined.JointRefinedSettings = joint_refined.JointRefinedSettings()
    #: lambda^2: the chi^2 a grid point must buy to enter (group penalty; F's ``add_chi2`` units).
    lambda_chi2: Annotated[float, Field(gt=0.)] = 25.
    #: ``group``: l2 over rows per grid point (shared support); ``l1``: per row and grid point.
    penalty: Literal["group", "l1"] = "group"
    #: Grid points per FFT bin (20: a step of 0.05 bin).
    grid_per_bin: Annotated[int, Field(ge=2)] = 20
    #: The grid's decays, in units of the start's median decay. One: more decays let a level split
    #: into flanking spikes of other decays (August, 10 rows, true offsets: 13 false poles against 3).
    decay_factors: tuple[float, ...] = (1.,)
    #: The grid spans the start poles' span plus this many bins on each side; also the noise's band.
    band_margin_bins: Annotated[float, Field(gt=0.)] = 4.
    #: Working set: grid points added per round (the worst violators of the optimality test).
    working_add: Annotated[int, Field(ge=1)] = 150
    #: A grid point violates if its statistic exceeds lambda (1 + this).
    kkt_tolerance: Annotated[float, Field(ge=0.)] = 0.02
    max_rounds: Annotated[int, Field(ge=1)] = 40
    #: A group is zero if its norm is under this fraction of the largest or of lambda (the solver's floor).
    zero_fraction: Annotated[float, Field(gt=0.)] = 1e-6
    solver: str = "CLARABEL"
    fallback_solver: str = "SCS"
    #: A run of non-zero grid points splits at an interior minimum under this fraction of the
    #: smaller of its two neighbouring maxima.
    split_ratio: Annotated[float, Field(gt=0., le=1.)] = 0.5
    #: F's refine and prune of the clustered poles (E, gamma, offsets); False: the grid poles.
    #: Slow from 35 poles (over 9 min, August 10 rows); F's own loop does it anyway (T3F).
    refine: bool = False
    #: Offsets <-> sparse fit: this many times the offsets per group (the grid poles fixed, a
    #: scan then a 1-D minimum with the prior), then the sparse fit again.
    alternations: Annotated[int, Field(ge=0)] = 0
    #: Merge two adjacent grid poles (closer than ``merge_bins``) or drop one while that costs less
    #: chi^2 than this (F's ``drop_chi2`` units; poles fixed, no refine). 0: off.
    merge_chi2: Annotated[float, Field(ge=0.)] = 25.
    merge_bins: Annotated[float, Field(gt=0.)] = 0.5
    #: ``scan``: the offsets alone, per group, the grid poles fixed; ``refine``: F's refine of the
    #: grid poles and the offsets together (no prune), its offsets kept.
    offset_step: Literal["scan", "refine"] = "scan"
    #: The offset scan: +- this many prior widths, in 2 x ``offset_scan_points`` steps.
    offset_scan_priors: Annotated[float, Field(gt=0.)] = 4.
    offset_scan_points: Annotated[int, Field(ge=1)] = 40
    #: F's settings for the refine (offset prior, c >= 0, drop_chi2).
    refine_settings: pursuit.PursuitSettings = pursuit.PursuitSettings()


@dataclass(frozen=True)
class Grid:
    """The dictionary: E_j (MHz, all G = grid_per_bin x samples aliases), gamma_d, the band mask."""
    frequencies_MHz: np.ndarray
    decays_per_us: np.ndarray
    band: np.ndarray


@dataclass(frozen=True)
class SparseSolution:
    """The grid amplitudes x (decay x row x grid point, real >= 0, in units of a_b), and the solver's
    record: working-set rounds, the final set's size, chi^2, and the largest statistic / lambda."""
    grid: Grid
    amplitudes: np.ndarray
    rounds: int
    working_set: int
    chi2: float
    worst_kkt: float
    #: 1/2 chi^2 + lambda P(u), the convex objective at the solution.
    objective: float


def fit(A, time_us, settings=SparseFitSettings(), row_groups=None, start=None, offsets_MHz=None):
    """-> the PoleFit of rows ``A``; ``row_groups`` labels rows sharing one offset (default: each row).

    ``start``: a PoleFit of the same rows (C's, a cache), for the offsets, the decay scale and the
    band; else C runs with ``settings.start``. ``offsets_MHz``: per row, replaces the start's
    offsets (e.g. the true ones, an oracle)."""
    dt_us = sample_time(time_us)
    a = normalize_to_initial_return(check_shape(A, time_us))
    time_us = np.asarray(time_us, dtype=float)
    groups = np.unique(np.arange(len(a)) if row_groups is None else np.asarray(row_groups), return_inverse=True)[1]
    start = start or joint_refined.fit(A, time_us, settings.start, row_groups)
    row_offsets = start.row_offsets_MHz if offsets_MHz is None else np.asarray(offsets_MHz, dtype=float)
    offsets = np.bincount(groups, weights=row_offsets) / np.bincount(groups)
    noise = row_noise(a, dt_us, start.frequencies_MHz, settings.band_margin_bins)
    grid = make_grid(start, len(time_us), dt_us, settings)
    data = (a, time_us, groups, noise)
    for alternation in range(settings.alternations + 1):
        solution = sparse_amplitudes(remove_offsets(a, time_us, offsets[groups]), noise, time_us, grid, settings,
                                     start.frequencies_MHz)
        frequencies, decays = cluster(solution, settings.split_ratio)
        if settings.merge_chi2:
            frequencies, decays = merge_and_drop(data, frequencies, decays, offsets, settings)
        if alternation < settings.alternations and settings.offset_step == "scan":
            offsets = fit_offsets(data, frequencies, decays, offsets, settings)
        elif alternation < settings.alternations:
            offsets = pursuit.refine(data, frequencies, decays, offsets, settings.refine_settings).offsets_MHz
    if settings.refine:
        poles = pursuit.prune(pursuit.refine(data, frequencies, decays, offsets, settings.refine_settings), data,
                              settings.refine_settings)
        frequencies, decays, offsets = poles.frequencies_MHz, poles.decays_per_us, poles.offsets_MHz
    return grid_pole_fit(data, frequencies, decays, offsets, settings.refine_settings.nonnegative)


def fit_offsets(data, frequencies_MHz, decays_per_us, offsets_MHz, settings):
    """-> per group the offset minimizing sum_{b in g} chi^2_b(delta) + (delta / sigma)^2, the poles fixed and
    c >= 0 refitted (the rows of a group are independent of the others'): a scan over
    +- ``offset_scan_priors`` sigma, then a bounded 1-D minimum around the best point."""
    a, time_us, groups, noise = data
    sigma = settings.refine_settings.offset_prior_MHz
    nonnegative = settings.refine_settings.nonnegative

    def cost(delta, rows):
        r = pursuit.amplitudes(remove_offsets(a[rows], time_us, np.full(len(rows), delta)), time_us, frequencies_MHz,
                               decays_per_us, nonnegative)[0]
        return float(np.sum(np.abs(r / noise[rows, None]) ** 2) + (delta / sigma) ** 2)

    scan = np.linspace(-1, 1, 2 * settings.offset_scan_points + 1) * settings.offset_scan_priors * sigma
    step = scan[1] - scan[0]
    new = np.array(offsets_MHz, dtype=float)
    for g in range(len(new)):
        rows = np.flatnonzero(groups == g)
        best = scan[np.argmin([cost(delta, rows) for delta in scan])]
        new[g] = minimize_scalar(cost, bounds=(best - step, best + step), args=(rows,), method="bounded",
                                 options=dict(xatol=1e-3 * step)).x
    return new


def chi2_of(data, frequencies_MHz, decays_per_us, offsets_MHz, nonnegative=True):
    """-> sum_b |a_b e^{2 pi i delta t} - V c_b|^2 / s_b^2, c real (>= 0) by least squares (no prior)."""
    a, time_us, groups, noise = data
    r = pursuit.amplitudes(remove_offsets(a, time_us, offsets_MHz[groups]), time_us, frequencies_MHz, decays_per_us,
                           nonnegative)[0]
    return float(np.sum(np.abs(r / noise[:, None]) ** 2))


def merge_and_drop(data, frequencies_MHz, decays_per_us, offsets_MHz, settings):
    """-> (E, gamma) after merging and dropping, one at a time, the cheapest while it costs less
    than ``merge_chi2`` of chi^2 (poles and offsets fixed, c >= 0 refitted): merge two adjacent
    poles closer than ``merge_bins`` into one at their weight-weighted mean, or drop one pole.
    Undoes the lasso's split of one level into two spikes, and its small spikes."""
    time_us = data[1]
    bin_MHz = 1 / (len(time_us) * (time_us[1] - time_us[0]))
    nonnegative = settings.refine_settings.nonnegative
    E, gamma = np.asarray(frequencies_MHz, dtype=float), np.asarray(decays_per_us, dtype=float)
    order = np.argsort(E)
    E, gamma = E[order], gamma[order]
    while len(E) > 1:
        a, _, groups, _ = data
        c = pursuit.amplitudes(remove_offsets(a, time_us, offsets_MHz[groups]), time_us, E, gamma, nonnegative)[1]
        weight = np.maximum(c.sum(axis=0), 1e-12)
        now = chi2_of(data, E, gamma, offsets_MHz, nonnegative)
        trials = [(np.delete(E, k), np.delete(gamma, k)) for k in range(len(E))]
        for k in np.flatnonzero(np.diff(E) < settings.merge_bins * bin_MHz):
            w = weight[k:k + 2] / weight[k:k + 2].sum()
            trials.append((np.r_[E[:k], w @ E[k:k + 2], E[k + 2:]], np.r_[gamma[:k], w @ gamma[k:k + 2], gamma[k + 2:]]))
        costs = [chi2_of(data, *trial, offsets_MHz, nonnegative) - now for trial in trials]
        best = int(np.argmin(costs))
        if costs[best] >= settings.merge_chi2:
            break
        E, gamma = trials[best]
    return E, gamma


def make_grid(start, samples, dt_us, settings):
    """-> the Grid: E_j = j / (G dt) aliased (G = grid_per_bin x samples), gamma_d = factor x the
    start's median decay, the band = the start poles' span +- ``band_margin_bins``."""
    size = settings.grid_per_bin * samples
    frequencies = np.fft.fftfreq(size, d=dt_us)
    margin = settings.band_margin_bins / (samples * dt_us)
    E = wrap_frequency(start.frequencies_MHz, dt_us)
    band = (frequencies >= E.min() - margin) & (frequencies <= E.max() + margin)
    gamma = max(float(np.median(start.decays_per_us)), 1e-6)
    return Grid(frequencies, gamma * np.asarray(settings.decay_factors, dtype=float), band)


def sparse_amplitudes(y, noise, time_us, grid, settings, start_MHz=()):
    """-> the SparseSolution of min 1/2 sum_b |y_b / s_b - V^ u_b|^2 + lambda P(u), u >= 0 (P the
    group or the l1 norm), by cvxpy on a working set grown by the optimality test; the set starts
    at the grid points nearest ``start_MHz`` (all decays)."""
    size, decays, rows = len(grid.frequencies_MHz), len(grid.decays_per_us), len(y)
    envelopes = np.exp(-np.outer(grid.decays_per_us, time_us))
    norms = np.linalg.norm(envelopes, axis=1)
    envelopes /= norms[:, None]                                                  # |v^| = 1
    target = y / noise[:, None]
    lam = np.sqrt(settings.lambda_chi2)
    band = np.tile(grid.band, decays)                                            # flat index d * G + j

    def statistic(r):                                  # per grid point (flat), in units of lambda
        s = size * np.fft.ifft(envelopes[:, None, :] * r[None], n=size, axis=-1).real      # Re V^H r
        s = np.maximum(s, 0.)
        value = np.linalg.norm(s, axis=1) if settings.penalty == "group" else s.max(axis=1)
        return np.where(band, value.ravel(), 0.) / lam

    nearest = np.abs(wrap_frequency(np.subtract.outer(np.asarray(start_MHz, dtype=float), grid.frequencies_MHz),
                                    time_us[1] - time_us[0])).argmin(axis=1) if len(start_MHz) else np.array([], int)
    working = np.unique((np.arange(decays)[:, None] * size + nearest[None, :]).ravel())
    working = working[band[working]]
    u = np.zeros((0, rows))
    for round_ in range(1, settings.max_rounds + 1):
        if len(working):
            u = solve_working_set(dictionary(working, grid, envelopes, time_us), target, lam, settings)
            keep = np.linalg.norm(u, axis=1) > settings.zero_fraction * max(np.linalg.norm(u, axis=1).max(), lam)
            working, u = working[keep], u[keep]
        residual = target - (dictionary(working, grid, envelopes, time_us) @ u).T if len(working) else target
        test = statistic(residual)
        test[working] = 0.
        worst = float(test.max())
        if worst <= 1 + settings.kkt_tolerance:
            break
        new = np.argsort(test)[::-1][:settings.working_add]
        working = np.r_[working, new[test[new] > 1 + settings.kkt_tolerance]]
        u = np.r_[u, np.zeros((len(working) - len(u), rows))]
    x = np.zeros((decays * size, rows))
    x[working] = u * noise[None, :] / norms[working // size][:, None]
    chi2 = float(np.sum(np.abs(residual) ** 2))
    penalty = np.linalg.norm(u, axis=1).sum() if settings.penalty == "group" else u.sum()
    return SparseSolution(grid, x.T.reshape(rows, decays, size).transpose(1, 0, 2), round_, len(working), chi2, worst,
                          0.5 * chi2 + lam * float(penalty))


def dictionary(indices, grid, envelopes, time_us):
    """-> V^ on the flat grid indices d * G + j: columns exp(-2 pi i E_j t) envelope_d(t), |v^| = 1."""
    size = len(grid.frequencies_MHz)
    E = grid.frequencies_MHz[indices % size]
    return envelopes[indices // size].T * np.exp(-2j * np.pi * np.outer(time_us, E))


def solve_working_set(V, target, lam, settings):
    """-> u >= 0 (point x row) minimizing 1/2 |M u - Y|^2 + lambda P(u) by cvxpy, M = [Re V; Im V]
    compressed by its QR (M = Q R: |M u - Y|^2 = |R u - Q^T Y|^2 + const)."""
    M = np.vstack([V.real, V.imag])
    Y = np.vstack([target.T.real, target.T.imag])
    Q, R = np.linalg.qr(M)
    u = cp.Variable((M.shape[1], Y.shape[1]), nonneg=True)
    penalty = cp.sum(cp.norm(u, 2, axis=1)) if settings.penalty == "group" else cp.sum(u)
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(R @ u - Q.T @ Y) + lam * penalty))
    try:
        problem.solve(solver=settings.solver)
    except cp.error.SolverError:                     # Clarabel fails rarely (1 of 80 cases); SCS then
        problem.solve(solver=settings.fallback_solver)
    return np.maximum(u.value, 0.)


def grid_weights(solution):
    """-> (E, weight, gamma) per band grid point in order of E: weight = sum over rows and decays
    of x, gamma = the x-weighted mean decay."""
    grid = solution.grid
    order = np.flatnonzero(grid.band)[np.argsort(grid.frequencies_MHz[grid.band])]
    per_decay = solution.amplitudes.sum(axis=1)[:, order]                    # decay x point
    weight = per_decay.sum(axis=0)
    gamma = (grid.decays_per_us @ per_decay) / np.maximum(weight, 1e-300)
    return grid.frequencies_MHz[order], weight, gamma


def cluster(solution, split_ratio=0.5):
    """-> (E, gamma) of the poles: runs of adjacent non-zero grid points, split at an interior
    minimum under ``split_ratio`` of the smaller neighbouring maximum; each run's weighted means."""
    E, weight, gamma = grid_weights(solution)
    frequencies, decays = [], []
    for run in np.split(np.arange(len(weight)), np.flatnonzero(np.diff(weight > 0)) + 1):
        if weight[run[0]] <= 0:
            continue
        for part in split_run(weight[run], split_ratio):
            w = weight[run][part]
            frequencies.append(np.sum(w * E[run][part]) / w.sum())
            decays.append(np.sum(w * gamma[run][part]) / w.sum())
    return np.asarray(frequencies), np.asarray(decays)


def split_run(weight, split_ratio):
    """-> index arrays of the parts of one run: cut at each interior local minimum that is under
    ``split_ratio`` x the smaller of the maxima on its two sides."""
    cuts, begin = [], 0
    for k in range(1, len(weight) - 1):
        if weight[k] <= weight[k - 1] and weight[k] < weight[k + 1]:
            if weight[k] < split_ratio * min(weight[begin:k].max(), weight[k + 1:].max()):
                cuts.append(k)
                begin = k
    edges = [0, *cuts, len(weight)]
    return [np.arange(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]


def grid_pole_fit(data, frequencies_MHz, decays_per_us, offsets_MHz, nonnegative=True):
    """-> the PoleFit of these poles and group offsets, amplitudes real (>= 0) by least squares."""
    a, time_us, groups, _ = data
    c = pursuit.amplitudes(remove_offsets(a, time_us, offsets_MHz[groups]), time_us, frequencies_MHz, decays_per_us,
                           nonnegative)[1]
    E = wrap_frequency(np.asarray(frequencies_MHz), time_us[1] - time_us[0])
    order = np.argsort(E)
    return PoleFit(E[order], np.asarray(decays_per_us)[order], c[:, order], len(E), None, offsets_MHz[groups])
