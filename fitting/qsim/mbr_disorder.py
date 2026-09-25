"""Numerics of the diagonal-disorder campaign: realizations, selection, level statistics.

Taken from the 7-1 diagonal-disorder notebook cells (jonginn's
``qsim_experiments.ipynb`` cells 323-334 and ``data_postprocess.ipynb`` cells
246-249), which held them inline, in MBR redesign step 7b (2026-09-24). The
arithmetic is unchanged. Two small differences, both noted where they apply:
the duplicate-level floor of :func:`adjacent_gap_ratios` scales with the
level magnitude (the preview cells' version), and :func:`match_levels` takes
the matching tolerance directly.

Pure numerics: arrays in, arrays out. ``experiments/qsim/mbr_disorder_ensemble.py``
and ``notebook_helpers/mbr_disorder_campaign.py`` use these.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

from slab import AttrDict

#: Mean adjacent-gap ratio of uncorrelated (Poisson) levels.
POISSON_MEAN = 2 * np.log(2) - 1
#: Mean adjacent-gap ratio of the GOE (Wigner-like surmise).
GOE_MEAN = 4 - 2 * np.sqrt(3)


def poisson_pdf(ratio):
    """Adjacent-gap-ratio density of Poisson levels, on [0, 1]."""
    ratio = np.asarray(ratio, dtype=float)
    return 2 / (1 + ratio) ** 2


def goe_pdf(ratio):
    """Adjacent-gap-ratio density of the GOE (Wigner-like surmise), on [0, 1]."""
    ratio = np.asarray(ratio, dtype=float)
    return (27 / 4) * (ratio + ratio ** 2) / (1 + ratio + ratio ** 2) ** 2.5


def disorder_direction(seed, leaf_count):
    """-> a random zero-mean unit vector over the storage modes (7-1 realizations).

    ``np.random.default_rng(seed).normal``, minus its mean, divided by its
    norm. The onsite energy of realization r is ``strength * direction`` and
    its pulse detunings are ``-strength * direction``.
    """
    direction = np.random.default_rng(seed).normal(size=int(leaf_count))
    direction -= np.mean(direction)
    direction /= np.linalg.norm(direction)
    return direction


def select_diagonal_rows(weights, number_to_select):
    """Max-min LDOS coverage selection of diagonal traces (7-1).

    ``weights[row, level]`` is |<row|level>|^2. Chooses ``number_to_select``
    rows so that the weakest level's best weight is as large as possible
    (a MILP, bisected over the floor), then, at that floor, the rows that cover
    the most levels. Returns ``(rows, floor, coverage)``: the sorted row
    indices, the floor reached, and each level's best weight over the rows.
    """
    weights = np.asarray(weights, dtype=float)
    row_count, level_count = weights.shape
    number_to_select = min(int(number_to_select), row_count)

    def solve(weight_floor, objective=None):
        visible = (weights.T >= weight_floor).astype(float)
        if np.any(visible.sum(axis=1) == 0):
            return None
        constraint_matrix = np.vstack([
            np.ones((1, row_count)),
            visible,
        ])
        result = milp(
            c=(
                np.zeros(row_count)
                if objective is None else objective
            ),
            integrality=np.ones(row_count, dtype=int),
            bounds=Bounds(np.zeros(row_count), np.ones(row_count)),
            constraints=LinearConstraint(
                constraint_matrix,
                np.r_[float(number_to_select), np.ones(level_count)],
                np.r_[
                    float(number_to_select),
                    np.full(level_count, np.inf),
                ],
            ),
            options={"disp": False},
        )
        if not result.success or result.x is None:
            return None
        rows = np.flatnonzero(result.x > 0.5)
        return rows if len(rows) == number_to_select else None

    floors = np.unique(np.r_[0.0, weights.ravel()])
    lower = 0
    upper = len(floors) - 1
    best_floor = 0.0
    selected_rows = None
    while lower <= upper:
        middle = (lower + upper) // 2
        trial_rows = solve(float(floors[middle]))
        if trial_rows is None:
            upper = middle - 1
        else:
            best_floor = float(floors[middle])
            selected_rows = trial_rows
            lower = middle + 1

    scale = max(best_floor, 1e-12)
    utility = np.sum(np.minimum(weights / scale, 1.0), axis=1)
    balanced_rows = solve(best_floor, objective=-utility)
    if balanced_rows is not None:
        selected_rows = balanced_rows
    selected_rows = np.asarray(sorted(selected_rows), dtype=int)
    coverage = weights[selected_rows].max(axis=0)
    return selected_rows, best_floor, coverage


def cycle_grid(max_abs_energy_MHz, floquet_cycle_us, max_cycle, min_time_points,
               nyquist_margin):
    """-> the 7-1 cycle grid: Nyquist-safe, at least ``min_time_points`` points.

    The step is the largest that keeps ``nyquist_margin * max|E|`` below
    Nyquist and still gives ``min_time_points`` points below ``max_cycle``;
    an odd step above 1 is made even. Raises if the margin cannot be met.

    Returns AttrDict with ``cycles``, ``step``, ``dt_us``, ``nyquist_MHz`` and
    ``fft_resolution_MHz``.
    """
    step_nyquist = int(np.floor(
        1.0 / (2.0 * nyquist_margin * max_abs_energy_MHz * floquet_cycle_us)))
    step_points = max(1, (max_cycle - 1) // (min_time_points - 1))
    step = max(1, min(step_nyquist, step_points))
    if step > 1 and step % 2:
        step -= 1
    cycles = np.arange(0, max_cycle, step, dtype=int)
    dt_us = step * floquet_cycle_us
    nyquist_MHz = 1.0 / (2.0 * dt_us)
    if nyquist_MHz < nyquist_margin * max_abs_energy_MHz:
        raise RuntimeError("the cycle step violates the requested Nyquist margin")
    return AttrDict(dict(
        cycles=cycles,
        step=int(step),
        dt_us=float(dt_us),
        nyquist_MHz=float(nyquist_MHz),
        fft_resolution_MHz=float(1.0 / (len(cycles) * dt_us)),
    ))


def trim_count(edge_fraction, dimension):
    """-> the number of levels cut at each spectrum edge: ceil(fraction * D)."""
    return int(np.ceil(edge_fraction * dimension))


def adjacent_gap_ratios(levels_MHz, trim_count):
    """-> r_n = min(s_n, s_n+1) / max(s_n, s_n+1) of the sorted bulk levels.

    ``trim_count`` levels are cut at each edge first. Raises ``ValueError`` if
    fewer than three bulk levels remain, or two levels are closer than
    ``100 eps * max(1, max|level|)`` (duplicate or unresolved poles).
    """
    levels_MHz = np.sort(np.asarray(levels_MHz, dtype=float))
    if trim_count:
        levels_MHz = levels_MHz[trim_count:-trim_count]
    gaps_MHz = np.diff(levels_MHz)
    if len(gaps_MHz) < 2:
        raise ValueError("at least three bulk levels are required")
    gap_floor_MHz = (
        100 * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(levels_MHz))))
    )
    if np.any(gaps_MHz <= gap_floor_MHz):
        raise ValueError("duplicate or unresolved levels")
    return (
        np.minimum(gaps_MHz[:-1], gaps_MHz[1:])
        / np.maximum(gaps_MHz[:-1], gaps_MHz[1:])
    )


def match_levels(poles_MHz, theory_levels_MHz, tolerance_MHz):
    """Match measured poles to theory levels one to one (Hungarian assignment).

    A pair counts only if it is within ``tolerance_MHz``; the 7-1 cells used
    ``match_tolerance_bins * fft_resolution_MHz``. Returns AttrDict with
    ``pole_rows`` and ``theory_rows`` (the matched pairs), ``missing_theory_MHz``,
    ``spurious_poles_MHz``, ``matched_count``, ``complete`` (as many poles as
    levels, all matched) and ``mae_MHz`` (NaN if nothing matched).
    """
    poles_MHz = np.asarray(poles_MHz, dtype=float)
    theory_levels_MHz = np.asarray(theory_levels_MHz, dtype=float)
    dimension = len(theory_levels_MHz)
    abs_error_MHz = np.abs(poles_MHz[:, None] - theory_levels_MHz[None, :])
    inside = abs_error_MHz <= tolerance_MHz
    if abs_error_MHz.size:
        penalty = (dimension + 1) * (float(np.max(abs_error_MHz)) + 1.0)
        rows, columns = linear_sum_assignment(np.where(inside, abs_error_MHz, penalty))
        valid = inside[rows, columns]
        pole_rows, theory_rows = rows[valid], columns[valid]
    else:
        pole_rows = theory_rows = np.array([], dtype=int)
    missing = np.setdiff1d(np.arange(dimension), theory_rows)
    spurious = np.setdiff1d(np.arange(len(poles_MHz)), pole_rows)
    return AttrDict(dict(
        pole_rows=pole_rows,
        theory_rows=theory_rows,
        missing_theory_MHz=theory_levels_MHz[missing],
        spurious_poles_MHz=poles_MHz[spurious],
        matched_count=int(len(pole_rows)),
        complete=bool(len(poles_MHz) == dimension and len(pole_rows) == dimension),
        tolerance_MHz=float(tolerance_MHz),
        mae_MHz=(float(np.mean(abs_error_MHz[pole_rows, theory_rows]))
                 if len(pole_rows) else np.nan),
    ))


def pooled_statistics(ratios_by_realization):
    """Pool gap ratios over realizations. ``{realization: ratios}`` -> AttrDict.

    ``mean`` is the mean of the realization means, ``sem`` its standard error
    (0 for one realization), and ``closer_to`` "GOE" or "Poisson".
    """
    labels = sorted(ratios_by_realization)
    if not labels:
        raise ValueError("no realization has gap ratios")
    means = np.asarray([np.mean(ratios_by_realization[label]) for label in labels])
    mean = float(np.mean(means))
    sem = (float(np.std(means, ddof=1) / np.sqrt(len(means)))
           if len(means) > 1 else 0.0)
    return AttrDict(dict(
        realizations=labels,
        pooled=np.concatenate([ratios_by_realization[label] for label in labels]),
        realization_means=means,
        mean=mean,
        sem=sem,
        closer_to=("GOE" if abs(mean - GOE_MEAN) < abs(mean - POISSON_MEAN)
                   else "Poisson"),
    ))


def trace_and_form_factor(normalized_returns):
    """Tr U(t) per realization and the spectral form factor, from diagonal returns.

    ``normalized_returns[r]`` is one realization's ``A_norm`` (occupation,
    cycle): every diagonal return of a *complete* basis, each divided by its
    t = 0 value. Tr U(t) = sum_a A_aa(t). Returns AttrDict with ``trace``
    (R, Q) complex, ``sff`` = <|Tr U|^2> over realizations (Q,), and
    ``sff_normalized`` = sff / D^2, which starts at 1.
    """
    stacked = np.asarray(normalized_returns, dtype=complex)
    if stacked.ndim != 3:
        raise ValueError("expected (realization, occupation, cycle) returns")
    dimension = stacked.shape[1]
    trace = stacked.sum(axis=1)
    sff = np.mean(np.abs(trace) ** 2, axis=0)
    return AttrDict(dict(trace=trace, sff=sff, sff_normalized=sff / dimension ** 2,
                         dimension=int(dimension)))
