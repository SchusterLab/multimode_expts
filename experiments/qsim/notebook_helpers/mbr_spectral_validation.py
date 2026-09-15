"""Spectral validation: does the recovered level statistics support a claim?

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
188, 252-255 and 277-306 by the stage-2 notebook decomposition. Primary
caller: `analysis_notebooks/202609_qsim_migration/mbr_spectral_validation.py`.

On the surface map this is the second "measurement and inference studies"
workspace, and its subject is stated by the source's own heading for cell 281:
*can the measured level statistics support a conclusion?* Everything here is a
different attempt to answer that without leaning on theory:

  188        global-only Matrix Pencil, using no Hamiltonian energies and no
             FFT peak positions
  252-255    joint block-Hankel shared-pole test at fixed rank
  277-280    alternative pole selection by row-normalized component
             singular values
  282        whether the recovered statistics are conclusive at all
  283-291    data-only shared-frequency refinement: clock correction, and
             fitting disjoint halves of the measured shots
  292-297    theory-free pole recovery, its 1 kHz reproducibility, and how
             sensitive the level statistics are to it
  298-303    independent-half reproducibility, then a posthoc comparison
             against the configured Hamiltonian
  304-306    an alternative row-pooled target-35 diagnostic

**These are competing methods, not a pipeline.** The stage-2 instructions say
to keep algorithm variants separate, and that matters more here than anywhere
else in the split: several of these selections disagree with each other, and
which one is right is the open question the workspace exists to settle. None
were merged, renamed to look alike, or reconciled with
`MBRSpectrumExperiment`'s own spectrum methods.

Most of these cells were already top-level `def`s, so they moved with their
bodies untouched. The procedural cells that drive them became one wrapper
function each, taking what the source read from the notebook namespace.

## Two handoffs the split had to make explicit

Cell 188 reads `encspec_reprocessed`, built by the N=3 reprocessing section of
what is now `analysis_notebooks/202609_qsim_migration/mbr.py`. Cells 279
onward read a long list of names from the disorder analysis section, now
`mbr_disorder.py`. Both are arguments here; the calling notebook builds them
by calling `mbr_n3_reprocess.reprocess_n3_spectroscopy` and the
`mbr_disorder_h5` loaders rather than relying on cell order.

Temporary home, per the stage-2 instructions.
"""

# Collected from the imports the source cells did individually: cell 188
# (sliding_window_view), 253 (linear_sum_assignment), 282 (inspect), 284
# (time, scipy.linalg, minimize, copy-as-joint_shallow_copy,
# MMAveragerProgram), 293 (sha256, nullcontext, threadpool_limits) and 297
# (pandas). They are one block here because the split turned per-cell imports
# into module imports.
import inspect
import math
import time
from contextlib import nullcontext
from copy import copy as joint_shallow_copy
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import linalg
from scipy.optimize import linear_sum_assignment, minimize

# TODO(stage2): threadpoolctl is imported by source cells 293 and 299, to
# bound BLAS threads during the precision search, but it is not installed in
# this pixi environment and is not declared in pyproject.toml. Those two cells
# therefore could not have run here as written -- a pre-existing break, not one
# the split introduced. Rather than add a dependency unilaterally or drop the
# thread limiting silently, the name is a shim that raises when used, so the
# rest of this module still imports and the affected functions fail loudly.
try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:
    def threadpool_limits(*args, **kwargs):  # noqa: D103
        raise ModuleNotFoundError(
            "threadpoolctl is not installed in this environment, but source "
            "cells 293 and 299 used it to bound BLAS threads during the "
            "precision search. Either add threadpoolctl to pyproject.toml, or "
            "decide that the thread limiting is not needed and remove it from "
            "the precision functions."
        )

precision_threadpool_limits = threadpool_limits

from slab import AttrDict

from IPython.display import display

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment

# Both of these were notebook globals here, defined by the *disorder* HDF5
# section: SavedSpectroscopyExperiment is cell 261's notebook-local loader and
# wrap_frequency is cell 274's. Importing them keeps one definition rather
# than a third copy -- see the TODO in mbr_disorder_h5 about that loader not
# yet being reconciled with the library.
from experiments.qsim.notebook_helpers.mbr_disorder_h5 import (
    SavedSpectroscopyExperiment,
    wrap_frequency,
)


# ------------------------------------------------------------------------
# Cell 188.
# ------------------------------------------------------------------------

def matrix_pencil_global_diagnostic(encspec_reprocessed):
    """Global-only Matrix Pencil (cell 188).

    Uses no Hamiltonian energies and no FFT peak positions. The next
    step performs the rowwise selection and merging."""
    # Global-only Matrix Pencil diagnostic; the next cell performs rowwise selection and merging.
    # No Hamiltonian energies or FFT peak positions are used.
    from numpy.lib.stride_tricks import sliding_window_view

    matrix_pencil_data = encspec_reprocessed
    matrix_pencil_requested_max_modes = 35
    matrix_pencil_numerical_floor = 1e-10
    matrix_pencil_min_persistence_fraction = 0.6

    reconstruction = matrix_pencil_data.reconstruction
    spectrum = matrix_pencil_data.spectrum
    if not hasattr(reconstruction, 'A'):
        raise ValueError('Matrix Pencil requires complex reconstruction.A; a magnitude-only merged spectrum cannot be used')

    time_us = np.asarray(spectrum.time_us, dtype=float)
    complex_return = np.asarray(reconstruction.A, dtype=complex)
    if complex_return.ndim != 2 or complex_return.shape[1] != len(time_us):
        raise ValueError('reconstruction.A must have shape (occupation, time point)')
    if len(time_us) < 5:
        raise ValueError('Matrix Pencil requires at least five time points')

    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
        raise ValueError('Matrix Pencil requires uniformly spaced time points')
    if np.any(np.abs(complex_return[:, 0]) < 1e-12):
        raise ValueError('at least one occupation has zero return at the first time point')
    complex_return = complex_return / complex_return[:, :1]

    # 1. Every row supplies a shifted pair of Hankel matrices with the same poles.
    sample_count = complex_return.shape[1]
    pencil_length = sample_count // 2
    unshifted_blocks = []
    shifted_blocks = []
    for row in complex_return:
        windows = sliding_window_view(row, pencil_length + 1)
        unshifted_blocks.append(windows[:, :-1])
        shifted_blocks.append(windows[:, 1:])
    unshifted = np.vstack(unshifted_blocks)
    shifted = np.vstack(shifted_blocks)

    # 2. SVD separates the shared exponential subspace from small residual directions.
    left_vectors, singular_values, right_vectors_h = np.linalg.svd(
        unshifted,
        full_matrices=False,
    )
    relative_singular_values = singular_values / singular_values[0]
    numerical_rank = np.count_nonzero(
        relative_singular_values > matrix_pencil_numerical_floor
    )
    maximum_rank = min(
        matrix_pencil_requested_max_modes,
        pencil_length,
        numerical_rank,
    )
    if maximum_rank < 1:
        raise RuntimeError('the Hankel matrix has no usable singular direction')

    # 3. Solve the reduced pencil at every allowed rank instead of fixing the number of tones.
    rank_results = {}
    for rank in range(1, maximum_rank + 1):
        left = left_vectors[:, :rank]
        right = right_vectors_h[:rank].conj().T
        shifted_reduced = left.conj().T @ shifted @ right
        inverse_singular_values = np.diag(1. / singular_values[:rank])
        reduced_pencil = inverse_singular_values @ shifted_reduced
        poles = np.linalg.eigvals(reduced_pencil)

        frequencies_MHz = -np.angle(poles) / (2. * np.pi * sample_time_us)
        pole_radii = np.abs(poles)
        decay_per_us = -np.log(np.maximum(pole_radii, np.finfo(float).tiny)) / sample_time_us
        order = np.argsort(frequencies_MHz)
        rank_results[rank] = dict(
            frequencies_MHz=frequencies_MHz[order],
            pole_radii=pole_radii[order],
            decay_per_us=decay_per_us[order],
        )

    # 4. A maximum-rank pole is more credible when lower-rank solutions return near it.
    resolution_MHz = 1. / (sample_count * sample_time_us)
    maximum_rank_result = rank_results[maximum_rank]
    maximum_rank_frequencies = maximum_rank_result['frequencies_MHz']
    anchor_groups = []
    for index, frequency in enumerate(maximum_rank_frequencies):
        if not anchor_groups:
            anchor_groups.append([index])
            continue

        previous_index = anchor_groups[-1][-1]
        previous_frequency = maximum_rank_frequencies[previous_index]
        if frequency - previous_frequency <= resolution_MHz:
            anchor_groups[-1].append(index)
        else:
            anchor_groups.append([index])

    anchor_indices = []
    for group in anchor_groups:
        group_radii = maximum_rank_result['pole_radii'][group]
        distance_from_unit_circle = np.abs(np.log(np.maximum(
            group_radii,
            np.finfo(float).tiny,
        )))
        representative = group[np.argmin(distance_from_unit_circle)]
        anchor_indices.append(representative)

    ranked_candidates = []
    for index in anchor_indices:
        frequency = maximum_rank_result['frequencies_MHz'][index]
        radius = maximum_rank_result['pole_radii'][index]
        decay = maximum_rank_result['decay_per_us'][index]
        matched_frequencies = []
        for result in rank_results.values():
            distances = np.abs(result['frequencies_MHz'] - frequency)
            nearest = np.argmin(distances)
            if distances[nearest] <= resolution_MHz:
                matched_frequencies.append(result['frequencies_MHz'][nearest])

        ranked_candidates.append(dict(
            frequency_MHz=float(np.median(matched_frequencies)),
            persistence=len(matched_frequencies),
            pole_radius=float(radius),
            decay_per_us=float(decay),
        ))
    ranked_candidates.sort(
        key=lambda candidate: (
            -candidate['persistence'],
            abs(np.log(max(candidate['pole_radius'], np.finfo(float).tiny))),
        )
    )
    minimum_persistence = int(np.ceil(
        matrix_pencil_min_persistence_fraction * maximum_rank
    ))
    stable_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate['persistence'] >= minimum_persistence
    ]

    encspec_matrix_pencil = dict(
        sample_time_us=sample_time_us,
        pencil_length=pencil_length,
        maximum_algebraic_rank=maximum_rank,
        resolution_MHz=resolution_MHz,
        singular_values=singular_values,
        relative_singular_values=relative_singular_values,
        rank_results=rank_results,
        ranked_candidates=ranked_candidates,
        minimum_persistence=minimum_persistence,
        stable_candidates=stable_candidates,
    )

    # 5. Plot the rank information without pretending that the maximum rank is the mode count.
    nyquist_MHz = 0.5 / sample_time_us
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    singular_index = np.arange(1, len(singular_values) + 1)
    axes[0].semilogy(singular_index, relative_singular_values, 'o-')
    axes[0].set(
        xlabel='singular-value index',
        ylabel='singular value / largest singular value',
        title='Hankel singular values; no rank selected',
    )

    plot_candidates = sorted(
        ranked_candidates,
        key=lambda candidate: candidate['frequency_MHz'],
    )
    candidate_frequencies = [candidate['frequency_MHz'] for candidate in plot_candidates]
    candidate_persistence = [candidate['persistence'] for candidate in plot_candidates]
    axes[1].vlines(
        candidate_frequencies,
        0.,
        candidate_persistence,
        color='0.75',
    )
    axes[1].scatter(
        candidate_frequencies,
        candidate_persistence,
        color='0.6',
        s=70,
        label='all maximum-rank candidates',
    )
    stable_frequencies = [candidate['frequency_MHz'] for candidate in stable_candidates]
    stable_persistence = [candidate['persistence'] for candidate in stable_candidates]
    axes[1].scatter(
        stable_frequencies,
        stable_persistence,
        color='tab:red',
        s=85,
        label='recurrent candidates',
    )
    for frequency, persistence in zip(stable_frequencies, stable_persistence):
        axes[1].annotate(
            f'{frequency:.4f}',
            (frequency, persistence),
            xytext=(0, 7),
            textcoords='offset points',
            ha='center',
            fontsize=8,
            rotation=45,
        )
    axes[1].axvline(0., color='0.7', linewidth=1.)
    axes[1].set(
        xlim=(-nyquist_MHz, nyquist_MHz),
        ylim=(0., maximum_rank + 1.),
        xlabel='energy E/h (MHz)',
        ylabel='number of ranks returning near this frequency',
        title=f'red: returned in at least {minimum_persistence} of {maximum_rank} ranks',
    )
    axes[1].legend()
    figure.suptitle(
        f'data-only multichannel Matrix Pencil; algebraic ceiling={maximum_rank}, not a selected mode count'
    )
    dict(
        minimum_persistence=minimum_persistence,
        stable_candidates=[
            (
                round(candidate['frequency_MHz'], 6),
                candidate['persistence'],
            )
            for candidate in stable_candidates
        ],
    )


# ------------------------------------------------------------------------
# Cell 253.
# ------------------------------------------------------------------------

def diag_fixed_rank_block_hankel(
    traces,
    time_us,
    rank,
    pencil_length=None,
):
    traces = np.asarray(traces, dtype=complex)
    if traces.ndim == 1:
        traces = traces[None, :]
    time_us = np.asarray(time_us, dtype=float)
    if traces.ndim != 2 or traces.shape[1] != len(time_us):
        raise ValueError("traces must have shape (channel, time)")
    if len(time_us) < 5:
        raise ValueError("at least five uniform time samples are required")

    sample_time_us = float(time_us[1] - time_us[0])
    if (
        sample_time_us <= 0.0
        or not np.allclose(np.diff(time_us), sample_time_us)
    ):
        raise ValueError("time_us must be uniformly increasing")
    if pencil_length is None:
        pencil_length = len(time_us) // 2
    if not 1 <= pencil_length < len(time_us):
        raise ValueError(
            "pencil_length must lie between 1 and sample_count - 1"
        )
    window_count = len(time_us) - pencil_length
    maximum_rank = min(
        pencil_length, len(traces) * window_count
    )
    if not 1 <= rank <= maximum_rank:
        raise ValueError(
            f"rank must be between 1 and {maximum_rank}"
        )

    row_scales = traces[:, 0].copy()
    if np.any(np.abs(row_scales) <= 100 * np.finfo(float).eps):
        raise ValueError("every diagonal trace must have nonzero A(0)")
    normalized_traces = traces / row_scales[:, None]
    row_norms = np.linalg.norm(normalized_traces, axis=1)
    svd_traces = normalized_traces / row_norms[:, None]

    unshifted_blocks = []
    shifted_blocks = []
    for row in svd_traces:
        windows = sliding_window_view(row, pencil_length + 1)
        unshifted_blocks.append(windows[:, :-1])
        shifted_blocks.append(windows[:, 1:])
    unshifted = np.vstack(unshifted_blocks)
    shifted = np.vstack(shifted_blocks)

    left, singular_values, right_h = np.linalg.svd(
        unshifted, full_matrices=False
    )
    left = left[:, :rank]
    right = right_h[:rank].conj().T
    shifted_reduced = left.conj().T @ shifted @ right
    reduced_pencil = np.linalg.solve(
        np.diag(singular_values[:rank]), shifted_reduced
    )
    poles = np.linalg.eigvals(reduced_pencil)
    frequencies_MHz = (
        -np.angle(poles) / (2 * np.pi * sample_time_us)
    )
    order = np.argsort(frequencies_MHz)
    poles = poles[order]
    frequencies_MHz = frequencies_MHz[order]
    pole_radii = np.abs(poles)
    decay_per_us = (
        -np.log(np.maximum(pole_radii, np.finfo(float).tiny))
        / sample_time_us
    )

    sample_index = np.arange(len(time_us))
    design = poles[None, :] ** sample_index[:, None]
    normalized_amplitudes, _, _, _ = np.linalg.lstsq(
        design, normalized_traces.T, rcond=None
    )
    normalized_fit = (design @ normalized_amplitudes).T
    fitted_traces = normalized_fit * row_scales[:, None]
    relative_residual = float(
        np.linalg.norm(traces - fitted_traces)
        / np.linalg.norm(traces)
    )
    discarded_singular_weight = float(
        np.linalg.norm(singular_values[rank:])
        / np.linalg.norm(singular_values)
    )
    singular_value_gap = (
        float(singular_values[rank - 1] / singular_values[rank])
        if rank < len(singular_values)
        else np.inf
    )
    return dict(
        rank=int(rank),
        channel_count=int(len(traces)),
        sample_count=int(len(time_us)),
        pencil_length=int(pencil_length),
        sample_time_us=sample_time_us,
        unshifted_shape=unshifted.shape,
        singular_values=singular_values,
        singular_value_gap=singular_value_gap,
        discarded_singular_weight=discarded_singular_weight,
        poles=poles,
        frequencies_MHz=frequencies_MHz,
        pole_radii=pole_radii,
        decay_per_us=decay_per_us,
        normalized_amplitudes=normalized_amplitudes.T,
        fitted_traces=fitted_traces,
        relative_residual=relative_residual,
        design_condition_number=float(np.linalg.cond(design)),
    )


def diag_match_levels(measured_MHz, theory_MHz, tolerance_MHz):
    measured_MHz = np.asarray(measured_MHz, dtype=float)
    theory_MHz = np.asarray(theory_MHz, dtype=float)
    distances_MHz = np.abs(
        measured_MHz[:, None] - theory_MHz[None, :]
    )
    n_measured = len(measured_MHz)
    n_theory = len(theory_MHz)
    unmatched_cost = 0.500001 * tolerance_MHz
    invalid_cost = 4.0 * max(tolerance_MHz, np.finfo(float).eps)
    assignment_cost = np.full(
        (n_measured + n_theory, n_theory + n_measured),
        invalid_cost,
        dtype=float,
    )
    assignment_cost[:n_measured, :n_theory] = np.where(
        distances_MHz <= tolerance_MHz,
        distances_MHz,
        invalid_cost,
    )
    assignment_cost[:n_measured, n_theory:] = unmatched_cost
    assignment_cost[n_measured:, :n_theory] = unmatched_cost
    assignment_cost[n_measured:, n_theory:] = 0.0
    assignment_rows, assignment_columns = linear_sum_assignment(
        assignment_cost
    )
    pairs = [
        (measured_index, theory_index)
        for measured_index, theory_index in zip(
            assignment_rows, assignment_columns
        )
        if measured_index < n_measured
        and theory_index < n_theory
        and distances_MHz[measured_index, theory_index]
        <= tolerance_MHz
    ]
    matched_measured = {pair[0] for pair in pairs}
    matched_theory = {pair[1] for pair in pairs}
    spurious = np.asarray([
        index for index in range(n_measured)
        if index not in matched_measured
    ], dtype=int)
    missing = np.asarray([
        index for index in range(n_theory)
        if index not in matched_theory
    ], dtype=int)
    errors_MHz = np.asarray([
        measured_MHz[measured_index] - theory_MHz[theory_index]
        for measured_index, theory_index in pairs
    ], dtype=float)
    return dict(
        pairs=pairs,
        matched_measured=np.asarray(
            sorted(matched_measured), dtype=int
        ),
        matched_theory=np.asarray(
            sorted(matched_theory), dtype=int
        ),
        spurious=spurious,
        missing=missing,
        errors_MHz=errors_MHz,
        mean_absolute_error_MHz=(
            float(np.mean(np.abs(errors_MHz)))
            if len(errors_MHz) else np.nan
        ),
    )


# ------------------------------------------------------------------------
# Cell 254.
# ------------------------------------------------------------------------

def report_block_hankel_matches(diag_stats_records):
    """Report the fixed-rank block-Hankel matches (cell 254)."""
    diag_joint_realization = 4
    diag_joint_rank = 35
    diag_joint_pencil_length = 70
    diag_joint_sum_pencil_length = None
    diag_joint_match_tolerance_bins = 0.2
    diag_joint_spectrum_xlim_kHz = None

    diag_joint_record = diag_stats_records[diag_joint_realization]
    if "data" not in diag_joint_record:
        raise RuntimeError(
            f"r={diag_joint_realization} is unavailable: "
            f"{diag_joint_record.get('error', 'unknown error')}"
        )
    diag_joint_data = diag_joint_record["data"]
    diag_joint_traces = np.asarray(
        diag_joint_data.reconstruction.A_norm, dtype=complex
    )
    diag_joint_time_us = np.asarray(
        diag_joint_data.spectrum.time_us, dtype=float
    )
    diag_joint_result = diag_fixed_rank_block_hankel(
        diag_joint_traces,
        diag_joint_time_us,
        rank=diag_joint_rank,
        pencil_length=diag_joint_pencil_length,
    )

    # Multiplying a trace by a constant does not change its poles.
    # The mean is used instead of the sum only for numerical scale.
    diag_joint_coherent_trace = np.mean(diag_joint_traces, axis=0)
    diag_joint_coherent_trace /= diag_joint_coherent_trace[0]
    diag_joint_sum_result = diag_fixed_rank_block_hankel(
        diag_joint_coherent_trace,
        diag_joint_time_us,
        rank=diag_joint_rank,
        pencil_length=diag_joint_sum_pencil_length,
    )

    diag_joint_theory_MHz = np.sort(np.asarray(
        diag_joint_record["theory_levels_MHz"], dtype=float
    ))
    diag_joint_rowwise_MHz = np.sort(np.asarray(
        diag_joint_record["poles_MHz"], dtype=float
    ))
    diag_joint_fft_resolution_MHz = float(
        diag_joint_data.spectrum.fft_resolution_MHz
    )
    diag_joint_match_tolerance_MHz = (
        diag_joint_match_tolerance_bins
        * diag_joint_fft_resolution_MHz
    )
    diag_joint_matches = {
        "rowwise": diag_match_levels(
            diag_joint_rowwise_MHz,
            diag_joint_theory_MHz,
            diag_joint_match_tolerance_MHz,
        ),
        "joint": diag_match_levels(
            diag_joint_result["frequencies_MHz"],
            diag_joint_theory_MHz,
            diag_joint_match_tolerance_MHz,
        ),
        "coherent mean": diag_match_levels(
            diag_joint_sum_result["frequencies_MHz"],
            diag_joint_theory_MHz,
            diag_joint_match_tolerance_MHz,
        ),
    }

    print(
        f"r={diag_joint_realization}; channels="
        f"{diag_joint_result['channel_count']}; samples="
        f"{diag_joint_result['sample_count']}; dt="
        f"{diag_joint_result['sample_time_us']:.6f} us; "
        f"joint H0 shape={diag_joint_result['unshifted_shape']}"
    )
    for label, result in [
        ("joint block-Hankel", diag_joint_result),
        ("coherent mean", diag_joint_sum_result),
    ]:
        print(
            f"{label}: rank={result['rank']}; "
            f"s35/s36={result['singular_value_gap']:.3g}; "
            f"discarded SVD weight="
            f"{result['discarded_singular_weight']:.3g}; "
            f"fit residual={result['relative_residual']:.3g}; "
            f"pole radius="
            f"[{np.min(result['pole_radii']):.3g}, "
            f"{np.max(result['pole_radii']):.3g}]; "
            f"growing poles="
            f"{np.count_nonzero(result['pole_radii'] > 1.0)}"
        )
    for label, measured_MHz in [
        ("rowwise", diag_joint_rowwise_MHz),
        ("joint", diag_joint_result["frequencies_MHz"]),
        ("coherent mean",
         diag_joint_sum_result["frequencies_MHz"]),
    ]:
        match = diag_joint_matches[label]
        print(
            f"{label}: matched={len(match['pairs'])}/"
            f"{len(diag_joint_theory_MHz)}; MAE="
            f"{1e3 * match['mean_absolute_error_MHz']:.3f} kHz"
        )
        print(
            "  missing theory (kHz):",
            np.round(1e3 * diag_joint_theory_MHz[match["missing"]], 3),
        )
        print(
            "  spurious experiment (kHz):",
            np.round(1e3 * measured_MHz[match["spurious"]], 3),
        )


# ------------------------------------------------------------------------
# Cell 255.
# ------------------------------------------------------------------------

def plot_block_hankel_levels(diag_joint_match_tolerance_MHz, diag_joint_matches, diag_joint_rank, diag_joint_realization, diag_joint_result, diag_joint_rowwise_MHz, diag_joint_spectrum_xlim_kHz, diag_joint_sum_result, diag_joint_theory_MHz):
    """Plot the shared-pole levels against theory (cell 255)."""
    fig, axes = plt.subplots(
        1, 3, figsize=(18.0, 4.8), constrained_layout=True
    )

    for label, result, color in [
        ("joint block-Hankel", diag_joint_result, "tab:blue"),
        ("coherent mean", diag_joint_sum_result, "tab:orange"),
    ]:
        singular_values = result["singular_values"]
        relative_singular_values = (
            singular_values / singular_values[0]
        )
        axes[0].semilogy(
            np.arange(1, len(singular_values) + 1),
            np.maximum(relative_singular_values, np.finfo(float).tiny),
            marker=".", markersize=4, color=color, label=label,
        )
    axes[0].plot(
        [diag_joint_rank, diag_joint_rank],
        [1e-8, 1.0],
        linestyle="--", color="black", linewidth=1.0,
        label=f"fixed rank {diag_joint_rank}",
    )
    axes[0].set(
        xlabel="singular-value index",
        ylabel=r"$s_j/s_1$",
        title="Hankel singular values",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    theory_kHz = 1e3 * diag_joint_theory_MHz
    joint_kHz = 1e3 * diag_joint_result["frequencies_MHz"]
    rowwise_kHz = 1e3 * diag_joint_rowwise_MHz
    sum_kHz = 1e3 * diag_joint_sum_result["frequencies_MHz"]
    joint_match = diag_joint_matches["joint"]
    for measured_index, theory_index in joint_match["pairs"]:
        axes[1].plot(
            [theory_kHz[theory_index], joint_kHz[measured_index]],
            [3.0, 1.0],
            color="tab:blue", alpha=0.16, linewidth=0.8,
        )
    axes[1].scatter(
        theory_kHz[joint_match["matched_theory"]],
        np.full(len(joint_match["matched_theory"]), 3.0),
        marker="|", s=145, color="tab:green",
        label="theory matched to joint",
    )
    axes[1].scatter(
        theory_kHz[joint_match["missing"]],
        np.full(len(joint_match["missing"]), 3.0),
        marker="|", s=165, linewidths=2.0, color="tab:red",
        label="theory missing from joint",
    )
    axes[1].scatter(
        rowwise_kHz, np.full(len(rowwise_kHz), 2.0),
        marker="x", s=24, color="black",
        label=(
            f"rowwise: {len(diag_joint_matches['rowwise']['pairs'])}"
            f"/{len(theory_kHz)} matched"
        ),
    )
    axes[1].scatter(
        joint_kHz[joint_match["matched_measured"]],
        np.full(len(joint_match["matched_measured"]), 1.0),
        marker="x", s=28, color="tab:blue",
        label=(
            f"joint: {len(joint_match['pairs'])}"
            f"/{len(theory_kHz)} matched"
        ),
    )
    axes[1].scatter(
        joint_kHz[joint_match["spurious"]],
        np.full(len(joint_match["spurious"]), 1.0),
        marker="x", s=38, linewidths=1.6, color="tab:red",
    )
    axes[1].scatter(
        sum_kHz, np.full(len(sum_kHz), 0.0),
        marker="+", s=40, color="tab:orange",
        label=(
            f"coherent mean: "
            f"{len(diag_joint_matches['coherent mean']['pairs'])}"
            f"/{len(theory_kHz)} matched"
        ),
    )
    axes[1].set_yticks(
        [0.0, 1.0, 2.0, 3.0],
        ["coherent mean", "joint", "rowwise", "theory"],
    )
    if diag_joint_spectrum_xlim_kHz is None:
        theory_span_kHz = float(np.ptp(theory_kHz))
        theory_margin_kHz = 0.08 * max(theory_span_kHz, 1.0)
        axes[1].set_xlim(
            float(np.min(theory_kHz) - theory_margin_kHz),
            float(np.max(theory_kHz) + theory_margin_kHz),
        )
    else:
        axes[1].set_xlim(*diag_joint_spectrum_xlim_kHz)
    axes[1].set(
        xlabel=r"energy $E/h$ (kHz)",
        title=(
            f"r={diag_joint_realization}; "
            f"tolerance={1e3 * diag_joint_match_tolerance_MHz:.1f} kHz"
        ),
    )
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend(fontsize=7, loc="upper right")

    axes[2].scatter(
        joint_kHz, diag_joint_result["pole_radii"],
        marker="x", s=28, color="tab:blue",
        label="joint",
    )
    axes[2].scatter(
        sum_kHz, diag_joint_sum_result["pole_radii"],
        marker="+", s=40, color="tab:orange",
        label="coherent mean",
    )
    frequency_limit_kHz = 1e3 / (2 * diag_joint_result["sample_time_us"])
    axes[2].plot(
        [-frequency_limit_kHz, frequency_limit_kHz],
        [1.0, 1.0],
        linestyle="--", color="black", linewidth=1.0,
    )
    axes[2].set(
        xlim=(-frequency_limit_kHz, frequency_limit_kHz),
        xlabel=r"frequency $E/h$ (kHz)",
        ylabel=r"pole radius $|z|$",
        title="pole-radius diagnostic",
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    plt.show()


# ------------------------------------------------------------------------
# Cell 278.
# ------------------------------------------------------------------------

def select_row_singular(mpm, traces, target_count=35):
    traces = np.asarray(traces, dtype=complex)
    sample_count = traces.shape[1]
    pencil_length = int(mpm.settings.pencil_length)
    hankel_height = sample_count - pencil_length
    sample_time_us = float(mpm.sampling.sample_time_us)
    rcond = mpm.settings.least_squares_rcond
    component_scores = {}
    row_scores = []
    row_diagnostics = []

    for row_index, trace in enumerate(traces):
        candidates = [c for c in mpm.candidates.per_row if c.row_index == row_index]
        if not candidates:
            continue
        frequencies = np.asarray([c.frequency_MHz for c in candidates])
        decay = np.asarray([c.decay_per_us for c in candidates])
        if mpm.settings.clip_growth:
            decay = np.maximum(decay, 0.)
        poles = np.exp((-decay - 2j * np.pi * frequencies) * sample_time_us)
        design = poles[None, :] ** np.arange(sample_count)[:, None]
        trace_norm = np.linalg.norm(trace)
        normalized_trace = trace / trace_norm if trace_norm > 0 else np.zeros_like(trace)

        # Fit only complex amplitudes at the measured MPM frequencies/decays.
        amplitudes, _, fit_rank, fit_singular = np.linalg.lstsq(design, normalized_trace, rcond=rcond)
        left_norm = np.linalg.norm(poles[None, :] ** np.arange(hankel_height)[:, None], axis=0)
        right_norm = np.linalg.norm(poles[None, :] ** np.arange(pencil_length)[:, None], axis=0)
        component_singular = np.abs(amplitudes) * left_norm * right_norm

        # A single component a*z**n has a rank-one Hankel matrix. Its only
        # singular value is |a|*||z**i||*||z**j||. Original row-SVD singular
        # values do NOT belong one-to-one to poles; this is a component score.
        largest = np.max(component_singular)
        normalized_singular = component_singular / largest if largest > 0 else np.zeros_like(component_singular)
        row_condition = float(fit_singular[0] / fit_singular[-1]) if fit_singular[-1] > 0 else np.inf
        row_diagnostics.append(dict(row_index=row_index, candidate_count=len(candidates),
                                    fit_rank=int(fit_rank), design_condition=row_condition))
        for candidate, amplitude, singular, score in zip(candidates, amplitudes, component_singular, normalized_singular):
            component_scores[id(candidate)] = float(score)
            row_scores.append(dict(row_index=row_index, frequency_kHz=1e3 * candidate.frequency_MHz,
                                   decay_per_us=candidate.decay_per_us, fitted_amplitude=amplitude,
                                   component_singular=float(singular), normalized_singular=float(score)))

    # Reuse the existing calibration-based merge result, including its centers.
    # Change ONLY the selection score. max() gives a pole visible in one row
    # full credit; repeated visibility in many rows is not an extra vote.
    merged_scores = []
    for pool_index, cluster in enumerate(mpm.candidates.merged):
        member_scores = [component_scores[id(member)] for member in cluster.members]
        merged_scores.append(dict(pool_index=pool_index, frequency_kHz=1e3 * cluster.frequency_MHz,
                                  decay_per_us=cluster.decay_per_us, row_score=max(member_scores),
                                  supporting_rows=list(cluster.supporting_rows), member_scores=member_scores,
                                  original_score=float(cluster.selection_score)))
    merged_scores.sort(key=lambda item: (-item['row_score'], item['frequency_kHz'], item['pool_index']))
    count = min(target_count, len(merged_scores), sample_count - 1)
    for rank, item in enumerate(merged_scores, start=1):
        item['score_rank'] = rank
        item['selected'] = rank <= count

    selected = sorted(merged_scores[:count], key=lambda item: item['frequency_kHz'])
    pool_indices = np.asarray([item['pool_index'] for item in selected], dtype=int)
    frequencies = np.asarray([mpm.candidates.merged[i].frequency_MHz for i in pool_indices])
    decay = np.asarray([mpm.candidates.merged[i].decay_per_us for i in pool_indices])
    time_us = np.arange(sample_count) * sample_time_us
    design = np.exp(time_us[:, None] * (-decay[None, :] - 2j * np.pi * frequencies[None, :]))
    amplitudes = np.linalg.lstsq(design, traces.T, rcond=rcond)[0]
    fitted_return = (design @ amplitudes).T
    residual = traces - fitted_return
    signal_norm = np.linalg.norm(traces)
    relative_residual = float(np.linalg.norm(residual) / signal_norm) if signal_norm > 0 else np.nan
    return dict(frequencies_MHz=frequencies, decay_per_us=decay, selected_pool_indices=pool_indices,
                row_scores=row_scores, merged_scores=merged_scores, row_diagnostics=row_diagnostics,
                fitted_return=fitted_return, relative_residual=relative_residual,
                target_count=target_count, selected_count=count)


# ------------------------------------------------------------------------
# Cell 279.
# ------------------------------------------------------------------------

def collect_row_singular_records(spectroscopy_records):
    """Run the row-singular selection over every realization (cell 279)."""
    row_singular_target_count = 35  # Top 35 measured clusters, NOT 35 theory-matched levels.

    # Run the current MPM section first. This cell reuses its measured candidates:
    # no HDF5 reload, no new jobs, no MPM rerun, and no overwrite of spectroscopy_records.
    row_singular_records = {}
    for realization, record in sorted(spectroscopy_records.items()):
        mpm = record.data.matrix_pencil
        traces = np.asarray(record.data.reconstruction.A, dtype=complex)
        selected = select_row_singular(mpm, traces, row_singular_target_count)
        row_singular_records[realization] = selected
        overlap = len(set(selected['selected_pool_indices']) & {
            i for i, cluster in enumerate(mpm.candidates.merged)
            if any(cluster is candidate for candidate in mpm.candidates.selected)
        })
        print(f"r={realization}: {len(selected['merged_scores'])} merged candidates -> "
              f"{selected['selected_count']} selected; retained from old selection={overlap}; "
              f"relative residual old/new={mpm.relative_residual:.3f}/{selected['relative_residual']:.3f}")

    # Optional inspection, without changing the selection:
    # pd.DataFrame(row_singular_records[0]['merged_scores'])  # score rank, frequency, contributing rows
    # pd.DataFrame(row_singular_records[0]['row_scores'])     # individual normalized component strengths
    # pd.DataFrame(row_singular_records[0]['row_diagnostics']) # ill-conditioned fits can inflate strengths


# ------------------------------------------------------------------------
# Cell 280.
# ------------------------------------------------------------------------

def row_singular_gap_ratios(frequencies_MHz, edge_fraction):
    frequencies = np.sort(np.asarray(frequencies_MHz, dtype=float))
    trim = int(np.ceil(edge_fraction * len(frequencies)))
    if trim:
        frequencies = frequencies[trim:-trim]
    gaps = np.diff(frequencies)
    denominator = np.maximum(gaps[:-1], gaps[1:])
    # Preserve exact duplicates (ratio 0); two zero gaps have undefined ratio.
    ratios = np.full(len(denominator), np.nan)
    np.divide(np.minimum(gaps[:-1], gaps[1:]), denominator, out=ratios, where=denominator > 0)
    return ratios[np.isfinite(ratios)]


def report_row_singular_statistics(match_levels, match_tolerance_bins, partial_realizations, row_singular_records, spectroscopy_records):
    """Level statistics from the row-singular selection (cell 280)."""
    # Theory is read ONLY here, after experimental selection has finished.
    row_singular_matches = {}
    fig, axes = plt.subplots(len(row_singular_records), 2, squeeze=False,
                             figsize=(15, 3.2 * len(row_singular_records)), constrained_layout=True)
    for row, (realization, selected) in enumerate(sorted(row_singular_records.items())):
        record = spectroscopy_records[realization]
        mpm = record.data.matrix_pencil
        tolerance_MHz = match_tolerance_bins * record.data.spectrum.fft_resolution_MHz
        measured_sets = [mpm.selected_frequencies_MHz, selected['frequencies_MHz']]
        for column, (label, measured) in enumerate(zip(['Current MPM score', 'Row-normalized component SV'], measured_sets)):
            match = match_levels(measured, record.theory_levels_MHz,
                                          mpm.sampling.sampling_frequency_MHz, tolerance_MHz)
            row_singular_matches[realization, column] = match
            axis = axes[row, column]
            theory_kHz, measured_kHz = 1e3 * match.theory_MHz, 1e3 * match.measured_MHz
            for measured_index, theory_index in match.pairs:
                axis.plot([theory_kHz[theory_index], measured_kHz[measured_index]], [1, 0],
                          color='tab:blue', alpha=0.25, linewidth=0.8)
            axis.scatter(theory_kHz, np.ones(len(theory_kHz)), marker='|', color='tab:green', s=130, label='configured H')
            axis.scatter(measured_kHz, np.zeros(len(measured_kHz)), marker='x', color='black', s=28, label='selected measured pole')
            axis.scatter(theory_kHz[match.missing], np.ones(len(match.missing)), marker='|', color='tab:red', s=140)
            axis.scatter(measured_kHz[match.spurious], np.zeros(len(match.spurious)), marker='x', color='tab:red', s=30, label='outside comparison tolerance')
            axis.set_yticks([0, 1], ['selected', 'theory'])
            axis.set(xlabel='principal-zone energy E/h (kHz)', ylim=(-0.1, 1.1),
                     title=f'r={realization}: {label}\n{len(match.pairs)}/{len(theory_kHz)} within {1e3 * tolerance_MHz:.2f} kHz; selected K={len(measured_kHz)}')
            axis.grid(axis='x', alpha=0.15)
        # Identical horizontal scale for old/new, including unmatched outliers.
        limits = [axis.get_xlim() for axis in axes[row]]
        for axis in axes[row]:
            axis.set_xlim(min(limit[0] for limit in limits), max(limit[1] for limit in limits))
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Same MPM candidates and merge groups; only the ranking score changes\nTheory and comparison tolerance do not choose the poles')
    plt.show()




    # Naive statistics of ALL selected measured poles, not just theory-matched ones.
    # Trim each measured/theory list independently; no theory-defined energy window.
    row_singular_edge_fraction = 0.10
    row_singular_statistics = {}
    for realization, selected in sorted(row_singular_records.items()):
        if partial_realizations.get(realization, {}).get('incomplete', False):
            print(f'r={realization}: incomplete acquisition shown above, omitted from pooled statistics')
            continue
        old = row_singular_matches[realization, 0]
        new = row_singular_matches[realization, 1]
        row_singular_statistics[realization] = {
            name: row_singular_gap_ratios(values, row_singular_edge_fraction)
            for name, values in [('Current MPM', old.measured_MHz), ('Row component SV', new.measured_MHz),
                                 ('Configured H', new.theory_MHz)]
        }

    if row_singular_statistics:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
        realization_ids = sorted(row_singular_statistics)
        for name, color, marker, offset in [('Current MPM', 'black', 'x', -0.12),
                                           ('Row component SV', 'tab:blue', 'o', 0),
                                           ('Configured H', 'tab:green', 's', 0.12)]:
            per_realization = [row_singular_statistics[r][name] for r in realization_ids]
            pooled = np.concatenate(per_realization)
            means = [np.mean(values) if len(values) else np.nan for values in per_realization]
            if len(pooled):
                axes[0].hist(pooled, bins=np.linspace(0, 1, 11), density=True, histtype='step', color=color,
                             linewidth=1.8, label=f'{name}: n={len(pooled)}, mean={np.mean(pooled):.3f}')
            axes[1].scatter(np.arange(len(realization_ids)) + offset, means, color=color, marker=marker, label=name)
        ratio_axis = np.linspace(0, 1, 500)
        axes[0].plot(ratio_axis, 2 / (1 + ratio_axis)**2, color='tab:purple', label='Poisson')
        axes[0].plot(ratio_axis, 27 / 4 * (ratio_axis + ratio_axis**2) / (1 + ratio_axis + ratio_axis**2)**2.5,
                     color='tab:orange', label='GOE surmise')
        axes[1].axhline(2 * np.log(2) - 1, color='tab:purple', linestyle='--', label='Poisson mean')
        axes[1].axhline(4 - 2 * np.sqrt(3), color='tab:orange', linestyle='--', label='GOE surmise mean')
        axes[0].set(xlim=(0, 1), xlabel='adjacent-gap ratio', ylabel='probability density', title='Naive pooled bulk statistics')
        axes[1].set(ylim=(0, 1), ylabel='mean adjacent-gap ratio', title='By realization')
        axes[1].set_xticks(np.arange(len(realization_ids)), [f'r={r}' for r in realization_ids])
        for axis in axes:
            axis.legend(fontsize=8)
        fig.suptitle('10% trimmed from each end of each spectrum; ranking comparison, not a validated GOE/Poisson claim')
        plt.show()


# ------------------------------------------------------------------------
# Cell 282.
# ------------------------------------------------------------------------

def assess_statistical_power(spectroscopy_records):
    """Ask whether the measured statistics can support a conclusion (cell 282).

    This is the workspace's own honesty check, and the reason the
    other selections are kept side by side rather than merged."""
    # Reuse the measured traces already analyzed above; do not load any jobs again.
    # Only the cross-row calibration merge multiplier changes. No theory levels,
    # GOE/Poisson target, or requirement to recover exactly 35 poles enters this test.
    import inspect

    stats_merge_sigmas = [0.5, 1.0, 2.0]
    stats_bootstrap_repeats = 10000
    stats_edge_fraction = 0.10
    stats_rng = np.random.default_rng(20260907)

    # Incomplete acquisition is different from incomplete spectral recovery.
    # Exclude interrupted acquisition groups, but retain complete measurements
    # even when MPM returns fewer than 35 levels. The existing r=3 preview is kept.
    stats_partial = globals().get('partial_realizations', {})
    stats_realizations = [
        realization for realization in sorted(spectroscopy_records)
        if not stats_partial.get(realization, {}).get('incomplete', False)
    ]
    stats_skipped = sorted(set(spectroscopy_records) - set(stats_realizations))
    print('Complete acquisition groups:', stats_realizations)
    print('Partial acquisition groups excluded from pooled statistics:', stats_skipped)

    stats_results = {}
    stats_parameter_names = inspect.signature(SavedSpectroscopyExperiment.analyze_matrix_pencil).parameters
    for realization in stats_realizations:
        record = spectroscopy_records[realization]
        baseline = record.data.matrix_pencil
        # Preserve all of the actual baseline's numerical settings, not today's
        # possibly edited settings cell; keep only arguments accepted by this method.
        options = {name: value for name, value in baseline.settings.items()
                   if name in stats_parameter_names}
        if options.get('row_frequency_standard_errors_MHz') is None:
            print(f'r={realization}: no calibration-SE merge in this baseline; skipped.')
            continue
        for sigma in stats_merge_sigmas:
            options['merge_frequency_tolerance_sigma'] = sigma
            if np.isclose(sigma, baseline.settings.merge_frequency_tolerance_sigma):
                mpm = baseline
            else:
                mpm = SavedSpectroscopyExperiment.analyze_matrix_pencil(
                    record.data.reconstruction, record.data.spectrum, **options)
            levels = np.sort(np.asarray(mpm.selected_frequencies_MHz, float))
            cut = int(np.ceil(stats_edge_fraction * len(levels)))
            bulk = levels[cut:-cut] if cut else levels
            gaps = np.diff(bulk)
            if len(gaps) < 2:
                print(f'r={realization}, sigma={sigma:g}: too few bulk levels for a gap ratio.')
                continue
            ratios = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
            stats_results[realization, sigma] = dict(
                levels_MHz=levels, ratios=ratios, mean_ratio=float(np.mean(ratios)),
                count=len(levels), relative_residual=float(mpm.relative_residual))
            print(f'r={realization}, merge sigma={sigma:g}: K={len(levels)}, '
                  f'mean gap ratio={np.mean(ratios):.4f}, residual={mpm.relative_residual:.4f}')

    # Use the same realizations at every setting. Resample entire realizations,
    # not individual adjacent ratios, which share levels and are not independent.
    stats_common = [r for r in stats_realizations
                       if all((r, sigma) in stats_results for sigma in stats_merge_sigmas)]
    stats_summary = {}
    if stats_common:
        stats_values = np.asarray([
            [stats_results[r, sigma]['mean_ratio'] for r in stats_common]
            for sigma in stats_merge_sigmas
        ])
        indices = stats_rng.integers(
            len(stats_common), size=(stats_bootstrap_repeats, len(stats_common)))
        averages = stats_values.mean(axis=1)
        intervals = np.quantile(stats_values[:, indices].mean(axis=2), [0.025, 0.975], axis=1)
        for index, sigma in enumerate(stats_merge_sigmas):
            stats_summary[sigma] = dict(
                mean_ratio=float(averages[index]), realization_count=len(stats_common),
                resampling_interval95=intervals[:, index].copy())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for index, realization in enumerate(stats_common):
            axes[0].plot(stats_merge_sigmas, stats_values[:, index], 'o-', label=f'r={realization}')
        axes[0].set(xlabel='Calibration merge sigma multiplier', ylabel='Mean adjacent-gap ratio',
                    title='Same measured traces; only merging changes')
        axes[0].legend(fontsize=8, ncols=2)
        axes[1].errorbar(stats_merge_sigmas, averages,
                         yerr=np.maximum(np.stack([averages - intervals[0], intervals[1] - averages]), 0),
                         fmt='o-', capsize=4)
        axes[1].set(xlabel='Calibration merge sigma multiplier', ylabel='Equal-realization mean ratio',
                    title=f'{len(stats_common)} realizations; 95% resampling range')
        fig.suptitle('Extraction sensitivity, not a GOE/Poisson classification')
        plt.show()
        print('Intervals cover realization sampling only, not missed/duplicated-pole bias.')
        if len(stats_common) < 5:
            print('Fewer than five realizations: these resampling intervals are unstable; '
                  'do not interpret them as a calibrated hypothesis test.')
    else:
        print('No complete calibrated acquisition groups are available for this comparison.')


# ------------------------------------------------------------------------
# Cell 284.
# ------------------------------------------------------------------------

def joint_infer_saved_frame_cycle_us(saved_correction_deg, final_occupations,
                               calibration_occupations, calibration_phase_deg,
                               saved_kerr_MHz, cycle_branches=0,
                               used_cycle_us_override=None):
    """Recover the acquisition-time value from immutable saved analyzer phases.

    gamma_saved = phase_cal + 180*branch + 360*K*choose(n_M1,2)*T_used.
    Supply the branches actually used in acquisition; zero is appropriate for
    these Sep05 campaigns. Do not replace them by a best-matching modulo branch.
    K is the signed Kerr used to build that analyzer correction, normally the
    saved d72_self_kerr_kHz/1000, not the current live station's Kerr.
    When no decoder has n_M1>=2, an independently established override is needed.
    """
    final = np.asarray(final_occupations)
    calibration_index = {tuple(state): index for index, state in enumerate(calibration_occupations)}
    phase = np.asarray([calibration_phase_deg[calibration_index[tuple(state)]] for state in final])
    if isinstance(cycle_branches, dict):
        branches = np.asarray([cycle_branches.get(tuple(state), 0) for state in final])
    else:
        branches = np.broadcast_to(cycle_branches, len(final))
    n = final[:, 0]
    coefficient = 360 * saved_kerr_MHz * n * (n - 1) / 2
    available = np.abs(coefficient) > 1e-14
    candidates = (np.asarray(saved_correction_deg)[available] - phase[available]
                  - 180 * branches[available]) / coefficient[available]
    if used_cycle_us_override is not None:
        result = float(used_cycle_us_override)
    elif len(candidates):
        result = float(np.median(candidates))
    else:
        raise ValueError("Saved phase data cannot determine acquisition T: give used_cycle_us_override from another verified job with the same timing")
    if not np.isfinite(result) or result <= 0 or not np.allclose(candidates, result, rtol=1e-6, atol=1e-8):
        raise ValueError("Saved correction, calibration, Kerr and branches do not give one consistent acquisition-time value")
    return result


def joint_decoder_clock_rotation(cycles, final_occupations, detunings_MHz,
                           used_cycle_us, compiled_cycle_us, reference_kerr_MHz=0.):
    """Return a known frame rotation, not a fitted spectral correction.

    The program applies disorder through physical DDS detuning, but advances
    decoding phases by 360*detuning*used_cycle_us per cycle. For onsite=-detuning,
    the excess decoder rotation is removed with the negative sign below.
    DDS rounding (<1 Hz here) is left at its configured value; no H is consulted.
    """
    final = np.asarray(final_occupations)
    reference_kerr = reference_kerr_MHz * final[:, 0] * (final[:, 0] - 1) / 2
    # Correct the saved gamma reference with the SAME saved Kerr; no spectral fit.
    phase_turns_per_cycle = (used_cycle_us - compiled_cycle_us) * (final[:, 1:] @ detunings_MHz - reference_kerr)
    return np.exp(-2j * np.pi * phase_turns_per_cycle[:, None] * np.asarray(cycles)[None, :])


def joint_rotate_covariance(covariance, rotation):
    real_rotation = np.stack((np.stack((rotation.real, -rotation.imag), axis=-1),
                              np.stack((rotation.imag, rotation.real), axis=-1)), axis=-2)
    return np.einsum("...ik,...kl,...jl->...ij", real_rotation, covariance, real_rotation)


def joint_shot_halves(child, rng):
    """Two disjoint shot sets; retain the saved readout offset in both means."""
    if child.cfg.expt.get('active_reset', False) and child.cfg.expt.get('pre_selection_reset', False):
        raise ValueError('This shot split needs unconditioned readout; pre-selection must be split with its selection mask')
    read_num = int(child.cfg.get('read_num', 0))
    if read_num < 1:
        read_num = 1 + int(child.cfg.expt.get('parity_check', False))
        if child.cfg.expt.get('active_reset', False):
            read_num += MMAveragerProgram.active_reset_read_num(**MMAveragerProgram.get_active_reset_params(child.cfg))
        read_num += int(child.cfg.expt.get('multiparity_readout', False))
    saved = np.asarray(child.data.avgi)
    points = [np.asarray(point, float).ravel()[read_num - 1::read_num] for point in child.data.idata]
    if len(points) != saved.size:
        raise ValueError('Raw shots and saved sweep points differ; do not guess their ordering')
    values = np.empty((2, saved.size))
    variance = np.empty(saved.size)
    for index, point in enumerate(points):
        # Equal halves make their average exactly the saved full mean.
        if len(point) < 4 or len(point) % 2:
            raise ValueError('Independent equal halves require an even number of shots >= 4')
        halves = np.array_split(rng.permutation(len(point)), 2)
        for half, indices in enumerate(halves):
            values[half, index] = saved.ravel()[index] + point[indices].mean() - point.mean()
        variance[index] = point.var(ddof=1) / len(point)
    children = []
    for value in values:
        duplicate = joint_shallow_copy(child)
        duplicate.data = AttrDict({key: value for key, value in child.data.items()
                                   if key not in ('idata', 'qdata')})
        duplicate.data.avgi = value.reshape(saved.shape)
        duplicate.data.pop('return_quadrature', None)
        duplicate.data.pop('Pe', None)
        children.append(duplicate)
    q = child.cfg.expt.qubits[0]
    contrast = float(child.cfg.device.readout.Ie[q] - child.cfg.device.readout.Ig[q])
    quadrature_var = variance.reshape(-1, 2).sum(axis=1).reshape(-1, 2) / contrast**2
    cycles = np.asarray(child.cfg.expt.offdiag_cycles)
    rotation = np.exp(-1j * np.deg2rad(child.cfg.expt.offdiag_decoder_phase_correction_deg * cycles))
    covariance = np.zeros((len(cycles), 2, 2))
    covariance[:, 0, 0] = quadrature_var[:, 0]
    covariance[:, 1, 1] = quadrature_var[:, 1]
    return children, joint_rotate_covariance(covariance, rotation)


# ------------------------------------------------------------------------
# Cell 286.
# ------------------------------------------------------------------------

def build_corrected_traces_and_halves(load_shots, loaded_dataset_name, loaded_spectroscopy, parameter_key, phase_calibration, read_shots, saved_floquet_timing, saved_parameters):
    """Build clock-corrected traces and disjoint shot halves (cell 286)."""
    # INPUTS ONLY: no frequency fitting and no Hamiltonian levels are read here.
    # `None` infers acquisition-time decoder clock from saved analyzer/calibration
    # phases, then shares it only between identical saved pulse configurations.
    # If a campaign has no decoder with >=2 M1 photons, set the known acquisition
    # clock explicitly; do not substitute the current pulse-generator method.
    joint_decoder_cycle_us = None
    joint_cycle_branches = 0
    joint_random_seed = 20260905
    first_cfg = next(iter(loaded_spectroscopy.values())).batch_expts[0].cfg.expt
    photon_number = sum(first_cfg.spectroscopy_occupations)
    joint_count = math.comb(photon_number + len(first_cfg.swap_stors), photon_number)  # 35 for N=3, four leaves.
    joint_compare_count = max(1, joint_count - 5)  # Smaller candidate model; not theory-based pole selection.
    joint_maxiter = 700
    joint_swap_attempts = 1          # Bounded replacement search, not exhaustive subset enumeration.
    joint_amplitude_bound = 1.0     # Assumes normalized unitary overlaps: sum_k |c[row,k]| <= 1.
                                   # None removes this constraint; no theory frequencies/weights are used.

    joint_inputs = {}
    joint_frame_clocks = {}
    cal = phase_calibration.data
    cal_index = {tuple(state): index for index, state in enumerate(cal.occupations)}

    # First pass can recover the decoder clock from a different realization with
    # the same pulses even if the requested realization has only n_M1=0/1 decoders.
    for realization, loaded in sorted(loaded_spectroscopy.items()):
        children = loaded.batch_expts
        acquired = SavedSpectroscopyExperiment.reconstruct_pair_spectroscopy(children)
        final = np.asarray(acquired.final_occupations)
        saved = SavedSpectroscopyExperiment._saved_correction(children)
        if saved.application_sign != -1 or saved.modes != {'final_analyzer'}:
            raise ValueError('This subsection requires the saved -1 final-analyzer convention, not a guessed legacy sign')
        gamma = np.asarray([saved.phase_by_occupation[tuple(state)] for state in final])
        kerr = float(children[0].cfg.expt.d72_self_kerr_kHz) / 1000
        fingerprint = parameter_key(saved_parameters(children[0]))
        clock_hint = joint_decoder_cycle_us
        if clock_hint is None and children[0].saved_hardware is not None:
            clock_hint = children[0].saved_hardware.get('decoder_cycle_us')
        if clock_hint is not None or (abs(kerr) > 1e-14 and np.any(final[:, 0] >= 2)):
            used = joint_infer_saved_frame_cycle_us(gamma, final, cal.occupations, cal.phase_mod180,
                                                   kerr, joint_cycle_branches, clock_hint)
            if fingerprint in joint_frame_clocks and not np.isclose(used, joint_frame_clocks[fingerprint]):
                raise ValueError('Identical pulse settings have different saved decoder clocks; split these acquisitions')
            joint_frame_clocks[fingerprint] = used

    rng = np.random.default_rng(joint_random_seed)
    for realization, loaded in sorted(loaded_spectroscopy.items()):
        children = loaded.batch_expts
        acquired = SavedSpectroscopyExperiment.reconstruct_pair_spectroscopy(children)
        cycles = np.asarray(acquired.cycles)
        final = np.asarray(acquired.final_occupations)
        timing = saved_floquet_timing(children[0])
        if timing is None:
            raise ValueError('No H5 or explicitly archived cycle clock is available; do not guess its time axis')
        actual = timing['cycle_us']
        fingerprint = parameter_key(saved_parameters(children[0]))
        if fingerprint not in joint_frame_clocks:
            raise ValueError('Set joint_decoder_cycle_us to the clock used during acquisition; these decoder states cannot determine it')
        used = joint_frame_clocks[fingerprint]
        half_children = [[], []]
        chunks = {}
        for child in children:
            other_timing = saved_floquet_timing(child)
            if other_timing is None or not np.isclose(other_timing['cycle_us'], actual):
                raise ValueError('Spectroscopy rows have different saved cycle clocks')
            if parameter_key(saved_parameters(child)) != fingerprint:
                raise ValueError('Spectroscopy rows have different saved pulse settings')
            read_shots(child)  # Direct H5 access, only when this validation is requested.
            split, covariance = joint_shot_halves(child, rng)
            if not load_shots:
                child.data.pop('idata', None)
                child.data.pop('qdata', None)
            for half in (0, 1):
                half_children[half].append(split[half])
            ecfg = child.cfg.expt
            pair = (tuple(ecfg.offdiag_decoder_occupation), tuple(ecfg.spectroscopy_occupations))
            chunks.setdefault(pair, []).append((np.asarray(ecfg.offdiag_cycles), covariance))
        covariance_rows = []
        for decoder, encoder in zip(acquired.final_occupations, acquired.occupations):
            parts = chunks[(tuple(decoder), tuple(encoder))]
            chunk_cycles = np.concatenate([part[0] for part in parts])
            order = np.argsort(chunk_cycles)
            if not np.array_equal(chunk_cycles[order], cycles):
                raise ValueError('Shot covariance and reconstructed time points do not align')
            covariance_rows.append(np.concatenate([part[1] for part in parts])[order])
        detunings = np.asarray(SavedSpectroscopyExperiment._saved_parameters(children).detunings)
        reference_kerr = float(children[0].cfg.expt.d72_self_kerr_kHz) / 1000
        rotation = joint_decoder_clock_rotation(cycles, final, detunings, used, actual, reference_kerr)
        half0 = SavedSpectroscopyExperiment.reconstruct_pair_spectroscopy(half_children[0]).A * rotation
        half1 = SavedSpectroscopyExperiment.reconstruct_pair_spectroscopy(half_children[1]).A * rotation
        covariance = joint_rotate_covariance(np.asarray(covariance_rows), rotation)
        indices = [cal_index[tuple(state)] for state in final]
        _, groups = np.unique(final, axis=0, return_inverse=True)
        joint_inputs[realization] = dict(
            A=acquired.A * rotation, half0=half0, half1=half1, cov=covariance,
            time_us=cycles * actual, cycles=cycles, occupations=np.asarray(acquired.occupations),
            final_occupations=final, decoder_groups=groups,
            row_calibration_se_MHz=np.asarray(cal.phase_error)[indices] / (360 * actual),
            decoder_cycle_us=used, actual_cycle_us=actual,
            job_ids=tuple(loaded.batch_job_ids), dataset=loaded_dataset_name,
        )
        difference = (half0 - half1) / 2
        noise_power = np.trace(covariance, axis1=-2, axis2=-1)
        noise_nmse = noise_power.sum() / np.sum(abs(acquired.A)**2)
        print(f'r={realization}: {len(cycles)} points, {cycles[-1]*actual:.3f} us; '
              f'decoder clock {used:.9f} -> physical {actual:.9f} us; expected shot-noise NMSE={noise_nmse:.4f}')
        print(f'  disjoint-half average error={np.max(abs((half0+half1)/2-joint_inputs[realization]["A"])):.2g}')


# ------------------------------------------------------------------------
# Cell 288.
# ------------------------------------------------------------------------

def test_kHz_scale_with_shot_halves(joint_inputs, match_levels, partial_realizations, spectroscopy_records):
    """Optional: test the 1 kHz scale with independent measured shot halves (cell 288)."""
    # Optional, inexpensive check after "Build corrected traces and disjoint shot halves".
    # Run the existing MPM separately on each half; no nonlinear fitter or theory input.
    import inspect

    half_agreement_kHz = 1.0  # Comparison scale, NOT a fit/merge threshold.
    half_results = {}
    parameter_names = inspect.signature(SavedSpectroscopyExperiment.analyze_matrix_pencil).parameters
    for realization, inputs in sorted(joint_inputs.items()):
        if partial_realizations.get(realization, {}).get('incomplete', False):
            continue
        record = spectroscopy_records[realization]
        if tuple(inputs['job_ids']) != tuple(record.job_ids):
            raise ValueError('The half-shot inputs and current MPM use different jobs; rerun the input cell.')
        baseline = record.data.matrix_pencil
        options = {name: value for name, value in baseline.settings.items() if name in parameter_names}
        half_levels, half_means = [], []
        for half in ('half0', 'half1'):
            reconstruction = AttrDict(dict(record.data.reconstruction))
            reconstruction.A = inputs[half]
            result = SavedSpectroscopyExperiment.analyze_matrix_pencil(reconstruction, record.data.spectrum, **options)
            levels = np.sort(np.asarray(result.selected_frequencies_MHz))
            cut = int(np.ceil(0.1 * len(levels)))
            bulk = levels[cut:-cut] if cut else levels
            gaps = np.diff(bulk)
            ratios = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
            half_levels.append(levels)
            half_means.append(float(np.mean(ratios)) if len(ratios) else np.nan)
        # Reuse the ordered one-to-one comparison algorithm; its second array here
        # is another MEASURED shot half, not the configured Hamiltonian spectrum.
        comparison = match_levels(
            half_levels[0], half_levels[1], baseline.sampling.sampling_frequency_MHz,
            half_agreement_kHz / 1000)
        half_results[realization] = dict(
            levels_MHz=half_levels, mean_ratios=half_means, comparison=comparison)
        print(f'r={realization}: {len(comparison.pairs)} one-to-one pairs within '
              f'{half_agreement_kHz:g} kHz; half counts={len(half_levels[0])}/{len(half_levels[1])}; '
              f'bulk mean ratios={half_means[0]:.4f}/{half_means[1]:.4f}')

    if half_results:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for realization, result in half_results.items():
            comparison = result['comparison']
            axes[0].scatter(1000 * comparison.measured_MHz[comparison.matched_measured],
                            1000 * comparison.theory_MHz[comparison.matched_theory],
                            s=18, label=f'r={realization}')
            axes[1].plot([0, 1], result['mean_ratios'], 'o-', label=f'r={realization}')
        lo, hi = axes[0].get_xlim()
        axes[0].plot([lo, hi], [lo, hi], 'k--', lw=1)
        axes[0].set(xlabel='First shot half (kHz)', ylabel='Second shot half (kHz)',
                    title=f'One-to-one agreement within {half_agreement_kHz:g} kHz')
        axes[0].legend(fontsize=8)
        axes[1].set(xticks=[0, 1], xticklabels=['First half', 'Second half'],
                    ylabel='Mean adjacent-gap ratio', title='All selected poles in each half; 10% edge trim')
        axes[1].legend(fontsize=8)
        fig.suptitle('Measured-shot reproducibility, not absolute accuracy')
        plt.show()
    print('Both halves share calibration and systematic errors. Agreement does not establish '
          'absolute 1 kHz accuracy, completeness, or unbiased GOE/Poisson classification.')


# ------------------------------------------------------------------------
# Cell 290.
# ------------------------------------------------------------------------

def joint_pencil(traces, dt, rank, length):
    windows = np.lib.stride_tricks.sliding_window_view(traces, length + 1, axis=1)
    h0 = windows[:, :, :-1].reshape(-1, length)
    h1 = windows[:, :, 1:].reshape(-1, length)
    u, s, vh = linalg.svd(h0, full_matrices=False, check_finite=False)
    k = min(rank, len(s))
    z = (u[:, :k].conj().T @ h1 @ vh[:k].conj().T) / s[:k, None]
    poles = linalg.eigvals(z, check_finite=False)
    f = -np.angle(poles) / (2 * np.pi * dt)
    gamma = -np.log(np.abs(poles)) / dt
    return f, np.clip(gamma, 0.0, 0.2), s


def joint_project_l1_columns(values, bound):
    magnitude = np.abs(values)
    sorted_magnitude = np.sort(magnitude, axis=0)[::-1]
    cumulative = np.cumsum(sorted_magnitude, axis=0) - bound
    thresholds = cumulative / np.arange(1, len(values) + 1)[:, None]
    active_count = np.sum(sorted_magnitude > thresholds, axis=0)
    theta = thresholds[np.maximum(active_count - 1, 0), np.arange(values.shape[1])]
    theta = np.maximum(theta, 0)
    return values * np.maximum(1 - theta / np.maximum(magnitude, 1e-300), 0)


def joint_amplitudes_for_design(design, traces, bound, initial=None):
    if bound is None:
        return linalg.lstsq(design, traces, cond=1e-9, check_finite=False)[0], 0
    gram = design.conj().T @ design
    rhs = design.conj().T @ traces
    lipschitz = linalg.eigvalsh(gram, check_finite=False, subset_by_index=[len(gram)-1, len(gram)-1])[0]
    values = np.zeros_like(rhs) if initial is None else initial.copy()
    momentum = values.copy()
    alpha = 1.0
    previous_cost = np.inf
    for iteration in range(1500):
        update = joint_project_l1_columns(momentum - (gram @ momentum - rhs) / lipschitz, bound)
        new_alpha = (1 + np.sqrt(1 + 4 * alpha * alpha)) / 2
        momentum = update + (alpha - 1) / new_alpha * (update - values)
        if iteration % 25 == 0:
            cost = float(np.real(np.vdot(update, gram @ update) - 2 * np.vdot(update, rhs)))
            if abs(previous_cost - cost) < 1e-9 * max(1, abs(cost)):
                values = update
                break
            previous_cost = cost
        values = update
        alpha = new_alpha
    return values, iteration


# ------------------------------------------------------------------------
# Cell 291.
# ------------------------------------------------------------------------

def joint_select_pool(traces, times, frequencies, decay, count, row_variance=None):
    """Orthogonal least-squares selection: maximize the joint residual decrease.

    Every supplied rowwise candidate is eligible, including candidates rejected
    by the previous score threshold. There is no frequency-merging tolerance.
    """
    traces = np.asarray(traces, complex)
    times = np.asarray(times, float)
    frequencies = np.asarray(frequencies, float)
    decay = np.maximum(np.asarray(decay, float), 0.)
    variance = np.ones(len(traces)) if row_variance is None else np.asarray(row_variance, float)
    residual = traces.T / np.sqrt(variance)[None, :]
    design = np.exp(times[:, None] * (-2j * np.pi * frequencies[None, :] - decay[None, :]))
    norm = np.linalg.norm(design, axis=0)
    dictionary = design / norm
    projected_norm2 = np.ones(len(frequencies))
    orthogonal = []
    chosen = []
    reductions = []
    for _ in range(count):
        score = np.sum(abs(dictionary.conj().T @ residual) ** 2, axis=1)
        score /= np.maximum(projected_norm2, 1e-14)
        score[projected_norm2 < 1e-10] = -np.inf
        score[chosen] = -np.inf
        index = int(np.argmax(score))
        if not np.isfinite(score[index]):
            raise ValueError("the supplied candidate pool does not span the requested model size")
        chosen.append(index)
        reductions.append(float(score[index]))
        q = dictionary[:, index].copy()
        if orthogonal:
            qmat = np.asarray(orthogonal).T
            # A second pass prevents accumulated loss of orthogonality.
            q -= qmat @ (qmat.conj().T @ q)
            q -= qmat @ (qmat.conj().T @ q)
        q /= np.linalg.norm(q)
        orthogonal.append(q)
        residual -= q[:, None] * (q.conj() @ residual)[None, :]
        projected_norm2 -= abs(q.conj() @ dictionary) ** 2
    return dict(indices=np.asarray(chosen), frequencies_MHz=frequencies[chosen],
                decay_per_us=decay[chosen], residual_decreases=np.asarray(reductions))


def joint_fit_shared(traces, times, initial_f, initial_decay, *, row_variance=None,
               decoder_groups=None, calibration_se_MHz=None, ridge=0.,
               max_decay_per_us=.2, maxiter=600, initial_offset=None,
               fixed_pair=None, pair_gap_MHz=None, amplitude_bound=None):
    """Variable projection: free frequencies/decays; exact complex amplitude fit.

    Calibration offsets are common to rows using the same decoder. Their
    standard errors come from the measured phase calibration, not theory.
    Ridge is optional numerical regularization, and must be validated on data.
    amplitude_bound optionally constrains each raw row's coefficient absolute
    sum. A bound of one assumes normalized unitary-access overlaps; it is not
    valid for every SPAM/readout model and is never silently enabled.
    No energy, amplitude fingerprint, level spacing or GOE/Poisson target enters.
    """
    traces, times = np.asarray(traces, complex), np.asarray(times, float)
    if amplitude_bound is not None and ridge:
        raise ValueError("use either an amplitude bound or ridge, not both")
    row_count, sample_count = traces.shape
    span = float(times[-1])
    dt = float(np.min(np.diff(times)))
    u = times / span
    count = len(initial_f)
    variance = np.ones(row_count) if row_variance is None else np.asarray(row_variance, float)
    groups = np.arange(row_count) if decoder_groups is None else np.asarray(decoder_groups, int)
    _, groups = np.unique(groups, return_inverse=True)
    group_count = int(groups.max()) + 1
    row_se = np.zeros(row_count) if calibration_se_MHz is None else np.asarray(calibration_se_MHz, float)
    group_se = np.array([np.max(row_se[groups == group]) for group in range(group_count)])
    # A zero-SE offset is exactly fixed; avoid optimizing a redundant coordinate.
    active_groups = np.flatnonzero(group_se > 0.)
    initial_z = np.zeros(group_count) if initial_offset is None else np.asarray(initial_offset, float)
    start = np.r_[2 * np.pi * span * np.asarray(initial_f),
                  span * np.clip(initial_decay, 0., max_decay_per_us), initial_z[active_groups]]
    nyquist_phase = np.pi * span / dt
    bounds = [(-nyquist_phase, nyquist_phase)] * count + [(0., max_decay_per_us * span)] * count
    bounds += [(None, None)] * len(active_groups)
    if fixed_pair is not None:
        first, second = fixed_pair
        gap_phase = 2 * np.pi * span * pair_gap_MHz
        start[first] = .5 * (start[first] + start[second])
        start = np.delete(start, second)
        bounds[first] = (-nyquist_phase + .5 * gap_phase, nyquist_phase - .5 * gap_phase)
        bounds.pop(second)
    scale = max(float(np.sum(abs(traces) ** 2 / variance[:, None])), 1.)
    previous_amplitudes = None

    def objective(parameters, extra=False):
        nonlocal previous_amplitudes
        if fixed_pair is not None:
            parameters = np.insert(parameters, second, 0.)
            center = parameters[first]
            parameters[first], parameters[second] = center - .5 * gap_phase, center + .5 * gap_phase
        theta, decay = parameters[:count], parameters[count:2 * count]
        z = np.zeros(group_count)
        z[active_groups] = parameters[2 * count:]
        offsets = group_se * z
        rotation = np.exp(-2j * np.pi * times[:, None] * offsets[groups][None, :])
        design = np.exp(u[:, None] * (-1j * theta[None, :] - decay[None, :]))
        target = traces.T * rotation.conj()
        inner_iterations = 0
        if amplitude_bound is not None:
            amplitude, inner_iterations = joint_amplitudes_for_design(
                design, target, amplitude_bound, previous_amplitudes)
            previous_amplitudes = amplitude
        elif ridge:
            augmented = np.vstack([design, np.sqrt(ridge) * np.eye(count)])
            augmented_target = np.vstack([target, np.zeros((count, row_count), complex)])
            amplitude = linalg.lstsq(augmented, augmented_target, cond=1e-12, check_finite=False)[0]
        else:
            amplitude = linalg.lstsq(design, target, cond=1e-12, check_finite=False)[0]
        joint_prediction = (design @ amplitude) * rotation
        residual = traces.T - joint_prediction
        data_cost = float(np.sum(abs(residual) ** 2 / variance[None, :]))
        ridge_cost = float(ridge * np.sum(abs(amplitude) ** 2 / variance[None, :]))
        prior_cost = .5 * float(z @ z)
        cross = (residual.conj() * rotation / variance[None, :]) @ amplitude.T
        gf = -2 * np.real(np.sum(cross * (-1j * u[:, None] * design), axis=0))
        gd = -2 * np.real(np.sum(cross * (-u[:, None] * design), axis=0))
        row_gradient = -2 * np.real(np.sum(
            residual.conj() * (-2j * np.pi * times[:, None] * joint_prediction) / variance[None, :], axis=0))
        go = np.zeros(group_count)
        np.add.at(go, groups, row_gradient * group_se[groups])
        go += z
        if extra:
            return dict(frequencies_MHz=theta / (2 * np.pi * span), decay_per_us=decay / span,
                        amplitudes=amplitude.T, fitted_return=joint_prediction.T, residual=residual.T,
                        data_cost=data_cost, ridge_cost=ridge_cost, calibration_cost=prior_cost,
                        decoder_offset_MHz=offsets, offset_standard_units=z,
                        inner_iterations=int(inner_iterations),
                        design_condition_number=float(np.linalg.cond(design)),
                        objective=(data_cost + ridge_cost + prior_cost) / scale)
        gradient = np.r_[gf, gd, go[active_groups]]
        if fixed_pair is not None:
            gradient[first] += gradient[second]
            gradient = np.delete(gradient, second)
        return (data_cost + ridge_cost + prior_cost) / scale, gradient / scale

    started = time.perf_counter()
    result = minimize(objective, start, jac=True, bounds=bounds, method="L-BFGS-B",
                      options=dict(maxiter=maxiter, ftol=1e-11, gtol=1e-8, maxls=40))
    output = objective(result.x, True)
    order = np.argsort(output["frequencies_MHz"])
    output["frequencies_MHz"] = output["frequencies_MHz"][order]
    output["decay_per_us"] = output["decay_per_us"][order]
    output["amplitudes"] = output["amplitudes"][:, order]
    output.update(seconds=time.perf_counter() - started, iterations=int(result.nit),
                  success=bool(result.success), message=str(result.message), ridge=float(ridge),
                  amplitude_bound=amplitude_bound,
                  nmse=float(np.sum(abs(output["residual"]) ** 2) / np.sum(abs(traces) ** 2)),
                  max_row_amplitude_sum=float(np.max(np.sum(abs(output["amplitudes"]), axis=1))))
    return output


def joint_prediction(fitted, times, decoder_groups=None):
    groups = np.arange(len(fitted["amplitudes"])) if decoder_groups is None else decoder_groups
    _, groups = np.unique(groups, return_inverse=True)
    design = np.exp(np.asarray(times)[:, None] *
                    (-2j * np.pi * fitted["frequencies_MHz"] - fitted["decay_per_us"]))
    rotation = np.exp(-2j * np.pi * np.asarray(times)[None, :] * fitted["decoder_offset_MHz"][groups, None])
    return (fitted["amplitudes"] @ design.T) * rotation


def joint_propose_swaps(traces, times, fitted, pool_f, pool_decay, *, row_variance=None,
                  decoder_groups=None, attempts=4):
    """Try replacing weak components with all pool candidates using joint RSS.

    Evaluates fixed-pole fits before the caller runs nonlinear refinement. This
    is a bounded search, not a claim of exhaustive optimization over subsets.
    """
    row_count = len(traces)
    variance = np.ones(row_count) if row_variance is None else np.asarray(row_variance)
    groups = np.arange(row_count) if decoder_groups is None else decoder_groups
    _, groups = np.unique(groups, return_inverse=True)
    rotation = np.exp(2j * np.pi * times[None, :] * fitted["decoder_offset_MHz"][groups, None])
    target = (traces * rotation).T / np.sqrt(variance)[None, :]
    frequencies, decays = fitted["frequencies_MHz"], fitted["decay_per_us"]
    design = np.exp(times[:, None] * (-2j * np.pi * frequencies - decays))
    dictionary = np.exp(times[:, None] * (-2j * np.pi * pool_f - np.maximum(pool_decay, 0.)))
    importance = []
    for removed in range(len(frequencies)):
        reduced = np.delete(design, removed, axis=1)
        amplitude = linalg.lstsq(reduced, target, cond=1e-12, check_finite=False)[0]
        importance.append(float(np.sum(abs(target - reduced @ amplitude) ** 2)))
    proposals = []
    for removed in np.argsort(importance)[:attempts]:
        reduced = np.delete(design, removed, axis=1)
        q, _ = linalg.qr(reduced, mode="economic", check_finite=False)
        residual = target - q @ (q.conj().T @ target)
        projected = dictionary - q @ (q.conj().T @ dictionary)
        norm2 = np.sum(abs(projected) ** 2, axis=0)
        score = np.sum(abs(projected.conj().T @ residual) ** 2, axis=1) / np.maximum(norm2, 1e-14)
        score[norm2 < 1e-10] = -np.inf
        added = int(np.argmax(score))
        new_f, new_g = frequencies.copy(), decays.copy()
        new_f[removed], new_g[removed] = pool_f[added], max(0., pool_decay[added])
        proposals.append(dict(frequencies_MHz=new_f, decay_per_us=new_g,
                              removed=int(removed), pool_index=added,
                              fixed_rss=float(np.sum(abs(residual) ** 2) - score[added])))
    return proposals


# ------------------------------------------------------------------------
# Cell 293.
# ------------------------------------------------------------------------

def prepare_precision_inputs(joint_count):
    """Set up the theory-free precision fit (cell 293)."""
    from hashlib import sha256
    from contextlib import nullcontext
    try:
        from threadpoolctl import threadpool_limits as precision_threadpool_limits
    except ImportError:
        precision_threadpool_limits = None  # Optional speed control, not a required analysis dependency.

    precision_target_kHz = 1.0       # Frequency reproducibility target, NOT an FFT-bin or merge threshold.
    precision_count = joint_count   # Known Hilbert-space size is a candidate model size, not a recovery claim.
    precision_maxiter = 1800        # Fast multi-start searches; non-convergence is reported, never hidden.
    precision_exact_maxiter = 800   # Final full-I/Q covariance refinement and local gap probes.
    precision_probe_pairs = 3       # Smallest fitted BULK gaps to stress-test; not all gaps are certified.
    precision_repeat = False        # True reruns even if exactly these input arrays/settings are cached.


# ------------------------------------------------------------------------
# Cell 294.
# ------------------------------------------------------------------------

def precision_fit_exact(traces, times, covariance, initialfit, calibration_se_MHz,
                        decoder_groups, maxiter=200, fixed_pair=None, pair_gap_MHz=None,
                        max_decay_per_us=.2):
    import time
    import numpy as np
    from scipy import linalg
    from scipy.optimize import minimize

    traces = np.asarray(traces, complex)
    times = np.asarray(times, float)
    covariance = np.asarray(covariance, float)
    row_count, sample_count = traces.shape
    count = len(initialfit['frequencies_MHz'])
    span = float(times[-1])
    unit_time = times / span
    sample_time = float(np.min(np.diff(times)))
    _, groups = np.unique(decoder_groups, return_inverse=True)
    group_count = int(groups.max()) + 1
    row_se = np.asarray(calibration_se_MHz, float)
    group_se = np.asarray([np.max(row_se[groups == g]) for g in range(group_count)])
    active_groups = np.flatnonzero(group_se > 0.)
    initial_z = np.asarray(initialfit.get('offset_standard_units', np.zeros(group_count)), float)

    # Covariance is that of each complex MEAN, ordered [Re A, Im A]. Preserve
    # time variation and IQ correlations; repair only numerical non-positivity.
    covariance = .5 * (covariance + np.swapaxes(covariance, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues[..., -1] <= 0.):
        raise ValueError('Each IQ covariance needs a nonzero measured noise scale')
    covariance_scale = np.maximum(eigenvalues[..., -1], np.finfo(float).tiny)
    if np.any(eigenvalues[..., 0] < -1e-10 * covariance_scale):
        raise ValueError('IQ covariance must be positive semidefinite')
    jitter = np.where(eigenvalues[..., 0] <= 0.,
                      100*np.finfo(float).eps*covariance_scale - eigenvalues[..., 0], 0.)
    eigenvalues = eigenvalues + jitter[..., None]
    whitening = (eigenvectors / np.sqrt(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    inverse_covariance = (eigenvectors / eigenvalues[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    observed_iq = np.stack((traces.real, traces.imag), axis=-1)
    whitened_observations = np.einsum('rtij,rtj->rti', whitening, observed_iq)
    scale = max(.5*float(np.sum(whitened_observations**2)), 1.)

    start = np.r_[2*np.pi*span*np.asarray(initialfit['frequencies_MHz']),
                  span*np.clip(initialfit['decay_per_us'], 0., max_decay_per_us),
                  initial_z[active_groups]]
    nyquist_phase = np.pi*span/sample_time
    bounds = [(-nyquist_phase, nyquist_phase)]*count + [(0., max_decay_per_us*span)]*count
    bounds += [(None, None)]*len(active_groups)
    if fixed_pair is not None:
        first, second = map(int, fixed_pair)
        if not (0 <= first < second < count) or pair_gap_MHz is None or pair_gap_MHz < 0.:
            raise ValueError('fixed_pair requires ordered input indices and a nonnegative pair_gap_MHz')
        gap_phase = 2*np.pi*span*pair_gap_MHz
        start[first] = .5*(start[first] + start[second])
        start = np.delete(start, second)
        bounds[first] = (-nyquist_phase + .5*gap_phase, nyquist_phase - .5*gap_phase)
        bounds.pop(second)

    def objective(parameters, details=False):
        if fixed_pair is not None:
            parameters = np.insert(parameters, second, 0.)
            center = parameters[first]
            parameters[first], parameters[second] = center-.5*gap_phase, center+.5*gap_phase
        theta, damping = parameters[:count], parameters[count:2*count]
        z = np.zeros(group_count)
        z[active_groups] = parameters[2*count:]
        offsets = group_se*z
        base_design = np.exp(unit_time[:, None]*(-1j*theta[None, :] - damping[None, :]))
        prediction = np.zeros_like(traces)
        amplitudes = np.zeros((row_count, count), complex)
        gradient_f, gradient_g = np.zeros(count), np.zeros(count)
        gradient_offsets = z.copy()
        data_cost = 0.
        ranks, conditions = [], []
        for row in range(row_count):
            rotation = np.exp(-2j*np.pi*times*offsets[groups[row]])
            design = base_design*rotation[:, None]
            real_design = np.empty((sample_count, 2, 2*count))
            real_design[:, 0, :count], real_design[:, 0, count:] = design.real, -design.imag
            real_design[:, 1, :count], real_design[:, 1, count:] = design.imag, design.real
            weighted_design = np.einsum('tij,tjk->tik', whitening[row], real_design).reshape(2*sample_count, 2*count)
            coefficients, _, rank, _ = linalg.lstsq(
                weighted_design, whitened_observations[row].reshape(-1),
                cond=1e-12, lapack_driver='gelsy', check_finite=False)
            amplitudes[row] = coefficients[:count] + 1j*coefficients[count:]
            prediction[row] = design @ amplitudes[row]
            residual = traces[row] - prediction[row]
            residual_iq = np.stack((residual.real, residual.imag), axis=-1)
            weighted_residual_iq = np.einsum('tij,tj->ti', inverse_covariance[row], residual_iq)
            data_cost += .5*float(np.sum(residual_iq*weighted_residual_iq))
            weighted_residual = weighted_residual_iq[:, 0] + 1j*weighted_residual_iq[:, 1]
            cross = weighted_residual.conj()[:, None]*design*amplitudes[row][None, :]
            gradient_f -= np.real(np.sum(cross*(-1j*unit_time[:, None]), axis=0))
            gradient_g -= np.real(np.sum(cross*(-unit_time[:, None]), axis=0))
            gradient_offsets[groups[row]] -= group_se[groups[row]]*np.real(np.sum(
                weighted_residual.conj()*(-2j*np.pi*times*prediction[row])))
            if details:
                ranks.append(int(rank))
                conditions.append(float(np.linalg.cond(weighted_design)))
        calibration_cost = .5*float(z @ z)
        gradient = np.r_[gradient_f, gradient_g, gradient_offsets[active_groups]]
        if fixed_pair is not None:
            gradient[first] += gradient[second]
            gradient = np.delete(gradient, second)
        if details:
            return dict(frequencies_MHz=theta/(2*np.pi*span), decay_per_us=damping/span,
                        amplitudes=amplitudes, fitted_return=prediction, residual=traces-prediction,
                        decoder_offset_MHz=offsets, offset_standard_units=z,
                        data_cost=data_cost, exact_chi2=2*data_cost, calibration_cost=calibration_cost,
                        total_nll=data_cost+calibration_cost, objective=(data_cost+calibration_cost)/scale,
                        amplitude_design_ranks=ranks, amplitude_design_conditions=conditions,
                        design_condition_number=max(conditions))
        return (data_cost+calibration_cost)/scale, gradient/scale

    started = time.perf_counter()
    optimizer = minimize(objective, start, jac=True, bounds=bounds, method='L-BFGS-B',
                         options=dict(maxiter=maxiter, ftol=1e-11, gtol=1e-8, maxls=30))
    result = objective(optimizer.x, details=True)
    original_frequencies = result['frequencies_MHz'].copy()
    order = np.argsort(original_frequencies)
    result['frequencies_MHz'] = original_frequencies[order]
    result['decay_per_us'] = result['decay_per_us'][order]
    result['amplitudes'] = result['amplitudes'][:, order]
    result['input_to_sorted_rank'] = np.argsort(order)
    if fixed_pair is not None:
        pair_ranks = np.argsort(order)[[first, second]]
        result.update(fixed_pair_input_indices=[first, second],
                      fixed_pair_sorted_ranks=pair_ranks,
                      fixed_pair_remains_adjacent=bool(abs(pair_ranks[1]-pair_ranks[0]) == 1),
                      achieved_pair_gap_MHz=float(original_frequencies[second]-original_frequencies[first]),
                      constrained_pair_sorted_indices=pair_ranks,
                      constrained_pair_adjacent=bool(abs(pair_ranks[1]-pair_ranks[0]) == 1),
                      achieved_gap_MHz=float(original_frequencies[second]-original_frequencies[first]))
    result.update(seconds=time.perf_counter()-started, iterations=int(optimizer.nit),
                  success=bool(optimizer.success), message=str(optimizer.message),
                  gradient_max_abs=float(np.max(abs(objective(optimizer.x)[1]))),
                  covariance_jitter_max=float(np.max(jitter)), covariance_jitter_count=int(np.count_nonzero(jitter)),
                  max_decay_per_us=max_decay_per_us, amplitude_bound=None, ridge=0., ridge_cost=0.,
                  nmse=float(np.linalg.norm(result['residual'])**2/np.linalg.norm(traces)**2),
                  max_row_amplitude_sum=float(np.max(np.sum(abs(result['amplitudes']), axis=1))))
    return result


# ------------------------------------------------------------------------
# Cell 295.
# ------------------------------------------------------------------------

def precision_candidates(traces, times, count):
    frequencies, decays = [], []
    for trace in traces:
        result = SavedSpectroscopyExperiment.analyze_matrix_pencil_trace(
            trace, times, requested_max_modes=count, minimum_consecutive_ranks=3,
            track_frequency_tolerance_bins=0.5, dedup_frequency_tolerance_MHz=0.0001,
            match_decay=False, rank_sweep_extra=2)
        for candidate in result.raw_candidates:
            frequencies.append(candidate.frequency_MHz)
            decays.append(max(0., candidate.decay_per_us))
    return np.asarray(frequencies), np.asarray(decays)


def precision_noise_cost(residual, covariance):
    """Gaussian negative log likelihood without parameter-independent constants."""
    vector = np.stack((residual.real, residual.imag), axis=-1)
    solved = np.linalg.solve(covariance, vector[..., None])[..., 0]
    return 0.5 * float(np.sum(vector * solved))


def precision_gap_ratios(frequencies, edge_fraction=0.10):
    ordered = np.sort(np.asarray(frequencies, dtype=float))
    trim = int(np.ceil(edge_fraction * len(ordered)))
    bulk = ordered[trim:-trim] if trim else ordered
    gaps = np.diff(bulk)
    denominator = np.maximum(gaps[:-1], gaps[1:])
    ratios = np.full(len(denominator), np.nan)
    np.divide(np.minimum(gaps[:-1], gaps[1:]), denominator, out=ratios, where=denominator > 0)
    return ratios[np.isfinite(ratios)]


def precision_search(traces, times, row_variance, groups, calibration_se, count, maxiter):
    """Training-only search: no held-out shots, H levels, or target gap statistic."""
    pool_f, pool_g = precision_candidates(traces, times, count)
    initial = joint_select_pool(traces, times, pool_f, pool_g, count, row_variance)
    block_f, block_g, _ = joint_pencil(traces, times[1] - times[0], count, len(times) // 2)
    options = dict(row_variance=row_variance, decoder_groups=groups,
                   calibration_se_MHz=calibration_se, amplitude_bound=None, maxiter=maxiter)
    trials = []
    for label, frequencies, decays in [('row candidates', initial['frequencies_MHz'], initial['decay_per_us']),
                                      ('joint Hankel', block_f, block_g)]:
        fitted = joint_fit_shared(traces, times, frequencies, decays, **options)
        fitted['initialization'] = label
        trials.append(fitted)
    best = min(trials, key=lambda fit: fit['objective'])
    for proposal in joint_propose_swaps(traces, times, best, pool_f, pool_g,
                                       row_variance=row_variance, decoder_groups=groups, attempts=1):
        fitted = joint_fit_shared(traces, times, proposal['frequencies_MHz'], proposal['decay_per_us'],
                                  initial_offset=best['offset_standard_units'], **options)
        fitted['initialization'] = 'candidate replacement'
        trials.append(fitted)
    best = min(trials, key=lambda fit: fit['objective'])
    return best, trials


def precision_merge_trial(traces, times, fitted, row_variance, groups, calibration_se, maxiter):
    """One smaller-model test. Choose the pair using training data only."""
    frequencies = fitted['frequencies_MHz']
    cut = int(np.ceil(0.10 * len(frequencies)))
    pairs = np.arange(cut, len(frequencies) - cut - 1)
    first = int(pairs[np.argmin(np.diff(frequencies)[pairs])])
    second = first + 1
    new_f = frequencies.copy()
    new_g = fitted['decay_per_us'].copy()
    new_f[first] = np.mean(new_f[[first, second]])
    new_g[first] = np.mean(new_g[[first, second]])
    merged = joint_fit_shared(traces, times, np.delete(new_f, second), np.delete(new_g, second),
                              row_variance=row_variance, decoder_groups=groups,
                              calibration_se_MHz=calibration_se, initial_offset=fitted['offset_standard_units'],
                              amplitude_bound=None, maxiter=maxiter)
    merged['initialization'] = f'merge nearest bulk pair {first}/{second}, then refit all frequencies'
    return merged


# ------------------------------------------------------------------------
# Cell 296.
# ------------------------------------------------------------------------

def run_precision_search(joint_inputs, loaded_spectroscopy, partial_realizations, precision_count, precision_exact_maxiter, precision_maxiter, precision_probe_pairs, precision_repeat, precision_target_kHz):
    """Theory-free pole recovery and its 1 kHz reproducibility (cell 296)."""
    # Prerequisites: HDF5 load, shared-frequency INPUTS, and the two fitting-definition
    # cells immediately above. The old long 'joint_results' runner is NOT required.
    # Inputs remain measurement-only. Changing the selected jobs requires rerunning
    # HDF5 load and INPUTS so that means, shot halves, covariance, and calibration agree.
    expected_jobs = {r: tuple(expt.batch_job_ids) for r, expt in loaded_spectroscopy.items()}
    actual_jobs = {r: tuple(data['job_ids']) for r, data in joint_inputs.items()}
    if actual_jobs != expected_jobs:
        raise RuntimeError('Selected jobs changed: rerun HDF5 load and shared-frequency INPUTS before this cell.')

    precision_cache = globals().get('precision_cache', {})
    precision_results = {}
    precision_options = (precision_target_kHz, precision_count, precision_maxiter,
                         precision_exact_maxiter, precision_probe_pairs, 'precision-v1')
    with precision_threadpool_limits(limits=1) if precision_threadpool_limits else nullcontext():
        for realization, inputs in sorted(joint_inputs.items()):
            if partial_realizations.get(realization, {}).get('incomplete', False):
                print(f'r={realization}: interrupted acquisition; not used for the precision/statistics check')
                continue
            fingerprint = sha256(repr((inputs['job_ids'], precision_options)).encode())
            for field in ('A', 'half0', 'half1', 'cov', 'time_us', 'row_calibration_se_MHz', 'decoder_groups'):
                values = np.ascontiguousarray(inputs[field])
                fingerprint.update(repr((values.shape, values.dtype.str)).encode())
                fingerprint.update(values.tobytes())
            cache_key = fingerprint.hexdigest()
            if not precision_repeat and cache_key in precision_cache:
                precision_results[realization] = precision_cache[cache_key]
                print(f'r={realization}: reused identical-data/settings precision result')
                continue

            started = time.perf_counter()
            times, covariance = inputs['time_us'], inputs['cov']
            groups, calibration_se = inputs['decoder_groups'], inputs['row_calibration_se_MHz']
            row_variance = np.mean(np.trace(covariance, axis1=-2, axis2=-1), axis=1)
            result = dict(inputs=inputs, halves={}, half_trials={}, full_trials=[], profiles=[])
            for half in (0, 1):
                train, heldout = inputs[f'half{half}'], inputs[f'half{1-half}']
                print(f'r={realization}, independent half {half}: free-frequency search, K={precision_count}...', flush=True)
                best, trials = precision_search(train, times, 2 * row_variance, groups, calibration_se,
                                                 precision_count, precision_maxiter)
                smaller = precision_merge_trial(train, times, best, 2 * row_variance, groups,
                                                 calibration_se, precision_maxiter)
                result['half_trials'][half] = trials
                for fitted in (best, smaller):
                    count = len(fitted['frequencies_MHz'])
                    heldout_residual = heldout - joint_prediction(fitted, times, groups)
                    fitted['heldout_noise_units'] = 2 * precision_noise_cost(heldout_residual, 2 * covariance) / (2 * heldout.size)
                    result['halves'][half, count] = fitted
                    print(f"  K={count}: training NMSE={fitted['nmse']:.5f}, held-out noise units="
                          f"{fitted['heldout_noise_units']:.3f}, converged={fitted['success']}", flush=True)

            for half in (0, 1):
                start = result['halves'][half, precision_count]
                print(f'r={realization}: full-data refinement from half {half}...', flush=True)
                fitted = joint_fit_shared(inputs['A'], times, start['frequencies_MHz'], start['decay_per_us'],
                                          row_variance=row_variance, decoder_groups=groups,
                                          calibration_se_MHz=calibration_se, initial_offset=start['offset_standard_units'],
                                          amplitude_bound=None, maxiter=precision_maxiter)
                result['full_trials'].append(fitted)
            start = min(result['full_trials'], key=lambda fit: precision_noise_cost(fit['residual'], covariance) + fit['calibration_cost'])
            print(f'r={realization}: exact per-time I/Q-noise refinement...', flush=True)
            best = precision_fit_exact(inputs['A'], times, covariance, start,
                                        calibration_se_MHz=calibration_se, decoder_groups=groups,
                                        maxiter=precision_exact_maxiter)
            result['best'] = best
            frequencies = best['frequencies_MHz']
            cut = int(np.ceil(0.10 * len(frequencies)))
            bulk_pairs = np.arange(cut, len(frequencies) - cut - 1)
            selected_pairs = bulk_pairs[np.argsort(np.diff(frequencies)[bulk_pairs])[:precision_probe_pairs]]
            base_cost = precision_noise_cost(best['residual'], covariance) + best['calibration_cost']
            for pair in selected_pairs:
                gap_MHz = frequencies[pair + 1] - frequencies[pair]
                # Change a gap by 2*target: at fixed midpoint each endpoint moves by
                # target. After nuisance refitting, inspect actual ordered shifts too.
                for change in (-2e-3 * precision_target_kHz, 2e-3 * precision_target_kHz):
                    new_gap = gap_MHz + change
                    if new_gap <= 0:
                        continue
                    print(f'r={realization}: pair {pair}/{pair+1}, gap {1e3*gap_MHz:.3f} -> {1e3*new_gap:.3f} kHz...', flush=True)
                    alternative = precision_fit_exact(inputs['A'], times, covariance, best,
                                                        calibration_se_MHz=calibration_se, decoder_groups=groups,
                                                        maxiter=precision_exact_maxiter, fixed_pair=(int(pair), int(pair+1)),
                                                        pair_gap_MHz=float(new_gap))
                    alternative['base_pair'] = (int(pair), int(pair+1))
                    alternative['base_gap_kHz'] = 1e3 * gap_MHz
                    alternative['requested_gap_kHz'] = 1e3 * new_gap
                    alternative['delta_cost'] = (precision_noise_cost(alternative['residual'], covariance)
                                                   + alternative['calibration_cost'] - base_cost)
                    prediction_difference = alternative['fitted_return'] - best['fitted_return']
                    # This is the separation of two predicted datasets in measured
                    # noise units, not an individual-pole SE or a chi-square p-value.
                    calibration_distance = np.sum((alternative['offset_standard_units'] - best['offset_standard_units'])**2)
                    alternative['prediction_distance'] = np.sqrt(2 * precision_noise_cost(prediction_difference, covariance)
                                                                   + calibration_distance)
                    alternative['max_ordered_shift_kHz'] = float(1e3 * np.max(abs(alternative['frequencies_MHz'] - frequencies)))
                    result['profiles'].append(alternative)
            result['seconds'] = time.perf_counter() - started
            result['options'] = precision_options
            precision_results[realization] = result
            precision_cache[cache_key] = result
            print(f"r={realization}: finished in {result['seconds']:.1f} s; candidate fits, not certified 1-kHz levels", flush=True)


# ------------------------------------------------------------------------
# Cell 297.
# ------------------------------------------------------------------------

def report_precision_statistics(precision_count, precision_results, precision_target_kHz):
    """Level-statistics sensitivity of the recovered poles (cell 297)."""
    import pandas as pd

    # Read-only reporting. A 1-kHz comparison is not a rule for merging frequencies.
    precision_summary_rows = []
    precision_profile_rows = []
    precision_statistics = {}
    fig, axes = plt.subplots(len(precision_results), 2, squeeze=False,
                             figsize=(14, 3.5 * len(precision_results)), constrained_layout=True)
    for row, (realization, result) in enumerate(sorted(precision_results.items())):
        first = result['halves'][0, precision_count]
        second = result['halves'][1, precision_count]
        best = result['best']
        differences_kHz = 1e3 * abs(first['frequencies_MHz'] - second['frequencies_MHz'])
        within_target = int(np.sum(differences_kHz <= precision_target_kHz))
        searches_converged = all(fit['success'] for fit in (first, second, best))
        if not searches_converged:
            status = 'optimization not converged; precision not established'
        elif within_target < precision_count:
            status = 'not all ordered levels reproducible at target in half-shots'
        else:
            status = 'half-shot reproducibility passed; absolute accuracy not certified'
        precision_summary_rows.append(dict(
            realization=realization, candidate_count=len(best['frequencies_MHz']),
            half_levels_within_target=within_target, target_kHz=precision_target_kHz,
            half_median_shift_kHz=float(np.median(differences_kHz)),
            half_max_shift_kHz=float(np.max(differences_kHz)),
            all_shot_NMSE=best['nmse'], max_amplitude_sum=best['max_row_amplitude_sum'],
            converged=searches_converged, status=status, seconds=result['seconds']))
        print(f"r={realization}: half 0 solver: {first['message']}; half 1 solver: {second['message']}; "
              f"full-data solver: {best['message']}")
        print(f"  all-shot max row sum|amplitude| = {best['max_row_amplitude_sum']:.3g}; "
              "very large values permit cancellation between fitted components, not independently observed peaks.")

        axis, error_axis = axes[row]
        for a, b in zip(first['frequencies_MHz'], second['frequencies_MHz']):
            axis.plot(1e3 * np.asarray([a, b]), [1, 0], color='0.8', lw=0.7)
        for fitted, height, color, label in [(first, 1, 'tab:blue', 'independent half 0'),
                                            (second, 0, 'tab:orange', 'independent half 1'),
                                            (best, -0.45, 'black', 'all-shot candidate fit')]:
            axis.scatter(1e3 * fitted['frequencies_MHz'], np.full(len(fitted['frequencies_MHz']), height),
                         marker='|', color=color, s=100, label=label)
        axis.set(yticks=[-0.45, 0, 1], yticklabels=['all', 'half 1', 'half 0'], xlabel='frequency (kHz)',
                 title=f'r={realization}: {within_target}/{precision_count} ordered pairs within {precision_target_kHz:g} kHz')
        if row == 0:
            axis.legend(fontsize=8, loc='upper left')
        for half, color in [(0, 'tab:blue'), (1, 'tab:orange')]:
            counts = sorted(count for h, count in result['halves'] if h == half)
            errors = [result['halves'][half, count]['heldout_noise_units'] for count in counts]
            error_axis.plot(counts, errors, 'o-', color=color, label=f'train half {half}, predict other half')
        error_axis.axhline(1, color='0.6', ls=':', label='held-out shot noise alone')
        error_axis.set(xticks=counts, xlabel='candidate model size', ylabel='held-out error /\nshot-noise expectation',
                       title='34 vs 35: independent prediction check')
        error_axis.legend(fontsize=8)

        variants = [('all-shot', best)]
        variants += [(f'half {half}', result['halves'][half, precision_count]) for half in (0, 1)]
        variants += [(f'full start {index}', fit) for index, fit in enumerate(result['full_trials'])]
        for index, fit in enumerate(result['profiles']):
            variants.append((f'gap probe {index}', fit))
            precision_profile_rows.append(dict(
                realization=realization, pair=fit['base_pair'], base_gap_kHz=fit['base_gap_kHz'],
                tested_gap_kHz=fit['requested_gap_kHz'], delta_noise_and_calibration_cost=fit['delta_cost'],
                prediction_distance_in_noise_units=fit['prediction_distance'],
                max_ordered_shift_kHz=fit['max_ordered_shift_kHz'], adjacent=fit['fixed_pair_remains_adjacent'],
                converged=fit['success'], mean_gap_ratio=float(np.mean(precision_gap_ratios(fit['frequencies_MHz'])))))
        precision_statistics[realization] = {
            label: precision_gap_ratios(fit['frequencies_MHz']) for label, fit in variants
        }
    fig.suptitle('Theory-free free-frequency fits: reproducibility and prediction, NOT certified 1-kHz accuracy')
    plt.show()
    precision_summary = pd.DataFrame(precision_summary_rows)
    precision_gap_probes = pd.DataFrame(precision_profile_rows)
    display(precision_summary)
    display(precision_gap_probes)
    print('Gap-probe cost changes are not p-values or confidence intervals. Crossed pairs and unfinished fits need further analysis.')
    print('Large-cost gray probes are not equally plausible spectra. A negative cost change means the baseline fit needs further optimization.')
    print('No failed or theory-unmatched levels are removed from the naive statistics below.')

    # Only now may configured-H values enter, and ONLY as a comparison if a matching
    # current MPM record already exists. No theory is computed or fed back to fitting.
    precision_theory_ratios = {}
    for realization, result in precision_results.items():
        reference = globals().get('spectroscopy_records', {}).get(realization)
        if reference is not None and tuple(reference.job_ids) == tuple(result['inputs']['job_ids']):
            sample_frequency = 1 / np.diff(result['inputs']['time_us'])[0]
            reference_f = (np.asarray(reference.theory_levels_MHz) + sample_frequency / 2) % sample_frequency - sample_frequency / 2
            precision_theory_ratios[realization] = precision_gap_ratios(reference_f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    ids = sorted(precision_statistics)
    colors = {'all-shot': 'black', 'half 0': 'tab:blue', 'half 1': 'tab:orange'}
    for label, color in colors.items():
        pooled = np.concatenate([precision_statistics[r][label] for r in ids])
        means = [np.mean(precision_statistics[r][label]) for r in ids]
        axes[0].hist(pooled, bins=np.linspace(0, 1, 11), density=True, histtype='step', color=color,
                     linewidth=1.8, label=f'{label}: mean={np.mean(pooled):.3f}')
        axes[1].plot(np.arange(len(ids)), means, 'o', color=color, label=label)
    for x, realization in enumerate(ids):
        alternative_means = [np.mean(values) for label, values in precision_statistics[realization].items()
                             if label.startswith(('full start', 'gap probe')) and len(values)]
        if alternative_means:
            axes[1].scatter(np.full(len(alternative_means), x) + 0.12, alternative_means,
                             marker='x', color='0.6', label='other starts / gap probes' if x == 0 else None)
    if precision_theory_ratios:
        theory_pooled = np.concatenate(list(precision_theory_ratios.values()))
        axes[0].hist(theory_pooled, bins=np.linspace(0, 1, 11), density=True, histtype='step',
                     color='tab:green', linewidth=1.8, label=f'configured H: mean={np.mean(theory_pooled):.3f}')
        axes[1].scatter([ids.index(r) for r in precision_theory_ratios],
                        [np.mean(values) for values in precision_theory_ratios.values()], color='tab:green',
                        marker='s', label='configured H (comparison only)')
    ratio_axis = np.linspace(0, 1, 500)
    axes[0].plot(ratio_axis, 2 / (1 + ratio_axis)**2, '--', color='tab:purple', label='Poisson')
    axes[0].plot(ratio_axis, 27 / 4 * (ratio_axis + ratio_axis**2) / (1 + ratio_axis + ratio_axis**2)**2.5,
                 '--', color='tab:red', label='GOE surmise')
    axes[1].axhline(2 * np.log(2) - 1, ls='--', color='tab:purple', label='Poisson mean')
    axes[1].axhline(4 - 2 * np.sqrt(3), ls='--', color='tab:red', label='GOE surmise mean')
    axes[0].set(xlim=(0, 1), xlabel='adjacent-gap ratio', ylabel='probability density', title='Naive candidate-spectrum statistics')
    axes[1].set(ylim=(0, 1), xlabel='realization', ylabel='mean adjacent-gap ratio', title='Fit sensitivity, NOT confidence intervals')
    axes[1].set_xticks(np.arange(len(ids)), [f'r={r}' for r in ids])
    for axis in axes:
        axis.legend(fontsize=8)
    fig.suptitle('Each spectrum trimmed 10% at each edge; fitting uncertainty and extraction bias are not removed')
    plt.show()


# ------------------------------------------------------------------------
# Cell 299.
# ------------------------------------------------------------------------

def joint_candidate_pool(traces, times, joint_count):
    """Keep all stable row-MPM candidates, before global-score selection."""
    frequencies, decays = [], []
    for trace in traces:
        result = SavedSpectroscopyExperiment.analyze_matrix_pencil_trace(
            trace, times, requested_max_modes=joint_count, minimum_consecutive_ranks=3,
            track_frequency_tolerance_bins=0.5, dedup_frequency_tolerance_MHz=0.0001,
            match_decay=False, rank_sweep_extra=2)
        for candidate in result.raw_candidates:
            frequencies.append(candidate.frequency_MHz)
            decays.append(max(0., candidate.decay_per_us))
    return np.asarray(frequencies), np.asarray(decays)


def fit_halves_independently(joint_amplitude_bound, joint_compare_count, joint_count, joint_inputs, joint_maxiter, joint_swap_attempts):
    """Fit each shot half independently, then refit all shots (cell 299)."""


    joint_results = {}
    with threadpool_limits(limits=1):
        for realization, inputs in sorted(joint_inputs.items()):
            times = inputs['time_us']
            covariance = inputs['cov']
            variance = np.mean(np.trace(covariance, axis1=-2, axis2=-1), axis=1)
            groups = inputs['decoder_groups']
            fit_options = dict(decoder_groups=groups, calibration_se_MHz=inputs['row_calibration_se_MHz'],
                               maxiter=joint_maxiter, amplitude_bound=joint_amplitude_bound)
            results = dict(halves={}, full_starts=[], inputs=inputs, amplitude_bound=joint_amplitude_bound)
            for half in (0, 1):
                train, test = inputs[f'half{half}'], inputs[f'half{1-half}']
                pool_f, pool_g = joint_candidate_pool(train, times)
                for count in sorted(set((joint_compare_count, joint_count))):
                    print(f'r={realization}, half={half}, K={count}: fitting {len(pool_f)} row candidates...', flush=True)
                    initial = joint_select_pool(train, times, pool_f, pool_g, count, 2 * variance)
                    block_f, block_g, _ = joint_pencil(train, times[1] - times[0], count, len(times)//2)
                    # Small numerical ridge ONLY generates a stable starting point.
                    # The reported fits below remove it (ridge=0); it supplies no spectral prior.
                    seed_options = dict(fit_options, amplitude_bound=None)
                    seed = joint_fit_shared(train, times, initial['frequencies_MHz'], initial['decay_per_us'],
                                            row_variance=2*variance, ridge=.01, **seed_options)
                    trials = []
                    for name, frequencies, decays in (
                        ('all-row candidate pool', initial['frequencies_MHz'], initial['decay_per_us']),
                        ('joint block-Hankel', block_f, block_g),
                        ('stabilized data-only start', seed['frequencies_MHz'], seed['decay_per_us']),
                    ):
                        fitted = joint_fit_shared(train, times, frequencies, decays,
                                                  row_variance=2*variance, **fit_options)
                        fitted['initialization'] = name
                        trials.append(fitted)
                    # Only training data chooses/replaces components. No validation or theory enters here.
                    best = min(trials, key=lambda trial: trial['objective'])
                    swaps = joint_propose_swaps(train, times, best, pool_f, pool_g,
                                                row_variance=2*variance, decoder_groups=groups,
                                                attempts=joint_swap_attempts)
                    for proposal in swaps:
                        fitted = joint_fit_shared(train, times, proposal['frequencies_MHz'], proposal['decay_per_us'],
                                                  row_variance=2*variance, initial_offset=best['offset_standard_units'],
                                                  **fit_options)
                        fitted['initialization'] = 'residual-driven replacement'
                        trials.append(fitted)
                    best = min(trials, key=lambda trial: trial['objective'])
                    test_residual = test - joint_prediction(best, times, groups)
                    best['heldout_nmse'] = float(np.sum(abs(test_residual)**2) / np.sum(abs(test)**2))
                    best['heldout_noise_units'] = float(np.mean(abs(test_residual)**2 / (2*variance[:, None])))
                    results['halves'][(half, count)] = best
                    print(f"  train NMSE={best['nmse']:.4f}, opposite-half NMSE={best['heldout_nmse']:.4f}, "
                          f"max amplitude sum={best['max_row_amplitude_sum']:.3g}, converged={best['success']}")
            # Refit all shots from BOTH independently obtained starts. Keep both answers.
            for half in (0, 1):
                start = results['halves'][(half, joint_count)]
                fitted = joint_fit_shared(inputs['A'], times, start['frequencies_MHz'], start['decay_per_us'],
                                          row_variance=variance, initial_offset=start['offset_standard_units'], **fit_options)
                results['full_starts'].append(fitted)
            results['best'] = min(results['full_starts'], key=lambda trial: trial['objective'])
            joint_results[realization] = results
            first = results['halves'][(0, joint_count)]['frequencies_MHz']
            second = results['halves'][(1, joint_count)]['frequencies_MHz']
            errors = 1000 * abs(first - second)
            print(f'r={realization}: independently fitted ordered levels within 1 / 2 / 5 kHz: '
                  f'{np.sum(errors<1)} / {np.sum(errors<2)} / {np.sum(errors<5)} of {joint_count}')
            print('  These are reproducibility diagnostics, NOT a resolved-level count or a confidence interval.')


# ------------------------------------------------------------------------
# Cell 301.
# ------------------------------------------------------------------------

def report_half_reproducibility(joint_count, joint_results):
    """Independent-half reproducibility (cell 301)."""
    # Plot-only: never changes frequencies, selects poles, or repeats a fit.
    fig, axes = plt.subplots(len(joint_results), 2, figsize=(14, 3.2*len(joint_results)), squeeze=False,
                             constrained_layout=True)
    for row, (realization, result) in enumerate(sorted(joint_results.items())):
        ax, error_ax = axes[row]
        first = 1000 * result['halves'][(0, joint_count)]['frequencies_MHz']
        second = 1000 * result['halves'][(1, joint_count)]['frequencies_MHz']
        full = 1000 * result['best']['frequencies_MHz']
        for a, b in zip(first, second):
            ax.plot([a, b], [1, 0], color='0.8', lw=0.7)
        ax.scatter(first, np.ones(len(first)), marker='|', color='tab:blue', label='disjoint shot half 0')
        ax.scatter(second, np.zeros(len(second)), marker='x', color='tab:orange', label='disjoint shot half 1')
        ax.scatter(full, np.full(len(full), -.35), marker='|', color='black', label='best all-shot candidate fit')
        other = next(fit for fit in result['full_starts'] if fit is not result['best'])
        ax.scatter(1000*other['frequencies_MHz'], np.full(len(full), -.7), marker='|', color='0.5',
                   label='other all-shot starting point')
        cost_difference = abs(other['data_cost'] + other['calibration_cost']
                              - result['best']['data_cost'] - result['best']['calibration_cost'])
        ax.set(yticks=[-.7, -.35, 0, 1], yticklabels=['all: other', 'all: best', 'half 1', 'half 0'], xlabel='frequency (kHz)',
               title=f'r={realization}: K={joint_count} candidates; all-shot objective difference={cost_difference:.3g}')
        if row == 0:
            ax.legend(fontsize=8, loc='upper left')
        for half, color in ((0, 'tab:blue'), (1, 'tab:orange')):
            counts = sorted(k for h, k in result['halves'] if h == half)
            scores = [result['halves'][(half, k)]['heldout_noise_units'] for k in counts]
            error_ax.plot(counts, scores, 'o-', color=color, label=f'train half {half}, predict other half')
        error_ax.set(xlabel='number of candidate components', ylabel='prediction error / held-out shot-noise power',
                     title='Does adding components predict new shots better?')
        error_ax.legend(fontsize=8)
    plt.show()


# ------------------------------------------------------------------------
# Cell 303.
# ------------------------------------------------------------------------

def compare_configured_hamiltonian(joint_results, loaded_spectroscopy):
    """Configured-Hamiltonian comparison -- posthoc only (cell 303)."""
    # Optional posthoc reference only: the fit above has already finished.
    # This is the configured static-H spectrum, not a claim of exact pulse-level Floquet theory.
    fig, axes = plt.subplots(len(joint_results), 1, figsize=(12, 2.8*len(joint_results)), squeeze=False,
                             constrained_layout=True)
    joint_reference = {}
    for row, (realization, result) in enumerate(sorted(joint_results.items())):
        loaded = loaded_spectroscopy[realization]
        if tuple(loaded.batch_job_ids) != result['inputs']['job_ids']:
            raise ValueError('Loaded jobs changed after fitting; rerun inputs/fit before the reference comparison')
        saved = SavedSpectroscopyExperiment._saved_parameters(loaded.batch_expts)
        rec = SavedSpectroscopyExperiment.reconstruct_pair_spectroscopy(loaded.batch_expts)
        rec.A = result['inputs']['A']
        actual = result['inputs']['actual_cycle_us']
        prog = loaded.batch_expts[0].prog
        modes = loaded.batch_expts[0].cfg.expt.swap_stors
        g = np.asarray([1/(4*prog.m1s_pi_fracs[mode-1]*actual) for mode in modes])
        kerr = float(loaded.batch_expts[0].cfg.expt.d72_self_kerr_kHz)/1000
        spectrum = SavedSpectroscopyExperiment.analyze_spectrum(rec, int(sum(rec.occupations[0])), saved.detunings,
                                                  g, actual, kerr, fft_window='raw', zero_padding=1)
        fs = 1/(result['inputs']['time_us'][1] - result['inputs']['time_us'][0])
        theory = np.sort((np.asarray(spectrum.energies_MHz)+fs/2) % fs - fs/2)*1000
        measured = np.sort(result['best']['frequencies_MHz'])*1000
        joint_reference[realization] = theory / 1000
        ax = axes[row, 0]
        if len(theory) == len(measured):
            for a, b in zip(theory, measured):
                ax.plot([a, b], [1, 0], color='0.8', lw=.7)
            mae = np.mean(abs(theory-measured))
            label = f'r={realization}: all ordered levels, MAE={mae:.2f} kHz'
        else:
            label = f'r={realization}: {len(measured)} fitted components vs {len(theory)} reference levels'
        ax.scatter(theory, np.ones(len(theory)), marker='|', color='tab:green')
        ax.scatter(measured, np.zeros(len(measured)), marker='x', color='black')
        ax.set(yticks=[0, 1], yticklabels=['candidate fit', 'configured static H'], xlabel='frequency (kHz)', title=label)
    plt.show()


# ------------------------------------------------------------------------
# Cell 305.
# ------------------------------------------------------------------------

def pool_circular_distance_MHz(left_MHz, right_MHz, sampling_frequency_MHz):
    difference_MHz = wrap_frequency(left_MHz - right_MHz, sampling_frequency_MHz)
    return abs(float(difference_MHz))


def pool_circular_mean_MHz(frequencies_MHz, weights, sampling_frequency_MHz):
    frequencies_MHz = np.asarray(frequencies_MHz, dtype=float)
    weights = np.asarray(weights, dtype=float)
    angles = 2.0 * np.pi * frequencies_MHz / sampling_frequency_MHz
    phasor = np.sum(weights * np.exp(1j * angles))
    if abs(phasor) < np.finfo(float).eps:
        return float(frequencies_MHz[np.argmax(weights)])
    return float(sampling_frequency_MHz * np.angle(phasor) / (2.0 * np.pi))


def select_row_pooled(matrix_pencil, target_count, merge_tolerance_kHz):
    sampling_frequency_MHz = float(matrix_pencil.sampling.sampling_frequency_MHz)
    merge_tolerance_MHz = 1e-3 * float(merge_tolerance_kHz)
    raw_candidates = list(matrix_pencil.candidates.raw_per_row)

    candidates_by_row = {}
    for candidate in raw_candidates:
        row_index = int(candidate.row_index)
        candidates_by_row.setdefault(row_index, []).append(candidate)

    pooled_candidates = []
    for row_index, row_candidates in candidates_by_row.items():
        row_strengths = np.asarray(
            [max(float(candidate.confidence), 0.0) for candidate in row_candidates]
        )
        row_scale = float(np.max(row_strengths))
        if row_scale == 0.0:
            row_scale = 1.0

        for candidate, strength in zip(row_candidates, row_strengths):
            pooled_candidates.append(
                {
                    'row_index': row_index,
                    'occupation': tuple(candidate.occupation),
                    'frequency_MHz': float(candidate.frequency_MHz),
                    'absolute_score': float(strength),
                    'confidence': float(candidate.confidence),
                    'row_normalized_score': float(strength / row_scale),
                }
            )

    pooled_candidates.sort(key=lambda item: (-item['row_normalized_score'], -item['confidence']))

    clusters = []
    for item in pooled_candidates:
        compatible_clusters = []
        for cluster_index, cluster in enumerate(clusters):
            if item['row_index'] in cluster['rows']:
                continue
            member_distances_MHz = [
                pool_circular_distance_MHz(
                    item['frequency_MHz'], member['frequency_MHz'], sampling_frequency_MHz
                )
                for member in cluster['members']
            ]
            distance_MHz = max(member_distances_MHz)
            if distance_MHz <= merge_tolerance_MHz:
                compatible_clusters.append((distance_MHz, -cluster['score'], cluster_index))

        if compatible_clusters:
            cluster_index = min(compatible_clusters)[2]
            cluster = clusters[cluster_index]
        else:
            cluster = {'members': [], 'rows': set(), 'frequency_MHz': item['frequency_MHz'], 'score': 0.0}
            clusters.append(cluster)

        cluster['members'].append(item)
        cluster['rows'].add(item['row_index'])
        member_scores = np.asarray([member['row_normalized_score'] for member in cluster['members']])
        member_weights = np.maximum(member_scores**2, 1e-12)
        cluster['frequency_MHz'] = pool_circular_mean_MHz(
            [member['frequency_MHz'] for member in cluster['members']], member_weights, sampling_frequency_MHz
        )
        absolute_scores = np.asarray([member['absolute_score'] for member in cluster['members']])
        cluster['score'] = float(np.sqrt(np.sum(member_scores**2)))
        cluster['absolute_score'] = float(np.sqrt(np.sum(absolute_scores**2)))

    for cluster in clusters:
        cluster['frequency_scatter_MHz'] = max(
            pool_circular_distance_MHz(
                member['frequency_MHz'], cluster['frequency_MHz'], sampling_frequency_MHz
            )
            for member in cluster['members']
        )

    clusters.sort(
        key=lambda cluster: (-cluster['score'], -cluster['absolute_score'], cluster['frequency_scatter_MHz'])
    )
    selected_clusters = clusters[:target_count]
    selected_frequencies_MHz = np.sort(
        np.asarray([cluster['frequency_MHz'] for cluster in selected_clusters])
    )

    return {
        'sampling_frequency_MHz': sampling_frequency_MHz,
        'merge_tolerance_MHz': merge_tolerance_MHz,
        'raw_candidates': raw_candidates,
        'pooled_candidates': pooled_candidates,
        'clusters': clusters,
        'selected_clusters': selected_clusters,
        'frequencies_MHz': selected_frequencies_MHz,
    }


def select_row_pooled_levels(match_levels, match_tolerance_bins, spectroscopy_records):
    """Alternative row-pooled target-35 selection (cell 305)."""
    pool_target_count = 35
    pool_merge_tolerance_kHz = 0.1
    pool_edge_fraction = 0.1




    row_pool_records = {}
    for realization, record in sorted(spectroscopy_records.items()):
        matrix_pencil = record.data.matrix_pencil
        result = select_row_pooled(matrix_pencil, pool_target_count, pool_merge_tolerance_kHz)
        match_tolerance_MHz = match_tolerance_bins * float(matrix_pencil.sampling.fft_resolution_MHz)
        result['match'] = match_levels(
            result['frequencies_MHz'],
            record.theory_levels_MHz,
            result['sampling_frequency_MHz'],
            match_tolerance_MHz,
        )
        result['current_frequencies_MHz'] = np.sort(
            wrap_frequency(matrix_pencil.selected_frequencies_MHz, result['sampling_frequency_MHz'])
        )
        row_pool_records[realization] = result

        print(
            f'r={realization}: raw={len(result["raw_candidates"])}, '
            f'clusters={len(result["clusters"])}, '
            f'selected={len(result["frequencies_MHz"])}, '
            f'within tolerance={len(result["match"].pairs)}/35'
        )


    realizations = sorted(row_pool_records)
    ncols = 2
    nrows = int(np.ceil(len(realizations) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), squeeze=False)

    for axis, realization in zip(axes.flat, realizations):
        record = spectroscopy_records[realization]
        result = row_pool_records[realization]
        theory_kHz = 1e3 * np.sort(wrap_frequency(record.theory_levels_MHz, result['sampling_frequency_MHz']))
        current_kHz = 1e3 * result['current_frequencies_MHz']
        row_pool_kHz = 1e3 * result['frequencies_MHz']

        axis.vlines(theory_kHz, 2.75, 3.25, color='tab:green', linewidth=1.2, label='exact theory')
        axis.scatter(
            current_kHz, np.full(len(current_kHz), 2.0), color='black', marker='x', s=22, label='current MPM'
        )
        axis.scatter(
            row_pool_kHz,
            np.full(len(row_pool_kHz), 1.0),
            color='tab:purple',
            marker='x',
            s=24,
            label='row-pooled',
        )
        axis.set_yticks([1.0, 2.0, 3.0], ['row-pooled', 'current', 'theory'])
        axis.set_ylim(0.65, 3.35)
        axis.set_xlabel('principal-zone energy E/h (kHz)')
        axis.set_title(
            f'r={realization}: current={len(current_kHz)}, '
            f'pooled={len(row_pool_kHz)}, '
            f'matched={len(result["match"].pairs)}/35'
        )
        axis.grid(axis='x', alpha=0.2)

    for axis in axes.flat[len(realizations) :]:
        axis.axis('off')

    axes.flat[0].legend(fontsize=8, ncol=3, loc='upper left')
    fig.suptitle(
        'independent row-pooled target-35 selection; ' f'tolerance={pool_merge_tolerance_kHz:g} kHz', y=1.0
    )
    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------
# Cell 306.
# ------------------------------------------------------------------------

def pool_gap_ratios(levels_MHz, lower_MHz, upper_MHz):
    levels_MHz = np.sort(np.asarray(levels_MHz, dtype=float))
    levels_MHz = levels_MHz[(levels_MHz >= lower_MHz) & (levels_MHz <= upper_MHz)]
    gaps_MHz = np.diff(levels_MHz)
    smaller_gaps_MHz = np.minimum(gaps_MHz[:-1], gaps_MHz[1:])
    larger_gaps_MHz = np.maximum(gaps_MHz[:-1], gaps_MHz[1:])
    valid = larger_gaps_MHz > 0.0
    return smaller_gaps_MHz[valid] / larger_gaps_MHz[valid]


def report_row_pooled_level_statistics(pool_edge_fraction, pool_merge_tolerance_kHz, row_pool_records, spectroscopy_records):
    """Level statistics of the row-pooled selection (cell 306)."""


    row_pool_ratios = {}
    row_pool_current_ratios = {}
    row_pool_theory_ratios = {}

    for realization, result in sorted(row_pool_records.items()):
        record = spectroscopy_records[realization]
        theory_MHz = wrap_frequency(record.theory_levels_MHz, result['sampling_frequency_MHz'])
        theory_MHz = np.sort(theory_MHz)
        trim_count = int(np.ceil(pool_edge_fraction * len(theory_MHz)))
        theory_bulk_MHz = theory_MHz[trim_count:-trim_count]
        lower_MHz = theory_bulk_MHz[0]
        upper_MHz = theory_bulk_MHz[-1]
        row_pool_ratios[realization] = pool_gap_ratios(result['frequencies_MHz'], lower_MHz, upper_MHz)
        row_pool_current_ratios[realization] = pool_gap_ratios(
            result['current_frequencies_MHz'], lower_MHz, upper_MHz
        )
        row_pool_theory_ratios[realization] = pool_gap_ratios(theory_MHz, lower_MHz, upper_MHz)

    row_pool_pooled = np.concatenate(list(row_pool_ratios.values()))
    current_pooled = np.concatenate(list(row_pool_current_ratios.values()))
    theory_pooled = np.concatenate(list(row_pool_theory_ratios.values()))

    ratio_axis = np.linspace(0.0, 1.0, 1000)
    poisson_pdf = 2.0 / (1.0 + ratio_axis) ** 2
    goe_pdf = (27.0 / 4.0) * (ratio_axis + ratio_axis**2) / (1.0 + ratio_axis + ratio_axis**2) ** 2.5
    poisson_mean = 2.0 * np.log(2.0) - 1.0
    goe_mean = 4.0 - 2.0 * np.sqrt(3.0)
    ratio_edges = np.linspace(0.0, 1.0, 21)

    realizations = np.asarray(sorted(row_pool_records))
    row_pool_means = np.asarray([np.mean(row_pool_ratios[realization]) for realization in realizations])
    current_means = np.asarray([np.mean(row_pool_current_ratios[realization]) for realization in realizations])
    theory_means = np.asarray([np.mean(row_pool_theory_ratios[realization]) for realization in realizations])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].hist(
        row_pool_pooled,
        bins=ratio_edges,
        density=True,
        color='tab:purple',
        alpha=0.35,
        label='row-pooled target-35',
    )
    axes[0].hist(
        current_pooled,
        bins=ratio_edges,
        density=True,
        histtype='step',
        color='black',
        linewidth=1.5,
        label='current MPM',
    )
    axes[0].hist(
        theory_pooled,
        bins=ratio_edges,
        density=True,
        histtype='step',
        color='tab:green',
        linewidth=1.7,
        label='exact theory',
    )
    axes[0].plot(ratio_axis, poisson_pdf, label='Poisson')
    axes[0].plot(ratio_axis, goe_pdf, label='GOE')
    axes[0].set(
        xlim=(0.0, 1.0),
        xlabel='adjacent-gap ratio',
        ylabel='probability density',
        title='pooled level statistics; common theory-defined 10% bulk window',
    )
    axes[0].legend(fontsize=8)

    x = np.arange(len(realizations), dtype=float)
    axes[1].scatter(x - 0.18, row_pool_means, color='tab:purple', label='row-pooled')
    axes[1].scatter(x, current_means, color='black', marker='x', label='current MPM')
    axes[1].scatter(x + 0.18, theory_means, color='tab:green', label='exact theory')
    axes[1].axhline(poisson_mean, color='tab:blue', linestyle='--', label=f'Poisson mean={poisson_mean:.3f}')
    axes[1].axhline(goe_mean, color='tab:orange', linestyle='--', label=f'GOE mean={goe_mean:.3f}')
    axes[1].set_xticks(x, [f'r={realization}' for realization in realizations])
    axes[1].set(ylim=(0.0, 1.0), ylabel='mean adjacent-gap ratio', title='mean gap ratio by realization')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    row_pool_level_statistics = {
        'edge_fraction': pool_edge_fraction,
        'merge_tolerance_kHz': pool_merge_tolerance_kHz,
        'row_pool_by_realization': row_pool_ratios,
        'current_by_realization': row_pool_current_ratios,
        'theory_by_realization': row_pool_theory_ratios,
        'row_pool_pooled': row_pool_pooled,
        'current_pooled': current_pooled,
        'theory_pooled': theory_pooled,
    }

    print(
        f'row-pooled mean={np.mean(row_pool_means):.3f}; '
        f'current MPM mean={np.mean(current_means):.3f}; '
        f'theory mean={np.mean(theory_means):.3f}; '
        f'Poisson={poisson_mean:.3f}; GOE={goe_mean:.3f}'
    )


