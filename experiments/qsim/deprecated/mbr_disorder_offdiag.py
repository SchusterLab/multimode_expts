"""Pairwise and D72 (occupation-constrained) disorder campaigns -- DEPRECATED.

Moved here from `experiments/qsim/notebook_helpers/mbr_disorder_campaign.py` on
2026-09-24 (MBR redesign step 7a), without changes except this docstring. See
`docs/qsim/mbr_step7_plan.md`, decision 2: off-diagonal time traces (init !=
final) have no valid Stark-shift phase calibration, and they add no levels that
the diagonals of a complete basis do not already give. Only diagonal disorder
(section 7-1, still in `notebook_helpers/mbr_disorder_campaign.py`) is canonical.

Not maintained; may break when live code changes. If it breaks, add a note here
and do not fix it. Notebook user: `measurement_notebooks/202609_qsim_migration/
dormant/mbr_disorder_offdiag.py`.

Known problems when moved (the code is notebook cells copied whole, with no data
flow connected): `globals()` reads here see this module's globals, not the
notebook's, so caches start empty and the D72 common-calibration fallback raises
unless `calibration_job_ids` is set; `build_pairwise_plan(selected_rows=...)` is
overwritten; `d72_analysis_error` is shadowed by the `except` name.

Original docstring follows.

Planning, submission and analysis for the disorder spectroscopy campaigns.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
318-349 by the stage-2 notebook decomposition. Callers:
`measurement_notebooks/202609_qsim_migration/mbr_disorder.py` for the planning
and submission steps, and
`analysis_notebooks/202609_qsim_migration/mbr_disorder.py` for the reports.

Section 7 of the source ran three campaigns, and the split follows the
headings' own "-- no jobs" / "-- submits jobs" annotations:

  7 (intro)  cells 319-320  a pairwise disorder preview and its batch
  7-1        cells 323-334  diagonal disorder: settings, plan, run, MPM,
                            pooled level statistics
  7-2        cells 338-348  occupation-constrained disorder: the same shape
                            again, with channel selection on top

## The settings prefixes

This theme is what the stage-2 instructions had in mind about grouping a
repeated settings prefix into one plainly named config object. Cell 338 set
**thirty** `d72_*` names at notebook scope, and cells 340, 342, 344 and 346
read them as globals. Cell 323 set sixteen `diag_disorder_*` names the same
way. Those are now `D72Config` and `DiagDisorderConfig`, with the source's
values and its own explanatory comments preserved as field comments.

Each function unpacks its config into the local names the moved body already
uses, so no line of those long bodies needed renaming. `mpm_rank_sweep_extra`
is a property rather than a field, because the source derived it
(`minimum_consecutive_ranks - 1`) rather than setting it.

The campaign base -- `EncSpec`, `encspec_defaults`, the calibration cache and
the rest -- comes from `mbr_campaign.build_campaign` instead of the live
kernel.

## One cross-theme input

Cell 325 reads `best_self_kerr_kHz`, which section 3-1 of the acquisition
notebook produced and which now lives in
`mbr_n3_reprocess.fit_self_kerr_from_peak_overlap`. It is an explicit argument
here. Passing None falls back the way the source did: to the signed Kerr saved
with the phase-calibration jobs.

## Submission stays in the notebook

An earlier pass wrapped the three submission cells whole, as
`submit_pairwise`, `submit_diag_disorder` and `submit_d72`. Each one built the
runner *and* called `runner.execute` behind a closed keyword list, so
the notebook could not reach `batch_size`, `log` or any other per-run
argument, and the runner it was submitting through was invisible. Worse, each
opened by unpacking its whole config object into local names, most of which it
never used.

What is here now is the build halves -- `build_pairwise_batch`,
`build_diag_realization_batch` -- which return `(batch, runner, ...)` and
submit nothing, plus `check_d72_visibility`, the guard cell 344 ran before
submitting. The `execute` calls and the two per-realization acquisition loops
are inline in the notebook. This is the contract
`mbr_campaign.build_spectroscopy_batch` already had; do not hoist submission
again.

Temporary home, per the stage-2 instructions. The three campaigns are kept as
three code paths rather than unified -- they select channels and match theory
differently, and deciding which approach wins is not this pass's job.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product
from math import comb
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    linear_sum_assignment,
    milp,
)

from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner, json_plain

from experiments.qsim import floquet_dark_mode_readout as d72_module
from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.deprecated.legacy_mbr import MBRSpectrumExperiment


@dataclass
class D72Config:
    """Cell 338's thirty `d72_*` knobs, with its values and its comments."""

    # Physics and disorder ensemble.
    N: int = 3
    realization_count: int = 2
    disorder_strength_kHz: float = 50.0
    master_seed: int = 20260903

    # State/channel constraints.
    # None removes the cap; 1 keeps only hard-core states; 2 excludes |3>.
    max_occupation: Optional[int] = 3
    forbidden_states: list = field(default_factory=list)
    channel_count: int = 15
    required_support: int = 1
    allow_diagonal: bool = True
    allow_offdiagonal: bool = True

    # A diagnostic/submit guard after calibration access is normalized.
    min_acceptable_visibility: float = 1e-4

    # Coherence-limited sampling and acquisition.
    max_time_us: float = 80.0
    min_time_points: int = 50
    nyquist_margin: float = 1.35
    step_autocalculate: bool = False
    cycle_step: int = 2  # used when step_autocalculate is False
    cycle_chunk_points: int = 200
    reps: int = 1000
    batch_size: int = 2

    # None uses the signed Kerr in the current hardware configuration.
    self_kerr_kHz: Optional[float] = None
    branch_overrides: dict = field(default_factory=dict)

    # None uses the active common calibration, then the common file map.
    # Alternatively pass an explicit list of phase-calibration job IDs.
    calibration_job_ids: Any = None

    # Diagnostics and Matrix-Pencil settings.
    theory_plot_realizations: int = 1
    match_tolerance_bins: float = 1.5

    # Rowwise MPM keeps poles that remain stable across consecutive ranks.
    # Rank persistence determines candidate confidence before shared-pole
    # merging.
    mpm_minimum_consecutive_ranks: int = 3
    mpm_minimum_supporting_rows: int = 1

    # Cross-row merge tolerance is max(0.1 kHz, 3 sigma) for the difference
    # of the calibration slope errors. Rows sharing one decoder occupation
    # share the same common-mode error, so it is not counted twice. The
    # 0.1-kHz scale is also used only for within-row deduplication, not
    # rank-history tracking.
    mpm_merge_frequency_tolerance: str = "calibration"
    mpm_calibration_sigma_multiplier: float = 3.0
    mpm_frequency_tolerance_floor_kHz: float = 0.1
    mpm_dedup_frequency_tolerance_kHz: float = 0.1

    @property
    def mpm_rank_sweep_extra(self):
        """Derived in the source rather than set (cell 338 lines 64-66)."""
        return self.mpm_minimum_consecutive_ranks - 1



# --------------------------------------------------------------------------
# Section 7 intro: the pairwise disorder preview (cells 319-320).
# --------------------------------------------------------------------------


def build_pairwise_plan(campaign, station, N=3, strength_kHz=50.0, seed=20260815,
                        pair_count=10, reps=1200, batch_size=2,
                        selected_rows=None):
    """Plan a pairwise-detuning disorder run (cell 319). Submits nothing.

    Returns a dict of the `disorder_*` names cell 320 needs.
    """
    disorder_N = N
    disorder_strength_kHz = strength_kHz
    disorder_seed = seed
    disorder_pair_count = pair_count
    disorder_reps = reps
    disorder_batch_size = batch_size


    EncSpec = campaign.EncSpec
    encspec_modes = campaign.modes
    encspec_mode_labels = campaign.mode_labels
    encspec_sync_cycles = campaign.sync_cycles
    encspec_defaults = campaign.defaults
    encspec_calibrations = campaign.calibrations
    encspec_calibration_files = campaign.calibration_files
    encspec_calibration_job_ids = campaign.calibration_job_ids

    if disorder_N not in encspec_calibrations:
        calibration_expt = MBRPhaseCorrectionExperiment.from_job_files(
            encspec_calibration_files[disorder_N],
            station=station,
        )
        calibration_expt.batch_job_ids = encspec_calibration_job_ids[disorder_N]
        calibration_expt.analyze()
        encspec_calibrations[disorder_N] = calibration_expt

    disorder_calibration = encspec_calibrations[disorder_N]
    disorder_hardware = EncSpec.hardware_parameters(
        station,
        encspec_modes,
        encspec_sync_cycles,
    )
    rng = np.random.default_rng(disorder_seed)
    disorder_onsite_kHz = rng.normal(size=len(encspec_modes))
    disorder_onsite_kHz -= disorder_onsite_kHz.mean()
    disorder_onsite_kHz *= (
        disorder_strength_kHz
        / np.sqrt(np.mean(disorder_onsite_kHz ** 2))
    )
    disorder_detunings_MHz = -1e-3 * disorder_onsite_kHz

    probe_occupation = [disorder_N] + [0] * len(encspec_modes)
    probe = AttrDict(dict(
        occupations=[probe_occupation],
        final_occupations=[probe_occupation.copy()],
        cycles=np.array([0, 1]),
        A=np.ones((1, 2), dtype=complex),
    ))
    disorder_theory = EncSpec.analyze_spectrum(
        probe,
        disorder_N,
        disorder_detunings_MHz,
        disorder_hardware.couplings_MHz,
        disorder_hardware.floquet_cycle_us,
        disorder_hardware.physical_kerr_MHz,
    )

    disorder_basis = [tuple(occupation) for occupation in disorder_theory.fock_basis]
    disorder_access = {
        tuple(result.occupation): abs(result.complex_return[0])
        for result in disorder_calibration.data.results
    }
    allowed_rows = [
        row
        for row, occupation in enumerate(disorder_basis)
        if occupation[0] < 2 and occupation in disorder_access
    ]
    candidate_rows = list(combinations_with_replacement(allowed_rows, 2))
    basis_weights = np.asarray(disorder_theory.basis_eigenstate_weights)
    candidate_visibility = np.asarray([
        np.sqrt(
            disorder_access[disorder_basis[decoder]]
            * disorder_access[disorder_basis[encoder]]
        )
        * np.sqrt(basis_weights[decoder] * basis_weights[encoder])
        for decoder, encoder in candidate_rows
    ])

    selected_rows = []
    level_coverage = np.zeros(len(disorder_theory.energies_MHz))
    for _ in range(disorder_pair_count):
        scores = np.sum(np.log(
            1e-12 + level_coverage[None, :] + candidate_visibility
        ), axis=1)
        scores[selected_rows] = -np.inf
        selected = int(np.argmax(scores))
        selected_rows.append(selected)
        level_coverage += candidate_visibility[selected]

    disorder_pairs = [
        (
            disorder_basis[candidate_rows[row][0]],
            disorder_basis[candidate_rows[row][1]],
        )
        for row in selected_rows
    ]
    minimum_gap_MHz = np.min(np.diff(disorder_theory.energies_MHz))
    maximum_energy_MHz = max(
        abs(float(energy)) for energy in disorder_theory.energies_MHz
    )
    disorder_cycle_step = min(
        4,
        max(1, int(0.45 / (
            maximum_energy_MHz * disorder_hardware.floquet_cycle_us
        ))),
    )
    disorder_sample_count = max(
        200,
        int(np.ceil(
            1.5 / (
                minimum_gap_MHz
                * disorder_cycle_step
                * disorder_hardware.floquet_cycle_us
            )
        )) + 1,
    )
    disorder_cycles = disorder_cycle_step * np.arange(
        disorder_sample_count,
        dtype=int,
    )
    disorder_cycle_chunks = [
        disorder_cycles[start:start + 50]
        for start in range(0, len(disorder_cycles), 50)
    ]
    disorder_fft_resolution_kHz = 1e3 / (
        len(disorder_cycles)
        * disorder_cycle_step
        * disorder_hardware.floquet_cycle_us
    )

    print("leaf onsite disorder (kHz):", dict(zip(
        [f"S{stor}" for stor in encspec_modes],
        np.round(disorder_onsite_kHz, 3),
    )))
    for decoder, encoder in disorder_pairs:
        print(f"<{decoder}|U(t)|{encoder}>")
    print(
        "minimum exact gap / FFT resolution (kHz):",
        1e3 * minimum_gap_MHz,
        disorder_fft_resolution_kHz,
    )
    print(
        f"{len(disorder_cycles)} cycle points, step {disorder_cycle_step}, "
        f"{len(disorder_cycle_chunks)} chunks per pair"
    )

    plt.figure(figsize=(11, 3.5), constrained_layout=True)
    plt.vlines(
        1e3 * disorder_theory.energies_MHz,
        0.,
        level_coverage,
    )
    plt.xlabel("exact eigenenergy E/h (kHz)")
    plt.ylabel("selected-pair cumulative visibility")
    plt.show()

    return {
        name: value
        for name, value in locals().items()
        if name.startswith("disorder_")
    }


def build_pairwise_batch(campaign, station, client, plan, use_queue=True):
    """Phase-correct the pairwise plan and build its batch and runner.

    Cell 320's build half. Submits nothing: the notebook calls
    `runner.execute`, so job submission stays a visible, separate step -- the
    same contract as `mbr_campaign.build_spectroscopy_batch`.

    Returns (batch, runner, cycle_branches).
    """
    disorder_pairs = plan["disorder_pairs"]

    disorder_decoders = [list(decoder) for decoder, encoder in disorder_pairs]
    disorder_encoders = [list(encoder) for decoder, encoder in disorder_pairs]
    cycle_branches = {tuple(decoder): 0 for decoder in disorder_decoders}
    correction = MBRPhaseCorrectionExperiment.phase_correction_from_calibration(
        plan["disorder_calibration"],
        cycle_branches=cycle_branches,
    )
    batch = MBRSpectrumExperiment.spectroscopy_batch(
        campaign.defaults,
        campaign.modes,
        disorder_encoders,
        plan["disorder_cycle_chunks"],
        correction.phase_by_occupation,
        detunings=plan["disorder_detunings_MHz"],
        sync_cycles=campaign.sync_cycles,
        reps=plan["disorder_reps"],
        final_occupations=disorder_decoders,
    )
    print(f"{len(batch.configs)} jobs")
    runner = CharacterizationRunner(
        station=station,
        ExptClass=campaign.EncSpec,
        ExptProgram=batch.program,
        default_expt_cfg=batch.default_expt_cfg,
        job_client=client,
        use_queue=use_queue,
        show=False,
    )
    return batch, runner, cycle_branches



# --------------------------------------------------------------------------
# 7-2 occupation-constrained disorder (cells 338-348).
# --------------------------------------------------------------------------



def _d72_select_channels(
    visibility,
    channel_count,
    required_support=1,
    numerical_floor=1e-12,
):
    """Maximize the weakest eigenlevel visibility at fixed channel count."""
    visibility = np.asarray(visibility, dtype=float)
    if visibility.ndim != 2 or not np.all(np.isfinite(visibility)):
        raise ValueError("visibility must be one finite channel-by-level matrix")
    candidate_count, level_count = visibility.shape
    if candidate_count == 0 or level_count == 0:
        raise ValueError("the constrained channel set is empty")

    selected_count = min(int(channel_count), candidate_count)
    required_support = int(required_support)
    if selected_count < 1:
        raise ValueError("d72_channel_count must be positive")
    if required_support < 1 or required_support > selected_count:
        raise ValueError(
            "d72_required_support must be between 1 and the selected count"
        )

    def solve(visibility_floor, objective=None):
        visible = (visibility >= visibility_floor).T.astype(float)
        if np.any(visible.sum(axis=1) < required_support):
            return None
        constraint_matrix = np.vstack([
            np.ones((1, candidate_count)),
            visible,
        ])
        result = milp(
            c=(
                np.zeros(candidate_count)
                if objective is None else objective
            ),
            integrality=np.ones(candidate_count, dtype=int),
            bounds=Bounds(
                np.zeros(candidate_count),
                np.ones(candidate_count),
            ),
            constraints=LinearConstraint(
                constraint_matrix,
                np.r_[
                    float(selected_count),
                    np.full(level_count, float(required_support)),
                ],
                np.r_[
                    float(selected_count),
                    np.full(level_count, np.inf),
                ],
            ),
            options={"disp": False},
        )
        if not result.success or result.x is None:
            return None
        rows = np.flatnonzero(result.x > 0.5)
        return rows if len(rows) == selected_count else None

    positive_floors = np.unique(
        visibility[visibility > numerical_floor]
    )
    if len(positive_floors) == 0:
        raise RuntimeError("all allowed channel visibilities are zero")

    lower = 0
    upper = len(positive_floors) - 1
    selected_rows = None
    best_floor = None
    while lower <= upper:
        middle = (lower + upper) // 2
        trial_floor = float(positive_floors[middle])
        trial_rows = solve(trial_floor)
        if trial_rows is None:
            upper = middle - 1
        else:
            selected_rows = trial_rows
            best_floor = trial_floor
            lower = middle + 1

    if selected_rows is None:
        strongest = visibility.max(axis=0)
        dark_levels = np.flatnonzero(strongest <= numerical_floor)
        raise RuntimeError(
            "allowed channels cannot positively cover every level "
            f"within the channel budget; dark levels={dark_levels.tolist()}"
        )

    utility = np.sum(
        np.minimum(visibility / best_floor, 1.0),
        axis=1,
    )
    deterministic_objective = (
        -utility
        + 1e-10
        * np.arange(candidate_count)
        / max(candidate_count, 1)
    )
    balanced_rows = solve(best_floor, deterministic_objective)
    if balanced_rows is not None:
        selected_rows = balanced_rows
    selected_rows = np.sort(selected_rows)

    coverage = visibility[selected_rows].max(axis=0)
    support = np.sum(
        visibility[selected_rows] >= best_floor,
        axis=0,
    )
    return AttrDict(dict(
        rows=selected_rows,
        visibility_floor=float(best_floor),
        coverage=coverage,
        support=support,
        selected_count=selected_count,
    ))


def _d72_disorder_direction(rng, leaf_count):
    """Return a zero-mean, unit-RMS direction when that is possible."""
    if leaf_count < 1:
        raise ValueError("at least one leaf mode is required")
    if leaf_count == 1:
        return np.asarray([rng.choice([-1.0, 1.0])])

    direction = rng.normal(size=leaf_count)
    direction -= np.mean(direction)
    rms = np.sqrt(np.mean(direction ** 2))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise RuntimeError("failed to draw a nonzero disorder direction")
    return direction / rms


def build_d72_plans(campaign, station, config=None):
    """Build the constrained theory plans and select channels (cell 340).

    Submits nothing. Returns a dict of the `d72_*` names the later steps need.
    """
    config = config or D72Config()
    # Cell knobs, from the config object.
    d72_N = config.N
    d72_realization_count = config.realization_count
    d72_disorder_strength_kHz = config.disorder_strength_kHz
    d72_master_seed = config.master_seed
    d72_max_occupation = config.max_occupation
    d72_forbidden_states = config.forbidden_states
    d72_channel_count = config.channel_count
    d72_required_support = config.required_support
    d72_allow_diagonal = config.allow_diagonal
    d72_allow_offdiagonal = config.allow_offdiagonal
    d72_min_acceptable_visibility = config.min_acceptable_visibility
    d72_max_time_us = config.max_time_us
    d72_min_time_points = config.min_time_points
    d72_nyquist_margin = config.nyquist_margin
    d72_step_autocalculate = config.step_autocalculate
    d72_cycle_step = config.cycle_step
    d72_cycle_chunk_points = config.cycle_chunk_points
    d72_reps = config.reps
    d72_batch_size = config.batch_size
    d72_self_kerr_kHz = config.self_kerr_kHz
    d72_branch_overrides = config.branch_overrides
    d72_calibration_job_ids = config.calibration_job_ids
    d72_theory_plot_realizations = config.theory_plot_realizations
    d72_match_tolerance_bins = config.match_tolerance_bins
    d72_mpm_minimum_consecutive_ranks = config.mpm_minimum_consecutive_ranks
    d72_mpm_minimum_supporting_rows = config.mpm_minimum_supporting_rows
    d72_mpm_merge_frequency_tolerance = config.mpm_merge_frequency_tolerance
    d72_mpm_calibration_sigma_multiplier = config.mpm_calibration_sigma_multiplier
    d72_mpm_frequency_tolerance_floor_kHz = config.mpm_frequency_tolerance_floor_kHz
    d72_mpm_dedup_frequency_tolerance_kHz = config.mpm_dedup_frequency_tolerance_kHz
    # Derived in the source, not a knob.
    d72_mpm_rank_sweep_extra = config.mpm_rank_sweep_extra
    D72EncSpec = campaign.EncSpec

    EncSpec = campaign.EncSpec
    encspec_modes = campaign.modes
    encspec_mode_labels = campaign.mode_labels
    encspec_sync_cycles = campaign.sync_cycles
    encspec_defaults = campaign.defaults
    encspec_calibrations = campaign.calibrations
    encspec_calibration_files = campaign.calibration_files
    encspec_calibration_job_ids = campaign.calibration_job_ids

    d72_modes = [int(mode) for mode in encspec_modes]
    d72_mode_labels = ["M1"] + [f"S{mode}" for mode in d72_modes]
    d72_mode_count = len(d72_mode_labels)
    if len(d72_modes) == 0 or len(set(d72_modes)) != len(d72_modes):
        raise ValueError("encspec_modes must contain distinct leaf modes")

    # Load one phase calibration without using any Section 7/7-1 object.
    if d72_calibration_job_ids is not None:
        d72_calibration_job_ids = [
            str(job_id) for job_id in d72_calibration_job_ids
        ]
        d72_calibration_files = [
            station.data_path
            / f"{job_id}_{D72EncSpec.__name__}.h5"
            for job_id in d72_calibration_job_ids
        ]
        d72_calibration = MBRPhaseCorrectionExperiment.from_job_files(
            d72_calibration_files,
            station=station,
        )
        d72_calibration.batch_job_ids = list(d72_calibration_job_ids)
    else:
        d72_common_calibrations = globals().get(
            "encspec_calibrations",
            {},
        )
        if d72_N in d72_common_calibrations:
            d72_calibration = d72_common_calibrations[d72_N]
        else:
            d72_common_files = globals().get(
                "encspec_calibration_files",
                {},
            )
            d72_common_job_ids = globals().get(
                "encspec_calibration_job_ids",
                {},
            )
            if d72_N not in d72_common_files:
                raise RuntimeError(
                    "provide d72_calibration_job_ids or load the common "
                    f"N={d72_N} phase calibration first"
                )
            d72_calibration = MBRPhaseCorrectionExperiment.from_job_files(
                d72_common_files[d72_N],
                station=station,
            )
            d72_calibration.batch_job_ids = list(
                d72_common_job_ids.get(d72_N, [])
            )

    if "phase_mod180" not in d72_calibration.data:
        d72_calibration.analyze()
    d72_calibration_data = d72_calibration.data

    if list(d72_calibration_data.mode_labels) != d72_mode_labels:
        raise ValueError(
            "the active calibration mode order does not match encspec_modes: "
            f"{list(d72_calibration_data.mode_labels)} != {d72_mode_labels}"
        )

    # Compare live and calibration timing with the same DAC and tProc clock rules,
    # then reuse the timing and couplings saved by the calibration.
    d72_saved_hardware = d72_calibration_data.hardware
    d72_calibration_ecfg = d72_calibration.batch_expts[0].cfg.expt
    d72_calibration_sync_cycles = int(
        d72_calibration_ecfg.get(
            "scramble_sync_cycles", encspec_sync_cycles
        )
    )
    d72_calibration_gauss_sigma = d72_calibration_ecfg.get(
        "floquet_gauss_sigma", None
    )
    d72_calibration_waveform = d72_calibration_ecfg.get(
        "floquet_waveform", None
    )
    d72_live_hardware = D72EncSpec.hardware_parameters(
        station,
        d72_modes,
        d72_calibration_sync_cycles,
        d72_calibration_gauss_sigma,
        d72_calibration_waveform,
    )
    d72_same_coupling_shape = (
        np.shape(d72_saved_hardware.couplings_MHz)
        == np.shape(d72_live_hardware.couplings_MHz)
    )
    d72_stale_hardware = (
        not np.isclose(
            d72_live_hardware.floquet_cycle_us,
            d72_saved_hardware.floquet_cycle_us,
            rtol=0.0,
            atol=5e-3,
        )
        or not d72_same_coupling_shape
        or (
            d72_same_coupling_shape
            and not np.allclose(
                d72_live_hardware.couplings_MHz,
                d72_saved_hardware.couplings_MHz,
                rtol=0.02,
                atol=0.0,
            )
        )
    )
    if d72_stale_hardware:
        raise RuntimeError(
            "live station Floquet timing/couplings do not match "
            "the phase calibration: "
            f"live T={d72_live_hardware.floquet_cycle_us:.9f} us, "
            f"cal T={d72_saved_hardware.floquet_cycle_us:.9f} us, "
            f"live J={np.asarray(d72_live_hardware.couplings_MHz)} MHz, "
            f"cal J={np.asarray(d72_saved_hardware.couplings_MHz)} MHz"
        )

    d72_hardware = deepcopy(d72_saved_hardware)
    # Kerr alone is intentionally taken from the current hardware config.
    d72_hardware.physical_kerr_MHz = float(
        d72_live_hardware.physical_kerr_MHz
    )
    d72_requested_self_kerr_kHz = d72_self_kerr_kHz
    if d72_requested_self_kerr_kHz is None:
        d72_resolved_self_kerr_MHz = float(
            d72_hardware.physical_kerr_MHz
        )
        d72_resolved_self_kerr_kHz = (
            1e3 * d72_resolved_self_kerr_MHz
        )
    else:
        d72_resolved_self_kerr_kHz = float(
            d72_requested_self_kerr_kHz
        )
        d72_resolved_self_kerr_MHz = (
            1e-3 * d72_resolved_self_kerr_kHz
        )

    d72_calibration_occupations = [
        tuple(map(int, occupation))
        for occupation in d72_calibration_data.occupations
    ]
    d72_branch_overrides = {
        tuple(map(int, occupation)): int(branch)
        for occupation, branch in d72_branch_overrides.items()
    }
    d72_calibration_branches = np.asarray([
        d72_branch_overrides.get(occupation, 0)
        for occupation in d72_calibration_occupations
    ])
    d72_correction = D72EncSpec.build_phase_correction(
        d72_calibration_data.occupations,
        d72_calibration_data.phase_mod180,
        d72_calibration_branches,
        d72_resolved_self_kerr_MHz,
        d72_hardware.floquet_cycle_us,
    )

    d72_access_by_occupation = {
        tuple(map(int, result.occupation)): abs(
            complex(result.complex_return[0])
        )
        for result in d72_calibration_data.results
    }
    d72_forbidden_states = {
        tuple(map(int, occupation))
        for occupation in d72_forbidden_states
    }
    for d72_state in d72_forbidden_states:
        if len(d72_state) != d72_mode_count:
            raise ValueError(
                f"forbidden state has the wrong mode count: {d72_state}"
            )
        if sum(d72_state) != d72_N:
            raise ValueError(
                f"forbidden state has the wrong photon number: {d72_state}"
            )

    if d72_max_occupation is not None:
        d72_max_occupation = int(d72_max_occupation)
        if d72_max_occupation < 0:
            raise ValueError("d72_max_occupation must be nonnegative or None")
    if not d72_allow_diagonal and not d72_allow_offdiagonal:
        raise ValueError("enable diagonal or off-diagonal channels")

    d72_plans = []
    d72_expected_dimension = comb(
        d72_N + len(d72_modes),
        d72_N,
    )

    for d72_realization in range(d72_realization_count):
        d72_seed = d72_master_seed + d72_realization
        d72_rng = np.random.default_rng(d72_seed)
        d72_direction = _d72_disorder_direction(
            d72_rng,
            len(d72_modes),
        )
        d72_target_onsite_MHz = (
            1e-3
            * d72_disorder_strength_kHz
            * d72_direction
        )
        # The pulse detuning has the opposite sign from the analyzer onsite.
        d72_pulse_detunings_MHz = -d72_target_onsite_MHz

        d72_probe_occupation = (
            [d72_N] + [0] * len(d72_modes)
        )
        d72_probe = AttrDict(dict(
            occupations=[d72_probe_occupation],
            final_occupations=[d72_probe_occupation.copy()],
            cycles=np.asarray([0, 1], dtype=int),
            A=np.ones((1, 2), dtype=complex),
        ))
        d72_theory = D72EncSpec.analyze_spectrum(
            d72_probe,
            d72_N,
            d72_pulse_detunings_MHz,
            d72_hardware.couplings_MHz,
            d72_hardware.floquet_cycle_us,
            d72_resolved_self_kerr_MHz,
        )
        if len(d72_theory.energies_MHz) != d72_expected_dimension:
            raise RuntimeError("unexpected fixed-N Hilbert-space dimension")

        d72_basis = [
            tuple(map(int, occupation))
            for occupation in d72_theory.fock_basis
        ]
        d72_basis_weights = np.asarray(
            d72_theory.basis_eigenstate_weights,
            dtype=float,
        )
        d72_allowed_rows = np.asarray([
            row
            for row, occupation in enumerate(d72_basis)
            if (
                occupation in d72_access_by_occupation
                and occupation not in d72_forbidden_states
                and (
                    d72_max_occupation is None
                    or max(occupation) <= d72_max_occupation
                )
            )
        ], dtype=int)
        if len(d72_allowed_rows) == 0:
            raise RuntimeError(
                f"r={d72_realization}: no calibrated occupation survives "
                "the state constraints"
            )

        d72_allowed_access = np.asarray([
            d72_access_by_occupation[d72_basis[row]]
            for row in d72_allowed_rows
        ], dtype=float)
        d72_access_scale = float(np.max(d72_allowed_access))
        if not np.isfinite(d72_access_scale) or d72_access_scale <= 0.0:
            raise RuntimeError(
                f"r={d72_realization}: allowed calibration access is zero"
            )
        d72_normalized_access = {
            int(row): float(access / d72_access_scale)
            for row, access in zip(
                d72_allowed_rows,
                d72_allowed_access,
            )
        }

        d72_candidate_pairs = []
        for local_decoder, decoder_row in enumerate(d72_allowed_rows):
            for encoder_row in d72_allowed_rows[local_decoder:]:
                is_diagonal = int(decoder_row) == int(encoder_row)
                if is_diagonal and not d72_allow_diagonal:
                    continue
                if not is_diagonal and not d72_allow_offdiagonal:
                    continue
                d72_candidate_pairs.append((
                    int(decoder_row),
                    int(encoder_row),
                ))
        if not d72_candidate_pairs:
            raise RuntimeError(
                f"r={d72_realization}: no channel survives the pair constraints"
            )

        # Geometric-mean endpoint access is a SPAM proxy, not a fitted
        # off-diagonal calibration.
        d72_candidate_visibility = np.asarray([
            np.sqrt(
                d72_normalized_access[decoder_row]
                * d72_normalized_access[encoder_row]
            )
            * np.sqrt(
                d72_basis_weights[decoder_row]
                * d72_basis_weights[encoder_row]
            )
            for decoder_row, encoder_row in d72_candidate_pairs
        ])
        d72_selection = _d72_select_channels(
            d72_candidate_visibility,
            d72_channel_count,
            required_support=d72_required_support,
        )
        d72_selected_row_pairs = [
            d72_candidate_pairs[row]
            for row in d72_selection.rows
        ]
        d72_selected_pairs = [
            (
                d72_basis[decoder_row],
                d72_basis[encoder_row],
            )
            for decoder_row, encoder_row in d72_selected_row_pairs
        ]
        d72_weakest_level = int(
            np.argmin(d72_selection.coverage)
        )

        d72_plan = AttrDict(dict(
            realization=d72_realization,
            seed=d72_seed,
            direction=d72_direction,
            target_onsite_MHz=d72_target_onsite_MHz,
            pulse_detunings_MHz=d72_pulse_detunings_MHz,
            theory=d72_theory,
            basis=d72_basis,
            allowed_rows=d72_allowed_rows,
            candidate_pairs=d72_candidate_pairs,
            candidate_visibility=d72_candidate_visibility,
            selected_pairs=d72_selected_pairs,
            selected_row_pairs=d72_selected_row_pairs,
            selection=d72_selection,
            weakest_level=d72_weakest_level,
            visibility_ok=(
                d72_selection.visibility_floor
                >= d72_min_acceptable_visibility
            ),
        ))
        d72_plans.append(d72_plan)

        print(
            f"r={d72_realization}: D={d72_expected_dimension}, "
            f"allowed states={len(d72_allowed_rows)}, "
            f"candidate channels={len(d72_candidate_pairs)}, "
            f"selected={len(d72_selected_pairs)}"
        )
        print(
            "  leaf onsite (kHz):",
            np.round(1e3 * d72_target_onsite_MHz, 3),
        )
        print(
            "  max-min visibility:",
            f"{d72_selection.visibility_floor:.6g}",
            "| weakest level:",
            d72_weakest_level,
        )
        if not d72_plan.visibility_ok:
            print(
                "  WARNING: visibility is below "
                f"d72_min_acceptable_visibility="
                f"{d72_min_acceptable_visibility:.3g}"
            )
        for pair_index, (decoder, encoder) in enumerate(
            d72_selected_pairs
        ):
            print(
                f"  c={pair_index}: "
                f"<{decoder}|U(qT_F)|{encoder}>"
            )

    return {
        name: value
        for name, value in locals().items()
        if name.startswith("d72_")
    }


def preview_d72_jobs(campaign, station, plan, config=None):
    """Choose the coherence-limited cycle grid, preview jobs, plot theory (cell 342).

    Submits nothing -- this is the guard before the batches go in.
    """
    config = config or D72Config()
    # Cell knobs, from the config object.
    d72_N = config.N
    d72_realization_count = config.realization_count
    d72_disorder_strength_kHz = config.disorder_strength_kHz
    d72_master_seed = config.master_seed
    d72_max_occupation = config.max_occupation
    d72_forbidden_states = config.forbidden_states
    d72_channel_count = config.channel_count
    d72_required_support = config.required_support
    d72_allow_diagonal = config.allow_diagonal
    d72_allow_offdiagonal = config.allow_offdiagonal
    d72_min_acceptable_visibility = config.min_acceptable_visibility
    d72_max_time_us = config.max_time_us
    d72_min_time_points = config.min_time_points
    d72_nyquist_margin = config.nyquist_margin
    d72_step_autocalculate = config.step_autocalculate
    d72_cycle_step = config.cycle_step
    d72_cycle_chunk_points = config.cycle_chunk_points
    d72_reps = config.reps
    d72_batch_size = config.batch_size
    d72_self_kerr_kHz = config.self_kerr_kHz
    d72_branch_overrides = config.branch_overrides
    d72_calibration_job_ids = config.calibration_job_ids
    d72_theory_plot_realizations = config.theory_plot_realizations
    d72_match_tolerance_bins = config.match_tolerance_bins
    d72_mpm_minimum_consecutive_ranks = config.mpm_minimum_consecutive_ranks
    d72_mpm_minimum_supporting_rows = config.mpm_minimum_supporting_rows
    d72_mpm_merge_frequency_tolerance = config.mpm_merge_frequency_tolerance
    d72_mpm_calibration_sigma_multiplier = config.mpm_calibration_sigma_multiplier
    d72_mpm_frequency_tolerance_floor_kHz = config.mpm_frequency_tolerance_floor_kHz
    d72_mpm_dedup_frequency_tolerance_kHz = config.mpm_dedup_frequency_tolerance_kHz
    # Derived in the source, not a knob.
    d72_mpm_rank_sweep_extra = config.mpm_rank_sweep_extra
    D72EncSpec = campaign.EncSpec
    d72_branch_overrides = plan["d72_branch_overrides"]
    d72_calibration = plan["d72_calibration"]
    d72_calibration_gauss_sigma = plan["d72_calibration_gauss_sigma"]
    d72_calibration_sync_cycles = plan["d72_calibration_sync_cycles"]
    d72_calibration_waveform = plan["d72_calibration_waveform"]
    d72_correction = plan["d72_correction"]
    d72_expected_dimension = plan["d72_expected_dimension"]
    d72_forbidden_states = plan["d72_forbidden_states"]
    d72_hardware = plan["d72_hardware"]
    d72_max_occupation = plan["d72_max_occupation"]
    d72_mode_count = plan["d72_mode_count"]
    d72_modes = plan["d72_modes"]
    d72_plans = plan["d72_plans"]
    d72_resolved_self_kerr_MHz = plan["d72_resolved_self_kerr_MHz"]
    d72_resolved_self_kerr_kHz = plan["d72_resolved_self_kerr_kHz"]

    EncSpec = campaign.EncSpec
    encspec_modes = campaign.modes
    encspec_mode_labels = campaign.mode_labels
    encspec_sync_cycles = campaign.sync_cycles
    encspec_defaults = campaign.defaults
    encspec_calibrations = campaign.calibrations
    encspec_calibration_files = campaign.calibration_files
    encspec_calibration_job_ids = campaign.calibration_job_ids

    d72_max_abs_energy_MHz = max(
        float(np.max(np.abs(plan.theory.energies_MHz)))
        for plan in d72_plans
    )
    d72_min_exact_gap_MHz = min(
        float(np.min(np.diff(plan.theory.energies_MHz)))
        for plan in d72_plans
        if len(plan.theory.energies_MHz) > 1
    )
    d72_max_cycle = int(np.floor(
        d72_max_time_us / d72_hardware.floquet_cycle_us
    ))
    if d72_max_cycle < 1:
        raise ValueError("d72_max_time_us is shorter than one Floquet cycle")

    if d72_max_abs_energy_MHz > 0.0:
        d72_step_nyquist = int(np.floor(
            1.0
            / (
                2.0
                * d72_nyquist_margin
                * d72_max_abs_energy_MHz
                * d72_hardware.floquet_cycle_us
            )
        ))
    else:
        d72_step_nyquist = d72_max_cycle
    if d72_step_nyquist < 1:
        raise RuntimeError(
            "one-cycle sampling already violates the requested Nyquist margin"
        )

    d72_step_points = max(
        1,
        d72_max_cycle
        // max(int(d72_min_time_points) - 1, 1),
    )
    if d72_step_autocalculate:
        d72_cycle_step = max(
            1,
            min(d72_step_nyquist, d72_step_points),
        )
    else:
        d72_cycle_step = int(d72_cycle_step)
        if d72_cycle_step < 1:
            raise ValueError("d72_cycle_step must be positive")
        if d72_cycle_step > d72_step_nyquist:
            raise RuntimeError(
                "d72_cycle_step violates the requested Nyquist margin"
            )
    d72_cycles = np.arange(
        0,
        d72_max_cycle + 1,
        d72_cycle_step,
        dtype=int,
    )
    if len(d72_cycles) < 2:
        raise RuntimeError("the coherence-limited cycle grid is too short")
    d72_cycle_chunks = [
        d72_cycles[start:start + int(d72_cycle_chunk_points)]
        for start in range(
            0,
            len(d72_cycles),
            int(d72_cycle_chunk_points),
        )
    ]
    d72_dt_us = (
        d72_cycle_step
        * d72_hardware.floquet_cycle_us
    )
    d72_nyquist_MHz = 1.0 / (2.0 * d72_dt_us)
    d72_fft_resolution_MHz = 1.0 / (
        len(d72_cycles) * d72_dt_us
    )

    d72_total_jobs = 0
    d72_total_sweep_settings = 0
    for plan in d72_plans:
        plan.decoders = [
            list(decoder)
            for decoder, encoder in plan.selected_pairs
        ]
        plan.encoders = [
            list(encoder)
            for decoder, encoder in plan.selected_pairs
        ]
        plan.cycle_branches = {
            tuple(decoder): int(
                d72_branch_overrides.get(tuple(decoder), 0)
            )
            for decoder in plan.decoders
        }

        plan.defaults = deepcopy(encspec_defaults)
        if d72_calibration_gauss_sigma is not None:
            plan.defaults.floquet_gauss_sigma = (
                d72_calibration_gauss_sigma
            )
        if d72_calibration_waveform is not None:
            plan.defaults.floquet_waveform = (
                d72_calibration_waveform
            )
        plan.defaults.update(dict(
            d72_realization=plan.realization,
            d72_seed=plan.seed,
            d72_disorder_strength_kHz=d72_disorder_strength_kHz,
            d72_target_onsite_MHz=(
                plan.target_onsite_MHz.tolist()
            ),
            d72_max_occupation=d72_max_occupation,
            d72_forbidden_states=[
                list(state) for state in sorted(d72_forbidden_states)
            ],
            d72_selected_pairs=[
                [list(decoder), list(encoder)]
                for decoder, encoder in plan.selected_pairs
            ],
            d72_visibility_floor=(
                plan.selection.visibility_floor
            ),
            d72_self_kerr_kHz=(
                d72_resolved_self_kerr_kHz
            ),
        ))
        plan.batch = MBRSpectrumExperiment.spectroscopy_batch(
            plan.defaults,
            d72_modes,
            plan.encoders,
            d72_cycle_chunks,
            d72_correction.phase_by_occupation,
            detunings=plan.pulse_detunings_MHz.tolist(),
            sync_cycles=d72_calibration_sync_cycles,
            reps=d72_reps,
            final_occupations=plan.decoders,
        )
        plan.job_count = len(plan.batch.configs)
        d72_total_jobs += plan.job_count
        plan.sweep_setting_count = 0
        for config in plan.batch.configs:
            if "offdiag_cycles" in config:
                plan.sweep_setting_count += (
                    4 * len(config["offdiag_cycles"])
                )
            else:
                plan.sweep_setting_count += (
                    2 * len(config["floquet_cycles"])
                )
        d72_total_sweep_settings += plan.sweep_setting_count

        d72_theory_reconstruction = AttrDict(dict(
            occupations=[
                list(encoder)
                for decoder, encoder in plan.selected_pairs
            ],
            final_occupations=[
                list(decoder)
                for decoder, encoder in plan.selected_pairs
            ],
            cycles=d72_cycles.copy(),
            A=np.zeros(
                (
                    len(plan.selected_pairs),
                    len(d72_cycles),
                ),
                dtype=complex,
            ),
        ))
        plan.selected_theory = D72EncSpec.analyze_spectrum(
            d72_theory_reconstruction,
            d72_N,
            plan.pulse_detunings_MHz,
            d72_hardware.couplings_MHz,
            d72_hardware.floquet_cycle_us,
            d72_resolved_self_kerr_MHz,
        )

    print(
        f"7-2: leaves={len(d72_modes)}, total modes={d72_mode_count}, "
        f"N={d72_N}, levels={d72_expected_dimension}"
    )
    print(
        f"realizations={len(d72_plans)}, "
        f"selected channels/realization="
        f"{[len(plan.selected_pairs) for plan in d72_plans]}"
    )
    print(
        f"cycles={d72_cycles[0]}:{d72_cycles[-1]}:"
        f"{d72_cycle_step}, points={len(d72_cycles)}, "
        f"chunks={len(d72_cycle_chunks)}, "
        f"max time={d72_cycles[-1] * d72_hardware.floquet_cycle_us:.3f} us"
    )
    print(
        f"Nyquist={1e3 * d72_nyquist_MHz:.3f} kHz, "
        f"FFT resolution={1e3 * d72_fft_resolution_MHz:.3f} kHz, "
        f"minimum exact gap={1e3 * d72_min_exact_gap_MHz:.3f} kHz"
    )
    print(
        f"exact queue jobs={d72_total_jobs}, "
        f"raw phase/preparation sweep settings={d72_total_sweep_settings}"
    )
    print(
        f"signed M1 self-Kerr="
        f"{d72_resolved_self_kerr_kHz:.3f} kHz"
    )

    for plan in d72_plans[:int(d72_theory_plot_realizations)]:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 4.2),
            constrained_layout=True,
        )
        energies_kHz = 1e3 * np.asarray(
            plan.theory.energies_MHz,
        )
        axes[0].vlines(
            energies_kHz,
            0.0,
            plan.selection.coverage,
            color="tab:blue",
        )
        axes[0].axhline(
            d72_min_acceptable_visibility,
            color="tab:red",
            linestyle="--",
            label="submit guard",
        )
        axes[0].set(
            xlabel="exact eigenenergy E/h (kHz)",
            ylabel="best selected-channel visibility",
            title=(
                f"r={plan.realization}: constrained level coverage"
            ),
        )
        axes[0].legend()

        expectation = np.abs(
            plan.selected_theory.theory_A
        )
        image = axes[1].imshow(
            expectation,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                d72_cycles[0] - 0.5 * d72_cycle_step,
                d72_cycles[-1] + 0.5 * d72_cycle_step,
                -0.5,
                len(plan.selected_pairs) - 0.5,
            ],
        )
        axes[1].set(
            xlabel="Floquet cycle q",
            ylabel="selected channel index",
            title=r"theory $|\langle d|e^{-iHqT_F}|e\rangle|$",
        )
        fig.colorbar(
            image,
            ax=axes[1],
            label="expected magnitude",
        )
        plt.show()

    d72_calibration_ids = tuple(map(
        str,
        getattr(d72_calibration, "batch_job_ids", []),
    ))
    d72_campaign_key = (
        int(d72_N),
        int(d72_realization_count),
        float(d72_disorder_strength_kHz),
        int(d72_master_seed),
        None if d72_max_occupation is None
        else int(d72_max_occupation),
        tuple(sorted(d72_forbidden_states)),
        int(d72_channel_count),
        int(d72_required_support),
        bool(d72_allow_diagonal),
        bool(d72_allow_offdiagonal),
        float(d72_min_acceptable_visibility),
        float(d72_resolved_self_kerr_kHz),
        float(d72_hardware.floquet_cycle_us),
        tuple(map(float, d72_hardware.couplings_MHz)),
        int(d72_calibration_sync_cycles),
        tuple(sorted(
            (occupation, int(branch))
            for occupation, branch
            in d72_branch_overrides.items()
        )),
        str(getattr(station, "hardware_config_file", "")),
        repr(json_plain(dict(encspec_defaults))),
        int(d72_reps),
        tuple(map(int, d72_cycles)),
        tuple(map(int, d72_modes)),
        tuple(
            (
                int(plan.realization),
                tuple(plan.selected_pairs),
            )
            for plan in d72_plans
        ),
        d72_calibration_ids,
    )
    d72_campaigns = globals().get("d72_campaigns", {})
    d72_records = d72_campaigns.setdefault(
        d72_campaign_key,
        {},
    )

    return d72_records


def check_d72_visibility(plan, min_acceptable_visibility=None):
    """Block 7-2 submission when a realization's theory coverage is too weak.

    Cell 344's guard. The submission loop it guarded is in the notebook, so
    this is a check the notebook calls before entering that loop. Raises
    RuntimeError naming the offending realizations; returns None otherwise.
    """
    low_visibility = [
        (realization_plan.realization,
         realization_plan.selection.visibility_floor)
        for realization_plan in plan["d72_plans"]
        if not realization_plan.visibility_ok
    ]
    if low_visibility:
        raise RuntimeError(
            "7-2 submission blocked by weak theory coverage: "
            f"{low_visibility}; adjust the state constraints, "
            "channel count, or d72_min_acceptable_visibility"
        )


def analyze_d72(plan, d72_records, config=None, d72_analysis_error="raise"):
    """MPM rank stability, calibration merge and theory matching (cell 346).

    Submits nothing.
    """
    config = config or D72Config()
    # Cell knobs, from the config object.
    d72_N = config.N
    d72_realization_count = config.realization_count
    d72_disorder_strength_kHz = config.disorder_strength_kHz
    d72_master_seed = config.master_seed
    d72_max_occupation = config.max_occupation
    d72_forbidden_states = config.forbidden_states
    d72_channel_count = config.channel_count
    d72_required_support = config.required_support
    d72_allow_diagonal = config.allow_diagonal
    d72_allow_offdiagonal = config.allow_offdiagonal
    d72_min_acceptable_visibility = config.min_acceptable_visibility
    d72_max_time_us = config.max_time_us
    d72_min_time_points = config.min_time_points
    d72_nyquist_margin = config.nyquist_margin
    d72_step_autocalculate = config.step_autocalculate
    d72_cycle_step = config.cycle_step
    d72_cycle_chunk_points = config.cycle_chunk_points
    d72_reps = config.reps
    d72_batch_size = config.batch_size
    d72_self_kerr_kHz = config.self_kerr_kHz
    d72_branch_overrides = config.branch_overrides
    d72_calibration_job_ids = config.calibration_job_ids
    d72_theory_plot_realizations = config.theory_plot_realizations
    d72_match_tolerance_bins = config.match_tolerance_bins
    d72_mpm_minimum_consecutive_ranks = config.mpm_minimum_consecutive_ranks
    d72_mpm_minimum_supporting_rows = config.mpm_minimum_supporting_rows
    d72_mpm_merge_frequency_tolerance = config.mpm_merge_frequency_tolerance
    d72_mpm_calibration_sigma_multiplier = config.mpm_calibration_sigma_multiplier
    d72_mpm_frequency_tolerance_floor_kHz = config.mpm_frequency_tolerance_floor_kHz
    d72_mpm_dedup_frequency_tolerance_kHz = config.mpm_dedup_frequency_tolerance_kHz
    # Derived in the source, not a knob.
    d72_mpm_rank_sweep_extra = config.mpm_rank_sweep_extra
    d72_calibration = plan["d72_calibration"]
    d72_expected_dimension = plan["d72_expected_dimension"]
    d72_plans = plan["d72_plans"]
    d72_resolved_self_kerr_MHz = plan["d72_resolved_self_kerr_MHz"]

    for d72_realization, record in sorted(d72_records.items()):
        plan = record.plan
        try:
            d72_data = record.expt.analyze(
                calibration=d72_calibration,
                cycle_branches=record.cycle_branches,
                phase_frame="manual_kerr",
                manual_kerr_MHz=d72_resolved_self_kerr_MHz,
                spectrum_method="mpm",
                fft_window="raw",
                zero_padding=1,
                mpm_requested_max_modes=d72_expected_dimension,
                mpm_match_decay=False,
                mpm_minimum_consecutive_ranks=(
                    d72_mpm_minimum_consecutive_ranks
                ),
                mpm_rank_sweep_extra=d72_mpm_rank_sweep_extra,
                mpm_minimum_supporting_rows=(
                    d72_mpm_minimum_supporting_rows
                ),
                mpm_merge_frequency_tolerance_bins=(
                    d72_mpm_merge_frequency_tolerance
                ),
                mpm_calibration_sigma_multiplier=(
                    d72_mpm_calibration_sigma_multiplier
                ),
                mpm_merge_frequency_tolerance_floor_kHz=(
                    d72_mpm_frequency_tolerance_floor_kHz
                ),
                mpm_dedup_frequency_tolerance_MHz=(
                    1e-3 * d72_mpm_dedup_frequency_tolerance_kHz
                ),
            )
        except Exception as d72_analysis_error:
            record.analysis_error = d72_analysis_error
            print(
                f"r={d72_realization} analysis failed: "
                f"{d72_analysis_error}"
            )
            continue

        record.data = d72_data
        if "analysis_error" in record:
            record.pop("analysis_error")

        d72_sampling_frequency_MHz = float(
            d72_data.matrix_pencil.sampling.sampling_frequency_MHz
        )
        d72_principal_nyquist_MHz = (
            0.5 * d72_sampling_frequency_MHz
        )
        d72_poles_MHz = np.asarray(
            d72_data.matrix_pencil.selected_frequencies_MHz,
            dtype=float,
        )
        d72_poles_MHz = (
            (
                d72_poles_MHz
                + d72_principal_nyquist_MHz
            )
    #         % d72_sampling_frequency_MHz
            - d72_principal_nyquist_MHz
        )
        d72_theory_levels_MHz = np.asarray(
            plan.theory.energies_MHz,
            dtype=float,
        )

        d72_error_MHz = np.abs(
            d72_poles_MHz[:, None]
            - d72_theory_levels_MHz[None, :]
        )
        d72_tolerance_MHz = (
            d72_match_tolerance_bins
            * float(d72_data.spectrum.fft_resolution_MHz)
        )
        d72_inside_tolerance = (
            d72_error_MHz <= d72_tolerance_MHz
        )
        if len(d72_poles_MHz):
            d72_penalty = (
                (d72_expected_dimension + 1)
                * (float(np.max(d72_error_MHz)) + 1.0)
            )
            d72_match_rows, d72_match_columns = (
                linear_sum_assignment(
                    np.where(
                        d72_inside_tolerance,
                        d72_error_MHz,
                        d72_penalty,
                    )
                )
            )
            d72_valid = d72_inside_tolerance[
                d72_match_rows,
                d72_match_columns,
            ]
            d72_matched_pole_rows = d72_match_rows[d72_valid]
            d72_matched_theory_rows = (
                d72_match_columns[d72_valid]
            )
        else:
            d72_matched_pole_rows = np.asarray([], dtype=int)
            d72_matched_theory_rows = np.asarray([], dtype=int)

        d72_missing_theory_rows = np.setdiff1d(
            np.arange(d72_expected_dimension),
            d72_matched_theory_rows,
        )
        d72_spurious_pole_rows = np.setdiff1d(
            np.arange(len(d72_poles_MHz)),
            d72_matched_pole_rows,
        )
        d72_complete = bool(
            len(d72_poles_MHz) == d72_expected_dimension
            and len(d72_matched_theory_rows)
            == d72_expected_dimension
        )
        d72_match_mae_MHz = (
            float(np.mean(d72_error_MHz[
                d72_matched_pole_rows,
                d72_matched_theory_rows,
            ]))
            if len(d72_matched_pole_rows)
            else np.nan
        )

        record.mpm_poles_MHz = d72_poles_MHz
        record.theory_match = AttrDict(dict(
            complete=d72_complete,
            tolerance_MHz=d72_tolerance_MHz,
            matched_count=len(d72_matched_theory_rows),
            missing_theory_MHz=(
                d72_theory_levels_MHz[d72_missing_theory_rows]
            ),
            spurious_poles_MHz=(
                d72_poles_MHz[d72_spurious_pole_rows]
            ),
            mae_MHz=d72_match_mae_MHz,
        ))

        print(
            f"r={d72_realization}: poles={len(d72_poles_MHz)}, "
            f"matched={len(d72_matched_theory_rows)}/"
            f"{d72_expected_dimension}, "
            f"missing={len(d72_missing_theory_rows)}, "
            f"spurious={len(d72_spurious_pole_rows)}, "
            f"MAE={1e3 * d72_match_mae_MHz:.3f} kHz"
        )
        if len(d72_missing_theory_rows):
            print(
                "  missing theory (kHz):",
                np.round(
                    1e3
                    * d72_theory_levels_MHz[
                        d72_missing_theory_rows
                    ],
                    3,
                ),
            )
        if len(d72_spurious_pole_rows):
            print(
                "  unmatched poles (kHz):",
                np.round(
                    1e3
                    * d72_poles_MHz[d72_spurious_pole_rows],
                    3,
                ),
            )

        record.expt.display(
            data=d72_data,
            spectrum_method="fft",
        )
        record.expt.display(
            data=d72_data,
            spectrum_method="mpm",
        )
        plt.show()

    d72_completed_matches = [
        realization
        for realization, record in sorted(d72_records.items())
        if (
            record.get("theory_match") is not None
            and record.theory_match.complete
        )
    ]
    print(
        "7-2 complete theory matches:",
        f"{len(d72_completed_matches)}/{len(d72_plans)}",
        d72_completed_matches,
    )

    return d72_records
