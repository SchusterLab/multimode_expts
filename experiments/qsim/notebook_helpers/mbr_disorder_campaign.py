"""Planning, submission and analysis for the disorder spectroscopy campaigns.

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
class DiagDisorderConfig:
    """Cell 323's sixteen `diag_disorder_*` knobs, with its values."""

    N: int = 3
    realization_count: int = 20
    strength_kHz: float = 50.0
    master_seed: int = 20260816
    selected_states: int = 10
    max_cycle: int = 200
    min_time_points: int = 100
    nyquist_margin: float = 1.35
    reps: int = 1200
    batch_size: int = 2
    edge_fraction: float = 0.10
    gap_ratio_bins: int = 15
    match_tolerance_bins: float = 1.5
    require_complete_match: bool = False

    # None uses best_self_kerr_kHz from section 3-1 when available, then falls
    # back to the signed Kerr saved with the phase-calibration jobs.
    self_kerr_kHz: Optional[float] = None

    # Add only occupations whose calibration phase needs a nonzero 180-deg
    # branch.
    branch_overrides: dict = field(default_factory=dict)


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
# 7-1 diagonal disorder (cells 323-334).
# --------------------------------------------------------------------------



def _diag_disorder_select_rows(weights, number_to_select):
    """Historical max-min LDOS coverage selection for diagonal traces."""
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


def build_diag_disorder_plans(campaign, station, config=None,
                              best_self_kerr_kHz=None):
    """Build the theory-selected diagonal-disorder plans (cell 325).

    Submits nothing. `best_self_kerr_kHz` is section 3-1's fitted Kerr, which
    the source read from the live kernel; pass None to fall back to the signed
    Kerr saved with the phase-calibration jobs, as the source also did.

    Returns a dict of the `diag_disorder_*` names the later steps need.
    """
    config = config or DiagDisorderConfig()
    # Cell knobs, from the config object.
    diag_disorder_N = config.N
    diag_disorder_realization_count = config.realization_count
    diag_disorder_strength_kHz = config.strength_kHz
    diag_disorder_master_seed = config.master_seed
    diag_disorder_selected_states = config.selected_states
    diag_disorder_max_cycle = config.max_cycle
    diag_disorder_min_time_points = config.min_time_points
    diag_disorder_nyquist_margin = config.nyquist_margin
    diag_disorder_reps = config.reps
    diag_disorder_batch_size = config.batch_size
    diag_disorder_edge_fraction = config.edge_fraction
    diag_disorder_gap_ratio_bins = config.gap_ratio_bins
    diag_disorder_match_tolerance_bins = config.match_tolerance_bins
    diag_disorder_require_complete_match = config.require_complete_match
    diag_disorder_self_kerr_kHz = config.self_kerr_kHz
    diag_disorder_branch_overrides = config.branch_overrides

    EncSpec = campaign.EncSpec
    encspec_modes = campaign.modes
    encspec_mode_labels = campaign.mode_labels
    encspec_sync_cycles = campaign.sync_cycles
    encspec_defaults = campaign.defaults
    encspec_calibrations = campaign.calibrations
    encspec_calibration_files = campaign.calibration_files
    encspec_calibration_job_ids = campaign.calibration_job_ids

    if diag_disorder_N not in encspec_calibrations:
        diag_disorder_calibration = MBRPhaseCorrectionExperiment.from_job_files(
            encspec_calibration_files[diag_disorder_N],
            station=station,
        )
        diag_disorder_calibration.batch_job_ids = (
            encspec_calibration_job_ids[diag_disorder_N]
        )
        diag_disorder_calibration.analyze()
        encspec_calibrations[diag_disorder_N] = diag_disorder_calibration

    diag_disorder_calibration = encspec_calibrations[diag_disorder_N]
    diag_disorder_calibration_data = diag_disorder_calibration.data
    diag_disorder_hardware = diag_disorder_calibration_data.hardware

    if diag_disorder_self_kerr_kHz is None:
        if "best_self_kerr_kHz" in globals():
            diag_disorder_self_kerr_kHz = float(best_self_kerr_kHz)
            diag_disorder_kerr_source = "section 3-1 fit"
        else:
            diag_disorder_self_kerr_kHz = 1e3 * float(
                diag_disorder_hardware.physical_kerr_MHz
            )
            diag_disorder_kerr_source = "phase-calibration jobs"
    else:
        diag_disorder_self_kerr_kHz = float(diag_disorder_self_kerr_kHz)
        diag_disorder_kerr_source = "manual value in this cell"

    diag_disorder_self_kerr_MHz = 1e-3 * diag_disorder_self_kerr_kHz
    diag_disorder_mode_count = len(encspec_modes) + 1
    diag_disorder_plans = []

    for diag_realization in range(diag_disorder_realization_count):
        diag_seed = diag_disorder_master_seed + diag_realization
        diag_rng = np.random.default_rng(diag_seed)
        diag_direction = diag_rng.normal(size=len(encspec_modes))
        diag_direction -= np.mean(diag_direction)
        diag_direction /= np.linalg.norm(diag_direction)

        # The analyzer Hamiltonian uses onsite=-detunings.
        diag_target_onsite_MHz = (
            1e-3 * diag_disorder_strength_kHz * diag_direction
        )
        diag_pulse_detunings_MHz = -diag_target_onsite_MHz
        diag_probe_occupation = [
            diag_disorder_N,
            *([0] * (diag_disorder_mode_count - 1)),
        ]
        diag_probe = AttrDict(dict(
            occupations=[diag_probe_occupation],
            final_occupations=[diag_probe_occupation],
            cycles=np.array([0, 1]),
            A=np.ones((1, 2), dtype=complex),
        ))
        diag_theory = EncSpec.analyze_spectrum(
            diag_probe,
            diag_disorder_N,
            diag_pulse_detunings_MHz,
            diag_disorder_hardware.couplings_MHz,
            diag_disorder_hardware.floquet_cycle_us,
            diag_disorder_self_kerr_MHz,
        )
        diag_rows, diag_floor, diag_coverage = (
            _diag_disorder_select_rows(
                diag_theory.basis_eigenstate_weights,
                diag_disorder_selected_states,
            )
        )
        diag_occupations = [
            list(diag_theory.fock_basis[row]) for row in diag_rows
        ]
        diag_disorder_plans.append(dict(
            realization=diag_realization,
            seed=diag_seed,
            direction=diag_direction,
            target_onsite_MHz=diag_target_onsite_MHz,
            pulse_detunings_MHz=diag_pulse_detunings_MHz,
            theory=diag_theory,
            occupations=diag_occupations,
            selection_floor=float(diag_floor),
            coverage=diag_coverage,
        ))

    diag_disorder_dimension = len(
        diag_disorder_plans[0]["theory"].energies_MHz
    )
    diag_disorder_max_abs_energy_MHz = max(
        np.max(np.abs(plan["theory"].energies_MHz))
        for plan in diag_disorder_plans
    )
    diag_disorder_step_nyquist = int(np.floor(
        1.0 / (
            2.0
            * diag_disorder_nyquist_margin
            * diag_disorder_max_abs_energy_MHz
            * diag_disorder_hardware.floquet_cycle_us
        )
    ))
    diag_disorder_step_points = max(
        1,
        (diag_disorder_max_cycle - 1)
        // (diag_disorder_min_time_points - 1),
    )
    diag_disorder_cycle_step = max(
        1,
        min(diag_disorder_step_nyquist, diag_disorder_step_points),
    )
    if diag_disorder_cycle_step > 1 and diag_disorder_cycle_step % 2:
        diag_disorder_cycle_step -= 1

    diag_disorder_cycles = np.arange(
        0,
        diag_disorder_max_cycle,
        diag_disorder_cycle_step,
        dtype=int,
    )
    diag_disorder_cycle_chunks = [diag_disorder_cycles]
    diag_disorder_dt_us = (
        diag_disorder_cycle_step
        * diag_disorder_hardware.floquet_cycle_us
    )
    diag_disorder_nyquist_MHz = 1.0 / (2.0 * diag_disorder_dt_us)
    diag_disorder_fft_resolution_MHz = 1.0 / (
        len(diag_disorder_cycles) * diag_disorder_dt_us
    )
    if diag_disorder_nyquist_MHz < (
        diag_disorder_nyquist_margin
        * diag_disorder_max_abs_energy_MHz
    ):
        raise RuntimeError(
            "diagonal-disorder cycle step violates the requested "
            "Nyquist margin"
        )

    diag_calibration_occupations = [
        tuple(occupation)
        for occupation in diag_disorder_calibration_data.occupations
    ]
    diag_calibration_branches = np.asarray([
        int(diag_disorder_branch_overrides.get(occupation, 0))
        for occupation in diag_calibration_occupations
    ])
    diag_disorder_correction = EncSpec.build_phase_correction(
        diag_disorder_calibration_data.occupations,
        diag_disorder_calibration_data.phase_mod180,
        diag_calibration_branches,
        diag_disorder_self_kerr_MHz,
        diag_disorder_hardware.floquet_cycle_us,
    )

    diag_disorder_total_jobs = (
        diag_disorder_realization_count
        * 2
        * diag_disorder_selected_states
        * len(diag_disorder_cycle_chunks)
    )
    diag_disorder_estimated_hours = (
        diag_disorder_total_jobs
        * len(diag_disorder_cycles)
        * 3.76
        * (diag_disorder_reps / 1200.0)
        / 3600.0
    )

    print(
        f"diagonal disorder: N={diag_disorder_N}, "
        f"realizations={diag_disorder_realization_count}, "
        f"jobs={diag_disorder_total_jobs}"
    )
    print(
        f"cycles=0:{diag_disorder_max_cycle}:{diag_disorder_cycle_step}, "
        f"points={len(diag_disorder_cycles)}, "
        f"Nyquist={1e3 * diag_disorder_nyquist_MHz:.3f} kHz, "
        f"FFT resolution={1e3 * diag_disorder_fft_resolution_MHz:.3f} kHz"
    )
    print(
        f"signed M1 self-Kerr={diag_disorder_self_kerr_kHz:.3f} kHz "
        f"({diag_disorder_kerr_source})"
    )
    print(f"estimated single-worker time={diag_disorder_estimated_hours:.1f} h")
    for plan in diag_disorder_plans:
        print(
            f"r={plan['realization']} seed={plan['seed']} | "
            f"leaf onsite={np.round(1e3 * plan['target_onsite_MHz'], 3)} kHz"
        )
        print("  diagonal occupations:", [
            tuple(occupation) for occupation in plan["occupations"]
        ])

    # Cache one campaign per parameter/calibration signature in this kernel.
    diag_disorder_campaigns = globals().get("diag_disorder_campaigns", {})
    diag_disorder_campaign_key = (
        int(diag_disorder_N),
        int(diag_disorder_realization_count),
        float(diag_disorder_strength_kHz),
        int(diag_disorder_master_seed),
        int(diag_disorder_selected_states),
        int(diag_disorder_max_cycle),
        int(diag_disorder_cycle_step),
        int(diag_disorder_reps),
        float(diag_disorder_self_kerr_kHz),
        tuple(sorted(
            (tuple(map(int, occupation)), int(branch))
            for occupation, branch
            in diag_disorder_branch_overrides.items()
        )),
        tuple(
            tuple(map(int, occupation))
            for plan in diag_disorder_plans
            for occupation in plan["occupations"]
        ),
        tuple(map(int, encspec_modes)),
        tuple(map(str, getattr(
            diag_disorder_calibration, "batch_job_ids", []
        ))),
    )
    diag_disorder_records = diag_disorder_campaigns.setdefault(
        diag_disorder_campaign_key,
        {},
    )

    return {
        name: value
        for name, value in locals().items()
        if name.startswith("diag_disorder_")
    }


def build_diag_realization_batch(campaign, station, client, plan, config,
                                 realization_plan, use_queue=True):
    """Build one diagonal-disorder realization's batch and runner.

    Cell 327's build half, for a single realization. Submits nothing: the
    notebook loops over `plan["diag_disorder_plans"]` and calls
    `runner.execute` itself, so the acquisition loop and its batch size stay
    visible.

    Returns (batch, runner, cycle_branches).
    """
    diag_defaults = deepcopy(campaign.defaults)
    diag_defaults.update(dict(
        diagonal_disorder_realization=realization_plan["realization"],
        diagonal_disorder_seed=realization_plan["seed"],
        diagonal_disorder_strength_kHz=config.strength_kHz,
        diagonal_disorder_direction=realization_plan["direction"].tolist(),
        diagonal_disorder_target_onsite_MHz=(
            realization_plan["target_onsite_MHz"].tolist()
        ),
        diagonal_disorder_selected_occupations=realization_plan["occupations"],
        diagonal_disorder_self_kerr_kHz=config.self_kerr_kHz,
    ))
    cycle_branches = {
        tuple(occupation): int(
            config.branch_overrides.get(tuple(occupation), 0)
        )
        for occupation in realization_plan["occupations"]
    }
    batch = MBRSpectrumExperiment.spectroscopy_batch(
        diag_defaults,
        campaign.modes,
        realization_plan["occupations"],
        plan["diag_disorder_cycle_chunks"],
        plan["diag_disorder_correction"].phase_by_occupation,
        detunings=realization_plan["pulse_detunings_MHz"].tolist(),
        sync_cycles=campaign.sync_cycles,
        reps=config.reps,
        final_occupations=realization_plan["occupations"],
    )
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


def analyze_diag_disorder(plan, config=None, diag_analysis_error="raise"):
    """Matrix-Pencil analysis and theory matching per realization (cell 332).

    Submits nothing. `diag_analysis_error` keeps the source's switch for
    whether one failed realization aborts the pool or is skipped.
    """
    config = config or DiagDisorderConfig()
    # Cell knobs, from the config object.
    diag_disorder_N = config.N
    diag_disorder_realization_count = config.realization_count
    diag_disorder_strength_kHz = config.strength_kHz
    diag_disorder_master_seed = config.master_seed
    diag_disorder_selected_states = config.selected_states
    diag_disorder_max_cycle = config.max_cycle
    diag_disorder_min_time_points = config.min_time_points
    diag_disorder_nyquist_margin = config.nyquist_margin
    diag_disorder_reps = config.reps
    diag_disorder_batch_size = config.batch_size
    diag_disorder_edge_fraction = config.edge_fraction
    diag_disorder_gap_ratio_bins = config.gap_ratio_bins
    diag_disorder_match_tolerance_bins = config.match_tolerance_bins
    diag_disorder_require_complete_match = config.require_complete_match
    diag_disorder_self_kerr_kHz = config.self_kerr_kHz
    diag_disorder_branch_overrides = config.branch_overrides
    diag_disorder_calibration = plan["diag_disorder_calibration"]
    diag_disorder_dimension = plan["diag_disorder_dimension"]
    diag_disorder_records = plan["diag_disorder_records"]
    diag_disorder_self_kerr_MHz = plan["diag_disorder_self_kerr_MHz"]

    # Analyze only after all hardware acquisition is complete.
    for diag_realization, record in sorted(diag_disorder_records.items()):
        try:
            diag_data = record["expt"].analyze(
                calibration=diag_disorder_calibration,
                occupations=record["plan"]["occupations"],
                cycle_branches=record["cycle_branches"],
                phase_frame="manual_kerr",
                manual_kerr_MHz=diag_disorder_self_kerr_MHz,
                spectrum_method="mpm",
                fft_window="raw",
                zero_padding=1,
                mpm_requested_max_modes=diag_disorder_dimension,
                mpm_match_decay=False,
                mpm_track_frequency_tolerance_bins=1.0,
                mpm_merge_frequency_tolerance_bins=0.10,#0.20
                mpm_dedup_frequency_tolerance_bins=0.10, #0.20
                mpm_minimum_supporting_rows=1,
                # mpm_rank_sweep_extra=2,
            )
        except Exception as diag_analysis_error:
            record["analysis_error"] = diag_analysis_error
            print(f"r={diag_realization} analysis failed: {diag_analysis_error}")
            continue

        record["data"] = diag_data
        record.pop("analysis_error", None)

        # Historical diagnostic: match principal-interval MPM poles to theory.
        diag_sampling_frequency_MHz = float(
            diag_data.matrix_pencil.sampling.sampling_frequency_MHz
        )
        diag_principal_nyquist_MHz = 0.5 * diag_sampling_frequency_MHz
        diag_poles_MHz = np.asarray(
            diag_data.matrix_pencil.selected_frequencies_MHz,
            dtype=float,
        )
        diag_poles_MHz = (
            (diag_poles_MHz + diag_principal_nyquist_MHz)
    #         % diag_sampling_frequency_MHz
            - diag_principal_nyquist_MHz
        )
        diag_theory_levels_MHz = np.asarray(
            record["plan"]["theory"].energies_MHz,
            dtype=float,
        )
        if np.any(
            np.abs(diag_theory_levels_MHz)
            > diag_principal_nyquist_MHz
        ):
            raise RuntimeError(
                f"r={diag_realization}: theory levels lie outside Nyquist"
            )
        diag_match_tolerance_MHz = (
            diag_disorder_match_tolerance_bins
            * float(diag_data.spectrum.fft_resolution_MHz)
        )
        diag_abs_match_error_MHz = np.abs(
            diag_poles_MHz[:, None] - diag_theory_levels_MHz[None, :]
        )
        diag_inside_tolerance = (
            diag_abs_match_error_MHz <= diag_match_tolerance_MHz
        )
        diag_match_penalty = (
            (diag_disorder_dimension + 1)
            * (float(np.max(diag_abs_match_error_MHz)) + 1.0)
        )
        diag_match_rows, diag_match_columns = linear_sum_assignment(
            np.where(
                diag_inside_tolerance,
                diag_abs_match_error_MHz,
                diag_match_penalty,
            )
        )
        diag_valid_matches = diag_inside_tolerance[
            diag_match_rows, diag_match_columns
        ]
        diag_matched_pole_rows = diag_match_rows[diag_valid_matches]
        diag_matched_theory_rows = diag_match_columns[diag_valid_matches]
        diag_missing_theory_rows = np.setdiff1d(
            np.arange(diag_disorder_dimension),
            diag_matched_theory_rows,
        )
        diag_spurious_pole_rows = np.setdiff1d(
            np.arange(len(diag_poles_MHz)),
            diag_matched_pole_rows,
        )
        diag_complete_match = bool(
            len(diag_poles_MHz) == diag_disorder_dimension
            and len(diag_matched_pole_rows) == diag_disorder_dimension
        )
        diag_match_mae_MHz = (
            float(np.mean(diag_abs_match_error_MHz[
                diag_matched_pole_rows, diag_matched_theory_rows
            ]))
            if len(diag_matched_pole_rows) else np.nan
        )
        record["mpm_poles_MHz"] = diag_poles_MHz
        record["theory_match"] = dict(
            complete=diag_complete_match,
            tolerance_MHz=diag_match_tolerance_MHz,
            matched_count=len(diag_matched_pole_rows),
            missing_theory_MHz=(
                diag_theory_levels_MHz[diag_missing_theory_rows]
            ),
            spurious_poles_MHz=diag_poles_MHz[diag_spurious_pole_rows],
            mae_MHz=diag_match_mae_MHz,
        )
        print(
            f"r={diag_realization}: poles={len(diag_poles_MHz)}, "
            f"matched={len(diag_matched_pole_rows)}/"
            f"{diag_disorder_dimension}, "
            f"missing={len(diag_missing_theory_rows)}, "
            f"spurious={len(diag_spurious_pole_rows)}, "
            f"MAE={1e3 * diag_match_mae_MHz:.3f} kHz"
        )
        if len(diag_missing_theory_rows):
            print(
                "  missing theory (kHz):",
                np.round(
                    1e3
                    * diag_theory_levels_MHz[diag_missing_theory_rows],
                    3,
                ),
            )
        if len(diag_spurious_pole_rows):
            print(
                "  unmatched poles (kHz):",
                np.round(
                    1e3 * diag_poles_MHz[diag_spurious_pole_rows],
                    3,
                ),
            )
        if (
            diag_disorder_require_complete_match
            and not diag_complete_match
        ):
            raise RuntimeError(
                f"r={diag_realization}: incomplete theory-pole match"
            )
        record["expt"].display(data=diag_data, spectrum_method="fft")
        record["expt"].display(data=diag_data, spectrum_method="mpm")
        plt.show()

    return diag_disorder_records


def plot_diag_level_statistics(plan, config=None):
    """Pooled adjacent-gap-ratio level statistics (cell 334). Submits nothing."""
    config = config or DiagDisorderConfig()
    # Cell knobs, from the config object.
    diag_disorder_N = config.N
    diag_disorder_realization_count = config.realization_count
    diag_disorder_strength_kHz = config.strength_kHz
    diag_disorder_master_seed = config.master_seed
    diag_disorder_selected_states = config.selected_states
    diag_disorder_max_cycle = config.max_cycle
    diag_disorder_min_time_points = config.min_time_points
    diag_disorder_nyquist_margin = config.nyquist_margin
    diag_disorder_reps = config.reps
    diag_disorder_batch_size = config.batch_size
    diag_disorder_edge_fraction = config.edge_fraction
    diag_disorder_gap_ratio_bins = config.gap_ratio_bins
    diag_disorder_match_tolerance_bins = config.match_tolerance_bins
    diag_disorder_require_complete_match = config.require_complete_match
    diag_disorder_self_kerr_kHz = config.self_kerr_kHz
    diag_disorder_branch_overrides = config.branch_overrides
    diag_disorder_dimension = plan["diag_disorder_dimension"]
    diag_disorder_records = plan["diag_disorder_records"]

    # Historical MPM level statistics: require all D poles, trim 10% at each edge.
    diag_missing_analyses = [
        realization
        for realization in range(diag_disorder_realization_count)
        if (
            realization not in diag_disorder_records
            or diag_disorder_records[realization].get("data") is None
        )
    ]
    if diag_missing_analyses:
        raise RuntimeError(
            "cannot pool level statistics; missing successful analyses for "
            f"realizations {diag_missing_analyses}"
        )

    diag_disorder_ratio_by_realization = {}
    diag_disorder_trim_count = int(np.ceil(
        diag_disorder_edge_fraction * diag_disorder_dimension
    ))

    for diag_realization, record in sorted(diag_disorder_records.items()):
        diag_data = record["data"]
        diag_levels_MHz = np.sort(np.asarray(
            record["mpm_poles_MHz"],
            dtype=float,
        ))
        if len(diag_levels_MHz) != diag_disorder_dimension:
            raise RuntimeError(
                f"r={diag_realization}: expected "
                f"{diag_disorder_dimension} MPM poles, "
                f"got {len(diag_levels_MHz)}"
            )
        if (
            diag_disorder_require_complete_match
            and not record["theory_match"]["complete"]
        ):
            raise RuntimeError(
                f"r={diag_realization}: incomplete theory-pole match"
            )

        diag_bulk_levels_MHz = diag_levels_MHz[
            diag_disorder_trim_count:-diag_disorder_trim_count
        ]
        diag_gaps_MHz = np.diff(diag_bulk_levels_MHz)
        diag_gap_floor_MHz = 100 * np.finfo(float).eps
        if np.any(diag_gaps_MHz <= diag_gap_floor_MHz):
            raise RuntimeError(
                f"r={diag_realization}: duplicate/unresolved MPM poles "
                "prevent level-statistics pooling"
            )
        diag_ratios = np.minimum(
            diag_gaps_MHz[:-1], diag_gaps_MHz[1:]
        ) / np.maximum(
            diag_gaps_MHz[:-1], diag_gaps_MHz[1:]
        )
        diag_disorder_ratio_by_realization[diag_realization] = diag_ratios
        record["bulk_levels_MHz"] = diag_bulk_levels_MHz
        record["gap_ratios"] = diag_ratios
        print(
            f"r={diag_realization}: ratios={len(diag_ratios)}, "
            f"mean r={np.mean(diag_ratios):.4f}"
        )

    if diag_disorder_ratio_by_realization:
        diag_pooled_ratios = np.concatenate(
            list(diag_disorder_ratio_by_realization.values())
        )
        diag_realization_labels = list(
            diag_disorder_ratio_by_realization
        )
        diag_realization_means = np.asarray([
            np.mean(diag_disorder_ratio_by_realization[label])
            for label in diag_realization_labels
        ])
        diag_disorder_mean = float(np.mean(diag_realization_means))
        diag_disorder_sem = (
            float(
                np.std(diag_realization_means, ddof=1)
                / np.sqrt(len(diag_realization_means))
            )
            if len(diag_realization_means) > 1 else 0.0
        )
        diag_poisson_mean = 2 * np.log(2) - 1
        diag_goe_mean = 4 - 2 * np.sqrt(3)
        diag_closer_to = (
            "GOE"
            if abs(diag_disorder_mean - diag_goe_mean)
            < abs(diag_disorder_mean - diag_poisson_mean)
            else "Poisson"
        )

        diag_ratio_axis = np.linspace(0.0, 1.0, 1000)
        diag_poisson_pdf = 2 / (1 + diag_ratio_axis) ** 2
        diag_goe_pdf = (
            (27 / 4)
            * (diag_ratio_axis + diag_ratio_axis ** 2)
            / (1 + diag_ratio_axis + diag_ratio_axis ** 2) ** 2.5
        )
        diag_gap_ratio_edges = np.linspace(
            0.0,
            1.0,
            diag_disorder_gap_ratio_bins + 1,
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.5, 4.5),
            constrained_layout=True,
        )
        axes[0].hist(
            diag_pooled_ratios,
            bins=diag_gap_ratio_edges,
            density=True,
            alpha=0.35,
            color="black",
            edgecolor="black",
            label=(
                f"measured: n={len(diag_pooled_ratios)}, "
                f"mean={np.mean(diag_pooled_ratios):.3f}"
            ),
        )
        axes[0].plot(
            diag_ratio_axis,
            diag_poisson_pdf,
            color="tab:blue",
            linewidth=2,
            label="Poisson",
        )
        axes[0].plot(
            diag_ratio_axis,
            diag_goe_pdf,
            color="tab:orange",
            linewidth=2,
            label="GOE (Wigner surmise)",
        )
        axes[0].set(
            xlim=(0, 1),
            xlabel=r"adjacent-gap ratio $\tilde r$",
            ylabel="probability density",
            title="pooled diagonal-disorder gap ratios",
        )
        axes[0].legend()

        diag_x = np.arange(len(diag_realization_labels))
        axes[1].scatter(
            diag_x,
            diag_realization_means,
            s=65,
            color="black",
            zorder=3,
        )
        axes[1].axhline(
            diag_poisson_mean,
            color="tab:blue",
            linestyle="--",
            label=f"Poisson mean={diag_poisson_mean:.3f}",
        )
        axes[1].axhline(
            diag_goe_mean,
            color="tab:orange",
            linestyle="--",
            label=f"GOE mean={diag_goe_mean:.3f}",
        )
        axes[1].errorbar(
            len(diag_x),
            diag_disorder_mean,
            yerr=diag_disorder_sem,
            fmt="D",
            capsize=4,
            color="tab:red",
            label=(
                f"disorder mean={diag_disorder_mean:.3f} "
                f"+/- {diag_disorder_sem:.3f}"
            ),
        )
        axes[1].set_xticks(
            np.append(diag_x, len(diag_x)),
            [f"r={label}" for label in diag_realization_labels] + ["mean"],
        )
        axes[1].set(
            ylim=(0, 1),
            ylabel=r"mean adjacent-gap ratio $\langle\tilde r\rangle$",
            title=f"realization means; closer to {diag_closer_to}",
        )
        axes[1].legend()
        plt.show()

        diag_disorder_level_statistics = dict(
            ratios_by_realization=diag_disorder_ratio_by_realization,
            pooled_ratios=diag_pooled_ratios,
            realization_means=diag_realization_means,
            disorder_mean=diag_disorder_mean,
            disorder_sem=diag_disorder_sem,
            closer_to=diag_closer_to,
            trim_count=diag_disorder_trim_count,
        )
    else:
        diag_disorder_level_statistics = None
        print("No realization produced all required MPM poles for level statistics.")

    return plt.gcf()


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
