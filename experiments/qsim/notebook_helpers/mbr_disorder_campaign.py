"""Planning, submission and analysis for the disorder spectroscopy campaigns.

MBR redesign step 7a (2026-09-24): only section 7-1 (diagonal disorder) is
left here. The pairwise preview and 7-2 (D72) moved without changes to
`experiments/qsim/deprecated/mbr_disorder_offdiag.py`; see
`docs/qsim/mbr_step7_plan.md`. The text below describes the file before the
split.

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
