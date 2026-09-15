"""Disorder-averaged spectral form factor: the 2000-realization ensemble.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
359-369 by the stage-2 notebook decomposition. Primary caller:
`measurement_notebooks/202609_qsim_migration/mbr_sff.py`.

Named `mbr_sff_campaign` rather than `mbr_sff` because
`experiments/qsim/mbr_sff.py` already exists and is the library side --
`DisorderSFFExperiment` and its `batch`/`analyze_ensemble` live there. This
module is only the notebook's campaign planning and reporting around it.

Cell 361 set thirteen `sff_*` knobs at notebook scope before doing any work.
The stage-2 instructions call for grouping a repeated settings prefix like
that into one plainly named config object, so those are `SFFConfig`, with the
source's values as defaults. Note what `total_reps_per_realization` means:
4 gives two shots in each of the two independent replicas, and the full SFF
uses the cross product of those replicas.

This theme reads the campaign base built by `mbr_campaign.build_campaign`
rather than depending on the acquisition notebook having run.

Temporary home, per the stage-2 instructions.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict

from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment


@dataclass
class SFFConfig:
    """Cell 361's thirteen `sff_*` knobs, with its values as defaults."""

    N: int = 3
    realization_count: int = 2000
    # 4 = two shots in each of the two independent replicas.
    total_reps_per_realization: int = 4
    disorder_strength_kHz: float = 50.0
    master_seed: int = 20260901

    # The coherence-limited grid of section 7-2, copied rather than imported
    # so this theme does not depend on the disorder campaign.
    max_time_us: float = 80.0
    cycle_step: int = 2

    # The q=0 encoder/decoder visibility is measured once and reused for
    # every realization.
    visibility_reps: int = 1000
    realizations_per_job: int = 5
    batch_size: int = 14
    bootstrap_samples: int = 300
    branch_overrides: dict = field(default_factory=dict)


def build_sff_plan(campaign, station, client, config=None):
    """Build the ensemble plan and print the workload preview (cell 361).

    Submits nothing. Raises if the N-photon calibration is absent or if its
    mode order disagrees with the campaign's, rather than proceeding on a
    mismatched basis.

    Returns a dict of the `sff_*` names the later steps need.
    """
    config = config or SFFConfig()

    sff_N = config.N
    sff_realization_count = config.realization_count
    sff_total_reps_per_realization = config.total_reps_per_realization
    sff_disorder_strength_kHz = config.disorder_strength_kHz
    sff_master_seed = config.master_seed
    sff_max_time_us = config.max_time_us
    sff_cycle_step = config.cycle_step
    sff_visibility_reps = config.visibility_reps
    sff_realizations_per_job = config.realizations_per_job
    sff_batch_size = config.batch_size
    sff_bootstrap_samples = config.bootstrap_samples
    sff_branch_overrides = config.branch_overrides

    encspec_calibrations = campaign.calibrations
    encspec_calibration_files = campaign.calibration_files
    encspec_calibration_job_ids = campaign.calibration_job_ids
    encspec_defaults = campaign.defaults
    encspec_mode_labels = campaign.mode_labels
    encspec_modes = campaign.modes
    encspec_sync_cycles = campaign.sync_cycles
    floquet_dark_mode_readout = campaign.floquet_dark_mode_readout
    EncSpec = campaign.EncSpec
    SFFExperiment = floquet_dark_mode_readout.DisorderSFFExperiment

    if sff_total_reps_per_realization < 2 or sff_total_reps_per_realization % 2:
        raise ValueError("sff_total_reps_per_realization must be a positive even integer")
    sff_shots_per_replica = sff_total_reps_per_realization // 2

    # Use the common full-basis phase calibration from Section 1.
    if sff_N not in encspec_calibrations:
        if sff_N not in encspec_calibration_files:
            raise RuntimeError(
                f"Run or load the full N={sff_N} phase calibration in Section 1 first."
            )
        sff_calibration = MBRPhaseCorrectionExperiment.from_job_files(
            encspec_calibration_files[sff_N],
            station=station,
        )
        sff_calibration.batch_job_ids = encspec_calibration_job_ids[sff_N]
        sff_calibration.analyze()
        encspec_calibrations[sff_N] = sff_calibration

    sff_calibration = encspec_calibrations[sff_N]
    if "phase_mod180" not in sff_calibration.data:
        sff_calibration.analyze()
    sff_calibration_data = sff_calibration.data

    if list(sff_calibration_data.mode_labels) != encspec_mode_labels:
        raise ValueError(
            "The N=3 calibration mode order does not match encspec_modes: "
            f"{list(sff_calibration_data.mode_labels)} != {encspec_mode_labels}"
        )

    # Reuse the Floquet waveform and timing convention of the phase calibration.
    sff_calibration_ecfg = sff_calibration.batch_expts[0].cfg.expt
    sff_sync_cycles = int(
        sff_calibration_ecfg.get("scramble_sync_cycles", encspec_sync_cycles)
    )
    sff_gauss_sigma = sff_calibration_ecfg.get("floquet_gauss_sigma", None)
    sff_waveform = sff_calibration_ecfg.get("floquet_waveform", None)
    sff_live_hardware = EncSpec.hardware_parameters(
        station,
        encspec_modes,
        sff_sync_cycles,
        sff_gauss_sigma,
        sff_waveform,
    )
    sff_hardware = deepcopy(sff_calibration_data.hardware)

    # Kerr is an analysis-frame parameter, so use its current hardware-config value.
    sff_hardware.physical_kerr_MHz = float(sff_live_hardware.physical_kerr_MHz)

    sff_defaults = deepcopy(encspec_defaults)
    if sff_gauss_sigma is not None:
        sff_defaults.floquet_gauss_sigma = sff_gauss_sigma
    if sff_waveform is not None:
        sff_defaults.floquet_waveform = sff_waveform

    sff_occupations = [
        list(state)
        for state in product(
            range(sff_N + 1),
            repeat=len(encspec_mode_labels),
        )
        if sum(state) == sff_N
    ]
    sff_occupations.sort(reverse=True)
    sff_dimension = len(sff_occupations)

    sff_calibration_occupations = [
        tuple(map(int, occupation))
        for occupation in sff_calibration_data.occupations
    ]
    sff_calibration_branches = np.asarray([
        int(sff_branch_overrides.get(occupation, 0))
        for occupation in sff_calibration_occupations
    ])
    sff_correction = EncSpec.build_phase_correction(
        sff_calibration_data.occupations,
        sff_calibration_data.phase_mod180,
        sff_calibration_branches,
        sff_hardware.physical_kerr_MHz,
        sff_hardware.floquet_cycle_us,
    )
    sff_missing_calibrations = [
        tuple(occupation)
        for occupation in sff_occupations
        if tuple(occupation) not in sff_correction.phase_by_occupation
    ]
    if sff_missing_calibrations:
        raise ValueError(f"Missing N=3 calibration rows: {sff_missing_calibrations}")

    sff_max_cycle = int(np.floor(
        sff_max_time_us / sff_hardware.floquet_cycle_us
    ))
    sff_cycles = np.arange(
        0,
        sff_max_cycle + 1,
        sff_cycle_step,
        dtype=int,
    )

    sff_realization_ids = np.arange(sff_realization_count, dtype=np.int64)
    sff_realization_seeds = sff_master_seed + sff_realization_ids
    sff_disorder_directions = np.empty(
        (sff_realization_count, len(encspec_modes)),
        dtype=float,
    )
    for sff_row, sff_seed in enumerate(sff_realization_seeds):
        sff_rng = np.random.default_rng(int(sff_seed))
        sff_direction = sff_rng.normal(size=len(encspec_modes))
        sff_direction -= np.mean(sff_direction)
        sff_direction /= np.sqrt(np.mean(sff_direction ** 2))
        sff_disorder_directions[sff_row] = sff_direction

    sff_target_onsite_MHz = (
        1e-3 * sff_disorder_strength_kHz * sff_disorder_directions
    )
    # The pulse detuning has the opposite sign from the analyzer onsite energy.
    sff_pulse_detunings_MHz = -sff_target_onsite_MHz

    sff_plan = SFFExperiment.batch(
        default_expt_cfg=sff_defaults,
        swap_stors=encspec_modes,
        occupations=sff_occupations,
        cycles=sff_cycles,
        phase_by_occupation=sff_correction.phase_by_occupation,
        realization_detunings_MHz=sff_pulse_detunings_MHz,
        realization_seeds=sff_realization_seeds,
        realization_ids=sff_realization_ids,
        sync_cycles=sff_sync_cycles,
        shots_per_replica=sff_shots_per_replica,
        visibility_reps=sff_visibility_reps,
        realizations_per_job=sff_realizations_per_job,
        include_visibility=True,
    )

    # Section 7-1 measured about 3.76 s per two-phase, 1200-rep cycle point.
    sff_seconds_per_elementary_shot = 3.76 / (2 * 1200)
    sff_estimated_acquisition_hours = (
        sff_plan.total_elementary_shots
        * sff_seconds_per_elementary_shot
        / 3600
    )
    sff_disorder_program_loads = sff_realization_count * sff_dimension
    sff_visibility_program_loads = 4 * sff_dimension

    print(
        f"N={sff_N}, D={sff_dimension}, realizations={sff_realization_count}, "
        f"total reps/realization={sff_total_reps_per_realization} "
        f"({sff_shots_per_replica}+{sff_shots_per_replica} A/B)"
    )
    print(
        f"requested ensemble reps={sff_realization_count * sff_total_reps_per_realization:,}"
    )
    print(
        f"cycles={sff_cycles[0]}:{sff_cycles[-1]}:{sff_cycle_step}, "
        f"positive depths={len(sff_plan.positive_cycles)}, "
        f"max time={sff_cycles[-1] * sff_hardware.floquet_cycle_us:.3f} us"
    )
    print(
        f"queue jobs=1 visibility + {len(sff_plan.configs) - 1} disorder; "
        f"disorder program loads={sff_disorder_program_loads:,}"
    )
    print(
        f"expanded science shots={sff_plan.disorder_elementary_shots:,}; "
        f"visibility shots={sff_plan.visibility_elementary_shots:,}"
    )
    print(
        f"rough acquisition-only lower bound={sff_estimated_acquisition_hours:.1f} h"
    )
    print(
        "The estimate excludes 70,000 program compile/load operations, queue delay, "
        "and other overhead; batch_size does not parallelize one physical station."
    )
    print(
        "calibration/live cycle estimates (us):",
        f"{sff_hardware.floquet_cycle_us:.9f}",
        f"{sff_live_hardware.floquet_cycle_us:.9f}",
    )
    print(
        "calibration/live couplings (MHz):",
        np.asarray(sff_hardware.couplings_MHz),
        np.asarray(sff_live_hardware.couplings_MHz),
    )
    print(
        f"current hardware-config M1 Kerr={1e3 * sff_hardware.physical_kerr_MHz:.3f} kHz"
    )

    return {
        "SFFExperiment": SFFExperiment,
        "N": sff_N,
        "plan": sff_plan,
        "occupations": sff_occupations,
        "dimension": sff_dimension,
        "hardware": sff_hardware,
        "pulse_detunings_MHz": sff_pulse_detunings_MHz,
        "realization_count": sff_realization_count,
        "total_reps_per_realization": sff_total_reps_per_realization,
        "master_seed": sff_master_seed,
        "bootstrap_samples": sff_bootstrap_samples,
        "batch_size": sff_batch_size,
    }


def measure_visibility(campaign, station, client, sff):
    """Measure the depth-zero encoder/decoder visibility (cell 363).

    Submits one job. Run this before the full ensemble: it plots all 35
    encoder/decoder returns, and the ensemble analysis divides by it.

    Returns the visibility experiment.
    """
    BatchRunner = campaign.BatchRunner
    SFFExperiment = sff["SFFExperiment"]
    sff_plan = sff["plan"]
    sff_dimension = sff["dimension"]
    sff_occupations = sff["occupations"]

    sff_visibility_runner = BatchRunner(
        station=station,
        ExptClass=SFFExperiment,
        ExptProgram=sff_plan.program,
        default_expt_cfg=sff_plan.default_expt_cfg,
        job_client=client,
        show=False,
    )
    sff_visibility_batch = sff_visibility_runner.execute(
        sff_plan.configs[:1],
        batch_size=1,
        log=True,
        show=False,
    )
    sff_visibility_expt = sff_visibility_batch.batch_expts[0]
    sff_visibility = (
        np.asarray(sff_visibility_expt.data.visibility_real, dtype=float)
        + 1j * np.asarray(sff_visibility_expt.data.visibility_imag, dtype=float)
    )
    sff_visibility_magnitude = np.abs(sff_visibility)

    plt.figure(figsize=(12, 4.2))
    plt.plot(
        np.arange(sff_dimension),
        sff_visibility_magnitude,
        "o-",
    )
    plt.axhline(0.0, color="0.7", linewidth=0.8)
    plt.xlabel("occupation index in sff_occupations")
    plt.ylabel(r"$|V_j(0)|$")
    plt.title("depth-zero encoder/decoder visibility")
    plt.grid(alpha=0.25)
    plt.show()

    print("visibility job:", sff_visibility_batch.batch_job_ids)
    print("five weakest occupations:")
    for sff_index in np.argsort(sff_visibility_magnitude)[:5]:
        print(
            tuple(sff_occupations[sff_index]),
            f"|V(0)|={sff_visibility_magnitude[sff_index]:.6g}",
        )

    return sff_visibility_expt


def run_ensemble(campaign, station, client, sff):
    """Submit the positive-depth disorder jobs (cell 365).

    Only the positive-depth jobs; `sff_plan.configs[0]` is the visibility
    configuration, already measured above. The source wrapped this in a bare
    `except BaseException` so an interrupted submission still leaves the
    partial batch reachable, which is preserved.

    Returns the disorder batch.
    """
    BatchRunner = campaign.BatchRunner
    SFFExperiment = sff["SFFExperiment"]
    sff_plan = sff["plan"]
    sff_batch_size = sff["batch_size"]

    sff_disorder_runner = BatchRunner(
        station=station,
        ExptClass=SFFExperiment,
        ExptProgram=sff_plan.program,
        default_expt_cfg=sff_plan.default_expt_cfg,
        job_client=client,
        show=False,
    )
    try:
        sff_disorder_batch = sff_disorder_runner.execute(
            sff_plan.configs[1:],
            batch_size=sff_batch_size,
            log=True,
            show=False,
        )
    except BaseException:
        print(
            "jobs submitted before the stop/failure:",
            sff_disorder_runner.last_job_ids,
        )
        raise

    print("completed disorder jobs:", len(sff_disorder_batch.batch_job_ids))
    print("first job:", sff_disorder_batch.batch_job_ids[0])
    print("last job:", sff_disorder_batch.batch_job_ids[-1])

    return sff_disorder_batch


def analyze_and_plot_sff(sff, sff_disorder_batch, sff_visibility_expt):
    """Analyze the ensemble and plot the measured SFF (cell 367).

    The full SFF uses the cross product of the two independent replicas, so
    the disconnected part is estimated without a same-shot bias.

    Returns a dict with `time_us`, `full`, `disconnected`, `connected` and
    `standard_error`.
    """
    SFFExperiment = sff["SFFExperiment"]
    sff_N = sff["N"]
    sff_dimension = sff["dimension"]
    sff_master_seed = sff["master_seed"]
    sff_bootstrap_samples = sff["bootstrap_samples"]
    sff_total_reps_per_realization = sff["total_reps_per_realization"]

    sff_data = SFFExperiment.analyze_ensemble(
        sff_disorder_batch,
        visibility=sff_visibility_expt,
        bootstrap_samples=sff_bootstrap_samples,
        bootstrap_seed=sff_master_seed,
    )

    sff_time_us = np.asarray(sff_data.time_us, dtype=float)
    sff_full = np.asarray(sff_data.sff_full, dtype=float)
    sff_disconnected = np.asarray(sff_data.sff_disconnected, dtype=float)
    sff_connected = np.asarray(sff_data.sff_connected, dtype=float)
    sff_standard_error = np.asarray(
        sff_data.sff_full_standard_error,
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 4.8),
        constrained_layout=True,
    )
    axes[0].plot(sff_time_us, sff_full, label="full SFF")
    axes[0].fill_between(
        sff_time_us,
        sff_full - sff_standard_error,
        sff_full + sff_standard_error,
        alpha=0.22,
        label="full ± 1 SE",
    )
    axes[0].plot(
        sff_time_us,
        sff_disconnected,
        label="disconnected",
    )
    axes[0].plot(
        sff_time_us,
        np.asarray(sff_data.sff_naive),
        linestyle=":",
        color="0.45",
        label="naive $|z|^2$",
    )
    axes[0].axhline(
        1.0 / sff_dimension,
        color="black",
        linestyle="--",
        linewidth=1,
        label=r"$1/D$",
    )
    axes[0].set(
        xlabel="time (us)",
        ylabel=r"$K(t)$",
        title="disorder-averaged SFF",
    )
    axes[0].legend(fontsize=8)

    axes[1].plot(
        sff_time_us,
        sff_connected,
        color="tab:purple",
        label="connected SFF",
    )
    if len(sff_data.bootstrap_connected_95_low):
        axes[1].fill_between(
            sff_time_us,
            sff_data.bootstrap_connected_95_low,
            sff_data.bootstrap_connected_95_high,
            color="tab:purple",
            alpha=0.2,
            label="bootstrap 95% interval",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="time (us)",
        ylabel=r"$K_c(t)$",
        title="connected SFF",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"N={sff_N}, D={sff_dimension}, "
        f"R={int(sff_data.realization_count[0])}, "
        f"reps={sff_total_reps_per_realization}"
    )
    plt.show()

    return {
        "time_us": sff_time_us,
        "full": sff_full,
        "disconnected": sff_disconnected,
        "connected": sff_connected,
        "standard_error": sff_standard_error,
    }


def compare_effective_hamiltonian_ensemble(campaign, sff, measured):
    """Compare the measurement with the effective-Hamiltonian ensemble (cell 369).

    Diagonalizes the same 35-dimensional effective Hamiltonian for every
    realization's detunings, so this is a theory curve for the exact ensemble
    that was measured -- not a random-matrix surrogate.

    Submits nothing. Returns the figure.
    """
    EncSpec = campaign.EncSpec
    encspec_modes = campaign.modes
    sff_N = sff["N"]
    sff_dimension = sff["dimension"]
    sff_hardware = sff["hardware"]
    sff_pulse_detunings_MHz = sff["pulse_detunings_MHz"]
    sff_realization_count = sff["realization_count"]
    sff_time_us = measured["time_us"]
    sff_full = measured["full"]
    sff_connected = measured["connected"]
    sff_standard_error = measured["standard_error"]

    sff_probe_occupation = [sff_N] + [0] * len(encspec_modes)
    sff_probe = AttrDict(dict(
        occupations=[sff_probe_occupation],
        final_occupations=[sff_probe_occupation.copy()],
        cycles=np.asarray([0, 1], dtype=int),
        A=np.ones((1, 2), dtype=complex),
    ))
    sff_theory_traces = np.empty(
        (sff_realization_count, len(sff_time_us)),
        dtype=complex,
    )

    for sff_row, sff_detunings_MHz in enumerate(sff_pulse_detunings_MHz):
        sff_theory = EncSpec.analyze_spectrum(
            sff_probe,
            sff_N,
            sff_detunings_MHz,
            sff_hardware.couplings_MHz,
            sff_hardware.floquet_cycle_us,
            sff_hardware.physical_kerr_MHz,
        )
        sff_energies_MHz = np.asarray(sff_theory.energies_MHz, dtype=float)
        sff_theory_traces[sff_row] = np.mean(
            np.exp(
                -2j
                * np.pi
                * sff_energies_MHz[:, None]
                * sff_time_us[None, :]
            ),
            axis=0,
        )
        if (sff_row + 1) % 250 == 0:
            print(f"diagonalized {sff_row + 1}/{sff_realization_count}")

    sff_theory_same = np.abs(sff_theory_traces) ** 2
    sff_theory_full = np.mean(sff_theory_same, axis=0)
    sff_theory_disconnected = (
        np.abs(np.sum(sff_theory_traces, axis=0)) ** 2
        - np.sum(sff_theory_same, axis=0)
    ) / (sff_realization_count * (sff_realization_count - 1))
    sff_theory_connected = sff_theory_full - sff_theory_disconnected

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 4.8),
        constrained_layout=True,
    )
    axes[0].plot(sff_time_us, sff_full, label="experiment")
    axes[0].plot(
        sff_time_us,
        sff_theory_full,
        linestyle="--",
        label="effective-H theory",
    )
    axes[0].fill_between(
        sff_time_us,
        sff_full - sff_standard_error,
        sff_full + sff_standard_error,
        alpha=0.2,
    )
    axes[0].axhline(
        1.0 / sff_dimension,
        color="black",
        linestyle=":",
        linewidth=1,
    )
    axes[0].set(
        xlabel="time (us)",
        ylabel=r"$K(t)$",
        title="full SFF",
    )
    axes[0].legend()

    axes[1].plot(sff_time_us, sff_connected, label="experiment")
    axes[1].plot(
        sff_time_us,
        sff_theory_connected,
        linestyle="--",
        label="effective-H theory",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="time (us)",
        ylabel=r"$K_c(t)$",
        title="connected SFF",
    )
    axes[1].legend()
    plt.show()

    return plt.gcf()
