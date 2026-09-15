"""Hamiltonian tomography: the N=1 three-depth propagator pilot.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
350-358 by the stage-2 notebook decomposition. Primary caller:
`measurement_notebooks/202609_qsim_migration/mbr_tomography.py`.

What this measures, per cell 350's markdown: besides the ordinary generalized
eigenproblem it fits one shared s-cycle transfer matrix F to both
`M_s ~= F M_0` and `M_2s ~= F M_s`. Comparing that fit for raw versus
phase-corrected matrices tests whether the fixed-depth phase correction
preserves a time-homogeneous matrix model. The reported
`E/h = -arg(lambda) / (2 pi s T)` is in the principal Floquet zone and is
defined modulo `1/(sT)`.

Run the three steps in order. An older experiment acquired without all three
cycles cannot be analyzed here -- `analyze_tomography` will not synthesize the
missing depth.

`fit_shared_step` was a `def` nested inside cell 356 that closed over four
notebook names (`hamtom_data`, `hamtom_step`, `hamtom_depth`,
`hamtom_cycle_us`). It is a module-level function with those as arguments now,
because the move is what broke that closure.

This theme reads the campaign base built by `mbr_campaign.build_campaign`
rather than depending on the acquisition notebook having run.

Temporary home, per the stage-2 instructions.
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_propagator import MBRPropagatorExperiment


def build_tomography_plan(campaign, station, client, N=1, step=10,
                          reps=1000, batch_size=5):
    """Build the q=[0, s, 2s] plan and print the workload (cell 352).

    Submits nothing. Raises if the N-photon calibration has not been loaded,
    rather than silently acquiring one -- a missing calibration is a missing
    scientific input.

    Returns a dict of the `hamtom_*` names the later steps need.
    """
    hamtom_N = N
    hamtom_step = step
    hamtom_reps = reps
    hamtom_batch_size = batch_size
    # Derived in cell 352 lines 9-10.
    hamtom_depth = 2 * hamtom_step
    hamtom_cycles = [0, hamtom_step, hamtom_depth]

    encspec_calibrations = campaign.calibrations
    encspec_defaults = campaign.defaults
    encspec_mode_labels = campaign.mode_labels
    encspec_modes = campaign.modes
    encspec_sync_cycles = campaign.sync_cycles
    floquet_dark_mode_readout = campaign.floquet_dark_mode_readout
    EncSpec = campaign.EncSpec
    BatchRunner = campaign.BatchRunner

    if hamtom_N not in encspec_calibrations:
        raise RuntimeError(
            "Run or load the N=1 calibration in Section 1 first."
        )

    hamtom_calibration = encspec_calibrations[hamtom_N]
    hamtom_cycle_us = float(
        hamtom_calibration.data.hardware.floquet_cycle_us
    )
    hamtom_occupations = [
        list(occupation)
        for occupation in product(
            range(hamtom_N + 1),
            repeat=len(encspec_mode_labels),
        )
        if sum(occupation) == hamtom_N
    ]
    hamtom_occupations.sort(reverse=True)
    hamtom_correction = MBRPhaseCorrectionExperiment.phase_correction_from_calibration(
        hamtom_calibration,
        cycle_branches={
            tuple(occupation): 0 for occupation in hamtom_occupations
        },
    )
    hamtom_batch = MBRPropagatorExperiment.propagator_batch(
        encspec_defaults,
        encspec_modes,
        hamtom_occupations,
        hamtom_cycles,
        phase_by_occupation=hamtom_correction.phase_by_occupation,
        sync_cycles=encspec_sync_cycles,
        reps=hamtom_reps,
    )
    hamtom_runner = BatchRunner(
        station=station,
        ExptClass=EncSpec,
        ExptProgram=(
            floquet_dark_mode_readout.EncodingPropagatorProgram
        ),
        default_expt_cfg=hamtom_batch.default_expt_cfg,
        job_client=client,
        show=False,
    )

    # Rough scale from the 3.76 s/point estimate used in Section 7-1.
    hamtom_serial_minutes = (
        hamtom_batch.total_points * 3.76
        * hamtom_reps / 1200.0 / 60.0
    )
    hamtom_parallel_jobs = min(
        hamtom_batch_size, len(hamtom_batch.configs)
    )
    print("basis order:", [
        tuple(occupation) for occupation in hamtom_occupations
    ])
    print(
        f"Floquet cycle={hamtom_cycle_us:.6f} us; "
        f"q={hamtom_cycles}; step time="
        f"{hamtom_step * hamtom_cycle_us:.6f} us; max time="
        f"{hamtom_depth * hamtom_cycle_us:.6f} us"
    )
    print(
        f"jobs={len(hamtom_batch.configs)}, "
        f"points/job={hamtom_batch.points_per_job}, "
        f"total points={hamtom_batch.total_points}, "
        f"program repetitions={hamtom_batch.total_points * hamtom_reps:,}"
    )
    print(
        f"rough time={hamtom_serial_minutes:.1f} min serialized, "
        f"or {hamtom_serial_minutes / hamtom_parallel_jobs:.1f} min "
        f"with {hamtom_parallel_jobs} concurrent workers, plus overhead"
    )

    return {
        "N": hamtom_N,
        "step": hamtom_step,
        "depth": hamtom_depth,
        "cycles": hamtom_cycles,
        "cycle_us": hamtom_cycle_us,
        "reps": hamtom_reps,
        "batch_size": hamtom_batch_size,
        "calibration": hamtom_calibration,
        "occupations": hamtom_occupations,
        "batch": hamtom_batch,
        "runner": hamtom_runner,
    }


def fit_shared_step(matrices, cycles, step, depth, cycle_us):
    """Fit one s-cycle transfer matrix to both depth steps (cell 356).

    Solves `F M_0 ~= M_s` and `F M_s ~= M_2s` simultaneously in least
    squares, and also reports the semigroup residual of the parameter-free
    prediction `M_2s = M_s M_0^-1 M_s`. Lower residuals mean the matrix model
    is more nearly time-homogeneous.

    Returns a dict with the transfer matrix, its poles sorted by frequency,
    the pole radii, the frequencies in MHz, and both residuals.
    """
    hamtom_step = step
    hamtom_depth = depth
    hamtom_cycle_us = cycle_us

    def hamtom_fit_shared_step(matrices):
        matrices = np.asarray(matrices, dtype=complex)
        cycle_index = {
            int(cycle): index
            for index, cycle in enumerate(cycles)
        }
        M0 = matrices[cycle_index[0]]
        M1 = matrices[cycle_index[hamtom_step]]
        M2 = matrices[cycle_index[hamtom_depth]]

        # Fit F M0 ~= M1 and F M1 ~= M2 simultaneously.
        design = np.hstack((M0, M1))
        target = np.hstack((M1, M2))
        transfer = np.linalg.lstsq(
            design.T, target.T, rcond=None
        )[0].T
        recurrence_residual = (
            np.linalg.norm(target - transfer @ design)
            / np.linalg.norm(target)
        )
        predicted_M2 = M1 @ np.linalg.solve(M0, M1)
        semigroup_residual = (
            np.linalg.norm(M2 - predicted_M2)
            / np.linalg.norm(M2)
        )

        poles = np.linalg.eigvals(transfer)
        frequencies_MHz = -np.angle(poles) / (
            2 * np.pi * hamtom_step * hamtom_cycle_us
        )
        order = np.argsort(frequencies_MHz)
        return {
            "transfer": transfer,
            "poles": poles[order],
            "pole_radii": np.abs(poles[order]),
            "frequencies_MHz": frequencies_MHz[order],
            "recurrence_residual": float(recurrence_residual),
            "semigroup_residual": float(semigroup_residual),
        }


def analyze_tomography(hamtom_expt, plan):
    """Analyze the three depths and fit both raw and corrected (cell 356).

    Returns (data, raw_fit, corrected_fit, theory_frequencies_MHz).
    """
    hamtom_occupations = plan["occupations"]
    hamtom_calibration = plan["calibration"]
    hamtom_cycles = plan["cycles"]
    hamtom_step = plan["step"]
    hamtom_depth = plan["depth"]
    hamtom_cycle_us = plan["cycle_us"]

    hamtom_data = hamtom_expt.analyze(
        occupations=hamtom_occupations,
        calibration=hamtom_calibration,
        finite_difference_cycles=hamtom_cycles,
        eigenphase_cycle=hamtom_step,
    )
    hamtom_eigenphase = hamtom_data.eigenphase
    hamtom_couplings_MHz = np.asarray(
        hamtom_data.hardware.couplings_MHz, dtype=float
    )
    hamtom_bright_energy_MHz = np.linalg.norm(hamtom_couplings_MHz)
    hamtom_theory_frequencies_MHz = np.asarray([
        -hamtom_bright_energy_MHz, 0.0, 0.0, 0.0,
        hamtom_bright_energy_MHz,
    ])

    hamtom_raw_fit = fit_shared_step(
        hamtom_data.raw_matrices, hamtom_data.cycles,
        hamtom_step, hamtom_depth, hamtom_cycle_us,
    )
    hamtom_corrected_fit = fit_shared_step(
        hamtom_data.matrices, hamtom_data.cycles,
        hamtom_step, hamtom_depth, hamtom_cycle_us,
    )

    print("finite difference used:", hamtom_data.finite_difference is not None)
    print(f"cond(M0)={hamtom_data.zero_cycle_condition_number:.3g}")
    print(
        "endpoint-normalized M0 identity residual="
        f"{hamtom_data.endpoint_normalized_zero_cycle_identity_residual:.3g}"
    )
    print(
        "calibration diagonal mismatch="
        f"{hamtom_data.calibration_diagonal_relative_mismatch:.3g}"
    )
    print(
        f"q={hamtom_step} two-depth GEVP E/h (kHz):",
        np.round(1e3 * hamtom_eigenphase.eigenfrequencies_MHz, 3),
    )
    print("three-depth shared-step fits (lower residual is better):")
    for label, fit in (
        ("raw", hamtom_raw_fit),
        ("phase corrected", hamtom_corrected_fit),
    ):
        print(
            f"  {label}: recurrence={fit['recurrence_residual']:.3g}, "
            f"semigroup={fit['semigroup_residual']:.3g}"
        )
        print(  # TODO(stage2): source cell Q356 read `print(s`
            "    E/h (kHz):",
            np.round(1e3 * fit["frequencies_MHz"], 3),
        )
        print("    |lambda|:", np.round(fit["pole_radii"], 5))
    print(
        "zero-detuning theory E/h (kHz):",
        np.round(1e3 * hamtom_theory_frequencies_MHz, 3),
    )
    print(
        f"step-fit alias period="
        f"{1e3 / (hamtom_step * hamtom_cycle_us):.3f} kHz; "
        f"principal zone=+/-"
        f"{0.5e3 * hamtom_eigenphase.alias_period_MHz:.3f} kHz"
    )

    return (
        hamtom_data,
        hamtom_raw_fit,
        hamtom_corrected_fit,
        hamtom_theory_frequencies_MHz,
    )


def plot_tomography_diagnostics(hamtom_data, hamtom_raw_fit,
                                hamtom_corrected_fit,
                                hamtom_theory_frequencies_MHz, plan):
    """The five diagnostic panels (cell 358).

    Three matrix magnitudes, the shared-step eigenvalues on the unit circle,
    and the three-depth least-squares spectrum against theory.
    """
    hamtom_step = plan["step"]
    hamtom_depth = plan["depth"]

    hamtom_cycle_index = {
        int(cycle): index
        for index, cycle in enumerate(hamtom_data.cycles)
    }
    hamtom_M0 = hamtom_data.matrices[hamtom_cycle_index[0]]
    hamtom_Mstep = hamtom_data.matrices[
        hamtom_cycle_index[hamtom_step]
    ]
    hamtom_Mdepth = hamtom_data.matrices[
        hamtom_cycle_index[hamtom_depth]
    ]
    hamtom_raw_values = hamtom_raw_fit["poles"]
    hamtom_corrected_values = hamtom_corrected_fit["poles"]

    fig, axes = plt.subplots(
        1, 5, figsize=(22, 4.2), constrained_layout=True, squeeze=False
    )
    axes = axes[0]
    for axis, matrix, title in (
        (axes[0], hamtom_M0, r"$|M_0|$"),
        (axes[1], hamtom_Mstep, rf"$|M_{{{hamtom_step}}}|$"),
        (axes[2], hamtom_Mdepth, rf"$|M_{{{hamtom_depth}}}|$"),
    ):
        image = axis.imshow(
            np.abs(matrix), origin="upper", cmap="magma", vmin=0
        )
        axis.set_title(title)
        axis.set_xticks(
            np.arange(len(hamtom_data.mode_labels)),
            hamtom_data.mode_labels,
            rotation=45,
        )
        axis.set_yticks(
            np.arange(len(hamtom_data.mode_labels)),
            hamtom_data.mode_labels,
        )
        axis.set_xlabel("encoder")
        axis.set_ylabel("decoder")
        fig.colorbar(image, ax=axis)

    hamtom_angle = np.linspace(0, 2 * np.pi, 500)
    axes[3].plot(
        np.cos(hamtom_angle), np.sin(hamtom_angle),
        color="black", linestyle=":", linewidth=1,
    )
    axes[3].scatter(
        hamtom_raw_values.real, hamtom_raw_values.imag,
        s=55, marker="x", label="raw",
    )
    axes[3].scatter(
        hamtom_corrected_values.real, hamtom_corrected_values.imag,
        s=55, facecolors="none", edgecolors="C1",
        label="phase corrected",
    )
    axes[3].axhline(0, color="0.8", linewidth=0.8)
    axes[3].axvline(0, color="0.8", linewidth=0.8)
    axes[3].set_aspect("equal")
    axes[3].set(
        xlabel=r"$\mathrm{Re}\,\lambda$",
        ylabel=r"$\mathrm{Im}\,\lambda$",
        title="shared-step eigenvalues",
    )
    axes[3].legend(fontsize=8)

    axes[4].scatter(
        np.zeros(len(hamtom_raw_fit["frequencies_MHz"])),
        1e3 * hamtom_raw_fit["frequencies_MHz"],
        s=55, marker="x", label="raw fit",
    )
    axes[4].scatter(
        np.ones(len(hamtom_corrected_fit["frequencies_MHz"])),
        1e3 * hamtom_corrected_fit["frequencies_MHz"],
        s=55, facecolors="none", edgecolors="C1",
        label="phase-corrected fit",
    )
    axes[4].scatter(
        2 * np.ones(len(hamtom_theory_frequencies_MHz)),
        1e3 * hamtom_theory_frequencies_MHz,
        marker="x", s=65, color="black", label="theory",
    )
    axes[4].set_xticks(
        [0, 1, 2], ["raw", "corrected", "theory"]
    )
    axes[4].set(
        xlim=(-0.5, 2.5),
        ylabel=r"$E/h$ (kHz)",
        title="three-depth least-squares spectrum",
    )
    plt.show()
