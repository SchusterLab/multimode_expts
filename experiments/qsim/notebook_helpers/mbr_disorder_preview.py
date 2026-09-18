"""Preview a diagonal-disorder campaign from job IDs, and pool its statistics.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
241-251 by the stage-2 notebook decomposition. Primary caller:
`analysis_notebooks/202609_qsim_migration/mbr_disorder.py`.

Three steps, each a `diag_*` settings prefix in the source:

  242-244  `diag_preview_*`  load one realization from job IDs and report it
  246-249  `diag_stats_*`    analyze every completed realization, then pool
                             the adjacent-gap-ratio level statistics
  251      `diag_level_*`    experimental and theoretical levels per
                             realization

Those prefixes are function arguments here rather than notebook globals. They
were not collapsed into a config dataclass the way the far larger `d72_*` and
`diag_disorder_*` blocks were, because each of these three has only a handful
of knobs and they are genuinely per-step rather than shared.

`error` appears as a cross-cell name in the source: cells 246 and 248 catch
per-realization failures and leave the exception bound for the next cell to
inspect. Each function returns its failures explicitly instead.

Temporary home, per the stage-2 instructions.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict

from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.saved_jobs import load_aggregate


def diag_preview_job_range(date, first, last):
    return [
        f"JOB-{date}-{job:05d}"
        for job in range(first, last + 1)
    ]


def diag_preview_job_ids(branch_overrides=None):
    """The calibration and per-realization job IDs of the campaign (cell 242).

    Returns (calibration_job_ids, job_ids_by_realization, branch_overrides).
    """
    diag_preview_branch_overrides = (
        {} if branch_overrides is None else branch_overrides
    )


    # N=3 calibration documented in qsim_experiments.ipynb.
    # Replace this range if the running campaign used another calibration.
    diag_preview_calibration_job_ids = diag_preview_job_range(
        20260828, 289, 358
    )

    # Saved Section 7-1 output: r=0..18 complete; r=19 has 11 completed jobs.
    diag_preview_job_ids_by_realization = {
        0: diag_preview_job_range(20260829, 16, 35),
        1: diag_preview_job_range(20260829, 36, 55),
        2: diag_preview_job_range(20260829, 56, 75),
        3: diag_preview_job_range(20260829, 76, 95),
        4: diag_preview_job_range(20260829, 96, 115),
        5: diag_preview_job_range(20260829, 116, 135),
        6: diag_preview_job_range(20260829, 136, 155),
        7: diag_preview_job_range(20260829, 156, 175),
        8: diag_preview_job_range(20260829, 176, 195),
        9: diag_preview_job_range(20260829, 196, 215),
        10: diag_preview_job_range(20260829, 216, 235),
        11: diag_preview_job_range(20260829, 236, 255),
        12: (
            diag_preview_job_range(20260829, 256, 271)
            + diag_preview_job_range(20260830, 1, 4)
        ),
        13: diag_preview_job_range(20260830, 5, 24),
        14: diag_preview_job_range(20260830, 25, 44),
        15: diag_preview_job_range(20260830, 45, 64),
        16: diag_preview_job_range(20260830, 65, 84),
        17: diag_preview_job_range(20260830, 85, 104),
        18: diag_preview_job_range(20260830, 105, 124),
        19: diag_preview_job_range(20260830, 125, 135),
    }

    diag_preview_branch_overrides = {}

    return (
        diag_preview_calibration_job_ids,
        diag_preview_job_ids_by_realization,
        diag_preview_branch_overrides,
    )


def load_preview_realization(diag_preview_calibration_job_ids,
                             diag_preview_job_ids_by_realization,
                             diag_preview_branch_overrides,
                             realization=0,
                             timing=None):
    """Load and analyze one realization (cell 243).

    Reads HDF5 only; `timing` is the escape hatch for files with no
    provenance, as everywhere else.

    Returns a dict of the `diag_preview_*` names the report step needs.
    """
    diag_preview_realization = realization

    import importlib
    from math import comb

    import matplotlib.pyplot as plt
    import numpy as np
    from experiments.qsim import floquet_dark_mode_readout

    importlib.reload(floquet_dark_mode_readout)
    DiagPreviewEncSpec = MBRSpectrumExperiment

    diag_preview_calibration = load_aggregate(
        diag_preview_calibration_job_ids,
        owner=MBRPhaseCorrectionExperiment,
        timing=timing,
        analyze=True,
    )

    diag_preview_realization = 12


    diag_preview_job_ids = diag_preview_job_ids_by_realization[
        diag_preview_realization
    ]
    diag_preview_expt = load_aggregate(
        diag_preview_job_ids, owner=DiagPreviewEncSpec, timing=timing,
    )
    diag_preview_cfg = diag_preview_expt.batch_expts[0].cfg.expt
    diag_preview_saved_realizations = sorted({
        int(child.cfg.expt.diagonal_disorder_realization)
        for child in diag_preview_expt.batch_expts
    })
    diag_preview_occupations = [
        list(map(int, occupation))
        for occupation in (
            diag_preview_cfg.diagonal_disorder_selected_occupations
        )
    ]
    diag_preview_N = sum(diag_preview_occupations[0])
    diag_preview_dimension = comb(
        diag_preview_N + len(diag_preview_occupations[0]) - 1,
        diag_preview_N,
    )
    diag_preview_manual_kerr_MHz = 1e-3 * float(
        diag_preview_cfg.diagonal_disorder_self_kerr_kHz
    )
    diag_preview_cycle_branches = {
        tuple(occupation): int(
            diag_preview_branch_overrides.get(tuple(occupation), 0)
        )
        for occupation in diag_preview_occupations
    }

    print(
        f"loaded r={diag_preview_saved_realizations}; "
        f"jobs={len(diag_preview_job_ids)}, N={diag_preview_N}, "
        f"dimension={diag_preview_dimension}"
    )
    print(
        "onsite disorder (kHz):",
        np.round(
            1e3 * np.asarray(
                diag_preview_cfg.diagonal_disorder_target_onsite_MHz
            ),
            3,
        ),
    )
    print("occupations:", [
        tuple(occupation) for occupation in diag_preview_occupations
    ])
    print(
        f"manual Kerr={1e3 * diag_preview_manual_kerr_MHz:.4f} kHz; "
        f"calibration jobs={len(diag_preview_calibration_job_ids)}"
    )

    return {
        name: value
        for name, value in locals().items()
        if name.startswith("diag_preview_")
    }


def report_preview_realization(preview):
    """Print and plot one previewed realization (cell 244)."""
    diag_preview_calibration = preview["diag_preview_calibration"]
    diag_preview_cycle_branches = preview["diag_preview_cycle_branches"]
    diag_preview_dimension = preview["diag_preview_dimension"]
    diag_preview_expt = preview["diag_preview_expt"]
    diag_preview_manual_kerr_MHz = preview["diag_preview_manual_kerr_MHz"]
    diag_preview_occupations = preview["diag_preview_occupations"]

    diag_preview_data = diag_preview_expt.analyze(
        calibration=diag_preview_calibration,
        occupations=diag_preview_occupations,
        cycle_branches=diag_preview_cycle_branches,
        phase_frame="manual_kerr",
        manual_kerr_MHz=diag_preview_manual_kerr_MHz,
        spectrum_method="mpm",
        fft_window="raw",
        zero_padding=1,
        mpm_requested_max_modes=diag_preview_dimension,
        mpm_match_decay=False,
        mpm_track_frequency_tolerance_bins=0.10,
        mpm_merge_frequency_tolerance_bins=0.10,
        mpm_dedup_frequency_tolerance_bins=0.10,
        mpm_minimum_supporting_rows=1,
    )

    diag_preview_poles_kHz = 1e3 * np.sort(np.asarray(
        diag_preview_data.matrix_pencil.selected_frequencies_MHz,
        dtype=float,
    ))
    print(
        f"MPM poles: {len(diag_preview_poles_kHz)}/"
        f"{diag_preview_dimension}"
    )
    print("MPM frequencies (kHz):", np.round(diag_preview_poles_kHz, 3))
    print(
        "FFT resolution (kHz):",
        1e3 * float(diag_preview_data.spectrum.fft_resolution_MHz),
    )

    diag_preview_expt.display(
        data=diag_preview_data,
        spectrum_method="fft",
    )
    diag_preview_expt.display(
        data=diag_preview_data,
        spectrum_method="mpm",
    )
    plt.show()


def analyze_every_realization(diag_preview_job_ids_by_realization,
                              edge_fraction=0.10, gap_ratio_bins=15,
                              excluded_occupations=None, timing=None):
    """Analyze every completed disorder realization (cell 246).

    Returns (records, failures). `failures` replaces the source's habit of
    leaving the last exception bound in the notebook namespace for the next
    cell to look at.
    """
    diag_stats_edge_fraction = edge_fraction
    diag_stats_gap_ratio_bins = gap_ratio_bins
    diag_stats_excluded_occupations = (
        [] if excluded_occupations is None else excluded_occupations
    )
    diag_stats_failures = []

    import importlib
    from math import comb

    import matplotlib.pyplot as plt
    import numpy as np
    from slab import AttrDict
    from experiments.qsim import floquet_dark_mode_readout

    importlib.reload(floquet_dark_mode_readout)
    DiagStatsEncSpec = MBRSpectrumExperiment

    diag_stats_records = {}
    diag_stats_edge_fraction = 0.10
    diag_stats_gap_ratio_bins = 15
    diag_stats_excluded_occupations = {(0, 3, 0, 0, 0)}

    for expected_realization, job_ids in sorted(
        diag_preview_job_ids_by_realization.items()
    ):
        try:
            loaded_expt = load_aggregate(
                job_ids, owner=DiagStatsEncSpec, timing=timing,
            )
            phases_by_occupation = {}
            for child in loaded_expt.batch_expts:
                occupation = tuple(
                    child.cfg.expt.spectroscopy_occupations
                )
                if occupation in diag_stats_excluded_occupations:
                    continue
                phases_by_occupation.setdefault(occupation, set()).add(
                    float(child.cfg.expt.spectroscopy_analyzer_phase)
                )
            complete_occupations = {
                occupation
                for occupation, phases in phases_by_occupation.items()
                if {0.0, 90.0}.issubset(phases)
            }

            kept_children = []
            kept_job_ids = []
            for child, job_id in zip(
                loaded_expt.batch_expts,
                loaded_expt.batch_job_ids,
            ):
                occupation = tuple(
                    child.cfg.expt.spectroscopy_occupations
                )
                if occupation not in complete_occupations:
                    continue
                kept_children.append(child)
                kept_job_ids.append(job_id)
            expt = MBRSpectrumExperiment._from_expts(
                kept_children,
                job_ids=kept_job_ids,
            )
            saved_realizations = sorted({
                int(child.cfg.expt.diagonal_disorder_realization)
                for child in expt.batch_expts
            })
            if saved_realizations != [expected_realization]:
                raise RuntimeError(
                    f"expected r={expected_realization}, got "
                    f"{saved_realizations}"
                )

            cfg = expt.batch_expts[0].cfg.expt
            occupations = [
                list(map(int, occupation))
                for occupation in (
                    cfg.diagonal_disorder_selected_occupations
                )
                if tuple(occupation) in complete_occupations
            ]
            photon_number = sum(occupations[0])
            dimension = comb(
                photon_number + len(occupations[0]) - 1,
                photon_number,
            )
            data = expt.analyze(
                occupations=occupations,
                phase_frame="as_acquired",
                spectrum_method="mpm",
                fft_window="raw",
                zero_padding=1,
                mpm_requested_max_modes=dimension,
                mpm_match_decay=False,
                mpm_track_frequency_tolerance_bins=0.50,
                mpm_merge_frequency_tolerance_bins=0.50,
                mpm_dedup_frequency_tolerance_bins=0.50,
                mpm_minimum_supporting_rows=1,
            )

            sampling_frequency_MHz = float(
                data.matrix_pencil.sampling.sampling_frequency_MHz
            )
            nyquist_MHz = 0.5 * sampling_frequency_MHz
            poles_MHz = np.asarray(
                data.matrix_pencil.selected_frequencies_MHz,
                dtype=float,
            )
            poles_MHz = np.sort(
                (poles_MHz + nyquist_MHz)
    #             % sampling_frequency_MHz
                - nyquist_MHz
            )
            recorded_kerr_MHz = 1e-3 * float(
                cfg.diagonal_disorder_self_kerr_kHz
            )
            hardware_kerr_MHz = float(
                data.hardware.physical_kerr_MHz
            )
            probe_occupation = [
                photon_number,
                *([0] * (len(occupations[0]) - 1)),
            ]
            theory_probe = AttrDict(dict(
                occupations=[probe_occupation],
                final_occupations=[probe_occupation],
                cycles=np.array([0, 1]),
                A=np.ones((1, 2), dtype=complex),
            ))
            recorded_theory = DiagStatsEncSpec.analyze_spectrum(
                theory_probe,
                photon_number,
                data.detunings,
                data.hardware.couplings_MHz,
                data.hardware.floquet_cycle_us,
                recorded_kerr_MHz,
            )
            theory_levels_MHz = np.sort(np.asarray(
                recorded_theory.energies_MHz,
                dtype=float,
            ))

            diag_stats_records[expected_realization] = dict(
                expt=expt,
                data=data,
                job_ids=list(job_ids),
                occupations=occupations,
                photon_number=photon_number,
                dimension=dimension,
                poles_MHz=poles_MHz,
                theory_levels_MHz=theory_levels_MHz,
                recorded_kerr_MHz=recorded_kerr_MHz,
                hardware_kerr_MHz=hardware_kerr_MHz,
            )
            print(
                f"r={expected_realization}: "
                f"MPM poles={len(poles_MHz)}/{dimension}; "
                f"rows={len(occupations)}; "
                f"recorded/hardware Kerr="
                f"{1e3 * recorded_kerr_MHz:.4f}/"
                f"{1e3 * hardware_kerr_MHz:.4f} kHz"
            )
        except Exception as error:
            diag_stats_records[expected_realization] = dict(
                job_ids=list(job_ids),
                error=error,
            )
            print(f"r={expected_realization}: analysis failed: {error}")

    return diag_stats_records, diag_stats_failures


def diag_stats_adjacent_gap_ratios(levels_MHz, trim_count):
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


def pool_level_statistics(diag_stats_records, diag_stats_edge_fraction=0.10):
    """Pool the adjacent-gap ratios across realizations (cell 248).

    Returns a dict with the measured and theory pooled ratios, the per
    realization ratio lists, the incomplete-pole counts and the trim count.
    """
    diag_stats_failures = []

    successful_records = {
        realization: record
        for realization, record in diag_stats_records.items()
        if "data" in record
    }
    if not successful_records:
        raise RuntimeError("no disorder realization was analyzed")

    dimensions = {
        record["dimension"] for record in successful_records.values()
    }
    if len(dimensions) != 1:
        raise RuntimeError(f"mixed Hilbert-space dimensions: {dimensions}")
    diag_stats_dimension = dimensions.pop()
    diag_stats_trim_count = int(np.ceil(
        diag_stats_edge_fraction * diag_stats_dimension
    ))

    diag_stats_measured_ratios = {}
    diag_stats_theory_ratios = {}
    diag_stats_incomplete_pole_counts = {}

    for realization, record in sorted(successful_records.items()):
        diag_stats_theory_ratios[realization] = (
            diag_stats_adjacent_gap_ratios(
                record["theory_levels_MHz"],
                diag_stats_trim_count,
            )
        )
        pole_count = len(record["poles_MHz"])
        # if pole_count != diag_stats_dimension:
        #     diag_stats_incomplete_pole_counts[realization] = pole_count
        #     print(
        #         f"skip measured r={realization}: "
        #         f"{pole_count}/{diag_stats_dimension} poles"
        #     )
        #     continue
        try:
            diag_stats_measured_ratios[realization] = (
                diag_stats_adjacent_gap_ratios(
                    record["poles_MHz"],
                    diag_stats_trim_count,
                )
            )
        except ValueError as error:
            diag_stats_incomplete_pole_counts[realization] = pole_count
            print(f"skip measured r={realization}: {error}")

    diag_stats_theory_pooled = np.concatenate(list(
        diag_stats_theory_ratios.values()
    ))
    diag_stats_measured_pooled = (
        np.concatenate(list(diag_stats_measured_ratios.values()))
        if diag_stats_measured_ratios else np.array([], dtype=float)
    )

    print(
        f"theory: {len(diag_stats_theory_ratios)} realizations, "
        f"{len(diag_stats_theory_pooled)} ratios"
    )
    print(
        f"measured complete: {len(diag_stats_measured_ratios)} "
        f"realizations, {len(diag_stats_measured_pooled)} ratios"
    )
    if diag_stats_incomplete_pole_counts:
        print("incomplete measured pole counts:", (
            diag_stats_incomplete_pole_counts
        ))

    return {
        name: value
        for name, value in locals().items()
        if name.startswith("diag_stats_")
    }


def plot_pooled_level_statistics(pooled, diag_stats_gap_ratio_bins=15):
    """The pooled adjacent-gap-ratio histogram and its comparisons (cell 249)."""
    diag_stats_measured_pooled = pooled["diag_stats_measured_pooled"]
    diag_stats_theory_pooled = pooled["diag_stats_theory_pooled"]
    diag_stats_measured_ratios = pooled["diag_stats_measured_ratios"]
    diag_stats_theory_ratios = pooled["diag_stats_theory_ratios"]
    diag_stats_incomplete_pole_counts = pooled[
        "diag_stats_incomplete_pole_counts"
    ]
    diag_stats_trim_count = pooled["diag_stats_trim_count"]

    diag_stats_ratio_axis = np.linspace(0.0, 1.0, 1000)
    diag_stats_poisson_pdf = 2 / (1 + diag_stats_ratio_axis) ** 2
    diag_stats_goe_pdf = (
        (27 / 4)
        * (diag_stats_ratio_axis + diag_stats_ratio_axis ** 2)
        / (
            1 + diag_stats_ratio_axis + diag_stats_ratio_axis ** 2
        ) ** 2.5
    )
    diag_stats_poisson_mean = 2 * np.log(2) - 1
    diag_stats_goe_mean = 4 - 2 * np.sqrt(3)
    diag_stats_edges = np.linspace(
        0.0, 1.0, diag_stats_gap_ratio_bins + 1
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.6), constrained_layout=True
    )
    if len(diag_stats_measured_pooled):
        axes[0].hist(
            diag_stats_measured_pooled,
            bins=diag_stats_edges,
            density=True,
            alpha=0.35,
            color="black",
            edgecolor="black",
            label=(
                f"measured: n={len(diag_stats_measured_pooled)}, "
                f"mean={np.mean(diag_stats_measured_pooled):.3f}"
            ),
        )
    axes[0].hist(
        diag_stats_theory_pooled,
        bins=diag_stats_edges,
        density=True,
        histtype="step",
        linewidth=2,
        color="tab:green",
        label=(
            f"theory: n={len(diag_stats_theory_pooled)}, "
            f"mean={np.mean(diag_stats_theory_pooled):.3f}"
        ),
    )
    axes[0].plot(
        diag_stats_ratio_axis,
        diag_stats_poisson_pdf,
        color="tab:blue",
        linewidth=2,
        label="Poisson",
    )
    axes[0].plot(
        diag_stats_ratio_axis,
        diag_stats_goe_pdf,
        color="tab:orange",
        linewidth=2,
        label="GOE",
    )
    axes[0].set(
        xlim=(0, 1),
        xlabel=r"adjacent-gap ratio $\tilde r$",
        ylabel="probability density",
        title="pooled disorder level statistics",
    )
    axes[0].legend()

    realizations = sorted(diag_stats_theory_ratios)
    x = np.arange(len(realizations))
    theory_means = np.asarray([
        np.mean(diag_stats_theory_ratios[r]) for r in realizations
    ])
    theory_disorder_mean = float(np.mean(theory_means))
    theory_disorder_sem = (
        float(np.std(theory_means, ddof=1) / np.sqrt(len(theory_means)))
        if len(theory_means) > 1 else 0.0
    )
    axes[1].scatter(
        x, theory_means, color="tab:green", label="theory"
    )
    measured_realizations = sorted(diag_stats_measured_ratios)
    if measured_realizations:
        measured_x = np.asarray([
            realizations.index(r) for r in measured_realizations
        ])
        measured_means = np.asarray([
            np.mean(diag_stats_measured_ratios[r])
            for r in measured_realizations
        ])
        axes[1].scatter(
            measured_x,
            measured_means,
            color="black",
            marker="x",
            s=70,
            label="measured (complete poles)",
        )
        measured_disorder_mean = float(np.mean(measured_means))
        measured_disorder_sem = (
            float(
                np.std(measured_means, ddof=1)
                / np.sqrt(len(measured_means))
            )
            if len(measured_means) > 1 else 0.0
        )
    else:
        measured_means = np.array([], dtype=float)
        measured_disorder_mean = np.nan
        measured_disorder_sem = np.nan
    mean_x = len(realizations)
    axes[1].errorbar(
        mean_x - 0.08,
        theory_disorder_mean,
        yerr=theory_disorder_sem,
        fmt="D",
        capsize=4,
        color="tab:green",
    )
    if measured_realizations:
        axes[1].errorbar(
            mean_x + 0.08,
            measured_disorder_mean,
            yerr=measured_disorder_sem,
            fmt="D",
            capsize=4,
            color="black",
        )
    axes[1].axhline(
        diag_stats_poisson_mean,
        color="tab:blue",
        linestyle="--",
        label=f"Poisson mean={diag_stats_poisson_mean:.3f}",
    )
    axes[1].axhline(
        diag_stats_goe_mean,
        color="tab:orange",
        linestyle="--",
        label=f"GOE mean={diag_stats_goe_mean:.3f}",
    )
    axes[1].set_xticks(
        np.append(x, mean_x),
        [f"r={r}" for r in realizations] + ["mean"],
    )
    axes[1].set(
        ylim=(0, 1),
        ylabel=r"mean adjacent-gap ratio $\langle\tilde r\rangle$",
        title="mean ratio by realization",
    )
    axes[1].legend(fontsize=8)
    plt.show()

    diag_stats_level_statistics = dict(
        measured_ratios_by_realization=diag_stats_measured_ratios,
        theory_ratios_by_realization=diag_stats_theory_ratios,
        measured_pooled=diag_stats_measured_pooled,
        theory_pooled=diag_stats_theory_pooled,
        measured_realization_means=measured_means,
        theory_realization_means=theory_means,
        measured_disorder_mean=measured_disorder_mean,
        measured_disorder_sem=measured_disorder_sem,
        theory_disorder_mean=theory_disorder_mean,
        theory_disorder_sem=theory_disorder_sem,
        incomplete_pole_counts=diag_stats_incomplete_pole_counts,
        trim_count=diag_stats_trim_count,
    )

    return plt.gcf()


def plot_levels_per_realization(diag_stats_records, realization=0,
                                comparison_xlim_kHz=None,
                                match_tolerance_bins=1.5):
    """Experimental and theoretical levels for every realization (cell 251)."""
    diag_level_realization = realization
    diag_level_comparison_xlim_kHz = comparison_xlim_kHz
    diag_level_match_tolerance_bins = match_tolerance_bins

    from matplotlib.lines import Line2D
    from scipy.optimize import linear_sum_assignment

    diag_level_realization = 14
    diag_level_comparison_xlim_kHz = None
    diag_level_match_tolerance_bins = 0.5

    for diag_level_realization in [diag_level_realization]:

        if diag_level_realization not in diag_stats_records:
            raise KeyError(
                f"r={diag_level_realization} is not loaded; available: "
                f"{sorted(diag_stats_records)}"
            )
        diag_level_selected_record = (
            diag_stats_records[diag_level_realization]
        )
        if (
            "poles_MHz" not in diag_level_selected_record
            or "theory_levels_MHz" not in diag_level_selected_record
        ):
            raise RuntimeError(
                f"r={diag_level_realization} analysis failed: "
                f"{diag_level_selected_record.get('error', 'unknown error')}"
            )
        diag_level_realizations = [int(diag_level_realization)]
        diag_level_successful_records = {
            realization: record
            for realization, record in diag_stats_records.items()
            if realization == diag_level_realization
        }

        if diag_level_comparison_xlim_kHz is None:
            all_comparison_levels_kHz = np.concatenate([
                1e3 * np.asarray(record[key], dtype=float)
                for record in diag_level_successful_records.values()
                for key in ("theory_levels_MHz", "poles_MHz")
            ])
            comparison_min_kHz = float(np.min(all_comparison_levels_kHz))
            comparison_max_kHz = float(np.max(all_comparison_levels_kHz))
            comparison_span_kHz = comparison_max_kHz - comparison_min_kHz
            comparison_margin_kHz = 0.03 * max(
                comparison_span_kHz, 1.0
            )
            diag_level_comparison_xlim_kHz = (
                comparison_min_kHz - comparison_margin_kHz,
                comparison_max_kHz + comparison_margin_kHz,
            )

        diag_level_ncols = 1
        diag_level_nrows = 1
        fig, axes = plt.subplots(
            diag_level_nrows,
            diag_level_ncols,
            figsize=(13.0, 4.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.asarray(axes, dtype=object).reshape(-1)

        for axis, realization in zip(axes, diag_level_realizations):
            record = diag_stats_records[realization]
            axis.set_xlim(*diag_level_comparison_xlim_kHz)
            axis.set_ylim(-0.75, 0.75)
            axis.set_yticks(
                [-0.35, 0.35],
                ["experiment", "theory"],
            )
            axis.grid(axis="x", alpha=0.2)

            if realization not in diag_level_successful_records:
                error = record.get("error", "unknown analysis error")
                axis.text(
                    0.5,
                    0.5,
                    f"analysis failed\n{error}",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                axis.set_title(f"r={realization}")
                continue

            theory_kHz = 1e3 * np.asarray(
                record["theory_levels_MHz"], dtype=float
            )
            measured_kHz = 1e3 * np.asarray(
                record["poles_MHz"], dtype=float
            )
            fft_resolution_kHz = 1e3 * float(
                record["data"].spectrum.fft_resolution_MHz
            )
            match_tolerance_kHz = (
                diag_level_match_tolerance_bins
                * fft_resolution_kHz
            )
            distances_kHz = np.abs(
                measured_kHz[:, None] - theory_kHz[None, :]
            )
            n_measured = len(measured_kHz)
            n_theory = len(theory_kHz)
            unmatched_cost = 0.500001 * match_tolerance_kHz
            invalid_cost = 4.0 * max(match_tolerance_kHz, 1.0)
            assignment_cost = np.full(
                (n_measured + n_theory, n_theory + n_measured),
                invalid_cost,
                dtype=float,
            )
            assignment_cost[:n_measured, :n_theory] = np.where(
                distances_kHz <= match_tolerance_kHz,
                distances_kHz,
                invalid_cost,
            )
            assignment_cost[:n_measured, n_theory:] = unmatched_cost
            assignment_cost[n_measured:, :n_theory] = unmatched_cost
            assignment_cost[n_measured:, n_theory:] = 0.0
            assignment_rows, assignment_columns = (
                linear_sum_assignment(assignment_cost)
            )
            matched_pairs = [
                (measured_index, theory_index)
                for measured_index, theory_index in zip(
                    assignment_rows, assignment_columns
                )
                if measured_index < n_measured
                and theory_index < n_theory
                and distances_kHz[measured_index, theory_index]
                <= match_tolerance_kHz
            ]
            matched_measured_indices = {
                measured_index
                for measured_index, _ in matched_pairs
            }
            matched_theory_indices = {
                theory_index
                for _, theory_index in matched_pairs
            }
            spurious_indices = np.asarray([
                index for index in range(n_measured)
                if index not in matched_measured_indices
            ], dtype=int)
            missing_indices = np.asarray([
                index for index in range(n_theory)
                if index not in matched_theory_indices
            ], dtype=int)
            matched_errors_kHz = np.asarray([
                measured_kHz[measured_index]
                - theory_kHz[theory_index]
                for measured_index, theory_index in matched_pairs
            ], dtype=float)
            for measured_index, theory_index in matched_pairs:
                axis.plot(
                    [theory_kHz[theory_index],
                     measured_kHz[measured_index]],
                    [0.35, -0.35],
                    color="tab:blue", alpha=0.18,
                    linewidth=0.8, zorder=1,
                )
            matched_theory_array = np.asarray(
                sorted(matched_theory_indices), dtype=int
            )
            matched_measured_array = np.asarray(
                sorted(matched_measured_indices), dtype=int
            )
            axis.scatter(
                theory_kHz[matched_theory_array],
                np.full(len(matched_theory_array), 0.35),
                marker="|",
                s=150,
                linewidths=1.5,
                color="tab:green",
                zorder=2,
            )
            axis.scatter(
                theory_kHz[missing_indices],
                np.full(len(missing_indices), 0.35),
                marker="|",
                s=170,
                linewidths=2.0,
                color="tab:red",
                zorder=3,
            )
            axis.scatter(
                measured_kHz[matched_measured_array],
                np.full(len(matched_measured_array), -0.35),
                marker="x",
                s=28,
                linewidths=1.2,
                color="black",
                zorder=3,
            )
            axis.scatter(
                measured_kHz[spurious_indices],
                np.full(len(spurious_indices), -0.35),
                marker="x",
                s=38,
                linewidths=1.6,
                color="tab:red",
                zorder=4,
            )
            mean_absolute_error_kHz = (
                float(np.mean(np.abs(matched_errors_kHz)))
                if len(matched_errors_kHz) else np.nan
            )
            axis.set_title(
                f"r={realization}: matched "
                f"{len(matched_pairs)}/{n_theory} within "
                f"{match_tolerance_kHz:.1f} kHz; "
                f"MAE={mean_absolute_error_kHz:.2f} kHz"
            )
            print(
                f"r={realization}: matched "
                f"{len(matched_pairs)}/{n_theory}; "
                f"FFT bin={fft_resolution_kHz:.3f} kHz; "
                f"tolerance={match_tolerance_kHz:.3f} kHz; "
                f"MAE={mean_absolute_error_kHz:.3f} kHz"
            )
            print(
                "missing theory (kHz):",
                np.round(theory_kHz[missing_indices], 3),
            )
            print(
                "spurious experiment (kHz):",
                np.round(measured_kHz[spurious_indices], 3),
            )

        for axis in axes[len(diag_level_realizations):]:
            axis.set_visible(False)

        comparison_handles = [
            Line2D(
                [], [], marker="|", linestyle="None",
                markersize=13, markeredgewidth=1.5,
                color="tab:green", label="matched theory",
            ),
            Line2D(
                [], [], marker="x", linestyle="None",
                markersize=6, markeredgewidth=1.2,
                color="black", label="matched experiment",
            ),
            Line2D(
                [], [], marker="|", linestyle="None",
                markersize=13, markeredgewidth=2.0,
                color="tab:red", label="missing theory",
            ),
            Line2D(
                [], [], marker="x", linestyle="None",
                markersize=6, markeredgewidth=1.6,
                color="tab:red", label="spurious experiment",
            ),
        ]
        axes[0].legend(
            handles=comparison_handles,
            loc="upper right",
            ncols=2,
        )
        fig.supxlabel(r"energy $E/h$ (kHz)")
        plt.show()

    return plt.gcf()
