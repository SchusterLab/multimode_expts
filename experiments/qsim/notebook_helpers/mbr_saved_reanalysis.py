"""Reload and reanalyze the saved Aug 2026 N=3 and disorder spectroscopy.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
213-240 by the stage-2 notebook decomposition. Primary caller:
`analysis_notebooks/202609_qsim_migration/mbr.py`.

The source had this work twice over, as two "Reproduce the Aug-15--17"
sections: one reading from the job server (cells 217-229) and one reading
local HDF5 files with a mock station (cells 230-240). That distinction is
real -- the two get their children from genuinely different places -- so both
survive here as `load_saved_remote` and `load_saved_local`. What was *not*
real:

- cells 213 and 218 defined `saved_job_range` byte-identically. One copy.
- cells 222 and 232 defined the same occupation-pairing check twice, with
  identical bodies and only the name different
  (`_saved_occupation_pairs` / `_offline_occupation_pairs`). One copy, named
  `occupation_pairs`.
- cells 226/234, 227/235 and 228/236 are the same dataset-picking and
  trace-plotting code, differing only in line wrapping and a trailing comma.
  They became `plot_occupation_trace_panels` plus, in the notebook, one
  `data_sets` dict.

The local loader took `saved_station` -- a mock station built at cell 231 --
from notebook scope. It is a `station` argument now, and building that mock
station stayed in the notebook, since which config versions to reconstruct
against is a scientific choice.

Temporary home, per the stage-2 instructions.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict

from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.notebook_helpers.mbr_loading import (
    job_id_generator,
    load_dark_experiments,
)


def saved_job_range(date, first, last):
    """Inclusive job-ID range (cells 213 and 218, identical)."""
    return [f"JOB-{date}-{number:05d}" for number in range(first, last + 1)]


def occupation_pairs(expt, label):
    """Group a batch's children by occupation and check both analyzer phases.

    Cells 222 and 232 defined this twice with identical bodies, under
    the names `_saved_occupation_pairs` and `_offline_occupation_pairs`.
    """
    grouped = {}
    for child in expt.batch_expts:
        cfg = child.cfg.expt
        occupation = tuple(map(int, cfg.spectroscopy_occupations))
        grouped.setdefault(occupation, []).append(
            float(cfg.spectroscopy_analyzer_phase)
        )
    for occupation, phases in grouped.items():
        if len(phases) != 2 or not np.allclose(sorted(phases), [0.0, 90.0]):
            raise RuntimeError(
                f"{label}: {occupation} has analyzer phases {phases}, expected 0/90"
            )
    return list(grouped)


def load_saved_remote(saved_n3_calibration_job_ids,
                      saved_n3_spectroscopy_job_ids,
                      saved_disorder_job_ids,
                      saved_n3_cycle_branches,
                      saved_n3_manual_kerr_MHz,
                      saved_job_client,
                      saved_fft_window="raw",
                      saved_zero_padding=1):
    """Load and reanalyze the completed jobs from the job server (cell 222).

    No `runner.execute()` appears here or anywhere below -- this path only
    reads jobs that already ran.

    `saved_n3_manual_kerr_MHz=None` selects the source's alternative branch,
    which derives the Kerr phase frame from the data instead of imposing one.

    The parameters keep the `saved_*` names the moved body already uses --
    they were cell 221's notebook globals -- so no line of the body needed
    renaming.

    Returns a dict of the `saved_*` names the following cells read, including
    `saved_disorder_records`.
    """
    saved_n3_calibration_expt = MBRPhaseCorrectionExperiment.from_job_ids(
        saved_n3_calibration_job_ids,
        client=saved_job_client,
    )
    saved_n3_calibration_expt.analyze()
    saved_n3_calibration_occupations = [
        tuple(map(int, occupation))
        for occupation in saved_n3_calibration_expt.data.occupations
    ]
    if (
        len(saved_n3_calibration_occupations) != 35
        or any(sum(occupation) != 3 for occupation in saved_n3_calibration_occupations)
    ):
        raise RuntimeError("calibration is not the complete 35-state N=3 sector")
    if saved_n3_calibration_expt.data.hardware.source != "saved program":
        raise RuntimeError("calibration did not recover timing from the saved program")

    saved_n3_spectroscopy_expt = MBRSpectrumExperiment.from_job_ids(
        saved_n3_spectroscopy_job_ids,
        client=saved_job_client,
    )
    saved_n3_spectroscopy_occupations = occupation_pairs(
        saved_n3_spectroscopy_expt,
        "N=3 spectroscopy",
    )
    if set(saved_n3_spectroscopy_occupations) != set(saved_n3_calibration_occupations):
        raise RuntimeError("N=3 spectroscopy and calibration occupations differ")

    # Exact reconstruction in the frame that was physically acquired.
    saved_n3_as_acquired_data = saved_n3_spectroscopy_expt.analyze(
        calibration=saved_n3_calibration_expt,
        occupations=saved_n3_calibration_occupations,
        phase_frame="as_acquired",
        fft_window=saved_fft_window,
        zero_padding=saved_zero_padding,
        spectrum_method="fft",
    )

    # Optional reframe used in the later -10.5-kHz comparison plots.
    if saved_n3_manual_kerr_MHz is None:
        saved_n3_data = saved_n3_as_acquired_data
    else:
        saved_n3_data = saved_n3_spectroscopy_expt.analyze(
            calibration=saved_n3_calibration_expt,
            occupations=saved_n3_calibration_occupations,
            phase_frame="manual_kerr",
            manual_kerr_MHz=float(saved_n3_manual_kerr_MHz),
            cycle_branches=saved_n3_cycle_branches,
            fft_window=saved_fft_window,
            zero_padding=saved_zero_padding,
            spectrum_method="fft",
        )

    saved_disorder_records = {}
    for realization, job_ids in sorted(saved_disorder_job_ids.items()):
        expt = MBRSpectrumExperiment.from_job_ids(job_ids, client=saved_job_client)
        cfg0 = expt.batch_expts[0].cfg.expt
        saved_realizations = {
            int(child.cfg.expt.disorder_realization)
            for child in expt.batch_expts
        }
        if saved_realizations != {int(realization)}:
            raise RuntimeError(
                f"manifest r={realization} contains saved realizations {saved_realizations}"
            )

        occupations = [
            tuple(map(int, occupation)) for occupation in cfg0.selected_occupations
        ]
        paired_occupations = occupation_pairs(expt, f"disorder r={realization}")
        if set(paired_occupations) != set(occupations):
            raise RuntimeError(f"disorder r={realization}: selected occupations differ")

        manual_kerr_MHz = float(cfg0.target_manual_kerr_MHz)
        cycle_branches = {
            occupation: saved_n3_cycle_branches.get(occupation, 0)
            for occupation in occupations
        }
        data = expt.analyze(
            calibration=saved_n3_calibration_expt,
            occupations=occupations,
            cycle_branches=cycle_branches,
            phase_frame="manual_kerr",
            manual_kerr_MHz=manual_kerr_MHz,
            fft_window=saved_fft_window,
            zero_padding=saved_zero_padding,
            spectrum_method="fft",
        )

        saved_theory_energies_MHz = np.asarray(cfg0.theory_energies_MHz, dtype=float)
        np.testing.assert_allclose(
            data.spectrum.energies_MHz,
            saved_theory_energies_MHz,
            rtol=0.0,
            atol=1e-10,
            err_msg=f"disorder r={realization}: saved theory and rebuilt theory differ",
        )
        plan = AttrDict(dict(
            realization=int(cfg0.disorder_realization),
            seed=int(cfg0.disorder_seed),
            strength_kHz=float(cfg0.disorder_strength_kHz),
            direction=np.asarray(cfg0.disorder_direction, dtype=float),
            target_onsite_MHz=np.asarray(
                cfg0.disorder_target_onsite_MHz, dtype=float
            ),
            pulse_detunings_MHz=np.asarray(
                cfg0.disorder_api_detunings_MHz, dtype=float
            ),
            occupations=[list(occupation) for occupation in occupations],
            theory_energies_MHz=saved_theory_energies_MHz,
            manual_kerr_MHz=manual_kerr_MHz,
        ))
        saved_disorder_records[realization] = AttrDict(dict(
            plan=plan,
            job_ids=list(job_ids),
            expt=expt,
            data=data,
        ))

    hardware = saved_n3_calibration_expt.data.hardware
    print(
        f"calibration: {len(saved_n3_calibration_job_ids)} jobs; "
        f"Tcycle={hardware.floquet_cycle_us:.9f} us; "
        f"g={1e3 * np.asarray(hardware.couplings_MHz)} kHz"
    )
    print(
        f"N=3 full spectroscopy: {len(saved_n3_spectroscopy_job_ids)} jobs, "
        f"{len(saved_n3_data.reconstruction.occupations)} occupations; "
        f"frame={saved_n3_data.phase_frame}; "
        f"K={1e3 * saved_n3_data.spectrum.physical_kerr_MHz:.4f} kHz"
    )
    for realization, record in saved_disorder_records.items():
        print(
            f"disorder r={realization}: {len(record.job_ids)} jobs, "
            f"{len(record.data.reconstruction.occupations)} occupations; "
            f"K={1e3 * record.data.spectrum.physical_kerr_MHz:.4f} kHz"
        )

    return {
        name: value
        for name, value in locals().items()
        if name.startswith("saved_") or name == "hardware"
    }


def make_local_loader(station, project_name,
                      EncSpec=MBRSpectrumExperiment):
    """Build a loader for local HDF5 children (cell 231).

    Returns a callable `(job_dates, job_starts, job_finishes, project_name=None)`.
    `station` was the mock `saved_station` that cell 231 built at notebook
    scope; it is explicit now because the move is what broke that.

    Note which class names the files versus which analyzes them: the saved
    HDF5 files are named for `EncodingHamiltonianSpectroscopyExperiment`, and
    `EncSpec` is what reassembles them.
    """
    SavedEncSpec = EncSpec

    def saved_local_experiment(job_dates, job_starts, job_finishes,
                               project_name=project_name):
        """Load local H5 children with the wrappers defined near the top."""
        job_ids = job_id_generator(job_dates, job_starts, job_finishes)
        child_expts = load_dark_experiments(
            project_name,
            job_dates,
            job_starts,
            job_finishes,
            ExpClass=EncodingHamiltonianSpectroscopyExperiment,  # names the saved files, not the analysis
        )
        if len(child_expts) != len(job_ids):
            raise FileNotFoundError(
                f"Expected {len(job_ids)} local H5 jobs but loaded "
                f"{len(child_expts)} from {project_name!r}."
            )
        return SavedEncSpec._from_expts(
            child_expts,
            job_ids=job_ids,
            station=station,
        )

    return saved_local_experiment


def load_saved_local(saved_local_experiment,
                     saved_four_realization_range,
                     offline_four_realization_branches,
                     saved_n3_range,
                     saved_disorder_ranges,
                     saved_four_project_name,
                     offline_fft_window="raw",
                     offline_zero_padding=1):
    """Same reanalysis as `load_saved_remote`, from local files (cell 232).

    Everything comes from HDF5 on disk; nothing touches the job queue.

    `saved_four_project_name` is separate from the loader's own default
    project because the four-realization data set lives in a different
    experiment directory than the N=3 and disorder data -- cell 231 kept both
    names for exactly that reason.

    Returns a dict of the `saved_*` and `data_*` names the following cells
    read.
    """
    data_four_realization = saved_local_experiment(
        *saved_four_realization_range,
        project_name=saved_four_project_name,
    )
    data_four_realization.analyze(
        phase_frame="as_acquired",
        cycle_branches=offline_four_realization_branches,
        fft_window=offline_fft_window,
        zero_padding=offline_zero_padding,
        spectrum_method="fft",
    )


    # Full N=3 spectroscopy: 35 occupations x analyzer phases 0/90.
    saved_n3_spectroscopy_expt = saved_local_experiment(*saved_n3_range)
    saved_n3_occupations = occupation_pairs(
        saved_n3_spectroscopy_expt, "N=3 spectroscopy"
    )
    if len(saved_n3_occupations) != 35 or any(
        sum(occupation) != 3 for occupation in saved_n3_occupations
    ):
        raise RuntimeError("spectroscopy is not the complete 35-state N=3 sector")
    saved_n3_data = saved_n3_spectroscopy_expt.analyze(
        occupations=saved_n3_occupations,
        phase_frame="as_acquired",
        fft_window=offline_fft_window,
        zero_padding=offline_zero_padding,
        spectrum_method="fft",
    )
    # Disorder spectroscopy: ten selected occupations x analyzer phases 0/90.
    saved_disorder_records = {}
    for realization, job_range in sorted(saved_disorder_ranges.items()):
        expt = saved_local_experiment(*job_range)
        cfg0 = expt.batch_expts[0].cfg.expt
        saved_realizations = {
            int(child.cfg.expt.disorder_realization) for child in expt.batch_expts
        }
        if saved_realizations != {realization}:
            raise RuntimeError(
                f"manifest r={realization} contains saved realizations "
                f"{saved_realizations}"
            )
        occupations = [
            tuple(map(int, occupation)) for occupation in cfg0.selected_occupations
        ]
        paired = occupation_pairs(expt, f"disorder r={realization}")
        if set(paired) != set(occupations):
            raise RuntimeError(f"disorder r={realization}: occupations differ")
        data = expt.analyze(
            occupations=occupations,
            phase_frame="as_acquired",
            fft_window=offline_fft_window,
            zero_padding=offline_zero_padding,
            spectrum_method="fft",
        )
        saved_disorder_records[realization] = AttrDict(
            dict(expt=expt, data=data, cfg=cfg0)
        )

    print(
        f"Loaded four-realization data locally: "
        f"{len(data_four_realization.batch_job_ids)} H5 jobs"
    )
    print(
        f"Loaded N=3 locally: {len(saved_n3_spectroscopy_expt.batch_job_ids)} H5 jobs, "
        f"{len(saved_n3_occupations)} occupations"
    )
    print(
        "Loaded disorder locally: "
        + ", ".join(
            f"r={realization} ({len(record.expt.batch_job_ids)} H5 jobs)"
            for realization, record in saved_disorder_records.items()
        )
    )

    return {
        name: value
        for name, value in locals().items()
        if name.startswith(("saved_", "data_", "offline_"))
    }


def coherent_normalized_trace_spectrum(data, scale_theory=True):
    """FFT of sum_n A_n(t)/A_n(0), using this data set's saved FFT convention."""
    if data.get("spectrum_only", False):
        raise ValueError(
            "Coherent trace FFT needs reconstruction.A; merged spectrum-only "
            "data have already discarded the complex time traces."
        )

    reconstruction = data.reconstruction
    spectrum = data.spectrum
    A = np.asarray(reconstruction.A, dtype=complex)
    time_us = np.asarray(spectrum.time_us, dtype=float)
    energy_MHz = np.asarray(spectrum.energy_MHz, dtype=float)

    if A.ndim != 2 or time_us.ndim != 1 or A.shape[1] != len(time_us):
        raise ValueError("reconstruction.A must have shape (occupation, time point)")
    if len(time_us) < 2 or not np.isclose(time_us[0], 0.0):
        raise ValueError("A/A(0) requires a time grid beginning at zero")
    if np.any(np.abs(A[:, 0]) < 1e-12):
        raise ValueError("at least one occupation has zero return amplitude at t=0")

    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0.0 or not np.allclose(
        np.diff(time_us), sample_time_us
    ):
        raise ValueError("coherent trace FFT requires a common uniform time grid")

    windows = {
        "raw": np.ones,
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
    }
    window_name = spectrum.get("fft_window", "raw") or "raw"
    if window_name not in windows:
        raise ValueError(f"unsupported FFT window: {window_name!r}")
    window = windows[window_name](len(time_us))
    if np.sum(window) <= 0.0:
        raise ValueError(f"{window_name} window has zero coherent gain")

    n_fft = len(energy_MHz)
    expected_energy_MHz = np.fft.fftshift(
        np.fft.fftfreq(n_fft, d=sample_time_us)
    )
    if energy_MHz.shape != expected_energy_MHz.shape or not np.allclose(
        energy_MHz, expected_energy_MHz
    ):
        raise ValueError("saved FFT energy grid does not match the time grid")

    fft_scale = n_fft / np.sum(window)
    A_normalized = A / A[:, :1]
    measured_trace = np.sum(A_normalized, axis=0)
    measured = fft_scale * np.abs(
        np.fft.fftshift(
            np.fft.ifft(measured_trace * window, n=n_fft)
        )
    )

    eigenenergies_MHz = np.asarray(spectrum.energies_MHz, dtype=float)
    eigenstate_weights = np.asarray(spectrum.eigenstate_weights, dtype=float)
    if (
        eigenstate_weights.ndim != 2
        or eigenstate_weights.shape[0] != A.shape[0]
        or eigenstate_weights.shape[1] != len(eigenenergies_MHz)
    ):
        raise ValueError(
            "theory eigenstate weights do not match occupations and energies"
        )
    theory_phase = np.exp(
        -2j * np.pi * np.outer(eigenenergies_MHz, time_us)
    )
    theory_A = eigenstate_weights @ theory_phase
    if np.any(np.abs(theory_A[:, 0]) < 1e-12):
        raise ValueError("at least one theoretical return is zero at t=0")
    theory_A_normalized = theory_A / theory_A[:, :1]
    theory_trace = np.sum(theory_A_normalized, axis=0)
    theory_unscaled = fft_scale * np.abs(
        np.fft.fftshift(
            np.fft.ifft(theory_trace * window, n=n_fft)
        )
    )

    theory_scale = 1.0
    if scale_theory and np.max(theory_unscaled) > 0.0:
        theory_scale = np.max(measured) / np.max(theory_unscaled)
    theory = theory_scale * theory_unscaled

    # display_result reads only measured/theory for the aggregate panel.
    # The two local heatmaps and exact-LDOS panel remain unchanged.
    display_spectrum = deepcopy(spectrum)
    display_spectrum.measured = measured
    display_spectrum.theory = theory
    display_spectrum.coherent_trace_fft = True

    return AttrDict(
        dict(
            energy_MHz=energy_MHz,
            A_normalized=A_normalized,
            measured_trace=measured_trace,
            theory_trace=theory_trace,
            measured=measured,
            theory=theory,
            theory_unscaled=theory_unscaled,
            theory_scale=theory_scale,
            fft_window=window_name,
            n_fft=n_fft,
            display_spectrum=display_spectrum,
        )
    )


def plot_occupation_trace_panels(pp_data, occupation_to_plot,
                                 realization_idx=0):
    """Real/imaginary return, its spectrum, and the theory, side by side (cell 237).

    Cells 227/235 and 228/236 were smaller versions of this same plot; this is
    the three-panel one. Returns (fig, axes).
    """
    data = pp_data.data
    data = pp_data.data
    reconstruction = data.reconstruction
    occupation_idx = reconstruction.occupations.index(occupation_to_plot)

    fig, axes = plt.subplots(1, 3, figsize=(25, 6), constrained_layout=True)

    axes[0].plot(
        data.spectrum.time_us,
        reconstruction.A[occupation_idx].real,
        label=r"Re$\langle n | U | n\rangle$",
    )
    axes[0].plot(
        data.spectrum.time_us,
        reconstruction.A[occupation_idx].imag,
        label=r"Im$\langle n | U | n\rangle$",
    )
    axes[0].set(
        xlabel="time (us)",
        ylabel="return amplitude",
        title="oscillation trace",
    )
    axes[0].legend(fontsize=20)

    measured = data.spectrum.measured_local[occupation_idx]
    theory = data.spectrum.theory_local[occupation_idx].copy()
    axes[1].plot(data.spectrum.energy_MHz, measured, color="black", label="measured")
    axes[1].plot(
        data.spectrum.energy_MHz,
        theory,
        color="tab:orange",
        label="theory (scaled)",
    )
    axes[1].set(
        xlim=(-data.spectrum.energy_limit_MHz, data.spectrum.energy_limit_MHz),
        xlabel="energy E/h (MHz)",
        ylabel="spectral magnitude",
        title="finite-time FFT",
    )
    axes[1].legend(loc="upper right", fontsize=20)

    ldos_weight_cutoff = 0.0
    ldos_energies_MHz, energy_indices = np.unique(
        np.round(data.spectrum.energies_MHz, 10), return_inverse=True
    )
    # eigenstate_weights follows the same order as reconstruction.occupations.
    ldos_weights = np.bincount(
        energy_indices,
        weights=data.spectrum.eigenstate_weights[occupation_idx],
        minlength=len(ldos_energies_MHz),
    )
    keep = ldos_weights >= ldos_weight_cutoff
    axes[2].vlines(
        ldos_energies_MHz[keep], 0.0, ldos_weights[keep], color="tab:blue"
    )
    axes[2].plot(
        ldos_energies_MHz[keep],
        ldos_weights[keep],
        "o",
        color="tab:blue",
        markersize=4,
    )
    axes[2].set(
        xlim=(-data.spectrum.energy_limit_MHz, data.spectrum.energy_limit_MHz),
        xlabel="eigenenergy E/h (MHz)",
        ylabel="spectral weight",
        title="exact LDOS weights",
    )
    fig.suptitle(rf"$|n\rangle$, n = {occupation_to_plot}")
    plt.show()

    return fig, axes
