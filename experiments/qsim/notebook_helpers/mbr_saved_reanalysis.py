"""Reload and reanalyze the saved Aug 2026 N=3 and disorder spectroscopy.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
213-240 by the stage-2 notebook decomposition. Primary caller:
`analysis_notebooks/202609_qsim_migration/mbr.py`.

The source had this work twice over, as two "Reproduce the Aug-15--17"
sections: one reading from the job server (cells 217-229) and one reading
local HDF5 files with a mock station (cells 230-240). Both now read HDF5
through :mod:`experiments.saved_jobs`, so where the data comes from is no
longer what separates them. What actually separates them is the analysis:

* `load_saved_calibrated` (was `load_saved_remote`) applies the N=3 phase
  calibration and reports the manual-Kerr frame;
* `load_saved_as_acquired` (was `load_saved_local`) applies no calibration at
  all and stays in the as-acquired frame.

They are named for that difference now, because naming them for their source
would be wrong -- and because a reader comparing their numbers needs to know
the frames differ. What was *not* real:

- cells 213 and 218 defined `saved_job_range` byte-identically. One copy.
- cells 222 and 232 defined the same occupation-pairing check twice, with
  identical bodies and only the name different
  (`_saved_occupation_pairs` / `_offline_occupation_pairs`). One copy, named
  `occupation_pairs` (old job layout only; now in `deprecated/`).
- cells 226/234, 227/235 and 228/236 are the same dataset-picking and
  trace-plotting code, differing only in line wrapping and a trailing comma.
  They became `plot_occupation_trace_panels` plus, in the notebook, one
  `data_sets` dict.

The local loader used to take `saved_station` -- a mock station built at cell
231 -- so that `_saved_parameters` could ask it for Floquet timing. That is
gone: a station answers with *today's* calibration, and the timing is now
recovered from the file's own provenance instead. Nothing here builds a
station, and `make_local_loader` is deleted along with the hard-coded
`C:\experiments` glob it wrapped.

Old and new classes (MBR redesign steps 6b and 7c). The N=3 and
four-realization sets are saved `MBRSpectrumExperiment` manifests, and the
disorder realizations a saved `MBRDisorderEnsembleExperiment` (old jobs
converted with `tools/migrate_mbr_jobs.py`). The old-class disorder loaders
and `occupation_pairs` moved to
`experiments/qsim/deprecated/mbr_saved_reanalysis_legacy.py`.

Temporary home, per the stage-2 instructions.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict

from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment


def saved_job_range(date, first, last):
    """Inclusive job-ID range (cells 213 and 218, identical)."""
    return [f"JOB-{date}-{number:05d}" for number in range(first, last + 1)]



def load_n3_calibrated(spectrum_manifest,
                       saved_n3_cycle_branches,
                       saved_n3_manual_kerr_MHz,
                       saved_fft_window="raw",
                       saved_zero_padding=1):
    """Reanalyze the saved N=3 spectrum *with* its phase calibration (cell 222).

    ``spectrum_manifest`` is a saved `MBRSpectrumExperiment` with its
    calibration set linked. No `runner.execute()` appears here -- this path
    only reads jobs that already ran.

    `saved_n3_manual_kerr_MHz=None` selects the source's alternative branch,
    which keeps the as-acquired frame instead of imposing a Kerr frame.

    Returns a dict of the `saved_*` names the following cells read.
    """
    saved_n3_spectroscopy_expt = MBRSpectrumExperiment.from_manifest(spectrum_manifest)
    saved_n3_calibration_expt = saved_n3_spectroscopy_expt.calibration
    if saved_n3_calibration_expt is None:
        raise RuntimeError(f"{spectrum_manifest} links no calibration set")
    saved_n3_calibration_expt.analyze()
    saved_n3_calibration_occupations = list(saved_n3_calibration_expt.occupations)
    if (
        len(saved_n3_calibration_occupations) != 35
        or any(sum(occupation) != 3 for occupation in saved_n3_calibration_occupations)
    ):
        raise RuntimeError("calibration is not the complete 35-state N=3 sector")
    # Assert the timing was *recovered* rather than guessed, and say which of
    # the recovered sources answered.
    hardware_source = saved_n3_calibration_expt.data.hardware.source
    if not hardware_source or "station" in hardware_source:
        raise RuntimeError(
            f"calibration timing came from {hardware_source!r}, which is not a "
            f"recovered historical value")
    print(f"Floquet timing recovered from: {hardware_source}")
    if set(saved_n3_spectroscopy_expt.occupations) != set(saved_n3_calibration_occupations):
        raise RuntimeError("N=3 spectroscopy and calibration occupations differ")

    # Exact reconstruction in the frame that was physically acquired.
    saved_n3_as_acquired_data = saved_n3_spectroscopy_expt.analyze(
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
            phase_frame="manual_kerr",
            manual_kerr_MHz=float(saved_n3_manual_kerr_MHz),
            cycle_branches=saved_n3_cycle_branches,
            fft_window=saved_fft_window,
            zero_padding=saved_zero_padding,
            spectrum_method="fft",
        )

    hardware = saved_n3_calibration_expt.data.hardware
    print(
        f"calibration: {len(saved_n3_calibration_occupations)} occupations; "
        f"Tcycle={hardware.floquet_cycle_us:.9f} us; "
        f"g={1e3 * np.asarray(hardware.couplings_MHz)} kHz"
    )
    print(
        f"N=3 full spectroscopy: {len(saved_n3_spectroscopy_expt.children)} traces, "
        f"{len(saved_n3_data.reconstruction.occupations)} occupations; "
        f"frame={saved_n3_data.phase_frame}; "
        f"K={1e3 * saved_n3_data.spectrum.physical_kerr_MHz:.4f} kHz"
    )
    return {
        name: value
        for name, value in locals().items()
        if name.startswith("saved_") or name == "hardware"
    }



def load_n3_as_acquired(four_realization_manifest,
                        offline_four_realization_branches,
                        n3_manifest,
                        offline_fft_window="raw",
                        offline_zero_padding=1):
    """The four-realization and N=3 sets in the as-acquired frame (cell 232).

    Same files as :func:`load_n3_calibrated`, so comparing the two compares
    phase frames and nothing else. Both arguments are saved
    `MBRSpectrumExperiment` manifests.

    Returns a dict of the `saved_*` and `data_*` names the following cells
    read.
    """
    data_four_realization = MBRSpectrumExperiment.from_manifest(four_realization_manifest)
    data_four_realization.analyze(
        phase_frame="as_acquired",
        cycle_branches=offline_four_realization_branches,
        fft_window=offline_fft_window,
        zero_padding=offline_zero_padding,
        spectrum_method="fft",
    )

    # Full N=3 spectroscopy: 35 occupations.
    saved_n3_spectroscopy_expt = MBRSpectrumExperiment.from_manifest(n3_manifest)
    saved_n3_occupations = list(saved_n3_spectroscopy_expt.occupations)
    if len(saved_n3_occupations) != 35 or any(
        sum(occupation) != 3 for occupation in saved_n3_occupations
    ):
        raise RuntimeError("spectroscopy is not the complete 35-state N=3 sector")
    saved_n3_data = saved_n3_spectroscopy_expt.analyze(
        phase_frame="as_acquired",
        fft_window=offline_fft_window,
        zero_padding=offline_zero_padding,
        spectrum_method="fft",
    )
    print(
        f"Four-realization data, as acquired: "
        f"{len(data_four_realization.children)} traces"
    )
    print(
        f"N=3, as acquired: {len(saved_n3_spectroscopy_expt.children)} traces, "
        f"{len(saved_n3_occupations)} occupations"
    )
    return {
        name: value
        for name, value in locals().items()
        if name.startswith(("saved_", "data_", "offline_"))
    }


def load_disorder_calibrated(ensemble_manifest, saved_n3_cycle_branches,
                             saved_fft_window="raw", saved_zero_padding=1):
    """The disorder realizations *with* the N=3 phase calibration (cell 222).

    ``ensemble_manifest`` is the converted `MBRDisorderEnsembleExperiment` of
    the August realizations, whose parts link the August N=3 calibration set.
    Each part is analyzed in the manual-Kerr frame at the Kerr its
    realization recorded (``target_manual_kerr_MHz`` in the old jobs), with
    the branches of ``saved_n3_cycle_branches`` for its occupations, and its
    rebuilt theory is checked against the theory saved with the jobs (to
    1e-10 MHz, as the source did).

    Returns `saved_disorder_records`: ``{realization: AttrDict(plan, job_ids,
    expt, data)}``, ``expt`` the part.
    """
    ensemble = MBRDisorderEnsembleExperiment.from_manifest(ensemble_manifest)
    saved_disorder_records = {}
    for record, part in zip(ensemble.realizations, ensemble.children):
        realization = record["realization"]
        cycle_branches = {
            occupation: saved_n3_cycle_branches.get(occupation, 0)
            for occupation in part.occupations
        }
        manual_kerr_MHz = 1e-3 * float(record["self_kerr_kHz"])
        data = part.analyze(
            cycle_branches=cycle_branches,
            phase_frame="manual_kerr",
            manual_kerr_MHz=manual_kerr_MHz,
            fft_window=saved_fft_window,
            zero_padding=saved_zero_padding,
            spectrum_method="fft",
        )
        saved_theory_energies_MHz = np.asarray(record["recorded_theory_energies_MHz"],
                                               dtype=float)
        np.testing.assert_allclose(
            data.spectrum.energies_MHz,
            saved_theory_energies_MHz,
            rtol=0.0,
            atol=1e-10,
            err_msg=f"disorder r={realization}: saved theory and rebuilt theory differ",
        )
        plan = AttrDict(dict(record, manual_kerr_MHz=manual_kerr_MHz,
                             pulse_detunings_MHz=np.asarray(part.detunings, dtype=float)))
        saved_disorder_records[realization] = AttrDict(dict(
            plan=plan, job_ids=list(part.job_ids), expt=part, data=data))

    for realization, record in saved_disorder_records.items():
        print(
            f"disorder r={realization}: {len(record.job_ids)} traces, "
            f"{len(record.data.reconstruction.occupations)} occupations; "
            f"K={1e3 * record.data.spectrum.physical_kerr_MHz:.4f} kHz"
        )
    return saved_disorder_records


def load_disorder_as_acquired(ensemble_manifest, offline_fft_window="raw",
                              offline_zero_padding=1):
    """The disorder realizations in the as-acquired frame (cell 232).

    Same ensemble as :func:`load_disorder_calibrated`, so comparing the two
    compares phase frames and nothing else.

    Returns `saved_disorder_records`: ``{realization: AttrDict(expt, data,
    record)}``.
    """
    ensemble = MBRDisorderEnsembleExperiment.from_manifest(ensemble_manifest)
    saved_disorder_records = {}
    for record, part in zip(ensemble.realizations, ensemble.children):
        data = part.analyze(
            phase_frame="as_acquired",
            fft_window=offline_fft_window,
            zero_padding=offline_zero_padding,
            spectrum_method="fft",
        )
        saved_disorder_records[record["realization"]] = AttrDict(
            dict(expt=part, data=data, record=record))
    print(
        "Disorder, as acquired: "
        + ", ".join(
            f"r={realization} ({len(entry.expt.job_ids)} traces)"
            for realization, entry in saved_disorder_records.items()
        )
    )
    return saved_disorder_records


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


def coherent_trace_report(data_sets, SavedEncSpec=MBRSpectrumExperiment):
    """Coherent normalized-trace FFT for every loaded data set (cell 240).

    For each data set: compute the coherent trace spectrum, draw the standard
    report figure, then relabel its aggregate axis so the title says whether
    it is a complete-basis DOS or a projected spectrum. Raises if it cannot
    find that axis, rather than silently mislabelling a figure.

    Returns (results, figures), both keyed by data-set index.
    """
    coherent_trace_results = {}
    coherent_trace_figures = {}

    for coherent_data_set_index, coherent_expt in data_sets.items():
        coherent_data = coherent_expt.data
        coherent_result = coherent_normalized_trace_spectrum(coherent_data)
        coherent_trace_results[coherent_data_set_index] = coherent_result

        fig = SavedEncSpec.display_result(
            coherent_data.reconstruction,
            coherent_result.display_spectrum,
            coherent_data.mode_labels,
        )
        aggregate_axes = [
            axis
            for axis in fig.axes
            if axis.get_title() in {"projected spectrum", "complete-basis DOS"}
        ]
        if len(aggregate_axes) != 1:
            raise RuntimeError(
                f"data_sets[{coherent_data_set_index}]: could not identify "
                "the aggregate-spectrum axis"
            )
        aggregate_axis = aggregate_axes[0]
        trace_kind = (
            "complete-basis trace DOS"
            if coherent_data.spectrum.complete_basis
            else "projected trace spectrum"
        )
        aggregate_axis.set_title(
            trace_kind
            + "\n"
            + r"$|\mathcal{F}[\sum_n A_n(t)/A_n(0)]|$"
        )
        aggregate_axis.lines[0].set_label("experiment")
        aggregate_axis.lines[1].set_label(
            "theory (same coherent FFT, peak-scaled)"
        )
        aggregate_axis.legend()

        coherent_trace_figures[coherent_data_set_index] = fig
        print(
            f"data_sets[{coherent_data_set_index}]: "
            f"{len(coherent_data.reconstruction.occupations)} rows, "
            f"window={coherent_result.fft_window}, n_fft={coherent_result.n_fft}, "
            f"complete_basis={coherent_data.spectrum.complete_basis}"
        )
        plt.show()

    return coherent_trace_results, coherent_trace_figures


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
