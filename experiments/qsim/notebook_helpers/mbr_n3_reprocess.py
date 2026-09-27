"""N=3 spectroscopy: FFT peak finding and the self-Kerr fit.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
174-192 by the stage-2 notebook decomposition. Caller:
`analysis_notebooks/202609_qsim_migration/mbr.py`.

- `compare_peak_finders` (cell 184): FFT peak finding three ways (raw,
  Savitzky-Golay smoothed, summed across occupations). Kept to show where
  FFT peak finding is weak, beside the Matrix-Pencil spectrum.
- `fit_self_kerr` (source `qsim_experiments.ipynb` cell 309): scan the signed
  M1 self-Kerr for the best experiment--theory peak overlap. Its result is an
  input of `plan_diagonal_disorder`. Kept here while the Kerr calibration
  procedure is being fixed (guan, 2026-09-26).

The other diagnostics of this range moved to
`experiments/qsim/deprecated/mbr_n3_diagnostics.py` in MBR redesign step 8C
(`docs/qsim/mbr_step8_plan.md`). The old-class functions are in
`experiments/qsim/deprecated/mbr_n3_reprocess_legacy.py` (step 7a).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


def compare_peak_finders(spectroscopy_expt, height=0.05, prominence=0.01):
    """Raw, smoothed and summed peak finding side by side (cell 184).

    Three stacked panels. Returns (fig, peak_list_raw, peak_list_smoothed).
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    # height and prominence are arguments.
    occupations = spectroscopy_expt.data.reconstruction.occupations

    x_data = np.array(spectroscopy_expt.data.spectrum.energy_MHz)
    y_collect_data = [0 for _ in x_data]
    peak_list_raw = []
    peak_list_smoothed = []

    nonzero_peaks_raw = np.zeros_like(x_data)
    nonzero_peaks_smoothed = np.zeros_like(x_data)

    energies = np.sort(np.asarray(spectroscopy_expt.data.spectrum.energies_MHz))
    unique_energies, multiplicities = np.unique(np.round(energies, 10), return_counts=True)

    pick = False

    if pick:
        states_to_probe = [
            (3, 0, 0, 0, 0),
            (2, 1, 0, 0, 0),
            (2, 0, 1, 0, 0),
            (2, 0, 0, 1, 0),
            (2, 0, 0, 0, 1),
            (1, 1, 1, 0, 0),

        ]

        idx_list = [spectroscopy_expt.data.reconstruction.occupations.index(state) for state in states_to_probe]
    else:
        states_to_probe = occupations
        idx_list = [spectroscopy_expt.data.reconstruction.occupations.index(state) for state in states_to_probe]
        print(idx_list)
    # for idx, state in enumerate(occupations):
    for idx in idx_list:
        state_label = spectroscopy_expt.data.reconstruction.occupations[idx]
        y_collect_data += spectroscopy_expt.data.spectrum.measured_local[idx, :]

        y_data = np.array(spectroscopy_expt.data.spectrum.measured_local[idx, :])

        y_smoothed = savgol_filter(y_data,
                                   window_length=5,
                                   polyorder=2)

        peaks_raw_data, __ = find_peaks(y_data,
                                        height=height,
                                        prominence=prominence)
        peaks_smoothed_data, _ = find_peaks(y_smoothed,
                                            height=height,
                                            prominence=prominence)
        for pr in peaks_raw_data:
            if not (pr in peak_list_raw):
                peak_list_raw.append(pr)

        for psd in peaks_smoothed_data:
            if not (psd in peak_list_smoothed):
                peak_list_smoothed.append(psd)

    for idx, _ in enumerate(x_data):
        if idx in peak_list_raw:
            nonzero_peaks_raw[idx] = 1


    for idx, _ in enumerate(x_data):
        if idx in peak_list_smoothed:
            nonzero_peaks_smoothed[idx] = 1

    axes[0].plot(x_data, y_collect_data, label='Accumulated Signal')
    axes[1].plot(x_data, y_collect_data, label='Accumulated Signal')
    axes[2].plot(x_data, y_collect_data, label='Accumulated Signal')

    axes[0].vlines(unique_energies, 0., multiplicities, color='tab:orange')
    axes[0].plot(unique_energies, multiplicities, 'o', color='tab:orange')

    y_collect_np = np.array(y_collect_data)

    axes[1].plot(x_data[peak_list_raw],
                 y_collect_np[peak_list_raw], 'rx',
                 markersize=8, label='Raw Peaks')

    axes[2].plot(x_data[peak_list_smoothed],
                 y_collect_np[peak_list_smoothed], 'bo',
                 markerfacecolor='none', markersize=10, label='Smoothed Peaks')


    axes[1].plot(x_data, nonzero_peaks_raw)
    axes[2].plot(x_data, nonzero_peaks_smoothed)

    for ax in axes:
        ax.legend()
        ax.set_xlim(np.min(x_data),
                    np.max(x_data))

    plt.tight_layout()
    plt.show()

    return plt.gcf(), peak_list_raw, peak_list_smoothed


def fit_self_kerr(spectrum, cycle_branches, legacy=None, **scan_options):
    """The self-Kerr scan on a loaded `MBRSpectrumExperiment` (see `_self_kerr_scan`).

    Uses the spectrum's own calibration set and its last analysis as the
    starting data. ``legacy`` goes to ``analyze``: jobs saved before the
    analyzer sign was recorded (the July 2026 sets) need ``legacy=True``,
    which the old scan could not pass. Returns (best_self_kerr_kHz,
    kerr_fit_scores, data).
    """
    def analyze_at(kerr_MHz):
        return spectrum.analyze(
            cycle_branches=cycle_branches,
            phase_frame="manual_kerr",
            manual_kerr_MHz=kerr_MHz,
            legacy=legacy,
            spectrum_method="fft",
        )

    if not spectrum.data:
        raise ValueError("run spectrum.analyze() first")
    return _self_kerr_scan(spectrum, spectrum.data, analyze_at, **scan_options)


def _self_kerr_scan(
        spectroscopy_expt, spectroscopy_data, analyze_at,
        kerr_grid_kHz=None, energy_limit_MHz=0.08,
        min_man_photons=2, baseline_quantile=0.20):
    """Scan the signed M1 self-Kerr for best experiment--theory peak overlap.

    ``analyze_at(kerr_MHz)`` re-analyzes in the manual-Kerr frame at one
    grid point and returns the data.

    From `qsim_experiments.ipynb` cell 309, not data_postprocess -- the
    surface map routes that notebook's cells 308-314 to the analysis side,
    and this is its N=3 analysis. Submits no jobs.

    Scores only traces whose encoder occupation has at least
    `min_man_photons` M1 photons. Rows are averaged within each
    M1-photon-number group and the groups are then weighted equally, so a
    lone trace such as (3, 0, 0, 0, 0) is not drowned out by the larger
    n_M1 = 2 group.

    Warns on stdout if the best point lands on a grid boundary, which means
    `kerr_grid_kHz` needs widening.

    Note the mutation: `analyze()` rewrites `spectroscopy_expt.data`, so this
    re-runs it at the winning grid point before returning, and hands back the
    restored data rather than leaving the caller holding a stale reference.

    Returns (best_self_kerr_kHz, kerr_fit_scores, spectroscopy_data). Both
    returns are None/unchanged if no trace qualifies.
    """
    if kerr_grid_kHz is None:
        kerr_grid_kHz = np.arange(-5.0, 0.0 + 1e-9, 0.005)
    kerr_fit_energy_limit_MHz = energy_limit_MHz
    kerr_fit_min_man_photons = min_man_photons
    kerr_fit_baseline_quantile = baseline_quantile
    best_self_kerr_kHz = None
    kerr_fit_scores = None

    def normalize_peak_rows(values):
        values = np.asarray(values, dtype=float)
        values = np.clip(
            values - np.quantile(
                values,
                kerr_fit_baseline_quantile,
                axis=1,
                keepdims=True,
            ),
            0.0,
            None,
        )
        return values / np.maximum(
            np.linalg.norm(values, axis=1, keepdims=True),
            1e-15,
        )

    kerr_fit_occupations_array = np.asarray(
        spectroscopy_data.reconstruction.occupations,
        dtype=int,
    )
    kerr_fit_rows = np.flatnonzero(
        kerr_fit_occupations_array[:, 0] >= kerr_fit_min_man_photons
    )
    kerr_fit_occupations = [
        tuple(spectroscopy_data.reconstruction.occupations[row])
        for row in kerr_fit_rows
    ]

    if len(kerr_fit_rows) == 0:
        print(
            f"self-Kerr fit skipped: no trace has "
            f"n_M1 >= {kerr_fit_min_man_photons}"
        )
    else:
        kerr_fit_scores = []
        kerr_fit_trace_scores = []
        kerr_fit_n_M1 = kerr_fit_occupations_array[kerr_fit_rows, 0]

        for kerr_kHz in kerr_grid_kHz:
            candidate_data = analyze_at(kerr_kHz * 1e-3)
            use_energy = (
                np.abs(candidate_data.spectrum.energy_MHz)
                < kerr_fit_energy_limit_MHz
            )
            measured_peaks = normalize_peak_rows(
                candidate_data.spectrum.measured_local[kerr_fit_rows][
                    :, use_energy
                ]
            )
            theory_peaks = normalize_peak_rows(
                candidate_data.spectrum.theory_local[kerr_fit_rows][
                    :, use_energy
                ]
            )
            trace_scores = np.sum(measured_peaks * theory_peaks, axis=1)
            photon_group_scores = [
                np.mean(trace_scores[kerr_fit_n_M1 == n_M1])
                for n_M1 in np.unique(kerr_fit_n_M1)
            ]
            kerr_fit_trace_scores.append(trace_scores)
            kerr_fit_scores.append(np.mean(photon_group_scores))

        kerr_fit_scores = np.asarray(kerr_fit_scores)
        kerr_fit_trace_scores = np.asarray(kerr_fit_trace_scores)
        best_kerr_index = int(np.argmax(kerr_fit_scores))
        best_self_kerr_kHz = float(kerr_grid_kHz[best_kerr_index])

        # analyze() mutates spectroscopy_expt.data, so restore the best grid point.
        spectroscopy_data = analyze_at(best_self_kerr_kHz * 1e-3)

        print("fit occupations:", kerr_fit_occupations)
        print(f"best signed M1 self-Kerr = {best_self_kerr_kHz:.3f} kHz")
        if best_kerr_index in (0, len(kerr_grid_kHz) - 1):
            print("best point is on the grid boundary; expand kerr_grid_kHz")

        plt.figure(figsize=(8, 4), constrained_layout=True)
        for column, fit_occupation in enumerate(kerr_fit_occupations):
            plt.plot(
                kerr_grid_kHz,
                kerr_fit_trace_scores[:, column],
                alpha=0.45,
                label=str(fit_occupation),
            )
        plt.plot(
            kerr_grid_kHz,
            kerr_fit_scores,
            color="black",
            linewidth=2.5,
            label="equal-weight mean by n_M1",
        )
        plt.axvline(best_self_kerr_kHz, color="tab:red", linestyle="--")
        plt.xlabel("signed M1 self-Kerr (kHz)")
        plt.ylabel("experiment--theory peak overlap")
        plt.legend(fontsize=8)
        plt.show()

        for kerr_fit_row in kerr_fit_rows:
            spectroscopy_expt.display_occupations(
                occupations=int(kerr_fit_row),
            )
        plt.show()

    return best_self_kerr_kHz, kerr_fit_scores, spectroscopy_data
