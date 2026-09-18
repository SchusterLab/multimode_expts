"""N=3 encoding-Hamiltonian spectroscopy reprocessing and FFT/MPM diagnostics.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
174-192 by the stage-2 notebook decomposition. Primary caller:
`analysis_notebooks/202609_qsim_migration/mbr.py`.

This range is a sequence of independent diagnostics on one reprocessed data
set, so each long cell became one function. The knobs each cell set at its top
are now that function's arguments, keeping their source values as defaults, and
the names later cells read are its returns.

Two hard-coded overrides in the source are preserved but named rather than
left as silent reassignments, since both look like debugging that stayed:

- cell 179 read the saved FFT window and then immediately set `'hann'`. That
  is now the `window_name` argument, defaulting to `'hann'` to match.
- cell 187's `ridge_*` settings and cell 184's peak `height`/`prominence` are
  ordinary arguments.

`encspec_reprocessed` is the object this module produces and that the
`mbr_sampling` and `mbr_spectral_validation` notebooks both need. Before the
split they read it out of the live kernel; now they call
`reprocess_n3_spectroscopy` themselves, or load the same jobs.

Temporary home, per the stage-2 instructions. In particular the ridge finder
and the peak-finding diagnostics were not reconciled with each other or with
`MBRSpectrumExperiment`'s own spectrum methods -- they are deliberately kept
as distinct algorithms, which the instructions ask for.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks, savgol_filter

from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.notebook_helpers.mbr_loading import (
    job_id_generator,
    load_encoding_spectroscopy,
)


def reprocess_n3_spectroscopy(calibration_expt, spectroscopy_expt,
                              cycle_branches=None, legacy=True,
                              manual_kerr_MHz=-19.756e-3):
    """Re-analyze the loaded N=3 jobs in the manual-Kerr phase frame (cell 175).

    `cycle_branches` defaults to empty, i.e. branch 0 for every occupation,
    as the source did. Note the source's own warning: these jobs used the old
    `+cycle*correction` analyzer convention, so a branch assignment copied
    from a `Q0 + iQ90` era notebook needs its sign flipped.

    Returns `encspec_reprocessed`.
    """
    encspec_cycle_branches = {} if cycle_branches is None else cycle_branches
    encspec_legacy = legacy
    encspec_manual_kerr_MHz = manual_kerr_MHz

    # Remove the complete analyzer correction used during acquisition. This is the uncorrected return in the current IQ convention.
    # encspec_uncorrected = spectroscopy_expt.analyze(
    #                                                 phase_frame='uncorrected',
    #                                                 cycle_branches=encspec_cycle_branches,
    #                                                 legacy=encspec_legacy,
    #                                                 fft_window='raw',
    #                                                 zero_padding=1,
    #                                                 spectrum_method='mpm')
    # print('saved analyzer correction removed')
    # spectroscopy_expt.display(data=encspec_uncorrected, spectrum_method='mpm')

    # Starting from the same saved data, undo the old correction and apply the calibration again with the selected signed Kerr.
    encspec_reprocessed = spectroscopy_expt.analyze(
                                                    calibration=calibration_expt,
                                                    phase_frame='manual_kerr',
                                                    manual_kerr_MHz=encspec_manual_kerr_MHz,
                                                    cycle_branches=encspec_cycle_branches,
                                                    legacy=encspec_legacy,
                                                    fft_window='raw',
                                                    zero_padding=1,
                                                    spectrum_method='mpm')
    spectroscopy_expt.display(data=encspec_reprocessed, spectrum_method='mpm')

    return encspec_reprocessed


def compare_incoherent_and_coherent_fft(spectroscopy_expt, window_name='hann'):
    """sum_i |FFT[A_i]| against |FFT[sum_i A_i]| (cell 179).

    Both are recomputed from the complex return rather than read back, so the
    comparison does not depend on which spectrum the previous analysis stored.
    Rows are normalized by A_i(0) first -- complex division removes each row's
    constant gain and phase, so every normalized A_i(0) is one.

    Returns (time_us, energy_MHz, fig).
    """
    encspec_trace_data = spectroscopy_expt.data
    encspec_trace_A = np.asarray(encspec_trace_data.reconstruction.A, dtype=complex)
    encspec_trace_time_us = np.asarray(encspec_trace_data.spectrum.time_us, dtype=float)
    encspec_trace_energy_MHz = np.asarray(encspec_trace_data.spectrum.energy_MHz, dtype=float)
    if encspec_trace_A.ndim != 2 or encspec_trace_A.shape[1] != len(encspec_trace_time_us):
        raise ValueError('complex return and time dimensions differ')
    if np.any(np.abs(encspec_trace_A[:, 0]) < 1e-12):
        raise ValueError('at least one occupation has zero return at t=0')

    # Complex division removes each row's constant gain and phase, so every normalized A_i(0) is one.
    encspec_trace_A_normalized = encspec_trace_A / encspec_trace_A[:, :1]
    encspec_trace_window_name = (
        window_name
        if window_name is not None
        else (encspec_trace_data.spectrum.fft_window or 'raw')
    )
    # Cell 179 read the saved window and then hard-coded 'hann' on the next
    # line. That override is the `window_name` default rather than a
    # silent reassignment.
    encspec_trace_windows = {'raw': np.ones,
                             'hann': np.hanning,
                             'hamming': np.hamming,
                             'blackman': np.blackman}

    encspec_trace_window = encspec_trace_windows[encspec_trace_window_name](len(encspec_trace_time_us))
    encspec_trace_n_fft = len(encspec_trace_energy_MHz)

    encspec_trace_sample_time_us = encspec_trace_time_us[1] - encspec_trace_time_us[0]

    if encspec_trace_sample_time_us <= 0. or not np.allclose(np.diff(encspec_trace_time_us), encspec_trace_sample_time_us):
        raise ValueError('FFT requires a common uniform time grid')
    encspec_trace_expected_energy_MHz = np.fft.fftshift(np.fft.fftfreq(encspec_trace_n_fft, d=encspec_trace_sample_time_us))
    if not np.allclose(encspec_trace_energy_MHz, encspec_trace_expected_energy_MHz):
        raise ValueError('saved FFT energy grid does not match the reconstructed time grid')
    encspec_trace_fft_scale = encspec_trace_n_fft / np.sum(encspec_trace_window)

    # The gray curve takes abs before summing; the black curve keeps the complex phase until after the trace is FFT'd.
    encspec_trace_local_fft = encspec_trace_fft_scale * np.fft.fftshift(np.fft.ifft(encspec_trace_A_normalized * encspec_trace_window, n=encspec_trace_n_fft, axis=1), axes=1)
    encspec_trace_incoherent_DOS = np.sum(np.abs(encspec_trace_local_fft), axis=0)
    encspec_trace_Z = np.sum(encspec_trace_A_normalized, axis=0)
    encspec_trace_coherent_DOS = encspec_trace_fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(encspec_trace_Z * encspec_trace_window, n=encspec_trace_n_fft)))
    encspec_trace_energy_limit_MHz = float(encspec_trace_data.spectrum.energy_limit_MHz)
    encspec_trace_plot_mask = np.abs(encspec_trace_energy_MHz) <= encspec_trace_energy_limit_MHz

    energies = np.sort(np.asarray(spectroscopy_expt.data.spectrum.energies_MHz))
    unique_energies, multiplicities = np.unique(np.round(energies, 10), return_counts=True)



    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
    for ax in axes:
        ax.plot(encspec_trace_energy_MHz, encspec_trace_incoherent_DOS,
                color='0.7', label=r'$\sum_i |\mathcal{F}[A_i]|$')
        ax.plot(encspec_trace_energy_MHz, encspec_trace_coherent_DOS,
                color='black', label=r'$|\mathcal{F}[\sum_i A_i]|$')
        ax.plot(unique_energies, multiplicities, 'o', color='tab:orange')
        ax.vlines(unique_energies, 0., multiplicities, color='tab:orange')
        ax.set_xlim(-encspec_trace_energy_limit_MHz, encspec_trace_energy_limit_MHz)
        ax.set(xlabel='energy E/h (MHz)', ylabel='spectral magnitude')
        ax.legend()
    axes[0].set_title('linear scale')
    axes[1].set_yscale('log')
    axes[1].set_title('log scale')
    encspec_trace_kind = 'complete-basis trace DOS' if encspec_trace_data.spectrum.complete_basis else 'projected trace spectrum'
    fig.suptitle(f'{encspec_trace_kind}; complex rows normalized by A_i(0) before summation')
    plt.show()

    print('number of occupation rows =', encspec_trace_A.shape[0])
    print('complete basis =', encspec_trace_data.spectrum.complete_basis)
    print('median over plotted range (rough floor proxy): incoherent =', np.median(encspec_trace_incoherent_DOS[encspec_trace_plot_mask]), ', coherent =', np.median(encspec_trace_coherent_DOS[encspec_trace_plot_mask]))

    return (
        encspec_trace_time_us,
        encspec_trace_energy_MHz,
        plt.gcf(),
    )


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


def find_ridge_peaks(encspec_reprocessed, max_candidates=35,
                     row_prominence=1., background_percentile=30.):
    """Peaks that repeat across occupations, with no theory input (cell 187).

    Deliberately uses no Hamiltonian energies and no expected peak count:
    `row_prominence` is in units of each row's own FFT noise. That
    independence is the point of the diagnostic, so do not add a theory
    comparison here.

    Returns (aligned_peaks, candidate_indices, fig).
    """
    ridge_data = encspec_reprocessed
    ridge_max_candidates = max_candidates
    ridge_row_prominence = row_prominence
    ridge_background_percentile = background_percentile

    spectrum = ridge_data.spectrum
    energy_MHz = np.asarray(spectrum.energy_MHz, dtype=float)
    local_fft = np.array(spectrum.measured_local, dtype=float, copy=True)
    if not np.allclose(np.sum(local_fft, axis=0), spectrum.measured):
        raise RuntimeError('measured_local was modified in memory; rerun execution 41 first')
    energy_step_MHz = float(np.median(np.diff(energy_MHz)))
    resolution_bins = max(1, int(np.ceil(spectrum.fft_resolution_MHz / abs(energy_step_MHz))))
    visible = np.abs(energy_MHz) <= spectrum.energy_limit_MHz

    # 1. Express every row in units of its own FFT noise.
    background_bins = 6 * resolution_bins + 1
    background = percentile_filter(
        local_fft,
        percentile=ridge_background_percentile,
        size=(1, background_bins),
        mode='nearest',
    )
    differences = np.diff(local_fft, axis=1)
    difference_center = np.median(differences, axis=1, keepdims=True)
    row_noise = 1.4826 * np.median(np.abs(differences - difference_center), axis=1) / np.sqrt(2.)
    valid_noise = row_noise[row_noise > 1e-12]
    row_noise[row_noise <= 1e-12] = np.median(valid_noise) if len(valid_noise) else 1.
    row_signal = np.maximum((local_fft - background) / row_noise[:, None], 0.)

    # 2. Mark local peaks in each row, then count peaks that align within one FFT resolution.
    row_peak_map = np.zeros_like(row_signal, dtype=bool)
    for row_index, signal in enumerate(row_signal):
        row_peaks = find_peaks(
            np.r_[0., signal, 0.],
            prominence=ridge_row_prominence,
            distance=resolution_bins,
        )[0] - 1
        row_peaks = row_peaks[(row_peaks >= 0) & (row_peaks < len(signal))]
        row_peak_map[row_index, row_peaks] = True
    alignment_offsets = np.arange(-resolution_bins, resolution_bins + 1)
    alignment_kernel = 1. - np.abs(alignment_offsets) / (resolution_bins + 1.)
    aligned_peaks = []
    for row in row_peak_map:
        aligned = np.convolve(row.astype(float), alignment_kernel, mode='same')
        aligned_peaks.append(aligned)
    ridge_score = np.sum(aligned_peaks, axis=0)

    # 3. Rank actual row-peak positions directly. A candidate may be a shoulder in the summed score.
    visible_indices = np.flatnonzero(visible)
    interior_indices = visible_indices[1:-1] # FFT display edges cannot establish a local maximum
    candidate_pool = interior_indices[np.any(row_peak_map[:, interior_indices], axis=0)]
    candidate_order = candidate_pool[np.argsort(ridge_score[candidate_pool])[::-1]]
    candidate_indices = []
    for index in candidate_order:
        distances = [abs(index - selected) for selected in candidate_indices]
        separated = all(distance > resolution_bins for distance in distances)
        if separated:
            candidate_indices.append(index)
        if len(candidate_indices) == ridge_max_candidates:
            break
    candidate_indices = np.sort(np.asarray(candidate_indices, dtype=int))
    candidate_energies_MHz = energy_MHz[candidate_indices]
    candidate_scores = ridge_score[candidate_indices]
    ridge_ranked_candidates = sorted(
        zip(candidate_energies_MHz, candidate_scores),
        key=lambda candidate: candidate[1],
        reverse=True,
    )
    encspec_fft_ridges = dict(
        energy_MHz=energy_MHz,
        score=ridge_score,
        row_peak_map=row_peak_map,
        candidate_indices=candidate_indices,
        candidate_energies_MHz=candidate_energies_MHz,
        candidate_scores=candidate_scores,
        ranked_candidates=ridge_ranked_candidates,
    )

    # 4. Show every candidate; line opacity and score indicate its strength.
    row_indices = np.arange(len(local_fft))
    labels = [str(tuple(occupation)) for occupation in ridge_data.reconstruction.occupations]
    extent = [energy_MHz[0], energy_MHz[-1], -0.5, len(local_fft) - 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)
    image = axes[0].imshow(
        local_fft,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        extent=extent,
        cmap='magma',
        vmin=0.,
        vmax=np.quantile(local_fft[:, visible], 0.995),
    )
    peak_rows, peak_columns = np.nonzero(row_peak_map)
    axes[0].plot(energy_MHz[peak_columns], peak_rows, '.', color='cyan', markersize=2, alpha=0.5)
    maximum_score = np.max(candidate_scores) if len(candidate_scores) else 1.
    for energy, score in zip(candidate_energies_MHz, candidate_scores):
        axes[0].axvline(
            energy,
            color='cyan',
            linewidth=1.,
            alpha=0.2 + 0.8 * score / maximum_score,
        )
    axes[0].set(
        xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz),
        xlabel='energy E/h (MHz)',
        ylabel='occupation',
        title='measured occupation-resolved FFT and candidate energies',
    )
    axes[0].set_yticks(row_indices)
    axes[0].set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=axes[0], label='spectral magnitude')
    axes[1].plot(energy_MHz[visible], ridge_score[visible], color='black')
    axes[1].plot(candidate_energies_MHz, candidate_scores, 'o', color='tab:red', markersize=8)
    for energy, score in zip(candidate_energies_MHz, candidate_scores):
        axes[1].annotate(
            f'{energy:.4f}',
            (energy, score),
            xytext=(0, 7),
            textcoords='offset points',
            ha='center',
            fontsize=8,
            rotation=45,
        )
    axes[1].set(
        xlabel='energy E/h (MHz)',
        ylabel='aligned row-peak votes',
        title=f'{len(candidate_indices)} ranked candidates; no fixed peak count',
    )
    fig.suptitle('data-only occupation-resolved FFT peak candidates')
    [
        (round(float(energy), 6), round(float(score), 2))
        for energy, score in ridge_ranked_candidates
    ]

    return aligned_peaks, candidate_indices, plt.gcf()


def load_and_analyze_n3(calibration_job_ids, spectroscopy_job_ids,
                        cycle_branches=None,
                        manual_kerr_MHz=-19.756e-3,
                        spectrum_method='matrix_pencil',
                        EncSpec=MBRSpectrumExperiment,
                        timing=None):
    """Load the N=3 jobs from HDF5 and analyze them (cell 189).

    Cell 189 called itself a tutorial: it is the worked path from job ranges
    to a choice of raw FFT or rowwise Matrix Pencil. Kept because it is the
    only place that path is written out end to end.

    Returns whatever the source's last binding produced, as a dict of the
    `encspec_N3_*` names the following cells read.
    """
    mpm_cycle_branches = {} if cycle_branches is None else cycle_branches
    mpm_manual_kerr_MHz = manual_kerr_MHz
    mpm_spectrum_method = spectrum_method

    # 1. Give complete job ranges. load_encoding_spectroscopy keeps only the requested calibration and spectroscopy programs and skips failed/unrelated jobs.
    mpm_calibration_job_ids = job_id_generator([20260722, 20260723], [683, 1], [712, 40])
    mpm_spectroscopy_job_ids = job_id_generator(20260723, [48, 87], [85, 149], step=[1, 2])
    mpm_calibration_expt, mpm_spectroscopy_expt = load_encoding_spectroscopy(
        EncSpec,
        mpm_calibration_job_ids,
        mpm_spectroscopy_job_ids,
        timing=timing,
        calibration_program_name='EntireFloquetCyclePhaseCalibrationProgram',
        spectroscopy_program_name='NPhotonHamiltonianSpectroscopyProgram',
    )

    # 2. Set the physical phase frame. Unlisted occupations use branch 0.
    mpm_cycle_branches = {
        # (3, 0, 0, 0, 0): 1,
    }
    mpm_manual_kerr_MHz = -19.756e-3
    mpm_spectrum_method = 'mpm' # use 'fft' for the existing FFT-only path

    # 3. These are the current MPM defaults. Keep them together so every heuristic choice is visible and editable.
    mpm_options = dict(
        mpm_requested_max_modes=35,
        mpm_pencil_length=None,
        mpm_minimum_consecutive_ranks=3,
        mpm_minimum_supporting_rows=1,
        mpm_track_frequency_tolerance_bins=1.5,
        mpm_merge_frequency_tolerance_bins=None,
        mpm_dedup_frequency_tolerance_bins=None,
        mpm_track_decay_tolerance_per_us=None,
        mpm_dedup_decay_tolerance_per_us=None,
        mpm_match_decay=True,
        mpm_numerical_floor=1e-10,
        mpm_noise_singular_value_factor=2.858,
        mpm_minimum_pole_radius=0.2,
        mpm_maximum_pole_radius=1.05,
        mpm_require_early_start=True,
        mpm_rank_sweep_extra=None,
        mpm_clip_growth=True,
        mpm_least_squares_rcond=None,
        mpm_store_rank_sweeps=False,
    )

    # 4. Reconstruct A(t), apply the requested Kerr/branch frame, calculate the reference FFT, then run MPM when selected.
    mpm_data = mpm_spectroscopy_expt.analyze(
        calibration=mpm_calibration_expt,
        phase_frame='manual_kerr',
        manual_kerr_MHz=mpm_manual_kerr_MHz,
        cycle_branches=mpm_cycle_branches,
        legacy=True,
        fft_window='raw',
        zero_padding=1,
        spectrum_method=mpm_spectrum_method,
        **mpm_options,
    )

    # 5. FFT keeps the original four-panel display. MPM shows raw FFT 2D, fitted-return FFT 2D, and the reconstructed DOS together.
    mpm_spectroscopy_expt.display(data=mpm_data, show_mpm_poles=True, plot_mpm_theory=False)

    if mpm_data.spectrum_method == 'matrix_pencil':
        print('selected frequencies (MHz):', np.round(mpm_data.matrix_pencil.selected_frequencies_MHz, 6))
        print('supporting occupations:', mpm_data.matrix_pencil.modes.supporting_row_counts)
        print('global relative residual:', mpm_data.matrix_pencil.relative_residual)
        print('resolved MPM settings:', dict(mpm_data.matrix_pencil.settings))

    return {
        name: value
        for name, value in locals().items()
        if name.startswith('encspec_N3_') or name.startswith('mpm_')
    }


def compare_trace_with_mpm(encspec_N3_spectrum, encspec_N3_time_us,
                           encspec_N3_trace, encspec_N3_trace_mpm):
    """Measured summed trace against its Matrix-Pencil reconstruction (cell 191).

    Also shows the extracted delta functions. Returns
    (energy_limit_MHz, fig).
    """
    encspec_N3_energy_MHz = np.asarray(encspec_N3_spectrum.energy_MHz, dtype=float)
    encspec_N3_fft_windows = {
        'raw': np.ones,
        'hann': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
    }
    encspec_N3_fft_window_name = encspec_N3_spectrum.fft_window or 'raw'
    encspec_N3_fft_window = encspec_N3_fft_windows[encspec_N3_fft_window_name](len(encspec_N3_time_us))
    encspec_N3_fft_scale = len(encspec_N3_energy_MHz) / np.sum(encspec_N3_fft_window)
    encspec_N3_trace_fft = encspec_N3_fft_scale * np.abs(
        np.fft.fftshift(
            np.fft.ifft(
                encspec_N3_trace * encspec_N3_fft_window,
                n=len(encspec_N3_energy_MHz),
            )
        )
    )
    encspec_N3_fitted_fft = encspec_N3_fft_scale * np.abs(
        np.fft.fftshift(
            np.fft.ifft(
                encspec_N3_trace_mpm.fitted_return * encspec_N3_fft_window,
                n=len(encspec_N3_energy_MHz),
            )
        )
    )
    encspec_N3_frequencies_MHz = encspec_N3_trace_mpm.selected_frequencies_MHz
    encspec_N3_DOS_weights = encspec_N3_trace_mpm.DOS_weights
    encspec_N3_pole_fft_heights = np.interp(
        encspec_N3_frequencies_MHz,
        encspec_N3_energy_MHz,
        encspec_N3_fitted_fft,
    )
    encspec_N3_energy_limit_MHz = float(encspec_N3_spectrum.energy_limit_MHz)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)

    axes[0].plot(encspec_N3_time_us, encspec_N3_trace.real, label='measured Re Z')
    axes[0].plot(encspec_N3_time_us, encspec_N3_trace.imag, label='measured Im Z')
    axes[0].plot(encspec_N3_time_us, encspec_N3_trace_mpm.fitted_return.real, '--', label='MPM Re Z')
    axes[0].plot(encspec_N3_time_us, encspec_N3_trace_mpm.fitted_return.imag, '--', label='MPM Im Z')
    axes[0].set(xlabel='time (us)', ylabel='trace amplitude', title='complete-basis trace')
    axes[0].legend()

    axes[1].plot(encspec_N3_energy_MHz, encspec_N3_trace_fft, color='black', label='measured trace FFT')
    axes[1].plot(encspec_N3_energy_MHz, encspec_N3_fitted_fft, '--', color='tab:blue', label='MPM reconstructed FFT')
    axes[1].plot(encspec_N3_frequencies_MHz, encspec_N3_pole_fft_heights, 'o', color='tab:blue', label='MPM poles')
    axes[1].set(
        xlim=(-encspec_N3_energy_limit_MHz, encspec_N3_energy_limit_MHz),
        xlabel='energy E/h (MHz)',
        ylabel='spectral magnitude',
        title='finite-time trace FFT',
    )
    axes[1].legend()

    axes[2].axhline(0., color='0.8', linewidth=1.)
    axes[2].vlines(encspec_N3_frequencies_MHz, 0., np.abs(encspec_N3_DOS_weights), color='tab:blue')
    axes[2].plot(encspec_N3_frequencies_MHz, np.abs(encspec_N3_DOS_weights), 'o', color='tab:blue')
    axes[2].set(
        xlim=(-encspec_N3_energy_limit_MHz, encspec_N3_energy_limit_MHz),
        xlabel='energy E/h (MHz)',
        ylabel='linear pole weight',
        title='Matrix-Pencil delta-functional DOS',
    )

    fig.suptitle(
        f'N=3 summed normalized-A Matrix Pencil; '
        f'K={len(encspec_N3_frequencies_MHz)}, '
        f'relative residual={encspec_N3_trace_mpm.relative_residual:.3f}'
    )

    return encspec_N3_energy_limit_MHz, plt.gcf()


def fit_self_kerr_from_peak_overlap(
        spectroscopy_expt, spectroscopy_data, calibration_expt,
        spectroscopy_occupations, cycle_branches,
        kerr_grid_kHz=None, energy_limit_MHz=0.08,
        min_man_photons=2, baseline_quantile=0.20):
    """Scan the signed M1 self-Kerr for best experiment--theory peak overlap.

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
            candidate_data = spectroscopy_expt.analyze(
                calibration=calibration_expt,
                occupations=spectroscopy_occupations,
                cycle_branches=cycle_branches,
                phase_frame="manual_kerr",
                manual_kerr_MHz=kerr_kHz * 1e-3,
                spectrum_method="fft",
            )
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
        spectroscopy_data = spectroscopy_expt.analyze(
            calibration=calibration_expt,
            occupations=spectroscopy_occupations,
            cycle_branches=cycle_branches,
            phase_frame="manual_kerr",
            manual_kerr_MHz=best_self_kerr_kHz * 1e-3,
            spectrum_method="fft",
        )

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
