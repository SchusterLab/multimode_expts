"""N=3 FFT and Matrix-Pencil diagnostics -- DEPRECATED.

Moved from `experiments/qsim/notebook_helpers/mbr_n3_reprocess.py` on 2026-09-26 (MBR
redesign step 8C), without changes. Why: `docs/qsim/mbr_step8_plan.md`, decision 3 (guan):
the exploratory diagnostics of the N=3 reprocessing leave the canonical analysis notebook.
Their notebook cells are `analysis_notebooks/202609_qsim_migration/dormant/mbr_n3_diagnostics.py`.
Not maintained; may break when live code changes. If it breaks, add a note here and do not
fix it.

Source: `measurement_notebooks/jonginn/data_postprocess.ipynb` cells 174-192. Cell 179 read
the saved FFT window and then set `'hann'`; that is `window_name`, default `'hann'`.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks, savgol_filter

from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment



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


def load_and_analyze_n3(spectrum_manifest,
                        cycle_branches=None,
                        manual_kerr_MHz=-19.756e-3,
                        spectrum_method='matrix_pencil'):
    """Load the saved N=3 spectrum and analyze it (cell 189).

    Cell 189 called itself a tutorial: it is the worked path from saved data
    to a choice of raw FFT or rowwise Matrix Pencil. Kept because it is the
    only place that path is written out end to end.

    ``spectrum_manifest`` is a saved `MBRSpectrumExperiment` with its
    calibration set linked (old jobs reach it through
    `tools/migrate_mbr_jobs.py`). Returns the `mpm_*` names as a dict.
    """
    # 1. Load the traces and their calibration set from the manifest.
    mpm_spectroscopy_expt = MBRSpectrumExperiment.from_manifest(spectrum_manifest)
    mpm_calibration_expt = mpm_spectroscopy_expt.calibration

    # 2. Set the physical phase frame. Unlisted occupations use branch 0.
    mpm_cycle_branches = {} if cycle_branches is None else cycle_branches
    mpm_manual_kerr_MHz = manual_kerr_MHz
    mpm_spectrum_method = spectrum_method  # 'fft' for the FFT-only path

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
    mpm_spectroscopy_expt.display(show_mpm_poles=True)

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


def matrix_pencil_global_diagnostic(encspec_reprocessed):
    """Global-only Matrix Pencil (cell 188).

    Moved from `mbr_spectral_validation` in MBR redesign step 7a
    (2026-09-24), without changes. Reads only `reconstruction.A` and
    `spectrum.time_us`, so it takes the new `MBRSpectrumExperiment.analyze()`
    result.

    Uses no Hamiltonian energies and no FFT peak positions. The next
    step performs the rowwise selection and merging."""
    # Global-only Matrix Pencil diagnostic; the next cell performs rowwise selection and merging.
    # No Hamiltonian energies or FFT peak positions are used.

    matrix_pencil_data = encspec_reprocessed
    matrix_pencil_requested_max_modes = 35
    matrix_pencil_numerical_floor = 1e-10
    matrix_pencil_min_persistence_fraction = 0.6

    reconstruction = matrix_pencil_data.reconstruction
    spectrum = matrix_pencil_data.spectrum
    if not hasattr(reconstruction, 'A'):
        raise ValueError('Matrix Pencil requires complex reconstruction.A; a magnitude-only merged spectrum cannot be used')

    time_us = np.asarray(spectrum.time_us, dtype=float)
    complex_return = np.asarray(reconstruction.A, dtype=complex)
    if complex_return.ndim != 2 or complex_return.shape[1] != len(time_us):
        raise ValueError('reconstruction.A must have shape (occupation, time point)')
    if len(time_us) < 5:
        raise ValueError('Matrix Pencil requires at least five time points')

    sample_time_us = time_us[1] - time_us[0]
    if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
        raise ValueError('Matrix Pencil requires uniformly spaced time points')
    if np.any(np.abs(complex_return[:, 0]) < 1e-12):
        raise ValueError('at least one occupation has zero return at the first time point')
    complex_return = complex_return / complex_return[:, :1]

    # 1. Every row supplies a shifted pair of Hankel matrices with the same poles.
    sample_count = complex_return.shape[1]
    pencil_length = sample_count // 2
    unshifted_blocks = []
    shifted_blocks = []
    for row in complex_return:
        windows = sliding_window_view(row, pencil_length + 1)
        unshifted_blocks.append(windows[:, :-1])
        shifted_blocks.append(windows[:, 1:])
    unshifted = np.vstack(unshifted_blocks)
    shifted = np.vstack(shifted_blocks)

    # 2. SVD separates the shared exponential subspace from small residual directions.
    left_vectors, singular_values, right_vectors_h = np.linalg.svd(
        unshifted,
        full_matrices=False,
    )
    relative_singular_values = singular_values / singular_values[0]
    numerical_rank = np.count_nonzero(
        relative_singular_values > matrix_pencil_numerical_floor
    )
    maximum_rank = min(
        matrix_pencil_requested_max_modes,
        pencil_length,
        numerical_rank,
    )
    if maximum_rank < 1:
        raise RuntimeError('the Hankel matrix has no usable singular direction')

    # 3. Solve the reduced pencil at every allowed rank instead of fixing the number of tones.
    rank_results = {}
    for rank in range(1, maximum_rank + 1):
        left = left_vectors[:, :rank]
        right = right_vectors_h[:rank].conj().T
        shifted_reduced = left.conj().T @ shifted @ right
        inverse_singular_values = np.diag(1. / singular_values[:rank])
        reduced_pencil = inverse_singular_values @ shifted_reduced
        poles = np.linalg.eigvals(reduced_pencil)

        frequencies_MHz = -np.angle(poles) / (2. * np.pi * sample_time_us)
        pole_radii = np.abs(poles)
        decay_per_us = -np.log(np.maximum(pole_radii, np.finfo(float).tiny)) / sample_time_us
        order = np.argsort(frequencies_MHz)
        rank_results[rank] = dict(
            frequencies_MHz=frequencies_MHz[order],
            pole_radii=pole_radii[order],
            decay_per_us=decay_per_us[order],
        )

    # 4. A maximum-rank pole is more credible when lower-rank solutions return near it.
    resolution_MHz = 1. / (sample_count * sample_time_us)
    maximum_rank_result = rank_results[maximum_rank]
    maximum_rank_frequencies = maximum_rank_result['frequencies_MHz']
    anchor_groups = []
    for index, frequency in enumerate(maximum_rank_frequencies):
        if not anchor_groups:
            anchor_groups.append([index])
            continue

        previous_index = anchor_groups[-1][-1]
        previous_frequency = maximum_rank_frequencies[previous_index]
        if frequency - previous_frequency <= resolution_MHz:
            anchor_groups[-1].append(index)
        else:
            anchor_groups.append([index])

    anchor_indices = []
    for group in anchor_groups:
        group_radii = maximum_rank_result['pole_radii'][group]
        distance_from_unit_circle = np.abs(np.log(np.maximum(
            group_radii,
            np.finfo(float).tiny,
        )))
        representative = group[np.argmin(distance_from_unit_circle)]
        anchor_indices.append(representative)

    ranked_candidates = []
    for index in anchor_indices:
        frequency = maximum_rank_result['frequencies_MHz'][index]
        radius = maximum_rank_result['pole_radii'][index]
        decay = maximum_rank_result['decay_per_us'][index]
        matched_frequencies = []
        for result in rank_results.values():
            distances = np.abs(result['frequencies_MHz'] - frequency)
            nearest = np.argmin(distances)
            if distances[nearest] <= resolution_MHz:
                matched_frequencies.append(result['frequencies_MHz'][nearest])

        ranked_candidates.append(dict(
            frequency_MHz=float(np.median(matched_frequencies)),
            persistence=len(matched_frequencies),
            pole_radius=float(radius),
            decay_per_us=float(decay),
        ))
    ranked_candidates.sort(
        key=lambda candidate: (
            -candidate['persistence'],
            abs(np.log(max(candidate['pole_radius'], np.finfo(float).tiny))),
        )
    )
    minimum_persistence = int(np.ceil(
        matrix_pencil_min_persistence_fraction * maximum_rank
    ))
    stable_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate['persistence'] >= minimum_persistence
    ]

    encspec_matrix_pencil = dict(
        sample_time_us=sample_time_us,
        pencil_length=pencil_length,
        maximum_algebraic_rank=maximum_rank,
        resolution_MHz=resolution_MHz,
        singular_values=singular_values,
        relative_singular_values=relative_singular_values,
        rank_results=rank_results,
        ranked_candidates=ranked_candidates,
        minimum_persistence=minimum_persistence,
        stable_candidates=stable_candidates,
    )

    # 5. Plot the rank information without pretending that the maximum rank is the mode count.
    nyquist_MHz = 0.5 / sample_time_us
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    singular_index = np.arange(1, len(singular_values) + 1)
    axes[0].semilogy(singular_index, relative_singular_values, 'o-')
    axes[0].set(
        xlabel='singular-value index',
        ylabel='singular value / largest singular value',
        title='Hankel singular values; no rank selected',
    )

    plot_candidates = sorted(
        ranked_candidates,
        key=lambda candidate: candidate['frequency_MHz'],
    )
    candidate_frequencies = [candidate['frequency_MHz'] for candidate in plot_candidates]
    candidate_persistence = [candidate['persistence'] for candidate in plot_candidates]
    axes[1].vlines(
        candidate_frequencies,
        0.,
        candidate_persistence,
        color='0.75',
    )
    axes[1].scatter(
        candidate_frequencies,
        candidate_persistence,
        color='0.6',
        s=70,
        label='all maximum-rank candidates',
    )
    stable_frequencies = [candidate['frequency_MHz'] for candidate in stable_candidates]
    stable_persistence = [candidate['persistence'] for candidate in stable_candidates]
    axes[1].scatter(
        stable_frequencies,
        stable_persistence,
        color='tab:red',
        s=85,
        label='recurrent candidates',
    )
    for frequency, persistence in zip(stable_frequencies, stable_persistence):
        axes[1].annotate(
            f'{frequency:.4f}',
            (frequency, persistence),
            xytext=(0, 7),
            textcoords='offset points',
            ha='center',
            fontsize=8,
            rotation=45,
        )
    axes[1].axvline(0., color='0.7', linewidth=1.)
    axes[1].set(
        xlim=(-nyquist_MHz, nyquist_MHz),
        ylim=(0., maximum_rank + 1.),
        xlabel='energy E/h (MHz)',
        ylabel='number of ranks returning near this frequency',
        title=f'red: returned in at least {minimum_persistence} of {maximum_rank} ranks',
    )
    axes[1].legend()
    figure.suptitle(
        f'data-only multichannel Matrix Pencil; algebraic ceiling={maximum_rank}, not a selected mode count'
    )
    dict(
        minimum_persistence=minimum_persistence,
        stable_candidates=[
            (
                round(candidate['frequency_MHz'], 6),
                candidate['persistence'],
            )
            for candidate in stable_candidates
        ],
    )
