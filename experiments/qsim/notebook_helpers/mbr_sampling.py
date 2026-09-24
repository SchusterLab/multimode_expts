"""Shot-count and randomized-occupation sampling studies on saved N=3 data.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
193-203 by the stage-2 notebook decomposition. Primary caller:
`analysis_notebooks/202609_qsim_migration/mbr_sampling.py`.

On the surface map this is one of the two "measurement and inference studies"
workspaces: it replays *already acquired* shots and asks what the spectrum
would have looked like with fewer of them, or with the occupation drawn at
random rather than scanned. Nothing here acquires data.

Two studies:

  193-197  saved-shot subsampling. Resample the stored shots at several
           counts, repeat with different seeds, and track how the FFT and the
           rowwise Matrix-Pencil spectra degrade.
  198-203  randomized-occupation replay. Draw the occupation per shot instead
           of sweeping it, under several protocols, and compare against the
           full scan.

The `encspec_N3_*` prefix is function arguments rather than notebook globals.
It was not collapsed into one config dataclass: unlike the disorder campaign's
thirty shared knobs, these split cleanly by study, and each function's
signature says what its own study needs.

`encspec_reprocessed`, `spectroscopy_expt` and `calibration_expt` came out of
the live kernel in the source -- the N=3 reprocessing section had to have been
run first. They are explicit arguments now; build them with
`mbr_n3_reprocess.reprocess_n3_spectroscopy`.

The source left the last caught exception bound in the notebook namespace
(`error`, `mpm_exception`) for a following cell to inspect. These functions
return their failures instead.

Temporary home, per the stage-2 instructions. The two sampling protocols are
deliberately kept distinct rather than unified, and neither was promoted to
`fitting/` -- the surface map says library promotion for this workspace can be
decided later.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict

from experiments.qsim.deprecated.legacy_mbr import MBRSpectrumExperiment


# --------------------------------------------------------------------------
# Saved-shot subsampling (cells 194-197).
# --------------------------------------------------------------------------


def run_shot_subsampling(spectroscopy_expt, calibration_expt,
                         encspec_reprocessed,
                         encspec_cycle_branches=None,
                         encspec_legacy=True,
                         encspec_manual_kerr_MHz=-19.756e-3,
                         repeats=3, base_seed=20260901,
                         axis_scale="log",
                         rowwise_mpm_kwargs=None):
    """Resample the stored shots at several counts (cell 194).

    Reads only saved shots -- no acquisition. Returns a dict of the
    `encspec_N3_*` names the summary and plotting steps need, plus a
    `failures` list in place of the source's dangling `error` binding.
    """
    encspec_N3_shot_repeats = repeats
    encspec_N3_shot_base_seed = base_seed
    encspec_N3_shot_axis_scale = axis_scale
    encspec_N3_shot_rowwise_mpm_kwargs = (
        {} if rowwise_mpm_kwargs is None else rowwise_mpm_kwargs
    )
    encspec_cycle_branches = (
        {} if encspec_cycle_branches is None else encspec_cycle_branches
    )
    encspec_N3_shot_failures = []

    # Re-wrap the already loaded N=3 child jobs with the current module class. This avoids downloading the jobs again after reloading the edited module.
    import importlib
    from experiments.qsim import floquet_dark_mode_readout

    importlib.reload(floquet_dark_mode_readout)
    EncSpec = MBRSpectrumExperiment
    encspec_N3_shot_sweep_expt = MBRSpectrumExperiment._from_expts(
        spectroscopy_expt.batch_expts,
        job_ids=spectroscopy_expt.batch_job_ids,
        station=getattr(spectroscopy_expt, '_analysis_station', None),
    )

    # shots_per_point is the number of final-readout shots retained for each saved cycle and preparation phase, not the total number used by one occupation.
    encspec_N3_shot_counts = np.arange(100, 1001, 100)#[50, 100, 200, 300, 500, 800, 1000]
    encspec_N3_shot_repeats = 10
    encspec_N3_shot_base_seed = 20260807
    encspec_N3_shot_axis_scale = 'linear'  # choose 'log' or 'linear'

    encspec_N3_shot_rowwise_mpm_kwargs = {}
    encspec_N3_shot_trace_mpm_kwargs = {
        'requested_max_modes': len(encspec_reprocessed.reconstruction.occupations),
        'track_frequency_tolerance_bins': 1.5,
        'dedup_frequency_tolerance_bins': 0.5,
    }

    encspec_N3_shot_reference = encspec_N3_shot_sweep_expt.analyze(
        calibration=calibration_expt,
        phase_frame='manual_kerr',
        manual_kerr_MHz=encspec_manual_kerr_MHz,
        cycle_branches=encspec_cycle_branches,
        legacy=encspec_legacy,
        fft_window='raw',
        zero_padding=1,
        spectrum_method='mpm',
        **encspec_N3_shot_rowwise_mpm_kwargs,
    )
    encspec_N3_reference_A_norm = np.asarray(
        encspec_N3_shot_reference.reconstruction.A_norm,
        dtype=complex,
    )
    encspec_N3_reference_trace = np.sum(encspec_N3_reference_A_norm, axis=0)

    encspec_N3_reference_energy_MHz = np.asarray(
        encspec_N3_shot_reference.spectrum.energy_MHz,
        dtype=float,
    )
    encspec_N3_reference_window = np.ones(len(encspec_N3_reference_trace))
    encspec_N3_reference_fft_scale = (
        len(encspec_N3_reference_energy_MHz)
        / np.sum(encspec_N3_reference_window)
    )
    encspec_N3_reference_trace_fft = encspec_N3_reference_fft_scale * np.abs(
        np.fft.fftshift(
            np.fft.ifft(
                encspec_N3_reference_trace * encspec_N3_reference_window,
                n=len(encspec_N3_reference_energy_MHz),
            )
        )
    )
    encspec_N3_reference_trace_mpm = EncSpec.analyze_matrix_pencil_trace(
        encspec_N3_reference_trace,
        encspec_N3_shot_reference.spectrum.time_us,
        **encspec_N3_shot_trace_mpm_kwargs,
    )



    #--------- Sampling single shot and repeat

    encspec_N3_shot_records = []

    for shot_count_index, shots_per_point in enumerate(encspec_N3_shot_counts):
        for repeat_index in range(encspec_N3_shot_repeats):
            shot_seed = (
                encspec_N3_shot_base_seed
                + 1000 * shot_count_index
                + repeat_index
            )
            print(
                f'{shots_per_point:4d} shots, repeat '
                f'{repeat_index + 1}/{encspec_N3_shot_repeats}',
                end='\r',
            )

            try:
                shot_data = encspec_N3_shot_sweep_expt.analyze(
                    calibration=calibration_expt,
                    phase_frame='manual_kerr',
                    manual_kerr_MHz=encspec_manual_kerr_MHz,
                    cycle_branches=encspec_cycle_branches,
                    legacy=encspec_legacy,
                    fft_window='raw',
                    zero_padding=1,
                    spectrum_method='mpm',
                    shots_per_point=shots_per_point,
                    shot_seed=shot_seed,
                    **encspec_N3_shot_rowwise_mpm_kwargs,
                )

                shot_A_norm = np.asarray(
                    shot_data.reconstruction.A_norm,
                    dtype=complex,
                )
                shot_trace = np.sum(shot_A_norm, axis=0)
                shot_trace_mpm = EncSpec.analyze_matrix_pencil_trace(
                    shot_trace,
                    shot_data.spectrum.time_us,
                    **encspec_N3_shot_trace_mpm_kwargs,
                )
                shot_trace_fft = encspec_N3_reference_fft_scale * np.abs(
                    np.fft.fftshift(
                        np.fft.ifft(
                            shot_trace * encspec_N3_reference_window,
                            n=len(encspec_N3_reference_energy_MHz),
                        )
                    )
                )

                encspec_N3_shot_records.append({
                    'shots_per_point': shots_per_point,
                    'repeat_index': repeat_index,
                    'seed': shot_seed,
                    'success': True,
                    'error': None,
                    'A_norm': shot_A_norm,
                    'trace': shot_trace,
                    'trace_fft': shot_trace_fft,
                    'trace_mpm': shot_trace_mpm,
                    'data': shot_data,
                })

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                encspec_N3_shot_records.append({
                    'shots_per_point': shots_per_point,
                    'repeat_index': repeat_index,
                    'seed': shot_seed,
                    'success': False,
                    'error': str(error),
                    'data': None,
                })

        records_at_count = [
            record
            for record in encspec_N3_shot_records
            if record['shots_per_point'] == shots_per_point
        ]
        successful_count = sum(record['success'] for record in records_at_count)
        print(
            f'{shots_per_point:4d} shots: '
            f'{successful_count}/{encspec_N3_shot_repeats} successful'
        )
        failures_at_count = [
            record for record in records_at_count if not record['success']
        ]
        if failures_at_count:
            print('    first failure:', failures_at_count[0]['error'])

    return {
        name: value
        for name, value in locals().items()
        if name.startswith("encspec_N3_")
    }


def summarize_shot_subsampling(shots):
    """Summarize and plot the shot-count sweep (cell 195).

    Returns a dict with the summary table, the colour map and the energy
    limit the following plots use.
    """
    encspec_N3_reference_A_norm = shots["encspec_N3_reference_A_norm"]
    encspec_N3_reference_energy_MHz = shots["encspec_N3_reference_energy_MHz"]
    encspec_N3_reference_trace_fft = shots["encspec_N3_reference_trace_fft"]
    encspec_N3_reference_trace_mpm = shots["encspec_N3_reference_trace_mpm"]
    encspec_N3_shot_axis_scale = shots["encspec_N3_shot_axis_scale"]
    encspec_N3_shot_counts = shots["encspec_N3_shot_counts"]
    encspec_N3_shot_records = shots["encspec_N3_shot_records"]
    encspec_N3_shot_reference = shots["encspec_N3_shot_reference"]
    encspec_N3_shot_repeats = shots["encspec_N3_shot_repeats"]

    # Measure repeat-to-repeat noise at each shot count. This does not assume that the full-shot reference is noiseless.
    encspec_N3_shot_summary = {}
    encspec_N3_reference_A_scale = np.sqrt(
        np.mean(np.abs(encspec_N3_reference_A_norm) ** 2)
    )
    encspec_N3_reference_fft_scale_RMS = np.sqrt(
        np.mean(encspec_N3_reference_trace_fft ** 2)
    )

    for shots_per_point in encspec_N3_shot_counts:
        records_at_count = [
            record
            for record in encspec_N3_shot_records
            if record['shots_per_point'] == shots_per_point
        ]
        successful_records = [
            record for record in records_at_count if record['success']
        ]

        if successful_records:
            sampled_A_norm = np.stack([
                record['A_norm'] for record in successful_records
            ])
            sampled_trace_fft = np.stack([
                record['trace_fft'] for record in successful_records
            ])
            pole_counts = np.asarray([
                len(record['trace_mpm'].selected_frequencies_MHz)
                for record in successful_records
            ])

            if len(successful_records) > 1:
                centered_A_norm = sampled_A_norm - np.mean(sampled_A_norm, axis=0)
                centered_trace_fft = (
                    sampled_trace_fft - np.mean(sampled_trace_fft, axis=0)
                )
                A_repeat_noise = (
                    np.sqrt(np.mean(np.abs(centered_A_norm) ** 2))
                    / encspec_N3_reference_A_scale
                )
                fft_repeat_noise = (
                    np.sqrt(np.mean(centered_trace_fft ** 2))
                    / encspec_N3_reference_fft_scale_RMS
                )
            else:
                A_repeat_noise = np.nan
                fft_repeat_noise = np.nan

            median_A_error = np.median([
                np.linalg.norm(record['A_norm'] - encspec_N3_reference_A_norm)
                / np.linalg.norm(encspec_N3_reference_A_norm)
                for record in successful_records
            ])
            median_fft_error = np.median([
                np.linalg.norm(record['trace_fft'] - encspec_N3_reference_trace_fft)
                / np.linalg.norm(encspec_N3_reference_trace_fft)
                for record in successful_records
            ])
        else:
            A_repeat_noise = np.nan
            fft_repeat_noise = np.nan
            median_A_error = np.nan
            median_fft_error = np.nan
            pole_counts = np.asarray([])

        encspec_N3_shot_summary[shots_per_point] = {
            'successful_records': successful_records,
            'failure_count': len(records_at_count) - len(successful_records),
            'A_repeat_noise': float(A_repeat_noise),
            'fft_repeat_noise': float(fft_repeat_noise),
            'median_A_error': float(median_A_error),
            'median_fft_error': float(median_fft_error),
            'pole_counts': pole_counts,
        }

        median_pole_count = np.median(pole_counts) if len(pole_counts) else np.nan
        print(
            f'{shots_per_point:4d} shots: '
            f'A noise={A_repeat_noise:.3g}, '
            f'trace-FFT noise={fft_repeat_noise:.3g}, '
            f'median K={median_pole_count:g}, '
            f'failures={encspec_N3_shot_summary[shots_per_point]["failure_count"]}'
        )


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    encspec_N3_shot_colors = plt.cm.viridis(
        np.linspace(0.1, 0.9, len(encspec_N3_shot_counts))
    )
    encspec_N3_energy_limit_MHz = float(
        encspec_N3_shot_reference.spectrum.energy_limit_MHz
    )

    # All successful trace FFTs remain visible, so the spread is not hidden by an average.
    ax1.plot(
        encspec_N3_reference_energy_MHz,
        encspec_N3_reference_trace_fft,
        color='black',
        linewidth=2,
        label='full saved shots',
    )
    for color, shots_per_point in zip(encspec_N3_shot_colors, encspec_N3_shot_counts):
        successful_records = encspec_N3_shot_summary[shots_per_point]['successful_records']
        for record_index, record in enumerate(successful_records):
            label = f'{shots_per_point} shots' if record_index == 0 else None
            ax1.plot(
                encspec_N3_reference_energy_MHz,
                record['trace_fft'],
                color=color,
                alpha=0.22,
                label=label,
            )
    ax1.set(
        xlim=(-encspec_N3_energy_limit_MHz, encspec_N3_energy_limit_MHz),
        xlabel='energy E/h (MHz)',
        ylabel='coherent trace-FFT magnitude',
        title='every reduced-shot trace FFT',
    )
    ax1.legend(ncol=2)

    # Every circle is one pole returned from one independently drawn shot subset.
    reference_frequencies_MHz = np.asarray(
        encspec_N3_reference_trace_mpm.selected_frequencies_MHz
    )

    for frequency_MHz in reference_frequencies_MHz:
        ax2.axvline(frequency_MHz, color='0.75', linewidth=1)
    for color, shots_per_point in zip(encspec_N3_shot_colors, encspec_N3_shot_counts):
        successful_records = encspec_N3_shot_summary[shots_per_point]['successful_records']
        for record in successful_records:
            frequencies_MHz = (
                record['trace_mpm'].selected_frequencies_MHz
            )
            repeat_offset = record['repeat_index'] - 0.5 * (encspec_N3_shot_repeats - 1)
            plotted_shot_count = shots_per_point * np.exp(0.025 * repeat_offset)
            ax2.scatter(
                frequencies_MHz,
                np.full(len(frequencies_MHz), plotted_shot_count),
                s=28,
                color=color,
                alpha=0.65,
            )
    ax2.plot([], [], 'o', color='0.4', label='subsampled trace-MPM poles')
    ax2.plot([], [], color='0.75', label='full-shot trace-MPM poles')
    ax2.set(
        xlim=(-encspec_N3_energy_limit_MHz, encspec_N3_energy_limit_MHz),
        xlabel='energy E/h (MHz)',
        ylabel='shots per saved sweep point',
        title='summed-trace MPM pole recurrence',
    )
    ax2.set_yscale(encspec_N3_shot_axis_scale)
    # axes[0, 1].legend()

    for color, shots_per_point in zip(encspec_N3_shot_colors, encspec_N3_shot_counts):
        summary = encspec_N3_shot_summary[shots_per_point]
        pole_counts = summary['pole_counts']
        if len(pole_counts):
            ax3.scatter(
                np.full(len(pole_counts), shots_per_point),
                pole_counts,
                color=color,
                alpha=0.7,
            )
        if summary['failure_count']:
            ax3.scatter(
                np.full(summary['failure_count'], shots_per_point),
                np.zeros(summary['failure_count']),
                marker='x',
                color='tab:red',
            )
    ax3.axhline(
        len(reference_frequencies_MHz),
        linestyle='--',
        color='black',
        label=f'full-shot K={len(reference_frequencies_MHz)}',
    )
    ax3.set(
        xlabel='shots per saved sweep point',
        ylabel='number of summed-trace MPM poles K',
        title='summed-trace pole count; red x means failure',
    )
    ax3.set_xscale(encspec_N3_shot_axis_scale)
    ax3.legend()

    fig.suptitle(
        'N=3 saved-shot subsampling; calibration kept at full statistics'
    )

    return {
        "encspec_N3_shot_summary": encspec_N3_shot_summary,
        "encspec_N3_shot_colors": encspec_N3_shot_colors,
        "encspec_N3_energy_limit_MHz": encspec_N3_energy_limit_MHz,
        "figure": plt.gcf(),
    }


def plot_rowwise_pole_counts(shots, summary):
    """How many rowwise MPM poles survive at each shot count (cell 196)."""
    encspec_N3_shot_axis_scale = shots["encspec_N3_shot_axis_scale"]
    encspec_N3_shot_counts = shots["encspec_N3_shot_counts"]
    encspec_N3_shot_reference = shots["encspec_N3_shot_reference"]
    encspec_N3_shot_repeats = shots["encspec_N3_shot_repeats"]
    encspec_N3_shot_summary = summary["encspec_N3_shot_summary"]
    encspec_N3_shot_colors = summary["encspec_N3_shot_colors"]
    encspec_N3_energy_limit_MHz = summary["encspec_N3_energy_limit_MHz"]

    # Show the complementary ordinary analyze() route without replacing the summed-trace cell above.
    # IMPORTANT: estimated_signal_rank is not a hard truncation of the rank sweep.
    # With rank_sweep_extra=None, every row is swept to its numerical/algebraic rank.
    # require_early_start only requires a pole history to begin by estimated_signal_rank,
    # and minimum_supporting_rows=1 allows a candidate supported by only one row.
    encspec_N3_rowwise_reference_frequencies_MHz = np.asarray(
        encspec_N3_shot_reference.matrix_pencil.selected_frequencies_MHz,
        dtype=float,
    )
    encspec_N3_rowwise_pole_counts = {}

    for shots_per_point in encspec_N3_shot_counts:
        successful_records = encspec_N3_shot_summary[shots_per_point][
            'successful_records'
        ]
        encspec_N3_rowwise_pole_counts[shots_per_point] = np.asarray([
            len(record['data'].matrix_pencil.selected_frequencies_MHz)
            for record in successful_records
        ])
        print(
            f'{shots_per_point:4d} shots: final K in each repeat = ',
            encspec_N3_rowwise_pole_counts[shots_per_point].tolist(),
        )

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), constrained_layout=True)

    for frequency_MHz in encspec_N3_rowwise_reference_frequencies_MHz:
        axes[0].axvline(frequency_MHz, color='0.75', linewidth=1)

    for color, shots_per_point in zip(
        encspec_N3_shot_colors,
        encspec_N3_shot_counts,
    ):
        successful_records = encspec_N3_shot_summary[shots_per_point][
            'successful_records'
        ]
        for record in successful_records:
            frequencies_MHz = np.asarray(
                record['data'].matrix_pencil.selected_frequencies_MHz,
                dtype=float,
            )
            repeat_offset = (
                record['repeat_index']
                - 0.5 * (encspec_N3_shot_repeats - 1)
            )
            plotted_shot_count = (
                shots_per_point * np.exp(0.025 * repeat_offset)
            )
            axes[0].scatter(
                frequencies_MHz,
                np.full(len(frequencies_MHz), plotted_shot_count),
                s=28,
                color=color,
                alpha=0.65,
            )

    axes[0].plot(
        [],
        [],
        'o',
        color='0.4',
        label='one dot = one final merged candidate',
    )
    axes[0].plot(
        [],
        [],
        color='0.75',
        label='full-shot rowwise-MPM poles',
    )
    axes[0].set(
        xlim=(-encspec_N3_energy_limit_MHz, encspec_N3_energy_limit_MHz),
        xlabel='energy E/h (MHz)',
        ylabel='shots per saved sweep point (repeat offsets)',
        title=(
            f'same analyze() call; {encspec_N3_shot_repeats} '
            'final pole set(s)'
        ),
    )
    axes[0].set_yscale(encspec_N3_shot_axis_scale)
    # axes[0].legend()

    for color, shots_per_point in zip(
        encspec_N3_shot_colors,
        encspec_N3_shot_counts,
    ):
        pole_counts = encspec_N3_rowwise_pole_counts[shots_per_point]
        axes[1].scatter(
            np.full(len(pole_counts), shots_per_point),
            pole_counts,
            color=color,
            alpha=0.7,
        )

    axes[1].axhline(
        len(encspec_N3_rowwise_reference_frequencies_MHz),
        linestyle='--',
        color='black',
        label=(
            'full-shot K='
            f'{len(encspec_N3_rowwise_reference_frequencies_MHz)}'
        ),
    )
    axes[1].set(
        xlabel='shots per saved sweep point',
        ylabel='number of selected shared candidates K',
        title='low-shot rowwise over-detection diagnostic',
    )
    axes[1].set_xscale(encspec_N3_shot_axis_scale)
    axes[1].legend()

    fig.suptitle(
        'N=3 ordinary rowwise MPM at reduced shots; each y-band '
        f'contains {encspec_N3_shot_repeats} independent subsample analysis(es)'
    )

    return plt.gcf()


def inspect_example_shots(shots, example_shots=None, example_repeat=0):
    """Look at one resampled spectrum in detail (cell 197)."""
    encspec_N3_shot_records = shots["encspec_N3_shot_records"]
    encspec_N3_shot_sweep_expt = shots["encspec_N3_shot_sweep_expt"]
    encspec_N3_example_shots = example_shots
    encspec_N3_example_repeat = example_repeat

    # Display one chosen reduced-shot realization with the existing 2-by-2 MPM figure.
    encspec_N3_example_shots = 200
    encspec_N3_example_repeat = 1

    example_matches = [
        record
        for record in encspec_N3_shot_records
        if (
            record['shots_per_point'] == encspec_N3_example_shots
            and record['repeat_index'] == encspec_N3_example_repeat
            and record['success']
        )
    ]
    if not example_matches:
        raise ValueError(
            'the selected shot-count realization did not finish successfully'
        )

    encspec_N3_shot_example = example_matches[0]['data']
    shot_metadata = encspec_N3_shot_example.shot_subsampling
    print(
        'used / minimum available shots per point:',
        shot_metadata.shots_per_point,
        '/',
        shot_metadata.minimum_available_shots,
    )
    print('random seed:', shot_metadata.seed)

    encspec_N3_shot_sweep_expt.display(
        data=encspec_N3_shot_example,
        spectrum_method='mpm',
    )

    return plt.gcf()


# --------------------------------------------------------------------------
# Randomized-occupation replay (cells 199-203).
# --------------------------------------------------------------------------


def build_random_occupation_pool(spectroscopy_expt, encspec_reprocessed,
                                 pool_seed=20260902):
    """Build the shot pool and reference spectra for the replay (cell 199).

    Returns a dict of the `encspec_N3_random_occ_*` names the sweep needs.
    """
    encspec_N3_random_occ_pool_seed = pool_seed

    # N=3 random-occupation trace sampling. Only the occupation is randomized in the first protocol; the existing four Ramsey settings remain fixed.
    # One complex occupation draw uses four final readouts: (phi, theta) = (0, 0), (0, 180), (90, 0), (90, 180).
    # The output is one randomized estimate of Z(t)=sum_i A_i(t)/A_i(0), not 35 separately reconstructed A_i traces.
    encspec_N3_random_occ_reference = encspec_reprocessed
    encspec_N3_random_occ_expts = list(spectroscopy_expt.batch_expts)
    encspec_N3_random_occ_occupations = [
        tuple(occupation)
        for occupation in encspec_N3_random_occ_reference.reconstruction.occupations
    ]
    encspec_N3_random_occ_cycles = np.asarray(
        encspec_N3_random_occ_reference.reconstruction.cycles
    )
    encspec_N3_random_occ_time_us = np.asarray(
        encspec_N3_random_occ_reference.spectrum.time_us,
        dtype=float,
    )
    encspec_N3_random_occ_dimension = len(encspec_N3_random_occ_occupations)
    encspec_N3_random_occ_time_count = len(encspec_N3_random_occ_time_us)
    encspec_N3_random_occ_occupation_index = {
        occupation: index
        for index, occupation in enumerate(encspec_N3_random_occ_occupations)
    }


    def _encspec_N3_random_occ_point_rows(values, point_count, name, job_index):
        # Match the saved-shot row handling used by EncSpec.subsample_spectroscopy_shots.
        try:
            array = np.asarray(values)
        except ValueError:
            array = None
        if array is not None and array.ndim >= 2 and array.shape[0] == point_count:
            return [
                np.asarray(array[index], dtype=float).reshape(-1)
                for index in range(point_count)
            ]
        if point_count == 1 and array is not None and array.dtype != object:
            return [np.asarray(array, dtype=float).reshape(-1)]
        if len(values) == point_count:
            return [
                np.asarray(values[index], dtype=float).reshape(-1)
                for index in range(point_count)
            ]
        raise ValueError(
            f'job {job_index} has {name} that does not match its '
            f'{point_count} saved sweep points'
        )


    # raw_soft_Pe[i, t, phi_index, theta_index] contains every saved final-I shot,
    # converted to the same Pe coordinate used by the ordinary averaged analysis.
    encspec_N3_random_occ_raw_soft_Pe = np.empty(
        (
            encspec_N3_random_occ_dimension,
            encspec_N3_random_occ_time_count,
            2,
            2,
        ),
        dtype=object,
    )
    encspec_N3_random_occ_raw_soft_Pe.fill(None)

    for job_index, expt in enumerate(encspec_N3_random_occ_expts):
        if (
            expt.cfg.expt.get('active_reset', False)
            and expt.cfg.expt.get('pre_selection_reset', False)
        ):
            raise ValueError('random-occupation replay does not support pre_selection_reset')

        occupation = tuple(expt.cfg.expt.spectroscopy_occupations)
        if occupation not in encspec_N3_random_occ_occupation_index:
            raise ValueError(f'unexpected spectroscopy occupation {occupation}')
        occupation_index = encspec_N3_random_occ_occupation_index[occupation]

        phi = float(expt.cfg.expt.spectroscopy_analyzer_phase)
        if phi not in (0.0, 90.0):
            raise ValueError(f'expected analyzer phase 0 or 90, received {phi}')
        phi_index = 0 if phi == 0.0 else 1

        local_cycles = np.asarray(expt.data['ypts'])
        preparation_phases = np.asarray(expt.data['xpts'], dtype=float)
        theta0 = np.flatnonzero(np.isclose(preparation_phases, 0.0))
        theta180 = np.flatnonzero(np.isclose(preparation_phases, 180.0))
        if len(theta0) != 1 or len(theta180) != 1:
            raise ValueError('every spectroscopy job must contain theta=0 and theta=180 once')
        theta_positions = [int(theta0[0]), int(theta180[0])]
        point_count = len(local_cycles) * len(preparation_phases)

        saved_avgi = np.asarray(expt.data['avgi'], dtype=float).reshape(
            len(local_cycles),
            len(preparation_phases),
        )
        idata_rows = _encspec_N3_random_occ_point_rows(
            expt.data['idata'],
            point_count,
            'idata',
            job_index,
        )

        read_num = int(expt.cfg.get('read_num', 0))
        if read_num < 1:
            raise ValueError(f'job {job_index} does not save a valid read_num')
        final_lane = read_num - 1
        q = int(expt.cfg.expt.qubits[0])
        Ig = float(expt.cfg.device.readout.Ig[q])
        Ie = float(expt.cfg.device.readout.Ie[q])
        if np.isclose(Ig, Ie):
            raise ValueError(f'job {job_index} has indistinguishable Ig and Ie')

        for local_time_index, cycle in enumerate(local_cycles):
            matching_times = np.flatnonzero(
                np.isclose(encspec_N3_random_occ_cycles, cycle)
            )
            if len(matching_times) != 1:
                raise ValueError(f'cycle {cycle} does not map to one reference time')
            time_index = int(matching_times[0])

            for theta_index, theta_position in enumerate(theta_positions):
                point_index = (
                    local_time_index * len(preparation_phases)
                    + theta_position
                )
                raw_i = idata_rows[point_index]
                if len(raw_i) % read_num:
                    raise ValueError(
                        f'job {job_index}, point {point_index}: raw-I length '
                        f'{len(raw_i)} is not divisible by read_num={read_num}'
                    )
                final_i = raw_i[final_lane::read_num]
                if len(final_i) < 2:
                    raise ValueError('at least two final-readout shots are required')

                # QICK's averaged avgi and collect_shots can differ by an ADC offset.
                # Add each raw fluctuation to the saved average before converting to Pe.
                centered_i = (
                    saved_avgi[local_time_index, theta_position]
                    + final_i
                    - np.mean(final_i)
                )
                soft_Pe = (centered_i - Ig) / (Ie - Ig)
                key = (occupation_index, time_index, phi_index, theta_index)
                if encspec_N3_random_occ_raw_soft_Pe[key] is not None:
                    raise ValueError(f'duplicate saved-shot pool at key {key}')
                encspec_N3_random_occ_raw_soft_Pe[key] = soft_Pe

    missing_random_occ_cells = [
        index
        for index in np.ndindex(encspec_N3_random_occ_raw_soft_Pe.shape)
        if encspec_N3_random_occ_raw_soft_Pe[index] is None
    ]
    if missing_random_occ_cells:
        raise ValueError(
            'saved spectroscopy archive is incomplete; first missing cells: '
            f'{missing_random_occ_cells[:5]}'
        )


    # Split every raw cell once into disjoint A/B pools. For independent fresh pools,
    # their cross product is an unbiased SFF estimator; in this finite-archive replay
    # it suppresses the positive self-shot-noise term in abs(Z_hat)**2.
    encspec_N3_random_occ_pool_seed = 20260810
    encspec_N3_random_occ_pool_rng = np.random.default_rng(
        encspec_N3_random_occ_pool_seed
    )
    encspec_N3_random_occ_raw_pools = np.empty(
        encspec_N3_random_occ_raw_soft_Pe.shape + (2,),
        dtype=object,
    )
    for index in np.ndindex(encspec_N3_random_occ_raw_soft_Pe.shape):
        values = np.asarray(encspec_N3_random_occ_raw_soft_Pe[index], dtype=float)
        permutation = encspec_N3_random_occ_pool_rng.permutation(len(values))
        midpoint = len(values) // 2
        encspec_N3_random_occ_raw_pools[index + (0,)] = values[permutation[:midpoint]]
        encspec_N3_random_occ_raw_pools[index + (1,)] = values[permutation[midpoint:]]

    encspec_N3_random_occ_available_per_half = np.asarray([
        len(encspec_N3_random_occ_raw_pools[index])
        for index in np.ndindex(encspec_N3_random_occ_raw_pools.shape)
    ])


    # Raw packets are in the acquired analyzer frame. Recover the deterministic
    # occupation/time phase multiplier already used by the full-stat analysis.
    encspec_N3_random_occ_acquired_A = np.asarray(
        encspec_N3_random_occ_reference.acquired_reconstruction.A,
        dtype=complex,
    )
    encspec_N3_random_occ_processed_A = np.asarray(
        encspec_N3_random_occ_reference.reconstruction.A,
        dtype=complex,
    )
    if np.any(np.abs(encspec_N3_random_occ_acquired_A) < 1e-12):
        raise ValueError('cannot infer the phase-frame multiplier at a zero acquired return')
    encspec_N3_random_occ_frame_factor = (
        encspec_N3_random_occ_processed_A
        / encspec_N3_random_occ_acquired_A
    )
    encspec_N3_random_occ_frame_factor /= np.abs(
        encspec_N3_random_occ_frame_factor
    )
    if not np.allclose(
        encspec_N3_random_occ_acquired_A * encspec_N3_random_occ_frame_factor,
        encspec_N3_random_occ_processed_A,
    ):
        raise ValueError('the inferred phase-frame multiplier does not reproduce reconstruction.A')

    # Use full-stat A_i(0) only as a fixed visibility calibration. This isolates the
    # state-selection and saved-shot noise; a future acquisition must budget this calibration.
    encspec_N3_random_occ_initial_return = (
        encspec_N3_random_occ_processed_A[:, 0].copy()
    )
    if np.any(np.abs(encspec_N3_random_occ_initial_return) < 1e-12):
        raise ValueError('every full-stat A_i(0) calibration must be nonzero')
    encspec_N3_random_occ_reference_A_norm = np.asarray(
        encspec_N3_random_occ_reference.reconstruction.A_norm,
        dtype=complex,
    )
    encspec_N3_random_occ_reference_Z = np.sum(
        encspec_N3_random_occ_reference_A_norm,
        axis=0,
    )
    encspec_N3_random_occ_reference_K = (
        np.abs(encspec_N3_random_occ_reference_Z) ** 2
        / encspec_N3_random_occ_dimension ** 2
    )


    # Self-contained summed-trace MPM settings. Change them here if the trace-MPM
    # settings above are changed; this cell does not require encspec_N3_trace_mpm.
    encspec_N3_random_occ_mpm_kwargs = {
        # A scalar 40-point trace has an algebraic MPM ceiling of 20 modes.
        'requested_max_modes': min(
            encspec_N3_random_occ_dimension,
            encspec_N3_random_occ_time_count // 2,
        ),
        'track_frequency_tolerance_bins': 1.5,
        'dedup_frequency_tolerance_bins': 1.5,
        # Restore the unknown-noise singular-value threshold. A value of 0.1
        # classifies nearly the entire noisy Hankel spectrum as signal.
        'noise_singular_value_factor': 2.858,
        # Sweep only far enough beyond the estimated rank to test the required
        # three-rank persistence, instead of following noise poles to full rank.
        'rank_sweep_extra': 2,
    }
    encspec_N3_random_occ_reference_mpm = spectroscopy_expt.analyze_matrix_pencil_trace(
        encspec_N3_random_occ_reference_Z,
        encspec_N3_random_occ_time_us,
        **encspec_N3_random_occ_mpm_kwargs,
    )

    encspec_N3_random_occ_energy_MHz = np.asarray(
        encspec_N3_random_occ_reference.spectrum.energy_MHz,
        dtype=float,
    )
    encspec_N3_random_occ_fft_window = np.ones(
        encspec_N3_random_occ_time_count
    )
    encspec_N3_random_occ_fft_scale = (
        len(encspec_N3_random_occ_energy_MHz)
        / np.sum(encspec_N3_random_occ_fft_window)
    )


    def _encspec_N3_random_occ_fft(trace):
        return encspec_N3_random_occ_fft_scale * np.abs(
            np.fft.fftshift(
                np.fft.ifft(
                    trace * encspec_N3_random_occ_fft_window,
                    n=len(encspec_N3_random_occ_energy_MHz),
                )
            )
        )


    def _encspec_N3_draw_state_packets(pool_index, draw_count, rng):
        # Occupation-only randomization: one draw uses the same occupation in all four fixed settings.
        trace = np.empty(encspec_N3_random_occ_time_count, dtype=complex)
        state_only_trace = np.empty_like(trace)
        occupation_counts = np.empty(
            (encspec_N3_random_occ_time_count, encspec_N3_random_occ_dimension),
            dtype=int,
        )
        probabilities = np.full(
            encspec_N3_random_occ_dimension,
            1.0 / encspec_N3_random_occ_dimension,
        )

        for time_index in range(encspec_N3_random_occ_time_count):
            counts = rng.multinomial(draw_count, probabilities)
            occupation_counts[time_index] = counts
            packet_sum = 0.0j

            for occupation_index, count in enumerate(counts):
                if count == 0:
                    continue
                selected = {}
                for phi_index in range(2):
                    for theta_index in range(2):
                        pool = encspec_N3_random_occ_raw_pools[
                            occupation_index,
                            time_index,
                            phi_index,
                            theta_index,
                            pool_index,
                        ]
                        if count > len(pool):
                            raise ValueError(
                                f'occupation draw count {count} exceeds disjoint '
                                f'raw-pool size {len(pool)}'
                            )
                        indices = rng.choice(len(pool), size=count, replace=False)
                        selected[phi_index, theta_index] = pool[indices]

                acquired_packets = (
                    selected[0, 0]
                    - selected[0, 1]
                    - 1j * (selected[1, 0] - selected[1, 1])
                )
                normalized_packets = (
                    encspec_N3_random_occ_frame_factor[occupation_index, time_index]
                    * acquired_packets
                    / encspec_N3_random_occ_initial_return[occupation_index]
                )
                packet_sum += np.sum(normalized_packets)

            trace[time_index] = packet_sum / draw_count
            state_only_trace[time_index] = (
                counts
                @ encspec_N3_random_occ_reference_A_norm[:, time_index]
                / draw_count
            )

        return trace, state_only_trace, occupation_counts


    def _encspec_N3_draw_all_random_shots(pool_index, shot_count, rng):
        # Full randomization: occupation, theta, and phi are redrawn for every physical readout.
        trace = np.empty(encspec_N3_random_occ_time_count, dtype=complex)
        occupation_counts = np.empty(
            (encspec_N3_random_occ_time_count, encspec_N3_random_occ_dimension),
            dtype=int,
        )
        category_count = 4 * encspec_N3_random_occ_dimension
        probabilities = np.full(category_count, 1.0 / category_count)
        analyzer_factor = np.asarray([1.0 + 0.0j, -1.0j])
        theta_sign = np.asarray([1.0, -1.0])

        for time_index in range(encspec_N3_random_occ_time_count):
            counts = rng.multinomial(shot_count, probabilities).reshape(
                encspec_N3_random_occ_dimension,
                2,
                2,
            )
            occupation_counts[time_index] = np.sum(counts, axis=(1, 2))
            sample_sum = 0.0j

            for occupation_index in range(encspec_N3_random_occ_dimension):
                frame_and_visibility = (
                    encspec_N3_random_occ_frame_factor[occupation_index, time_index]
                    / encspec_N3_random_occ_initial_return[occupation_index]
                )
                for phi_index in range(2):
                    for theta_index in range(2):
                        count = int(counts[occupation_index, phi_index, theta_index])
                        if count == 0:
                            continue
                        pool = encspec_N3_random_occ_raw_pools[
                            occupation_index,
                            time_index,
                            phi_index,
                            theta_index,
                            pool_index,
                        ]
                        if count > len(pool):
                            raise ValueError(
                                f'random setting count {count} exceeds disjoint '
                                f'raw-pool size {len(pool)}'
                            )
                        indices = rng.choice(len(pool), size=count, replace=False)
                        soft_Pe = pool[indices]
                        # The factor 2*(2*Pe-1) makes one uniformly random theta/phi
                        # shot an unbiased estimator of Q0-iQ90 in the current convention.
                        complex_samples = (
                            2.0
                            * analyzer_factor[phi_index]
                            * theta_sign[theta_index]
                            * (2.0 * soft_Pe - 1.0)
                            * frame_and_visibility
                        )
                        sample_sum += np.sum(complex_samples)

            trace[time_index] = sample_sum / shot_count

        return trace, occupation_counts

    print('random-occupation archive shape:', encspec_N3_random_occ_raw_soft_Pe.shape)
    print(
        'available final-I shots per disjoint half:',
        int(np.min(encspec_N3_random_occ_available_per_half)),
        'to',
        int(np.max(encspec_N3_random_occ_available_per_half)),
    )
    print(
        'This one Hamiltonian has only a few degenerate lines; it cannot validate '
        'recovery of 35 disorder-split levels.'
    )
    print(
        'With T=', encspec_N3_random_occ_time_count,
        'the scalar trace-MPM algebraic rank ceiling is',
        min(
            encspec_N3_random_occ_reference_mpm.settings.pencil_length,
            encspec_N3_random_occ_time_count
            - encspec_N3_random_occ_reference_mpm.settings.pencil_length,
        ),
        '; this offline replay cannot test drift suppression from online interleaving.',
    )

    pool = {
        name: value
        for name, value in locals().items()
        if name.startswith("encspec_N3_random_occ")
    }
    # The three closures the sweep and the plots call. They read this
    # function's locals, which is why they are not module-level.
    pool["_random_occ_fft"] = _encspec_N3_random_occ_fft
    pool["_draw_state_packets"] = _encspec_N3_draw_state_packets
    pool["_draw_all_random_shots"] = _encspec_N3_draw_all_random_shots
    return pool


def run_random_occupation_sweep(spectroscopy_expt, pool, repeats=3,
                                base_seed=20260903, protocols=None):
    """Replay the shots with the occupation drawn at random (cells 200-201).

    Returns a dict with the per-protocol records, the protocol list, the
    measurements-per-time axis, and a `failures` list in place of the
    source's dangling `error`/`mpm_exception` bindings.
    """
    encspec_N3_random_occ_repeats = repeats
    encspec_N3_random_occ_base_seed = base_seed
    encspec_N3_random_occ_dimension = pool["encspec_N3_random_occ_dimension"]
    encspec_N3_random_occ_mpm_kwargs = pool[
        "encspec_N3_random_occ_mpm_kwargs"
    ]
    encspec_N3_random_occ_reference_K = pool[
        "encspec_N3_random_occ_reference_K"
    ]
    encspec_N3_random_occ_reference_Z = pool[
        "encspec_N3_random_occ_reference_Z"
    ]
    encspec_N3_random_occ_time_us = pool["encspec_N3_random_occ_time_us"]
    encspec_N3_random_occ_failures = []
    _encspec_N3_random_occ_fft = pool["_random_occ_fft"]
    _encspec_N3_draw_state_packets = pool["_draw_state_packets"]
    _encspec_N3_draw_all_random_shots = pool["_draw_all_random_shots"]

    # This is the number of physical single-shot measurements used to form EACH time point.
    # MPM is run once, only after the complete Tr U(t) trace has been formed.
    encspec_N3_random_occ_measurements_per_time = np.arange(2000, 30001, 2000)#np.linspace(2000, 30000, 2000)
    encspec_N3_random_occ_repeats = 1
    encspec_N3_random_occ_base_seed = 20260811
    encspec_N3_random_occ_records = []
    encspec_N3_random_occ_protocols = ['state_only', 'state_theta_phi']

    for protocol_index, protocol in enumerate(encspec_N3_random_occ_protocols):
        print('protocol:', protocol)
        for measurement_index, measurements_per_time in enumerate(
            encspec_N3_random_occ_measurements_per_time
        ):
            if measurements_per_time % 4:
                raise ValueError('measurements_per_time must be divisible by four')
            occupation_packet_count = measurements_per_time // 4

            for repeat_index in range(encspec_N3_random_occ_repeats):
                seed = (
                    encspec_N3_random_occ_base_seed
                    + 100000 * protocol_index
                    + 1000 * measurement_index
                    + repeat_index
                )
                rng_A = np.random.default_rng(seed)
                rng_B = np.random.default_rng(seed + 500000)

                try:
                    if protocol == 'state_only':
                        packet_count_A = occupation_packet_count // 2
                        packet_count_B = occupation_packet_count - packet_count_A
                        z_A, state_only_A, counts_A = _encspec_N3_draw_state_packets(
                            0, packet_count_A, rng_A
                        )
                        z_B, state_only_B, counts_B = _encspec_N3_draw_state_packets(
                            1, packet_count_B, rng_B
                        )
                        z_hat = (
                            packet_count_A * z_A + packet_count_B * z_B
                        ) / occupation_packet_count
                        state_only_hat = (
                            packet_count_A * state_only_A
                            + packet_count_B * state_only_B
                        ) / occupation_packet_count
                    else:
                        shot_count_A = measurements_per_time // 2
                        shot_count_B = measurements_per_time - shot_count_A
                        z_A, counts_A = _encspec_N3_draw_all_random_shots(
                            0, shot_count_A, rng_A
                        )
                        z_B, counts_B = _encspec_N3_draw_all_random_shots(
                            1, shot_count_B, rng_B
                        )
                        z_hat = (
                            shot_count_A * z_A + shot_count_B * z_B
                        ) / measurements_per_time
                        state_only_hat = None

                    # The random single-shot results have now been combined into one
                    # Z_hat(t)=Tr U(t) estimate. MPM is applied only after this step.
                    Z_hat = encspec_N3_random_occ_dimension * z_hat
                    Z_A = encspec_N3_random_occ_dimension * z_A
                    Z_B = encspec_N3_random_occ_dimension * z_B
                    K_naive = (
                        np.abs(Z_hat) ** 2
                        / encspec_N3_random_occ_dimension ** 2
                    )
                    K_cross = (
                        np.real(Z_A * np.conj(Z_B))
                        / encspec_N3_random_occ_dimension ** 2
                    )
                    trace_fft = _encspec_N3_random_occ_fft(Z_hat)
                    try:
                        trace_mpm = spectroscopy_expt.analyze_matrix_pencil_trace(
                            Z_hat,
                            encspec_N3_random_occ_time_us,
                            **encspec_N3_random_occ_mpm_kwargs,
                        )
                        mpm_success = True
                        mpm_error = None
                        K_mpm_fit = (
                            np.abs(trace_mpm.fitted_return) ** 2
                            / encspec_N3_random_occ_dimension ** 2
                        )
                        level_trace = np.sum(
                            trace_mpm.DOS_weights[:, None]
                            * np.exp(
                                -2j
                                * np.pi
                                * trace_mpm.selected_frequencies_MHz[:, None]
                                * encspec_N3_random_occ_time_us[None, :]
                            ),
                            axis=0,
                        )
                        K_from_mpm_levels = (
                            np.abs(level_trace) ** 2
                            / encspec_N3_random_occ_dimension ** 2
                        )
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as mpm_exception:
                        trace_mpm = None
                        mpm_success = False
                        mpm_error = str(mpm_exception)
                        K_mpm_fit = np.full_like(K_cross, np.nan)
                        K_from_mpm_levels = np.full_like(K_cross, np.nan)
                    all_counts = counts_A + counts_B
                    mean_counts = np.mean(all_counts, axis=1)
                    count_cv = np.mean(
                        np.std(all_counts, axis=1)
                        / np.maximum(mean_counts, np.finfo(float).eps)
                    )

                    encspec_N3_random_occ_records.append({
                        'protocol': protocol,
                        'measurements_per_time': measurements_per_time,
                        'repeat_index': repeat_index,
                        'seed': seed,
                        'success': True,
                        'error': None,
                        'mpm_success': mpm_success,
                        'mpm_error': mpm_error,
                        'Z': Z_hat,
                        'state_only_Z': (
                            None
                            if state_only_hat is None
                            else encspec_N3_random_occ_dimension * state_only_hat
                        ),
                        'fft': trace_fft,
                        'mpm': trace_mpm,
                        'K_naive': K_naive,
                        'K_cross': K_cross,
                        'K_mpm_fit': K_mpm_fit,
                        'K_from_mpm_levels': K_from_mpm_levels,
                        'occupation_counts': all_counts,
                        'occupation_count_cv': float(count_cv),
                        'trace_relative_error': float(
                            np.linalg.norm(
                                Z_hat - encspec_N3_random_occ_reference_Z
                            )
                            / np.linalg.norm(encspec_N3_random_occ_reference_Z)
                        ),
                        'state_selection_relative_error': (
                            np.nan
                            if state_only_hat is None
                            else float(
                                np.linalg.norm(
                                    encspec_N3_random_occ_dimension * state_only_hat
                                    - encspec_N3_random_occ_reference_Z
                                )
                                / np.linalg.norm(encspec_N3_random_occ_reference_Z)
                            )
                        ),
                        'fft_relative_error': float(
                            np.linalg.norm(
                                trace_fft
                                - _encspec_N3_random_occ_fft(
                                    encspec_N3_random_occ_reference_Z
                                )
                            )
                            / np.linalg.norm(
                                _encspec_N3_random_occ_fft(
                                    encspec_N3_random_occ_reference_Z
                                )
                            )
                        ),
                        'sff_cross_relative_error': float(
                            np.linalg.norm(
                                K_cross - encspec_N3_random_occ_reference_K
                            )
                            / np.linalg.norm(encspec_N3_random_occ_reference_K)
                        ),
                        'sff_naive_relative_error': float(
                            np.linalg.norm(
                                K_naive - encspec_N3_random_occ_reference_K
                            )
                            / np.linalg.norm(encspec_N3_random_occ_reference_K)
                        ),
                        'sff_mpm_fit_relative_error': float(
                            np.linalg.norm(
                                K_mpm_fit - encspec_N3_random_occ_reference_K
                            )
                            / np.linalg.norm(encspec_N3_random_occ_reference_K)
                        ),
                        'sff_from_mpm_levels_relative_error': float(
                            np.linalg.norm(
                                K_from_mpm_levels
                                - encspec_N3_random_occ_reference_K
                            )
                            / np.linalg.norm(encspec_N3_random_occ_reference_K)
                        ),
                    })

                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    encspec_N3_random_occ_records.append({
                        'protocol': protocol,
                        'measurements_per_time': measurements_per_time,
                        'repeat_index': repeat_index,
                        'seed': seed,
                        'success': False,
                        'error': str(error),
                    })

            records_here = [
                record
                for record in encspec_N3_random_occ_records
                if (
                    record['protocol'] == protocol
                    and record['measurements_per_time'] == measurements_per_time
                )
            ]
            successful_here = [record for record in records_here if record['success']]
            successful_mpm_here = [
                record for record in successful_here if record['mpm_success']
            ]
            returned_K = [
                len(record['mpm'].selected_frequencies_MHz)
                for record in successful_mpm_here
            ]
            print(
                f'  single-shot measurements/time={measurements_per_time:,}; '
                f'complete averaged traces={len(successful_here)}/{encspec_N3_random_occ_repeats}; '
                f'MPM={len(successful_mpm_here)}/{len(successful_here)}; '
                f'K={returned_K}'
            )

    return {
        name: value
        for name, value in locals().items()
        if name.startswith("encspec_N3_random_occ")
    }


def plot_random_occupation_results(
        pool, sweep,
        overview_figsize=(16, 9), summary_figsize=(12, 7),
        fig_dpi=120, legend_fontsize=8, protocol_styles=None):
    """The FFT, MPM frequency and SFF comparisons per protocol (cells 202-203)."""
    RANDOM_OCC_OVERVIEW_FIGSIZE = overview_figsize
    RANDOM_OCC_SUMMARY_FIGSIZE = summary_figsize
    RANDOM_OCC_FIG_DPI = fig_dpi
    RANDOM_OCC_LEGEND_FONTSIZE = legend_fontsize
    protocol_styles = {} if protocol_styles is None else protocol_styles

    encspec_N3_random_occ_energy_MHz = pool[
        "encspec_N3_random_occ_energy_MHz"
    ]
    encspec_N3_random_occ_reference = pool["encspec_N3_random_occ_reference"]
    encspec_N3_random_occ_reference_K = pool[
        "encspec_N3_random_occ_reference_K"
    ]
    encspec_N3_random_occ_reference_Z = pool[
        "encspec_N3_random_occ_reference_Z"
    ]
    encspec_N3_random_occ_reference_mpm = pool[
        "encspec_N3_random_occ_reference_mpm"
    ]
    encspec_N3_random_occ_time_us = pool["encspec_N3_random_occ_time_us"]
    encspec_N3_random_occ_measurements_per_time = sweep[
        "encspec_N3_random_occ_measurements_per_time"
    ]
    encspec_N3_random_occ_protocols = sweep[
        "encspec_N3_random_occ_protocols"
    ]
    encspec_N3_random_occ_records = sweep["encspec_N3_random_occ_records"]
    _encspec_N3_random_occ_fft = pool["_random_occ_fft"]

    encspec_N3_random_occ_colors = plt.cm.viridis(
        np.linspace(0.08, 0.92, len(encspec_N3_random_occ_measurements_per_time))
    )
    encspec_N3_random_occ_reference_fft = _encspec_N3_random_occ_fft(
        encspec_N3_random_occ_reference_Z
    )
    encspec_N3_random_occ_reference_frequencies_MHz = np.asarray(
        encspec_N3_random_occ_reference_mpm.selected_frequencies_MHz,
        dtype=float,
    )
    encspec_N3_random_occ_energy_limit_MHz = float(
        encspec_N3_random_occ_reference.spectrum.energy_limit_MHz
    )

    def _encspec_N3_random_occ_successes(protocol, measurements_per_time, require_mpm=False):
        return [
            record
            for record in encspec_N3_random_occ_records
            if (
                record['protocol'] == protocol
                and record['measurements_per_time'] == measurements_per_time
                and record['success']
                and (not require_mpm or record['mpm_success'])
            )
        ]

    fig, axes = plt.subplots(
        3, 2,
        figsize=RANDOM_OCC_OVERVIEW_FIGSIZE,
        dpi=RANDOM_OCC_FIG_DPI,
        constrained_layout=True,
    )

    for column, protocol in enumerate(encspec_N3_random_occ_protocols):
        axes[0, column].plot(
            encspec_N3_random_occ_energy_MHz,
            encspec_N3_random_occ_reference_fft,
            color='black',
            linewidth=2,
            label='full saved data',
        )
        for color, measurements_per_time in zip(
            encspec_N3_random_occ_colors,
            encspec_N3_random_occ_measurements_per_time,
        ):
            records = _encspec_N3_random_occ_successes(protocol, measurements_per_time)
            if not records:
                continue
            ffts = np.stack([record['fft'] for record in records])
            median = np.median(ffts, axis=0)
            lower, upper = np.quantile(ffts, [0.1, 0.9], axis=0)
            axes[0, column].plot(
                encspec_N3_random_occ_energy_MHz,
                median,
                color=color,
                label=f'{measurements_per_time} shots/time',
            )
            axes[0, column].fill_between(
                encspec_N3_random_occ_energy_MHz,
                lower,
                upper,
                color=color,
                alpha=0.08,
            )
        axes[0, column].set(
            xlim=(-encspec_N3_random_occ_energy_limit_MHz, encspec_N3_random_occ_energy_limit_MHz),
            xlabel='energy E/h (MHz)',
            ylabel='trace-FFT magnitude',
            title=(
                'random occupation; four fixed settings'
                if protocol == 'state_only'
                else 'random occupation, theta, and phi per shot'
            ),
        )
        axes[0, column].legend(ncol=2, fontsize=RANDOM_OCC_LEGEND_FONTSIZE)

        for frequency_MHz in encspec_N3_random_occ_reference_frequencies_MHz:
            axes[1, column].axvline(frequency_MHz, color='0.75', linewidth=1)
        for color, measurements_per_time in zip(
            encspec_N3_random_occ_colors,
            encspec_N3_random_occ_measurements_per_time,
        ):
            records = _encspec_N3_random_occ_successes(
                protocol, measurements_per_time, require_mpm=True
            )
            for record in records:
                frequencies_MHz = record['mpm'].selected_frequencies_MHz
                plotted_measurements = measurements_per_time
                axes[1, column].scatter(
                    frequencies_MHz,
                    np.full(len(frequencies_MHz), plotted_measurements),
                    color=color,
                    s=22,
                    alpha=0.65,
                )
        axes[1, column].set(
            xlim=(-encspec_N3_random_occ_energy_limit_MHz, encspec_N3_random_occ_energy_limit_MHz),
            xlabel='energy E/h (MHz)',
            ylabel='single-shot measurements used per time point',
            title=r'MPM frequencies from each completed $\mathrm{Tr}\,U(t)$ trace',
        )
        axes[1, column].set_yscale('linear')

        # Direct cross-pool SFF from the randomized trace. The dashed red curve is
        # reconstructed from the MPM frequencies and fitted DOS weights at the
        # largest measurement count; it is model-dependent.
        axes[2, column].plot(
            encspec_N3_random_occ_time_us,
            encspec_N3_random_occ_reference_K,
            color='black',
            linewidth=2,
            label='full saved data',
        )
        sff_measurement_counts = [40, 200, 1000, 4000, 20000]
        for color, measurements_per_time in zip(
            encspec_N3_random_occ_colors,
            encspec_N3_random_occ_measurements_per_time,
        ):
            if measurements_per_time not in sff_measurement_counts:
                continue
            records = _encspec_N3_random_occ_successes(
                protocol,
                measurements_per_time,
            )
            if not records:
                continue
            cross_sff = np.stack([record['K_cross'] for record in records])
            mean_sff = np.mean(cross_sff, axis=0)
            lower_sff, upper_sff = np.quantile(
                cross_sff,
                [0.1, 0.9],
                axis=0,
            )
            axes[2, column].plot(
                encspec_N3_random_occ_time_us,
                mean_sff,
                color=color,
                label=f'{measurements_per_time} shots/time',
            )
            axes[2, column].fill_between(
                encspec_N3_random_occ_time_us,
                lower_sff,
                upper_sff,
                color=color,
                alpha=0.10,
            )

        highest_measurements = max(encspec_N3_random_occ_measurements_per_time)
        highest_mpm_records = _encspec_N3_random_occ_successes(
            protocol,
            highest_measurements,
            require_mpm=True,
        )
        if highest_mpm_records:
            axes[2, column].plot(
                encspec_N3_random_occ_time_us,
                np.mean(
                    np.stack([
                        record['K_from_mpm_levels']
                        for record in highest_mpm_records
                    ]),
                    axis=0,
                ),
                '--',
                color='tab:red',
                linewidth=1.5,
                label=f'{highest_measurements} shots/time: MPM levels',
            )
        axes[2, column].axhline(0.0, color='0.75', linewidth=1)
        axes[2, column].set(
            xlabel='time (us)',
            ylabel=r'SFF $|\mathrm{Tr}U/D|^2$',
            title='spectral form factor',
        )
        axes[2, column].set_yscale('linear')
        axes[2, column].legend(ncol=2, fontsize=RANDOM_OCC_LEGEND_FONTSIZE)

    fig.suptitle(
        r'At each time: combine random single shots into one $\mathrm{Tr}\,U(t)$ point; '
        'MPM runs once on each completed trace'
    )


    fig, axes = plt.subplots(
        1, 3,
        figsize=RANDOM_OCC_SUMMARY_FIGSIZE,
        dpi=RANDOM_OCC_FIG_DPI,
        constrained_layout=True,
    )
    protocol_styles = {
        'state_only': ('tab:blue', 'random occupation; four fixed settings'),
        'state_theta_phi': ('tab:orange', 'occupation, theta, phi all random'),
    }

    for protocol, (color, label) in protocol_styles.items():
        trace_medians = []
        trace_lowers = []
        trace_uppers = []
        cross_sff_medians = []
        naive_sff_medians = []
        state_selection_medians = []
        pole_counts_by_measurement_count = []

        for measurements_per_time in encspec_N3_random_occ_measurements_per_time:
            records = _encspec_N3_random_occ_successes(protocol, measurements_per_time)
            if not records:
                trace_medians.append(np.nan)
                trace_lowers.append(np.nan)
                trace_uppers.append(np.nan)
                cross_sff_medians.append(np.nan)
                naive_sff_medians.append(np.nan)
                state_selection_medians.append(np.nan)
                pole_counts_by_measurement_count.append(np.asarray([]))
                continue
            trace_errors = np.asarray([record['trace_relative_error'] for record in records])
            trace_medians.append(np.median(trace_errors))
            trace_lowers.append(np.quantile(trace_errors, 0.1))
            trace_uppers.append(np.quantile(trace_errors, 0.9))
            cross_sff_medians.append(np.median([
                record['sff_cross_relative_error'] for record in records
            ]))
            naive_sff_medians.append(np.median([
                record['sff_naive_relative_error'] for record in records
            ]))
            state_selection_medians.append(np.median([
                record['state_selection_relative_error'] for record in records
            ]))
            mpm_records = [record for record in records if record['mpm_success']]
            pole_counts_by_measurement_count.append(np.asarray([
                len(record['mpm'].selected_frequencies_MHz) for record in mpm_records
            ]))

        trace_medians = np.asarray(trace_medians)
        axes[0].plot(
            encspec_N3_random_occ_measurements_per_time,
            trace_medians,
            'o-',
            color=color,
            label=label,
        )
        axes[0].fill_between(
            encspec_N3_random_occ_measurements_per_time,
            trace_lowers,
            trace_uppers,
            color=color,
            alpha=0.15,
        )
        if protocol == 'state_only':
            axes[0].plot(
                encspec_N3_random_occ_measurements_per_time,
                state_selection_medians,
                'o--',
                color='tab:green',
                label='occupation-selection noise only',
            )
        axes[1].plot(
            encspec_N3_random_occ_measurements_per_time,
            cross_sff_medians,
            'o-',
            color=color,
            label=f'{label}: cross-pool SFF',
        )
        axes[1].plot(
            encspec_N3_random_occ_measurements_per_time,
            naive_sff_medians,
            'o--',
            color=color,
            alpha=0.65,
            label=f'{label}: naive |Z|^2',
        )

        for measurements_per_time, pole_counts in zip(
            encspec_N3_random_occ_measurements_per_time,
            pole_counts_by_measurement_count,
        ):
            axes[2].scatter(
                np.full(len(pole_counts), measurements_per_time),
                pole_counts,
                color=color,
                alpha=0.65,
            )

    axes[0].set(
        xscale='log',
        yscale='log',
        xlabel='single-shot measurements used per time point',
        ylabel='relative error of randomized Z(t)',
        title='trace convergence; bands are 10-90%',
    )
    axes[0].legend(fontsize=RANDOM_OCC_LEGEND_FONTSIZE)
    axes[1].set(
        xscale='log',
        yscale='log',
        xlabel='single-shot measurements used per time point',
        ylabel='relative SFF error',
        title='cross-pool estimate suppresses the positive self-noise term',
    )
    axes[1].legend(fontsize=RANDOM_OCC_LEGEND_FONTSIZE)
    axes[2].axhline(
        len(encspec_N3_random_occ_reference_frequencies_MHz),
        color='black',
        linestyle='--',
        label=f'full-trace K={len(encspec_N3_random_occ_reference_frequencies_MHz)}',
    )
    axes[2].set(
        xscale='log',
        xlabel='single-shot measurements used per time point',
        ylabel='returned MPM pole count K',
        title='MPM model-order stability',
    )
    axes[2].legend(fontsize=RANDOM_OCC_LEGEND_FONTSIZE)

    fig.suptitle(
        'Randomized protocol comparison at the same physical-readout budget; '
        'current data validate only this one Hamiltonian'
    )

    return plt.gcf()
