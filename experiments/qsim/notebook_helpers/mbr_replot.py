"""Report-figure machinery for the MBR spectroscopy replots.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
206-211 by the stage-2 notebook decomposition. Primary caller:
`analysis_notebooks/202609_qsim_migration/mbr.py`.

Most of that range was already top-level `def`s, so those moved with their
bodies intact. Two things changed around them.

First, the settings. Cell 206 defined nineteen `replot_*` names at notebook
scope and the functions read them as globals. The stage-2 instructions call for
grouping a repeated settings prefix like that into one plainly named config
object, so there is now `ReplotConfig`, passed explicitly. Its defaults are
cell 206's values, so a notebook wanting the original report figures can build
it with no arguments.

Second, the names lost the `replot_` prefix, which was only ever there to keep
one flat notebook namespace apart. The module is the namespace now.

Only the `def`s live here. Every call site stayed in the notebook -- the
overview-figure loop (cell 209), the panel export loop (cell 210) and the
time-trace request list (cell 211) are all dataset and figure choices.

`load_and_analyze_sectors` is the exception: it is cell 208's loop, which is
pure plumbing over the four photon-number sectors. Its N=2 supplement handling
is optional rather than assumed, because that supplement is a property of one
dataset -- one occupation was acquired on a different time grid, so only its
FFT rows can join the report spectrum, not its time traces.

Temporary home, per the stage-2 instructions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from slab import AttrDict

from experiments.qsim.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.legacy_mbr import MBRSpectrumExperiment
from experiments.saved_jobs import load_aggregate


@dataclass
class ReplotConfig:
    """The `replot_*` settings cell 206 kept at notebook scope."""

    manual_kerr_MHz: float = -19.756e-3
    cycle_branches: dict = field(
        default_factory=lambda: {1: {}, 2: {}, 3: {}, 4: {}}
    )
    fft_window: str = "raw"
    zero_padding: int = 1

    # Figure controls for every plot in the source's "Replot" section.
    overview_figsize: Any = (15, 12.3)
    single_panel_figsize: Any = None  # None keeps the automatic panel height.
    trace_figsize: Any = None  # None -> (10, 3.2 * number_of_traces).
    figure_dpi: Any = None  # Display DPI; None uses matplotlib.rcParams.
    save_dpi: int = 300
    legend_fontsize: int = 9
    suptitle_fontsize: int = 14
    overview_legend_ncols: int = 3

    # Cell 206 aliased this to MBRSpectrumExperiment. Kept configurable
    # because the source treated it as a knob.
    EncSpec: Any = MBRSpectrumExperiment


def expand_job_ranges(ranges):
    return [
        f'JOB-{date}-{number:05d}'
        for date, first, last, step in ranges
        for number in range(first, last + 1, step)
    ]


def load_sector(job_ranges, config, timing=None):
    """-> (calibration_expt, spectroscopy_expt, load_info) for one sector.

    Both aggregates come from HDF5 via :mod:`experiments.saved_jobs`. The job
    ranges for these sectors were submitted interleaved with other programs,
    so each is filtered by the program class recorded in the provenance
    sidecar -- which is checked before a file is opened, rather than by
    unpickling each job and inspecting its `prog`.
    """
    calibration_expt = load_aggregate(
        expand_job_ranges(job_ranges['calibration']),
        owner=MBRPhaseCorrectionExperiment,
        program_class='EntireFloquetCyclePhaseCalibrationProgram',
        timing=timing,
        analyze=True,
    )
    spectroscopy_expt = load_aggregate(
        expand_job_ranges(job_ranges['spectroscopy']),
        owner=config.EncSpec,
        program_class='NPhotonHamiltonianSpectroscopyProgram',
        timing=timing,
    )

    load_info = AttrDict(dict(
        calibration_loaded_ids=calibration_expt.batch_job_ids,
        spectroscopy_loaded_ids=spectroscopy_expt.batch_job_ids,
        calibration_skipped=calibration_expt.skipped_jobs,
        spectroscopy_skipped=spectroscopy_expt.skipped_jobs,
    ))
    return calibration_expt, spectroscopy_expt, load_info


def analyze_sector(N, calibration_expt, spectroscopy_expt, config):
    analyze_kwargs = dict(
        calibration=calibration_expt,
        phase_frame='manual_kerr',
        manual_kerr_MHz=config.manual_kerr_MHz,
        cycle_branches=config.cycle_branches[N],
        legacy=True,
        fft_window=config.fft_window,
        zero_padding=config.zero_padding,
        spectrum_method='matrix_pencil',
    )
    if N == 2:
        analyze_kwargs['mpm_merge_frequency_tolerance_bins'] = 0.5

    data = spectroscopy_expt.analyze(**analyze_kwargs)
    # if int(data.photon_number) != N:
    #     raise ValueError(f'expected N={N}, loaded N={data.photon_number}')
    return data


# ------------------------------------------------------------------------
# Cell 209: figure post-processing.
# ------------------------------------------------------------------------


def move_legends_outside(
        fig, *, ncols=3, figsize=None, legend_fontsize=None,
        save_path=None, save_dpi=300):
    # Collect and de-duplicate every axes legend, including the heatmap pole key.
    by_label = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith('_'):
                by_label.setdefault(label, handle)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if by_label:
        try:
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='outside lower center',
                ncols=ncols,
                frameon=False,
                fontsize=legend_fontsize,
            )
        except ValueError:
            # Fallback for Matplotlib versions without the outside-loc syntax.
            fig.subplots_adjust(bottom=0.18)
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='lower center',
                bbox_to_anchor=(0.5, 0.01),
                ncols=ncols,
                frameon=False,
                fontsize=legend_fontsize,
            )

    if figsize is not None:
        fig.set_size_inches(*figsize)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
    return fig


def relabel_artists(ax, replacements):
    for artist in [*ax.lines, *ax.collections]:
        label = artist.get_label()
        if label in replacements:
            artist.set_label(replacements[label])


def annotate_theory_scaling(fig, data, plot_kind):
    # analyze_spectrum globally multiplies the finite-time theory FFT so that
    # max(theory FFT sum) == max(measured FFT sum).  The exact Hamiltonian
    # delta-function/projector weights are physical weights and are not scaled.
    axes = fig.axes[:4]
    exact_weight_name = (
        'exact Hamiltonian DOS weights'
        if data.spectrum.complete_basis
        else 'exact projected spectral weights'
    )
    scale_note = (
        'finite-time theory FFT globally peak-rescaled to measured FFT; '
        f'{exact_weight_name} unscaled'
    )

    if plot_kind == 'mpm':
        axes[1].set_title(
            'theory finite-time FFT\n(globally peak-rescaled to measurement)'
        )
        axes[3].set_title(
            axes[3].get_title()
            + '\n(theory FFT curve rescaled; exact weights unscaled)'
        )
        relabel_artists(axes[3], {
            'theory FFT sum': 'theory FFT sum (globally peak-rescaled)',
            'exact Hamiltonian DOS weights': (
                'exact Hamiltonian DOS weights (unscaled)'
            ),
            'exact projected spectral weights': (
                'exact projected spectral weights (unscaled)'
            ),
        })
    elif plot_kind == 'fft':
        axes[1].set_title(
            axes[1].get_title()
            + '\n(globally peak-rescaled to measurement)'
        )
        axes[2].set_title(
            axes[2].get_title() + '\n(unscaled exact projector weights)'
        )
        axes[3].set_title(
            axes[3].get_title()
            + '\n(theory FFT curve globally peak-rescaled)'
        )
        relabel_artists(axes[3], {
            'theory': 'theory FFT sum (globally peak-rescaled)',
        })
    else:
        raise ValueError("plot_kind must be 'mpm' or 'fft'")

    return scale_note


# ------------------------------------------------------------------------
# Cell 210: export any report subplot as its own figure.
# ------------------------------------------------------------------------


def plot_single_spectroscopy_panel(
        N, plot_kind, panel, runs, panel_names_by_kind, *,
        EncSpec=MBRSpectrumExperiment, show_poles=True, figsize=None,
        figure_dpi=None, legend_fontsize=None, save_path=None,
        save_dpi=300):
    """Export one report subplot as its own figure (cell 210).

    `runs`, `panel_names_by_kind` and `ReplotEncSpec` were notebook globals
    that this body read; they are arguments now, because the move is what
    broke them. `panel_names_by_kind` is cell 210's own settings block, which
    stayed in the notebook.
    """
    plot_kind = str(plot_kind).lower()
    if plot_kind not in panel_names_by_kind:
        raise ValueError("plot_kind must be 'mpm' or 'fft'")
    if panel not in panel_names_by_kind[plot_kind]:
        raise ValueError(
            f'{plot_kind} panel must be one of '
            f'{panel_names_by_kind[plot_kind]}'
        )

    run = runs[N]
    data = run.mpm_data if plot_kind == 'mpm' else run.report_data
    if plot_kind == 'mpm' and data.get('spectrum_only', False):
        raise ValueError('MPM panels require rows on one common time grid')

    spectrum = data.spectrum
    matrix_pencil = (
        data.get('matrix_pencil', None) if plot_kind == 'mpm' else None
    )
    if plot_kind == 'mpm' and matrix_pencil is None:
        raise ValueError(
            "MPM data are unavailable; analyze with spectrum_method='matrix_pencil'"
        )

    occupations = [
        tuple(value) for value in data.reconstruction.occupations
    ]
    energy_MHz = np.asarray(spectrum.energy_MHz, dtype=float)
    energy_limit_MHz = float(spectrum.energy_limit_MHz)
    figure_height = (
        max(6.8, 0.28 * len(occupations) + 2.5)
        if panel == 'local_dos' else 6.8
    )
    resolved_figsize = (
        (9.5, figure_height + 0.5)
        if figsize is None else tuple(figsize)
    )
    fig, ax = plt.subplots(
        figsize=resolved_figsize,
        dpi=figure_dpi,
        constrained_layout=True,
    )

    if panel in ('measured_map', 'theory_map'):
        measured_local = np.asarray(spectrum.measured_local, dtype=float)
        theory_local = np.asarray(spectrum.theory_local, dtype=float)
        local = measured_local if panel == 'measured_map' else theory_local
        vmax = max(
            float(np.max(measured_local)), float(np.max(theory_local))
        )
        image = ax.imshow(
            local,
            origin='lower',
            aspect='auto',
            interpolation='nearest',
            extent=[
                energy_MHz[0], energy_MHz[-1],
                -0.5, len(occupations) - 0.5,
            ],
            cmap='magma',
            vmin=0.0,
            vmax=vmax,
        )
        if plot_kind == 'mpm' and show_poles:
            for frequency_index, frequency_MHz in enumerate(
                    matrix_pencil.selected_frequencies_MHz):
                ax.axvline(
                    frequency_MHz,
                    color='cyan',
                    linewidth=0.8,
                    alpha=0.45,
                    label=(
                        'selected shared Matrix-Pencil poles'
                        if frequency_index == 0 else None
                    ),
                )
            if panel == 'measured_map':
                ax.scatter(
                    [
                        candidate.frequency_MHz
                        for candidate in matrix_pencil.candidates.per_row
                    ],
                    [
                        candidate.row_index
                        for candidate in matrix_pencil.candidates.per_row
                    ],
                    s=24,
                    facecolors='none',
                    edgecolors='cyan',
                    linewidths=0.8,
                    label='rowwise Matrix-Pencil poles',
                )
        ax.set_yticks(np.arange(len(occupations)))
        ax.set_yticklabels(
            [str(value) for value in occupations],
            fontsize=7 if len(occupations) > 15 else 9,
        )
        ax.set(
            xlim=(-energy_limit_MHz, energy_limit_MHz),
            xlabel='energy E/h (MHz)',
            ylabel=f'occupation {data.mode_labels}',
            title=(
                'measured finite-time FFT'
                if panel == 'measured_map'
                else (
                    'theory finite-time FFT\n'
                    '(globally peak-rescaled to measurement)'
                )
            ),
        )
        fig.colorbar(image, ax=ax, label='spectral magnitude')

    elif plot_kind == 'mpm' and panel == 'measured_dos':
        ax.plot(
            energy_MHz,
            spectrum.measured,
            color='black',
            linewidth=1.7,
            label='measured FFT sum',
        )
        ax.plot(
            energy_MHz,
            matrix_pencil.reconstructed,
            color='tab:blue',
            linestyle='--',
            linewidth=1.5,
            label='Matrix-Pencil finite-time reconstruction',
        )
        ax.vlines(
            matrix_pencil.selected_frequencies_MHz,
            0.0,
            matrix_pencil.pole_DOS_weights,
            color='tab:blue',
            alpha=0.7,
            label='Matrix-Pencil linear pole DOS weights',
        )
        ax.plot(
            matrix_pencil.selected_frequencies_MHz,
            matrix_pencil.pole_DOS_weights,
            'o',
            color='tab:blue',
            markersize=5,
        )
        if show_poles:
            for frequency_MHz in matrix_pencil.selected_frequencies_MHz:
                ax.axvline(
                    frequency_MHz,
                    color='cyan',
                    linewidth=0.7,
                    alpha=0.35,
                )
        ax.set(
            xlim=(-energy_limit_MHz, energy_limit_MHz),
            xlabel='energy E/h (MHz)',
            ylabel='spectral magnitude / pole weight',
            title=(
                'measured FFT sum and Matrix-Pencil DOS'
                if spectrum.complete_basis
                else 'measured projected FFT sum and Matrix-Pencil weights'
            ),
        )

    elif plot_kind == 'mpm' and panel == 'theory_dos':
        exact_energies_MHz, exact_energy_indices = np.unique(
            np.round(np.asarray(spectrum.energies_MHz, dtype=float), 10),
            return_inverse=True,
        )
        exact_state_weights = np.sum(
            np.asarray(spectrum.eigenstate_weights, dtype=float), axis=0
        )
        exact_DOS_weights = np.bincount(
            exact_energy_indices,
            weights=exact_state_weights,
            minlength=len(exact_energies_MHz),
        )
        ax.plot(
            energy_MHz,
            spectrum.theory,
            color='tab:orange',
            linewidth=1.7,
            label='theory FFT sum (globally peak-rescaled)',
        )
        ax.vlines(
            exact_energies_MHz,
            0.0,
            exact_DOS_weights,
            color='tab:orange',
            alpha=0.7,
            label=(
                'exact Hamiltonian DOS weights (unscaled)'
                if spectrum.complete_basis
                else 'exact projected spectral weights (unscaled)'
            ),
        )
        ax.plot(
            exact_energies_MHz,
            exact_DOS_weights,
            'o',
            color='tab:orange',
            markersize=5,
        )
        if show_poles:
            for frequency_MHz in matrix_pencil.selected_frequencies_MHz:
                ax.axvline(
                    frequency_MHz,
                    color='cyan',
                    linewidth=0.7,
                    alpha=0.35,
                )
        ax.set(
            xlim=(-energy_limit_MHz, energy_limit_MHz),
            xlabel='energy E/h (MHz)',
            ylabel='spectral magnitude / DOS weight',
            title=(
                'rescaled theory FFT sum and unscaled exact DOS'
                if spectrum.complete_basis
                else (
                    'rescaled theory projected FFT sum and '
                    'unscaled exact spectral weights'
                )
            ),
        )

    elif plot_kind == 'fft' and panel == 'local_dos':
        EncSpec.display_local_density_of_states(
            spectrum, occupations, ax=ax
        )
        ax.set_title(
            ax.get_title() + '\n(unscaled exact projector weights)'
        )

    elif plot_kind == 'fft' and panel == 'total_dos':
        ax.plot(
            energy_MHz,
            spectrum.measured,
            color='black',
            linewidth=1.7,
            label='experiment',
        )
        ax.plot(
            energy_MHz,
            spectrum.theory,
            color='tab:orange',
            linewidth=1.7,
            label='theory FFT sum (globally peak-rescaled)',
        )
        ax.set(
            xlim=(-energy_limit_MHz, energy_limit_MHz),
            xlabel='energy E/h (MHz)',
            ylabel='spectral magnitude',
            title=(
                'complete-basis DOS\n'
                '(finite-time theory FFT globally peak-rescaled)'
                if spectrum.complete_basis
                else (
                    'projected spectrum\n'
                    '(finite-time theory FFT globally peak-rescaled)'
                )
            ),
        )

    ax.grid(alpha=0.15)
    scale_note = ''
    if panel == 'theory_map' or panel == 'total_dos':
        scale_note = '\nfinite-time theory FFT: globally peak-rescaled'
    elif panel in ('theory_dos', 'local_dos'):
        scale_note = (
            '\nfinite-time theory FFT: globally peak-rescaled; '
            'exact weights: unscaled'
            if panel == 'theory_dos'
            else '\nexact projector weights: unscaled'
        )
    fig.suptitle(
        f'N={N} {plot_kind.upper()} spectroscopy: {panel}{scale_note}'
    )
    return move_legends_outside(
        fig,
        ncols=2,
        figsize=resolved_figsize,
        legend_fontsize=legend_fontsize,
        save_path=save_path,
        save_dpi=save_dpi,
    )


# ------------------------------------------------------------------------
# Cell 211: time traces for the report.
# ------------------------------------------------------------------------


def get_time_trace(N, occupation, runs, *, normalized=True):
    """One occupation's time trace, from that sector's sources (cell 211).

    `runs` was a notebook global this body read.
    """
    occupation = tuple(occupation)
    available = []

    for source in runs[N].trace_sources:
        data = source.data
        occupations = [
            tuple(value) for value in data.reconstruction.occupations
        ]
        available.extend(occupations)
        if occupation not in occupations:
            continue

        row = occupations.index(occupation)
        time_us = np.asarray(data.spectrum.time_us, dtype=float)
        if normalized and 'A_norm' in data.reconstruction:
            trace = np.asarray(data.reconstruction.A_norm[row], dtype=complex)
        else:
            trace = np.asarray(data.reconstruction.A[row], dtype=complex)
            if normalized:
                if np.isclose(abs(trace[0]), 0.0):
                    raise ValueError(f'N={N}, {occupation}: A(0) is zero')
                trace = trace / trace[0]
        return time_us, trace, source.label

    raise KeyError(
        f'N={N} has no occupation {occupation}; available={sorted(set(available))}'
    )


def plot_time_traces(
        requests, runs, *, normalized=True, figsize=None, figure_dpi=None,
        legend_fontsize=None, title_fontsize=None, save_path=None,
        save_dpi=300):
    """One stacked panel per requested (N, occupation) trace (cell 211).

    `requests` is the notebook's own list of which traces to show. `runs`
    passes through to `get_time_trace`, which read it as a global.
    """
    resolved_figsize = (
        (10, 3.2 * len(requests))
        if figsize is None else tuple(figsize)
    )
    fig, axes = plt.subplots(
        len(requests),
        1,
        figsize=resolved_figsize,
        dpi=figure_dpi,
        sharex=False,
        squeeze=False,
    )
    axes = axes[:, 0]
    legend_handles = None
    legend_labels = None

    for ax, (N, occupation) in zip(axes, requests):
        time_us, trace, source_label = get_time_trace(
            N, occupation, runs, normalized=normalized
        )
        ax.plot(time_us, trace.real, label='Re[A(t)]', linewidth=1.8)
        ax.plot(time_us, trace.imag, label='Im[A(t)]', linewidth=1.8)
        ax.plot(time_us, np.abs(trace), label='|A(t)|', linewidth=1.8)
        ax.axhline(0.0, color='0.75', linewidth=0.8)
        ax.set(
            xlabel='Hamiltonian time (us)',
            ylabel='normalized return' if normalized else 'return',
            title=f'N={N}, occupation={tuple(occupation)} ({source_label})',
        )
        ax.grid(alpha=0.2)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.subplots_adjust(right=0.80, hspace=0.48)
    fig.legend(
        legend_handles,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(0.82, 0.5),
        frameon=False,
        fontsize=legend_fontsize,
    )
    fig.suptitle(
        'Selected Hamiltonian-spectroscopy time traces',
        fontsize=title_fontsize,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
    return fig


def load_and_analyze_sectors(config, job_ranges,
                             sectors=(1, 2, 3, 4),
                             n2_supplement_ranges=None,
                             n2_supplement_occupation=None,
                             timing=None):
    """Load every photon-number sector, then merge the N=2 supplement (cell 208).

    Returns `runs`, keyed by photon number. Each entry holds the calibration
    and spectroscopy experiments, the report and MPM data, the trace sources,
    and the load info.
    """
    runs = {}

    for N in sectors:
        calibration_expt, spectroscopy_expt, load_info = load_sector(
            job_ranges[N], config, timing=timing
        )
        sector_data = analyze_sector(
            N, calibration_expt, spectroscopy_expt, config
        )
        runs[N] = AttrDict(dict(
            calibration_expt=calibration_expt,
            spectroscopy_expt=spectroscopy_expt,
            report_data=sector_data,
            mpm_data=sector_data,
            trace_sources=[AttrDict(dict(label='main', data=sector_data))],
            load_info=load_info,
        ))
        print(
            f'N={N}:',
            len(load_info.calibration_loaded_ids), 'calibration jobs,',
            len(load_info.spectroscopy_loaded_ids), 'spectroscopy jobs,',
            len(sector_data.reconstruction.occupations), 'occupations',
        )

    if n2_supplement_ranges is None or n2_supplement_occupation is None:
        return runs

    # N=2 has one occupation on a different time grid. Analyze it separately,
    # then merge only the FFT rows for the complete report spectrum.
    supp_calibration_expt, supp_expt, supp_load_info = load_sector(
        n2_supplement_ranges, config, timing=timing
    )
    supp_data = supp_expt.analyze(
        calibration=supp_calibration_expt,
        phase_frame='manual_kerr',
        manual_kerr_MHz=config.manual_kerr_MHz,
        cycle_branches={n2_supplement_occupation: 0},
        legacy=True,
        fft_window=config.fft_window,
        zero_padding=config.zero_padding,
        spectrum_method='matrix_pencil',
    )
    supp_occupations = {
        tuple(occupation)
        for occupation in supp_data.reconstruction.occupations
    }
    if supp_occupations != {n2_supplement_occupation}:
        raise ValueError(
            f'N=2 supplement contains {supp_occupations}, '
            f'expected {n2_supplement_occupation}'
        )

    complete_data = config.EncSpec.merge_spectra([
        runs[2].mpm_data,
        supp_data,
    ])
    runs[2].report_data = complete_data
    runs[2].trace_sources.append(
        AttrDict(dict(label='supplement', data=supp_data))
    )
    runs[2].supplement_expt = supp_expt
    runs[2].supplement_load_info = supp_load_info

    print(
        'N=2 complete report:',
        len(complete_data.reconstruction.occupations),
        'occupations (FFT-only because the time grids differ)',
    )
    return runs
