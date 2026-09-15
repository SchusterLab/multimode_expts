# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MBR saved-data analysis
#
# Split out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
# 167-187, 189-192 and 205-240, plus `qsim_experiments.ipynb` cells 308-314,
# by the stage-2 notebook decomposition. This is the **saved-data** entry point
# for the four core MBR products; acquisition is
# `measurement_notebooks/202609_qsim_migration/mbr.py`.
#
# Four workspaces, in source order:
#
# 1. **N=2 from saved job files** (P167-172). Reconstructs the complex return
#    from analyzer-phase pairs and applies hand-entered rigid energy shifts.
# 2. **N=3 reprocessing and diagnostics** (P173-187, 189-192). One reprocessed
#    data set, then a series of independent FFT and Matrix-Pencil diagnostics.
# 3. **Report replots** (P205-211). The figure machinery for the four
#    photon-number sectors.
# 4. **Reproducing the Aug-15--17 datasets** (P212-240), twice over: once from
#    the job server and once from local HDF5 with a mock station.
#
# ## What moved, and what deliberately did not
#
# Long cells are now named functions in `experiments/qsim/notebook_helpers/`:
# `mbr_n2_spectroscopy.py`, `mbr_n3_reprocess.py`, `mbr_replot.py`,
# `mbr_saved_reanalysis.py` and `mbr_loading.py`. Each module's docstring
# records which source cells it came from and which apparent duplicates were
# checked before being merged.
#
# **Staying here:** every dataset choice and every hand-entered number. The
# job ID ranges, the `occupation_energy_shift_MHz` table, the branch
# assignments, the peak thresholds, the trace requests. Those are the
# scientific content of an analysis notebook.
#
# The 40 HDF5 paths that cell 168 held inline are now
# `n2_spectroscopy_files.yml`, beside this notebook. Read its own note: cell
# 167's prose says fifteen occupations but the list holds ten.
#
# ## Handoffs this split made explicit
#
# `encspec_reprocessed`, built in workspace 2, was read out of the live kernel
# by what are now two separate notebooks -- `mbr_sampling.py` and
# `mbr_spectral_validation.py`. Both now build it themselves by calling
# `reprocess_n3_spectroscopy`. Source cell 188, the global Matrix-Pencil
# diagnostic, went to `mbr_spectral_validation.py` and is not repeated here.
#
# Its neighbours: `mbr_disorder.py`, `mbr_sampling.py`,
# `mbr_spectral_validation.py`, and `dormant/`.

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
import pickle
import glob
import textwrap
from copy import deepcopy
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from tqdm.notebook import tqdm

import experiments as meas
from slab import AttrDict
from experiments import MultimodeStation
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display_classes import GeneralFitting
from fitting.wigner import WignerAnalysis
from job_server import JobClient

# The four aggregate MBR stages. `EncodingHamiltonianSpectroscopyExperiment`
# is still the loading layer and the shared numerics, and is still the class
# every job here was acquired under -- so it stays, and these four sit beside
# it. See analysis_notebooks/guan/MBR_analysis.py for the worked example.
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
from experiments.qsim.mbr_propagator import MBRPropagatorExperiment

from experiments.qsim.notebook_helpers.mbr_loading import (
    job_id_generator,
    load_encoding_spectroscopy,
)
from experiments.qsim.notebook_helpers import mbr_n2_spectroscopy as n2
from experiments.qsim.notebook_helpers import mbr_n3_reprocess as n3
from experiments.qsim.notebook_helpers import mbr_replot as replot
from experiments.qsim.notebook_helpers import mbr_saved_reanalysis as saved

NOTEBOOK_DIR = Path.cwd()
client = JobClient()

# %% [markdown]
# # 1. N=2 Hamiltonian spectroscopy from saved jobs
#
# Each file already contains both preparation phases
# $\theta=(0,180)^\circ$. The complex return is reconstructed as
#
# $$
# Q(\phi)=P_e(0,\phi)-P_e(180^\circ,\phi),\qquad
# A_{\mathbf n}=Q(0)+iQ(90^\circ).
# $$
#
# The decoder phase matrix saved in each HDF5 file was already applied during
# acquisition and is not applied again here. The last cell applies only the
# overall occupation-dependent shifts entered explicitly below.

# %%
# Acquisition constants for this dataset. One ordered Floquet cycle
# represents `hamiltonian_dt_us` of logical Hamiltonian time.
hamiltonian_dt_us = 0.08
physical_cycle_us = 0.4130059523809524
floquet_couplings_MHz = np.array([
    0.078125, 0.078125, 0.078125, 0.078125,
])
zero_padding = 8
energy_limit_MHz = 1.25

spectroscopy_fnames = n2.load_file_catalog(
    NOTEBOOK_DIR / "n2_spectroscopy_files.yml"
)

# %% [markdown]
# ## Reconstruct the complex return
#
# Files are placed by their saved analyzer phase rather than by list position.
# Only the reconstructed traces and saved acquisition metadata are retained in
# memory; the large shot arrays in each HDF5 file are released immediately.

# %%
# `kerr_override_MHz` reproduces a hard-coded line in cell 170, which read the
# saved M1 Kerr and then overwrote it with -10e-3. Pass None to use the value
# actually stored in the files.
n2_record = n2.reconstruct_complex_return(
    spectroscopy_fnames,
    kerr_override_MHz=-10e-3,
)
A_raw = n2_record["A_raw"]
occupation_strings = n2_record["occupation_strings"]
cycles = n2_record["cycles"]
N = n2_record["N"]
mode_labels = n2_record["mode_labels"]
swap_stors = n2_record["swap_stors"]
physical_kerr_MHz = n2_record["physical_kerr_MHz"]

print("N =", N)
print("A_raw shape =", A_raw.shape)
for occupation, amplitude in zip(occupation_strings, A_raw):
    print(occupation, "A(0) =", amplitude[0])

# %% [markdown]
# ## Manual occupation-resolved rigid-shift correction and DOS
#
# This does not infer a correction from the M1 row and does not fit a
# measured trace to theory. Enter one overall spectral shift for every
# occupation string in `occupation_energy_shift_MHz` below.
#
# The value is `desired peak position - measured peak position`, in MHz. A
# positive value moves the whole spectrum to the right and a negative value
# moves it to the left. The correction changes only the phase ramp of
# $A_{\mathbf n}(t)$; it does not fit, stretch, or rescale the spectrum.
#
# The target Hamiltonian uses the signed Kerr stored in the HDF5
# configuration, as in the preceding saved-job analysis cell.

# %%
n2_grid = n2.fft_grid_and_raw_dos(
    cycles=cycles,
    A_raw=A_raw,
    occupation_strings=occupation_strings,
    hamiltonian_dt_us=hamiltonian_dt_us,
    zero_padding=zero_padding,
)

n2_theory = n2.build_fixed_n_theory(
    N=N,
    mode_labels=mode_labels,
    swap_stors=swap_stors,
    floquet_couplings_MHz=floquet_couplings_MHz,
    physical_kerr_MHz=physical_kerr_MHz,
    physical_cycle_us=physical_cycle_us,
    hamiltonian_dt_us=hamiltonian_dt_us,
    occupation_strings=occupation_strings,
    grid=n2_grid,
)

# %%
# The one genuinely hand-entered input of this workspace. Example: if a
# measured peak sits at +0.45 MHz and should be at 0 MHz, enter -0.45.
one_fock = -0.45
two_fock = -2.17

occupation_energy_shift_MHz = {
    (2, 0, 0, 0, 0): two_fock,
    (1, 1, 0, 0, 0): one_fock,
    (1, 0, 1, 0, 0): one_fock,
    (1, 0, 0, 1, 0): one_fock,
    (1, 0, 0, 0, 1): one_fock,
    (0, 2, 0, 0, 0): two_fock,
    (0, 1, 1, 0, 0): one_fock,
    (0, 1, 0, 1, 0): one_fock,
    (0, 1, 0, 0, 1): one_fock,
    (0, 0, 2, 0, 0): two_fock,
    (0, 0, 1, 1, 0): one_fock,
    (0, 0, 1, 0, 1): one_fock,
    (0, 0, 0, 2, 0): two_fock,
    (0, 0, 0, 1, 1): one_fock,
    (0, 0, 0, 0, 2): two_fock,
}

n2_shifted = n2.apply_rigid_shifts(
    A_raw=A_raw,
    occupation_strings=occupation_strings,
    occupation_energy_shift_MHz=occupation_energy_shift_MHz,
    hamiltonian_dt_us=hamiltonian_dt_us,
    grid=n2_grid,
    theory=n2_theory,
)

# %%
n2.plot_shifted_spectra(
    occupation_strings=occupation_strings,
    mode_labels=mode_labels,
    N=N,
    cycles=cycles,
    energy_limit_MHz=energy_limit_MHz,
    grid=n2_grid,
    theory=n2_theory,
    shifted=n2_shifted,
    physical_kerr_MHz=physical_kerr_MHz,
)

# %% [markdown]
# # 2. N=3 encoding-Hamiltonian spectroscopy reprocessing
#
# Loads the full N=3 calibration and spectroscopy jobs once from the job
# queue, then runs a series of independent diagnostics on the result.

# %%
# Dataset choice: the complete N=3 calibration and spectroscopy job ranges.
calibration_job_ids = job_id_generator([20260722, 20260723], [683, 1], [712, 40])
spectroscopy_job_ids = job_id_generator(
    20260723, [48, 87], [85, 149], step=[1, 2]
)

calibration_expt, spectroscopy_expt = load_encoding_spectroscopy(
    MBRSpectrumExperiment,
    calibration_job_ids,
    spectroscopy_job_ids,
    client=client,
)

# %% [markdown]
# These jobs used the old `+cycle*correction` analyzer convention. The current
# reconstruction builds $A = Q_0 - iQ_{90}$, so a branch assignment copied
# from an old `Q0 + iQ90` notebook needs its sign flipped. Unspecified
# occupations use branch 0.

# %%
encspec_cycle_branches = {
    # (3, 0, 0, 0, 0): 1,
    # (0, 2, 0, 0, 1): -1,
}
encspec_legacy = True
encspec_manual_kerr_MHz = -19.756e-3  # signed; 0. for the zero-Kerr frame

encspec_reprocessed = n3.reprocess_n3_spectroscopy(
    calibration_expt=calibration_expt,
    spectroscopy_expt=spectroscopy_expt,
    cycle_branches=encspec_cycle_branches,
    legacy=encspec_legacy,
    manual_kerr_MHz=encspec_manual_kerr_MHz,
)

# %% [markdown]
# ### Incoherent versus coherent summation
#
# $\sum_i |\mathrm{FFT}[A_i]|$ against $|\mathrm{FFT}[\sum_i A_i]|$. Both are
# recomputed from the complex return, so the comparison does not depend on
# which spectrum the previous analysis happened to store.

# %%
encspec_trace_time_us, encspec_trace_energy_MHz, _fig = (
    n3.compare_incoherent_and_coherent_fft(
        spectroscopy_expt,
        # Cell 179 read the saved window then hard-coded 'hann'.
        window_name='hann',
    )
)

# %%
# Shapes, for orientation (source cells 177, 178, 183).
print(np.shape(spectroscopy_expt.data.reconstruction.A))
print(np.shape(encspec_trace_time_us))
spectroscopy_expt.data.spectrum.keys()

# %% [markdown]
# ### Small per-occupation inspections
#
# Source cells 176, 180, 181 and 185: short interactive pokes at one
# occupation at a time. Left as plain cells rather than wrapped -- they are
# each a handful of lines, and wrapping a one-off plot in a function makes it
# harder to edit, not easier.

# %%
# Cell 176: how bad it looks when every occupation is summed incoherently.
fig, ax = plt.subplots()

x_data = spectroscopy_expt.data.spectrum.energy_MHz
y_data = spectroscopy_expt.data.spectrum.measured_local[0, :].copy()
for idx in range(len(spectroscopy_expt.data.reconstruction.occupations) - 1):
    y_data += spectroscopy_expt.data.spectrum.measured_local[idx + 1, :]
ax.plot(x_data, y_data, label='summed over occupations')
ax.legend()

# %%
# Cell 180: two peak-finding methods on a single occupation. The three-panel
# version below (cell 184) generalizes this over every occupation, but this
# single-row comparison is where the thresholds get chosen.
from scipy.signal import find_peaks, savgol_filter

idx = spectroscopy_expt.data.reconstruction.occupations.index((2, 0, 1, 0, 0))
state_label = spectroscopy_expt.data.reconstruction.occupations[idx]

x_data = np.array(spectroscopy_expt.data.spectrum.energy_MHz)
y_data = np.array(spectroscopy_expt.data.spectrum.measured_local[idx, :])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

peaks, properties = find_peaks(y_data, height=0.04, prominence=0.006)

ax1.plot(x_data, y_data, label=f"{state_label} (Original)")
ax1.plot(x_data[peaks], y_data[peaks], "x", color='red', markersize=10, label="Detected Peaks")
ax1.set_title("Method 1: Direct find_peaks with Thresholds")
ax1.legend()


y_smoothed = savgol_filter(y_data, window_length=5, polyorder=2)

peaks_smooth, _ = find_peaks(y_smoothed, height=0.03, prominence=0.005)

ax2.plot(x_data, y_data, alpha=0.3, label="Original Data")
ax2.plot(x_data, y_smoothed, color='orange', label="Smoothed Signal")
ax2.plot(x_data[peaks_smooth], y_smoothed[peaks_smooth], "x", color='red', markersize=10, label="Detected Peaks")
ax2.set_title("Method 2: Savitzky-Golay Smoothing + find_peaks")
ax2.legend()

plt.tight_layout()
plt.show()

# %%
# Cell 181: exact level multiplicities against the theory curve.
energies = np.sort(np.asarray(spectroscopy_expt.data.spectrum.energies_MHz))
unique_energies, multiplicities = np.unique(np.round(energies, 10), return_counts=True)

plt.vlines(unique_energies, 0., multiplicities, color='tab:orange')
plt.plot(unique_energies, multiplicities, 'o', color='tab:orange')
plt.plot(spectroscopy_expt.data.spectrum.energy_MHz,
         spectroscopy_expt.data.spectrum.theory)

# %%
# Cell 185: one occupation's spectrum on its own.
idx = spectroscopy_expt.data.reconstruction.occupations.index((2, 1, 0, 0, 0))

fig, ax = plt.subplots()

state_label = spectroscopy_expt.data.reconstruction.occupations[idx]
x_data = spectroscopy_expt.data.spectrum.energy_MHz
y_data = spectroscopy_expt.data.spectrum.measured_local[idx, :]
ax.plot(x_data, y_data,
        label = f"{state_label}")
ax.legend()

# %% [markdown]
# ### Peak finding, three ways
#
# Raw, Savitzky-Golay smoothed, and summed-across-occupations.

# %%
_fig, peak_list_raw, peak_list_smoothed = n3.compare_peak_finders(
    spectroscopy_expt,
    height=0.05,
    prominence=0.01,
)

# %% [markdown]
# ### Theory-free ridge finding
#
# Peaks that repeat across occupations. No Hamiltonian energies and no
# expected peak count are used; `row_prominence` is in units of each row's own
# FFT noise. That independence is the point, so resist adding a theory
# comparison here.

# %%
aligned_peaks, candidate_indices, _fig = n3.find_ridge_peaks(
    encspec_reprocessed,
    max_candidates=35,
    row_prominence=1.,
    background_percentile=30.,
)

# %% [markdown]
# ### The worked path from job ranges to a spectrum
#
# Source cell 189 called itself a tutorial: it is the one place the route from
# job ranges to a choice of raw FFT or rowwise Matrix Pencil is written out
# end to end.

# %%
n3_result = n3.load_and_analyze_n3(
    calibration_job_ids=job_id_generator(
        [20260722, 20260723], [683, 1], [712, 40]
    ),
    spectroscopy_job_ids=job_id_generator(
        20260723, [48, 87], [85, 149], step=[1, 2]
    ),
    client=client,
    cycle_branches={},
    manual_kerr_MHz=-19.756e-3,
    spectrum_method='matrix_pencil',
)

# %%
encspec_N3_energy_limit_MHz, _fig = n3.compare_trace_with_mpm(
    encspec_N3_spectrum=n3_result['encspec_N3_spectrum'],
    encspec_N3_time_us=n3_result['encspec_N3_time_us'],
    encspec_N3_trace=n3_result['encspec_N3_trace'],
    encspec_N3_trace_mpm=n3_result['encspec_N3_trace_mpm'],
)

# %%
# Matrix-Pencil rank diagnostic for the summed trace (source cell 192).
diagnostic = n3_result['encspec_N3_trace_mpm'].diagnostic

threshold = 2.858 * np.median(diagnostic.singular_values)

print('estimated signal rank:', diagnostic.estimated_signal_rank)
print('maximum swept rank:', diagnostic.maximum_rank)
print('singular-value threshold:', threshold)

for solution in diagnostic.rank_solutions:
    if 4 <= solution.rank <= 9:
        print(solution.rank, np.round(solution.frequencies_MHz, 6))

# %% [markdown]
# # 3. M1 self-Kerr from experiment--theory peak overlap
#
# Source `qsim_experiments.ipynb` cell 309. Submits no jobs. It scans the
# signed M1 self-Kerr and scores only traces whose encoder occupation has at
# least two M1 photons. Rows are averaged within each M1-photon-number group
# and the groups are weighted equally, so a single trace such as
# `(3, 0, 0, 0, 0)` is not overwhelmed by the larger `n_M1 = 2` group.
#
# This is the one part of the acquisition notebook's numbered sections that is
# pure analysis, which is why the surface map routes cells 308-314 here.

# %%
# `spectroscopy_data` below is whatever the loaded spectroscopy experiment's
# last analyze() produced. The Kerr fit re-runs analyze() on a grid and then
# restores the best grid point, because analyze() mutates expt.data.
spectroscopy_data = spectroscopy_expt.data
spectroscopy_occupations = [
    tuple(state) for state in spectroscopy_data.reconstruction.occupations
]
cycle_branches = dict(encspec_cycle_branches)

best_self_kerr_kHz, kerr_fit_scores, spectroscopy_data = (
    n3.fit_self_kerr_from_peak_overlap(
        spectroscopy_expt=spectroscopy_expt,
        spectroscopy_data=spectroscopy_data,
        calibration_expt=calibration_expt,
        spectroscopy_occupations=spectroscopy_occupations,
        cycle_branches=cycle_branches,
        kerr_grid_kHz=np.arange(-5.0, 0.0 + 1e-9, 0.005),
        energy_limit_MHz=0.08,
        min_man_photons=2,
        baseline_quantile=0.20,
    )
)

# %% [markdown]
# ## Matrix Pencil analysis of the acquired spectroscopy
#
# Source cell 312. Reuses the jobs already held by `spectroscopy_expt`; submits
# nothing. FFT/theory and MPM come from the same analysis call and can be
# displayed independently.

# %%
spectroscopy_display_occupation = None

spectroscopy_expt.analyze(
    occupations=spectroscopy_occupations,
    cycle_branches=cycle_branches,
    spectrum_method="mpm",
)
if spectroscopy_display_occupation is None:
    spectroscopy_expt.display(
        spectrum_method="fft",
        level_statistics=False,
    )
    spectroscopy_expt.display(spectrum_method="mpm")
else:
    spectroscopy_expt.display(
        occupation=spectroscopy_display_occupation,
        spectrum_method="fft",
    )
    spectroscopy_expt.display(
        occupation=spectroscopy_display_occupation,
        spectrum_method="mpm",
    )
plt.show()

# %% [markdown]
# ## Reanalyze a saved spectroscopy batch
#
# Source cell 314. Loads four known-good interleaved off-diagonal jobs without
# acquiring new data. Replace the job list to analyze another saved batch.
#
# This needs a `station` only for its data path. Build a mock one, as the
# offline reproduce section further down does, rather than connecting to
# hardware from an analysis notebook.

# %%
saved_spectroscopy_job_ids = [
    f"JOB-20260823-{job:05d}" for job in range(5, 9)
]
saved_spectroscopy_expt = MBRSpectrumExperiment.from_job_ids(
    saved_spectroscopy_job_ids,
    client=client,
)
saved_spectroscopy_expt.analyze(
    cycle_branches={(0, 0, 2, 0, 0): 0},
)
saved_spectroscopy_expt.display(
    occupation=[0, 0, 1, 1, 0],
)
plt.show()

# %% [markdown]
# # 4. Report replots
#
# Source cells 205-211. The `replot_*` settings that cell 206 kept at notebook
# scope are now one `ReplotConfig`; its defaults are those values, so building
# it with no arguments reproduces the original figures.

# %%
replot_config = replot.ReplotConfig(
    manual_kerr_MHz=-19.756e-3,
    cycle_branches={1: {}, 2: {}, 3: {}, 4: {}},
    fft_window='raw',
    zero_padding=1,
    overview_figsize=(15, 12.3),
    save_dpi=300,
    legend_fontsize=9,
    suptitle_fontsize=14,
    overview_legend_ncols=3,
    EncSpec=MBRSpectrumExperiment,
)

# Dataset choice: (date, first job number, last job number, step).
replot_job_ranges = {
    1: dict(
        calibration=[(20260722, 557, 566, 1)],
        spectroscopy=[(20260722, 577, 595, 1)],
    ),
    2: dict(
        calibration=[(20260722, 35, 64, 1)],
        spectroscopy=[(20260722, 215, 244, 1)],
    ),
    3: dict(
        calibration=[
            (20260722, 683, 712, 1),
            (20260723, 1, 40, 1),
        ],
        spectroscopy=[
            (20260723, 48, 85, 1),
            (20260723, 87, 149, 2),
        ],
    ),
    4: dict(
        calibration=[(20260721, 423, 432, 1)],
        spectroscopy=[(20260722, 5, 14, 1)],
    ),
}

# N=2 has one occupation on a different time grid, so only its FFT rows can
# join the report spectrum -- not its time traces.
replot_N2_supplement_ranges = dict(
    calibration=[(20260722, 425, 426, 1)],
    spectroscopy=[(20260722, 452, 454, 1)],
)
replot_N2_supplement_occupation = (0, 0, 0, 0, 2)

# %%
replot_runs = replot.load_and_analyze_sectors(
    config=replot_config,
    job_ranges=replot_job_ranges,
    client=client,
    sectors=(1, 2, 3, 4),
    n2_supplement_ranges=replot_N2_supplement_ranges,
    n2_supplement_occupation=replot_N2_supplement_occupation,
)

# %% [markdown]
# ### Export one report subplot as its own figure
#
# `panel_names_by_kind` was cell 210's own settings block and stays here.

# %%
replot_panel_names_by_kind = {
    'mpm': (
        'measured_map',
        'theory_map',
        'measured_dos',
        'theory_dos',
    ),
    'fft': (
        'measured_map',
        'theory_map',
        'measured_dos',
        'theory_dos',
        'local_dos',
    ),
}

replot_single_panel_requests = [
    (3, 'mpm', 'measured_map'),
]

replot_single_panel_figures = {}
for replot_N, replot_plot_kind, replot_panel in replot_single_panel_requests:
    replot_single_panel_figures[
        (replot_N, replot_plot_kind, replot_panel)
    ] = replot.plot_single_spectroscopy_panel(
        replot_N,
        replot_plot_kind,
        replot_panel,
        runs=replot_runs,
        panel_names_by_kind=replot_panel_names_by_kind,
        EncSpec=replot_config.EncSpec,
        figsize=replot_config.single_panel_figsize,
        figure_dpi=replot_config.figure_dpi,
        legend_fontsize=replot_config.legend_fontsize,
        save_dpi=replot_config.save_dpi,
    )
plt.show()

# %% [markdown]
# ### Time traces for the report
#
# Edit only this list to choose which traces are shown. The N=2 supplemental
# occupation is valid here despite its different grid.

# %%
replot_trace_requests = [
    (1, (1, 0, 0, 0, 0)),
    # (2, (2, 0, 0, 0, 0)),
    # (3, (3, 0, 0, 0, 0)),
    # (2, (0, 0, 0, 0, 2)),
]
replot_normalize_traces = True

replot_time_trace_figure = replot.plot_time_traces(
    replot_trace_requests,
    runs=replot_runs,
    normalized=replot_normalize_traces,
    figsize=replot_config.trace_figsize,
    figure_dpi=replot_config.figure_dpi,
    legend_fontsize=replot_config.legend_fontsize,
    title_fontsize=replot_config.suptitle_fontsize,
    save_dpi=replot_config.save_dpi,
)
plt.show()

# %% [markdown]
# # 5. Reproduce the Aug-15--17 N=3 and disorder spectroscopy
#
# Load only -- no `runner.execute()` appears anywhere below. The source did
# this twice, once from the job server and once from local HDF5 files with a
# mock station, and both paths survive because they genuinely read from
# different places.
#
# What was *not* a real difference: cells 213/218 defined `saved_job_range`
# byte-identically, and cells 222/232 defined the same occupation-pairing
# check under two names with identical bodies. Both are single functions now.

# %% [markdown]
# ## 5a. From the job server

# %%
saved_job_client = JobClient()

# Complete N=3 phase calibration: 35 occupations x analyzer phases 0/90.
saved_n3_calibration_job_ids = saved.saved_job_range(20260815, 113, 182)

# Complete N=3 spectroscopy: 35 occupations x analyzer phases 0/90.
saved_n3_spectroscopy_job_ids = (
    saved.saved_job_range(20260815, 183, 242)
    + saved.saved_job_range(20260816, 1, 10)
)

# Ten theory-selected occupations x analyzer phases 0/90 per realization.
saved_disorder_job_ids = {
    0: saved.saved_job_range(20260816, 13, 32),
    1: saved.saved_job_range(20260816, 33, 52),
    2: saved.saved_job_range(20260816, 53, 72),
    3: saved.saved_job_range(20260817, 1, 20),
}

# Add later completed realizations here, for example:
# saved_disorder_job_ids[4] = saved.saved_job_range(20260818, FIRST, LAST)

# This reproduces the later plotted/rephased Hamiltonian. Set to None to keep
# only the exact as-acquired frame (saved device Kerr and analyzer correction).
saved_n3_manual_kerr_MHz = -10.5e-3
saved_n3_cycle_branches = {
    (2, 1, 0, 0, 0): 1,
    (2, 0, 1, 0, 0): 1,
    (1, 1, 0, 1, 0): 1,
    (1, 1, 0, 0, 1): 1,
    (1, 0, 1, 1, 0): 1,
    (1, 0, 1, 0, 1): 1,
}

saved_fft_window = "raw"
saved_zero_padding = 1

# %%
# Source cell 219: the four-realization dataset, loaded and analyzed on its
# own. It uses its own two-entry branch table and a plain FFT spectrum, not
# the manual-Kerr frame used for the N=3 set below.
data_four_realization = MBRSpectrumExperiment.from_job_ids(
    saved.saved_job_range(20260815, 9, 16),
    client=saved_job_client,
)

branches = {
    (3, 0, 0, 0, 0): 1,
    (2, 0, 0, 1, 0): 1,
}

_ = data_four_realization.analyze(
        cycle_branches=branches,
        fft_window=saved_fft_window,
        zero_padding=saved_zero_padding,
        spectrum_method="fft",
    )
data_four_realization.display()

# %%
saved_remote = saved.load_saved_remote(
    saved_n3_calibration_job_ids=saved_n3_calibration_job_ids,
    saved_n3_spectroscopy_job_ids=saved_n3_spectroscopy_job_ids,
    saved_disorder_job_ids=saved_disorder_job_ids,
    saved_n3_cycle_branches=saved_n3_cycle_branches,
    saved_n3_manual_kerr_MHz=saved_n3_manual_kerr_MHz,
    saved_job_client=saved_job_client,
    saved_fft_window=saved_fft_window,
    saved_zero_padding=saved_zero_padding,
)
saved_n3_calibration_expt = saved_remote["saved_n3_calibration_expt"]
saved_n3_spectroscopy_expt = saved_remote["saved_n3_spectroscopy_expt"]
saved_n3_data = saved_remote.get("saved_n3_data")
saved_disorder_records = saved_remote["saved_disorder_records"]

# %%
# Optional report figures from the reconstructed data.
saved_show_calibration = False
saved_show_n3 = True
saved_show_disorder = True

if saved_show_calibration:
    saved_n3_calibration_expt.display()
    plt.show()
if saved_show_n3:
    saved_n3_spectroscopy_expt.display(
        data=saved_n3_data, level_statistics=False
    )
    plt.show()

# %% [markdown]
# ## 5b. From local HDF5 files only
#
# Same reanalysis with nothing read from the job queue. The mock station
# exists only so the loaders have config versions and a data path; which
# versions to reconstruct against is a scientific choice, so it stays here.

# %%
saved_project_name = "260526_qsim_darkmode"
saved_four_project_name = "260814_qsim_encspec"
saved_station_config = {
    "hardware_config": "CFG-HW-20260815-00002",
}

saved_station = MultimodeStation(
    user="jonginn",
    mock=True,
    experiment_name=saved_project_name,
    hardware_config=saved_station_config["hardware_config"],
    log_measurements=False,
)

saved_local_experiment = saved.make_local_loader(
    station=saved_station,
    project_name=saved_project_name,
    EncSpec=MBRSpectrumExperiment,
)

saved_four_realization_range = ([20260815], [9], [16])
offline_four_realization_branches = {
    (3, 0, 0, 0, 0): 1,
    (2, 0, 0, 1, 0): 1,
}
saved_n3_range = ([20260815, 20260816], [183, 1], [242, 10])
saved_disorder_ranges = {
    0: ([20260816], [11], [80]),
}
offline_fft_window = "raw"
offline_zero_padding = 1

# %%
saved_local = saved.load_saved_local(
    saved_local_experiment=saved_local_experiment,
    saved_four_realization_range=saved_four_realization_range,
    offline_four_realization_branches=offline_four_realization_branches,
    saved_n3_range=saved_n3_range,
    saved_disorder_ranges=saved_disorder_ranges,
    saved_four_project_name=saved_four_project_name,
    offline_fft_window=offline_fft_window,
    offline_zero_padding=offline_zero_padding,
)
data_four_realization = saved_local["data_four_realization"]
saved_n3_spectroscopy_expt = saved_local["saved_n3_spectroscopy_expt"]
saved_n3_data = saved_local["saved_n3_data"]
saved_disorder_records = saved_local["saved_disorder_records"]

# %%
# Optional figures; all data above came from local H5 files.
offline_show_n3 = True
offline_show_disorder = True

if offline_show_n3:
    saved_n3_spectroscopy_expt.display(
        data=saved_n3_data, level_statistics=False
    )
    plt.show()

# %% [markdown]
# ## Picking a data set
#
# Cells 226 and 234 were the same dict under two sections. One copy.

# %%
data_sets = {
    0: data_four_realization,
    1: saved_n3_spectroscopy_expt,
    2: saved_disorder_records[0].expt,
}

data_set_index = 2
pp_data = data_sets[data_set_index]

# %%
occupation_to_plot = (2, 0, 0, 1, 0)
realization_idx = 0

fig, axes = saved.plot_occupation_trace_panels(
    pp_data,
    occupation_to_plot=occupation_to_plot,
    realization_idx=realization_idx,
)
plt.show()

# %% [markdown]
# ## Coherent normalized-trace FFT for every loaded data set
#
# The FFT of $\sum_n A_n(t)/A_n(0)$, using each data set's own saved FFT
# convention. Needs `reconstruction.A`, so a merged spectrum-only data set
# raises rather than silently producing something else.

# %%
coherent_trace_results, coherent_trace_figures = saved.coherent_trace_report(
    data_sets,
    SavedEncSpec=MBRSpectrumExperiment,
)
