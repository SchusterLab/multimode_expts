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
# # N=3 FFT and Matrix-Pencil diagnostics
#
# **DORMANT.** Moved from `analysis_notebooks/202609_qsim_migration/mbr.py` on 2026-09-26 (MBR
# redesign step 8C), without changes except imports. Why: `docs/qsim/mbr_step8_plan.md`,
# decision 3 (guan): exploratory diagnostics that answered method questions (window, peak thresholds, Matrix-Pencil reliability); they are not per-data-set analysis. Not maintained; may break when live code changes. If it breaks,
# add a note here and do not fix it.

# %%
import numpy as np
import matplotlib.pyplot as plt

from experiments.job_paths import data_root
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.deprecated import mbr_n3_diagnostics as n3_diagnostics


# %% [markdown]
# # 2. N=3 encoding-Hamiltonian spectroscopy reprocessing
#
# Loads the full N=3 spectrum (July 2026) and its calibration set once, then
# runs a series of independent diagnostics on the result.

# %%
# Dataset choice: the complete July N=3 sector (configs/datasets/mbr_datasets.yaml
# `july_N3`), converted to the new layout.
july_n3_manifest = data_root() / "260526_qsim_darkmode" / "assembled_data" / "260924_163508_MBRSpectrumExperiment.yaml"

spectroscopy_expt = MBRSpectrumExperiment.from_manifest(july_n3_manifest)
calibration_expt = spectroscopy_expt.calibration

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

# Undo the old correction and apply the calibration again with the selected
# signed Kerr (the spectrum's own calibration set).
encspec_reprocessed = spectroscopy_expt.analyze(
    phase_frame='manual_kerr',
    manual_kerr_MHz=encspec_manual_kerr_MHz,
    cycle_branches=encspec_cycle_branches,
    legacy=encspec_legacy,
    fft_window='raw',
    zero_padding=1,
    spectrum_method='mpm',
)
spectroscopy_expt.display(spectrum_method='mpm')

# %% [markdown]
# ### Global-only Matrix Pencil (source cell 188)
#
# Uses no Hamiltonian energies and no FFT peak positions. Moved here from
# `mbr_spectral_validation.py` in MBR redesign step 7a (the rest of that
# notebook is dormant).

# %%
n3_diagnostics.matrix_pencil_global_diagnostic(encspec_reprocessed)

# %% [markdown]
# ### Incoherent versus coherent summation
#
# $\sum_i |\mathrm{FFT}[A_i]|$ against $|\mathrm{FFT}[\sum_i A_i]|$. Both are
# recomputed from the complex return, so the comparison does not depend on
# which spectrum the previous analysis happened to store.

# %%
encspec_trace_time_us, encspec_trace_energy_MHz, _fig = (
    n3_diagnostics.compare_incoherent_and_coherent_fft(
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
# ### Theory-free ridge finding
#
# Peaks that repeat across occupations. No Hamiltonian energies and no
# expected peak count are used; `row_prominence` is in units of each row's own
# FFT noise. That independence is the point, so resist adding a theory
# comparison here.

# %%
aligned_peaks, candidate_indices, _fig = n3_diagnostics.find_ridge_peaks(
    encspec_reprocessed,
    max_candidates=35,
    row_prominence=1.,
    background_percentile=30.,
)

# %% [markdown]
# ### The worked path from job ranges to a spectrum
#
# Source cell 189 called itself a tutorial: it is the one place the route from
# saved data to a choice of raw FFT or rowwise Matrix Pencil is written out
# end to end.
#
# The two cells after it read `encspec_N3_spectrum`, `encspec_N3_time_us`,
# `encspec_N3_trace` and `encspec_N3_trace_mpm`. No code defines those names:
# the source cell that built the summed trace and its Matrix-Pencil fit did
# not survive the notebook split, and `load_and_analyze_n3` never returned
# them. Both cells are tagged `raises-exception` until someone rebuilds that
# step; it is analysis, not plumbing, so it was not guessed here.

# %%
n3_result = n3_diagnostics.load_and_analyze_n3(
    july_n3_manifest,
    cycle_branches={},
    manual_kerr_MHz=-19.756e-3,
    spectrum_method='matrix_pencil',
)

# %% tags=["raises-exception"]
encspec_N3_energy_limit_MHz, _fig = n3_diagnostics.compare_trace_with_mpm(
    encspec_N3_spectrum=n3_result['encspec_N3_spectrum'],
    encspec_N3_time_us=n3_result['encspec_N3_time_us'],
    encspec_N3_trace=n3_result['encspec_N3_trace'],
    encspec_N3_trace_mpm=n3_result['encspec_N3_trace_mpm'],
)

# %% tags=["raises-exception"]
# Matrix-Pencil rank diagnostic for the summed trace (source cell 192).
diagnostic = n3_result['encspec_N3_trace_mpm'].diagnostic

threshold = 2.858 * np.median(diagnostic.singular_values)

print('estimated signal rank:', diagnostic.estimated_signal_rank)
print('maximum swept rank:', diagnostic.maximum_rank)
print('singular-value threshold:', threshold)

for solution in diagnostic.rank_solutions:
    if 4 <= solution.rank <= 9:
        print(solution.rank, np.round(solution.frequencies_MHz, 6))

