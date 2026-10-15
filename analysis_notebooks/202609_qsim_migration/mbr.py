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
# The **saved-data** entry point for the MBR spectroscopy products; acquisition is
# `measurement_notebooks/202609_qsim_migration/mbr.py`. Every section is the same flow:
# load a saved manifest, `analyze(...)` with this data set's analysis choices, `display()`.
# The analysis choices (cycle branches, Kerr frame, FFT window) and the
# dataset paths are the scientific content of this notebook.
#
# 1. **N=3 spectrum** (July 2026), with FFT peak finding three ways beside the
#    Matrix-Pencil spectrum.
# 2. **M1 self-Kerr** from the experiment--theory peak overlap. Kept while the Kerr
#    calibration procedure is being fixed.
# 3. **Reproducing the Aug-15--17 data sets**, with the phase calibration and in the
#    as-acquired frame. Kept for the physics audit.
#
# Moved to `dormant/` in MBR redesign step 8C (`docs/qsim/mbr_step8_plan.md`, decision 3):
# the N=3 FFT/Matrix-Pencil diagnostics (`dormant/mbr_n3_diagnostics.py`), the report
# replots (`dormant/mbr_replot.py`), and the July N=2 reprocessing with hand-entered
# energy shifts (decoder-mode notebook retired Oct 5; its files were taken in the removed
# 'decoder' phase-correction mode).
#
# Split out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells 167-187,
# 189-192 and 205-240, plus `qsim_experiments.ipynb` cells 308-314, by the stage-2
# notebook decomposition. The N=1, N=2, N=3 and quickplot data sets are saved
# `MBRSpectrumExperiment` manifests (converted once with `tools/migrate_mbr_jobs.py`);
# the disorder realizations are a saved `MBRDisorderEnsembleExperiment`.
#
# Its neighbours: `mbr_disorder.py` and `dormant/`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt

from experiments.job_paths import data_root
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment


# %% [markdown]
# # 1. N=3 encoding-Hamiltonian spectroscopy reprocessing
#
# Loads the full N=3 spectrum (July 2026) and its calibration set once, then
# runs a series of independent diagnostics on the result.

# %%
# Dataset choice: the complete July N=3 sector (configs/datasets/mbr_datasets.yaml
# `july_N3`), converted to the new layout.
july_n3_manifest = data_root() / "260526_qsim_darkmode" / "assembled_data" / "261006_151633_MBRSpectrumExperiment.yaml"

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
encspec_manual_kerr_MHz = -19.756e-3  # signed; 0. for the zero-Kerr frame

# Undo the old correction and apply the calibration again with the selected
# signed Kerr (the spectrum's own calibration set).
encspec_reprocessed = spectroscopy_expt.analyze(
    phase_frame='manual_kerr',
    manual_kerr_MHz=encspec_manual_kerr_MHz,
    cycle_branches=encspec_cycle_branches,
    fft_window='raw',
    zero_padding=1,
    spectrum_method='mpm',
)
spectroscopy_expt.display(spectrum_method='mpm')

# %% [markdown]
# ### Peak finding, three ways
#
# Raw, Savitzky-Golay smoothed, and summed-across-occupations: where FFT peak
# finding is weak, beside the Matrix-Pencil spectrum displayed above.

# %%
_fig, peak_list_raw, peak_list_smoothed = spectroscopy_expt.display_peak_finders(
    height=0.05,
    prominence=0.01,
)

# %% [markdown]
# # 2. M1 self-Kerr from experiment--theory peak overlap
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

# The July jobs ran on code that added the analyzer correction (sign +1); the
# converted files record that sign, so the scan needs no flag.
best_self_kerr_kHz, kerr_fit_scores, spectroscopy_data = (
    spectroscopy_expt.fit_self_kerr(
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
# # 3. Reproduce the Aug-15--17 N=3 and disorder spectroscopy
#
# Load only -- no `runner.execute()` appears anywhere below. The source did
# this twice, once from the job server and once from local HDF5 files with a
# mock station. Both now read the same HDF5 files, so the loading difference
# is gone; what remains is a real difference in the *analysis*, and the two
# sections are named for it:
#
# - 3a applies the N=3 phase calibration and reports the manual-Kerr frame;
# - 3b applies no calibration and stays in the as-acquired frame.
#
# Because both read the same files, a disagreement between them is now a
# statement about phase frames and nothing else.
#
# Every cell below is the canonical flow: `from_manifest`, then `analyze(...)`
# with this data set's choices (the loader functions of stage 2 were removed in
# MBR redesign step 9B).

# %% [markdown]
# ## 3a. With the phase calibration applied

# %%
# Complete N=3 spectroscopy with its calibration set (configs/datasets/
# mbr_datasets.yaml `august_N3`), converted to the new layout.
saved_n3_manifest = data_root() / "260526_qsim_darkmode" / "assembled_data" / "260924_163516_MBRSpectrumExperiment.yaml"
# The four-realization quick-plot set (`august_quickplot`).
saved_four_realization_manifest = data_root() / "260814_qsim_encspec" / "assembled_data" / "260924_163516_MBRSpectrumExperiment.yaml"

# The four disorder realizations (`august_disorder` (former `august_disorder_r0`..`r3`): ten
# theory-selected occupations each), converted to one
# MBRDisorderEnsembleExperiment whose parts link the August N=3 calibration
# set (MBR redesign step 7c). A later realization joins by converting it into
# a new ensemble with tools/migrate_mbr_jobs.py (kind `disorder`).
saved_disorder_manifest = data_root() / "260526_qsim_darkmode" / "assembled_data" / "260924_195547_MBRDisorderEnsembleExperiment.yaml"

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
data_four_realization = MBRSpectrumExperiment.from_manifest(
    saved_four_realization_manifest,
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
saved_n3 = MBRSpectrumExperiment.from_manifest(saved_n3_manifest)
saved_n3_calibration = saved_n3.calibration
saved_n3_calibration.analyze()
# The complete N=3 sector, and the timing recovered from the files, not from
# today's station.
assert len(saved_n3_calibration.occupations) == 35 and all(
    sum(o) == 3 for o in saved_n3_calibration.occupations)
assert set(saved_n3.occupations) == set(saved_n3_calibration.occupations)
assert "station" not in saved_n3_calibration.data.hardware.source
print("Floquet timing recovered from:", saved_n3_calibration.data.hardware.source)

saved_n3_data = saved_n3.analyze(
    phase_frame="as_acquired" if saved_n3_manual_kerr_MHz is None else "manual_kerr",
    manual_kerr_MHz=saved_n3_manual_kerr_MHz,
    cycle_branches=saved_n3_cycle_branches if saved_n3_manual_kerr_MHz is not None else 0,
    fft_window=saved_fft_window,
    zero_padding=saved_zero_padding,
    spectrum_method="fft",
)
hardware = saved_n3_calibration.data.hardware
print(f"Tcycle={hardware.floquet_cycle_us:.9f} us; g={1e3 * np.asarray(hardware.couplings_MHz)} kHz; "
      f"frame={saved_n3_data.phase_frame}; K={1e3 * saved_n3_data.spectrum.physical_kerr_MHz:.4f} kHz")

# %%
# The disorder realizations, each in the manual-Kerr frame at the Kerr its
# record saved, with 3a's branches for its occupations.
saved_disorder = MBRDisorderEnsembleExperiment.from_manifest(saved_disorder_manifest)
for record, part in zip(saved_disorder.realizations, saved_disorder.children):
    part.analyze(
        phase_frame="manual_kerr",
        manual_kerr_MHz=1e-3 * float(record["self_kerr_kHz"]),
        cycle_branches={o: saved_n3_cycle_branches.get(o, 0) for o in part.occupations},
        fft_window=saved_fft_window,
        zero_padding=saved_zero_padding,
        spectrum_method="fft",
    )
    print(f"r={record['realization']}: {len(part.job_ids)} traces, "
          f"K={1e3 * part.data.spectrum.physical_kerr_MHz:.4f} kHz")

# %% tags=["raises-exception"]
# Known failure, also before the redesign: the rebuilt disorder theory
# differs from the theory saved with the jobs (up to 0.3 kHz, every level).
# A physics question (which Hamiltonian inputs changed), not a loading one.
saved_disorder_mismatch_MHz = saved_disorder.recorded_theory_mismatch_MHz()
print("max |rebuilt - recorded| theory level (kHz):",
      {r: round(1e3 * m, 4) for r, m in saved_disorder_mismatch_MHz.items()})
assert max(saved_disorder_mismatch_MHz.values()) <= 1e-10

# %%
# Optional report figures from the reconstructed data.
saved_show_calibration = False
saved_show_n3 = True

if saved_show_calibration:
    saved_n3_calibration.display()
    plt.show()
if saved_show_n3:
    # saved_n3_data is the spectrum's latest analysis, which display() shows.
    saved_n3.display(level_statistics=False)
    plt.show()

# %% [markdown]
# ## 3b. With no calibration applied, in the as-acquired frame
#
# The same files as 3a, reanalyzed without the phase calibration. The mock
# station this section used to build is gone: it existed only so the loader
# could ask *something* for the Floquet cycle time, and asking a station
# returns today's calibration rather than the one these jobs ran under. The
# timing now comes from each job's own provenance.
#
# The job ranges no longer carry a project name either -- `resolve_job_paths`
# finds each file from its job ID, so which experiment directory a dataset
# landed in is no longer something this notebook has to know. That mattered
# here, because the four-realization set sits in a different project than the
# N=3 and disorder sets.

# %%
offline_four_realization_branches = {
    (3, 0, 0, 0, 0): 1,
    (2, 0, 0, 1, 0): 1,
}
# The disorder realizations are 3a's ensemble. The source asked here for
# JOB-20260816-00011..80 as one realization; that range cannot load (jobs
# 73-80 do not exist, 11-12 are not spectroscopy jobs), so step 7c uses the
# 3a partition. Whether the source meant another partition is a dataset
# question for jonginn.
offline_fft_window = "raw"
offline_zero_padding = 1

# %%
data_four_realization.analyze(
    phase_frame="as_acquired",
    cycle_branches=offline_four_realization_branches,
    fft_window=offline_fft_window,
    zero_padding=offline_zero_padding,
    spectrum_method="fft",
)
saved_n3_data = saved_n3.analyze(
    phase_frame="as_acquired",
    fft_window=offline_fft_window,
    zero_padding=offline_zero_padding,
    spectrum_method="fft",
)
for part in saved_disorder.children:
    part.analyze(
        phase_frame="as_acquired",
        fft_window=offline_fft_window,
        zero_padding=offline_zero_padding,
        spectrum_method="fft",
    )
print("as acquired: four-realization", len(data_four_realization.children), "traces; N=3",
      len(saved_n3.children), "traces; disorder",
      [len(part.job_ids) for part in saved_disorder.children], "traces")

# %%
# Optional figures; all data above came from local H5 files.
offline_show_n3 = True

if offline_show_n3:
    # saved_n3_data is the spectrum's latest analysis, which display() shows.
    saved_n3.display(level_statistics=False)
    plt.show()

# %% [markdown]
# ## Picking a data set
#
# Cells 226 and 234 were the same dict under two sections. One copy.

# %%
data_sets = {
    0: data_four_realization,
    1: saved_n3,
    2: saved_disorder.part(0),
}
pp_data = data_sets[2]

# %%
occupation_to_plot = (2, 0, 0, 1, 0)

MBRSpectrumExperiment.display_occupation(pp_data.data.reconstruction, pp_data.data.spectrum,
                                         occupation_to_plot, ldos_weight_cutoff=0.)
plt.show()

# %% [markdown]
# ## Coherent normalized-trace FFT for every loaded data set
#
# The FFT of $\sum_n A_n(t)/A_n(0)$, using each data set's own saved FFT
# convention. Needs `reconstruction.A`, so a merged spectrum-only data set
# raises rather than silently producing something else.

# %%
coherent_trace_results = {}
for index, data_set in data_sets.items():
    coherent_trace_results[index], _fig = data_set.display_coherent_trace()
    plt.show()
