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
# # Disorder campaign analysis
#
# The 7-1 diagonal-disorder campaign (jonginn `qsim_experiments.ipynb` cells
# 323-334, source analysis `data_postprocess.ipynb` cells 241-251). Acquisition
# is `measurement_notebooks/202609_qsim_migration/mbr_disorder.py`.
#
# **MBR redesign step 7c (2026-09-24).** The saved realizations are one
# `MBRDisorderEnsembleExperiment`: the old jobs were converted with
# `tools/migrate_mbr_jobs.py` (kind `disorder`) into one
# `MBRSpectrumExperiment` per realization, sharing one calibration set. The
# manifest path is the dataset choice. The old preview helpers
# (`notebook_helpers/mbr_disorder_preview.py`) moved to
# `experiments/qsim/deprecated/`; on the same raw files they give the same
# poles, theory levels and gap ratios as the ensemble
# (`tests/test_mbr_disorder_ensemble.py`, baseline).
#
# The source cells overwrote some of their own arguments with hard-coded
# values: the preview always loaded realization 12, the pooled analysis
# always left out occupation (0, 3, 0, 0, 0), and the level plot always
# showed realization 14 at 0.5 bins. Those values are now the arguments below,
# so the results are the ones the source printed.
#
# The off-diagonal (D72) HDF5 reprocessing, which was section 2 here, is in
# `dormant/mbr_disorder_offdiag.py` (docs/qsim/mbr_step7_plan.md).
#
# Its neighbours: `mbr.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt

from experiments.job_paths import data_root
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment

# %% [markdown]
# # 1. The 7-1 diagonal-disorder campaign
#
# Realizations 0-18 of `diagonal_disorder_71` (tests/data/mbr_datasets.json);
# r=19 has 11 of 20 jobs and was not converted.

# %%
diag_disorder_manifest = data_root() / "260818_qsim_spectroscopy" / "assembled_data" / "260924_195637_MBRDisorderEnsembleExperiment.yaml"

ensemble = MBRDisorderEnsembleExperiment.from_manifest(diag_disorder_manifest)
print("realizations:", [record["realization"] for record in ensemble.realizations])
print("calibration set:", ensemble.calibration.manifest_path)

# %% [markdown]
# ## One realization, with the calibration applied (source cells 243-244)
#
# The manual-Kerr frame at the Kerr the realization recorded.

# %%
diag_preview_realization = 12
diag_preview_part = ensemble.part(diag_preview_realization)
diag_preview_record = ensemble.realizations[
    [record["realization"] for record in ensemble.realizations].index(diag_preview_realization)]
diag_preview_dimension = 35

diag_preview_data = diag_preview_part.analyze(
    phase_frame="manual_kerr",
    manual_kerr_MHz=1e-3 * float(diag_preview_record["self_kerr_kHz"]),
    cycle_branches={},
    spectrum_method="mpm",
    fft_window="raw",
    zero_padding=1,
    mpm_requested_max_modes=diag_preview_dimension,
    mpm_match_decay=False,
    mpm_track_frequency_tolerance_bins=0.10,
    mpm_merge_frequency_tolerance_bins=0.10,
    mpm_dedup_frequency_tolerance_bins=0.10,
    mpm_minimum_supporting_rows=1,
)
diag_preview_poles_kHz = 1e3 * np.sort(np.asarray(
    diag_preview_data.matrix_pencil.selected_frequencies_MHz, dtype=float))
print(f"r={diag_preview_realization}: MPM poles {len(diag_preview_poles_kHz)}/{diag_preview_dimension}")
print("MPM frequencies (kHz):", np.round(diag_preview_poles_kHz, 3))
print("FFT resolution (kHz):", 1e3 * float(diag_preview_data.spectrum.fft_resolution_MHz))
diag_preview_part.display(spectrum_method="fft")
diag_preview_part.display(spectrum_method="mpm")
plt.show()

# %% [markdown]
# ## Every realization, and the pooled level statistics (source cells 246-249)
#
# As acquired, theory at each realization's recorded Kerr, occupation
# (0, 3, 0, 0, 0) left out, 10 % of the levels cut at each edge.

# %%
diag_stats = ensemble.analyze(
    phase_frame="as_acquired",
    theory_kerr_MHz="recorded",
    excluded_occupations=[(0, 3, 0, 0, 0)],
    edge_fraction=0.10,
    on_error="skip",
    mpm_track_frequency_tolerance_bins=0.50,
    mpm_merge_frequency_tolerance_bins=0.50,
    mpm_dedup_frequency_tolerance_bins=0.50,
)
if diag_stats.ratio_failures:
    print("realizations without measured gap ratios:", diag_stats.ratio_failures)
print(f"theory: mean r = {diag_stats.theory.mean:.3f}")
if diag_stats.measured is not None:
    print(f"measured: mean r = {diag_stats.measured.mean:.3f} +/- {diag_stats.measured.sem:.3f} "
          f"({diag_stats.measured.closer_to})")

# %%
ensemble.display(gap_ratio_bins=15)
plt.show()

# %% [markdown]
# ## Experimental and theoretical levels of one realization (source cell 251)

# %%
ensemble.display_levels(realization=14, match_tolerance_bins=0.5)
plt.show()
