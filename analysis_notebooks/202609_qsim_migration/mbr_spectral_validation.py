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
# # Spectral validation
#
# Split out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
# 188, 252-255 and 277-306 by the stage-2 notebook decomposition. On the
# surface map this is the second "measurement and inference studies"
# workspace.
#
# Its subject is the source's own heading for cell 281: **can the measured
# level statistics support a conclusion?** Every section below is a different
# attempt to answer that without leaning on theory.
#
# | source cells | method |
# |---|---|
# | 188 | global-only Matrix Pencil — no Hamiltonian energies, no FFT peak positions |
# | 252-255 | joint block-Hankel shared-pole test at fixed rank |
# | 277-280 | pole selection by row-normalized component singular values |
# | 282 | whether the recovered statistics are conclusive at all |
# | 283-291 | data-only shared-frequency refinement: clock correction, disjoint shot halves |
# | 292-297 | theory-free pole recovery, 1 kHz reproducibility, level-statistics sensitivity |
# | 298-303 | independent-half reproducibility, then a posthoc configured-H comparison |
# | 304-306 | an alternative row-pooled target-35 diagnostic |
#
# ## These are competing methods, not a pipeline
#
# Several of these selections disagree with each other, and which one is right
# is the open question this workspace exists to settle. None were merged,
# renamed to look alike, or reconciled with `MBRSpectrumExperiment`'s own
# spectrum methods. Run the sections you want to compare; there is no intended
# top-to-bottom order beyond the numbered subsections within each method.
#
# ## Two handoffs, and one broken dependency
#
# Cell 188 read `encspec_reprocessed` from the live kernel, built by the N=3
# reprocessing section of what is now `mbr.py`. Cells 279 onward read a long
# list of names from the disorder analysis, now `mbr_disorder.py` — including
# `SavedSpectroscopyExperiment` and `wrap_frequency`, the notebook-local HDF5
# loader and frequency wrapper, which are imported from
# `mbr_disorder_h5` rather than copied a third time.
#
# **`threadpoolctl` is not installed in this environment and is not declared
# in `pyproject.toml`,** but source cells 293 and 299 imported it to bound
# BLAS threads during the precision search. Those cells could not have run
# here as written. The helper module keeps the name as a shim that raises when
# used, so everything else imports and the affected functions fail loudly
# rather than silently dropping the thread limiting. See the TODO in
# `mbr_spectral_validation.py`.
#
# Function signatures in that module were derived from what each cell actually
# read, not guessed, so several take long argument lists. That is honest about
# how entangled the source cells were; shortening them means deciding which
# inputs are really one object, which this pass does not do.
#
# Its neighbours: `mbr.py`, `mbr_disorder.py`, `mbr_sampling.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import experiments as meas
from slab import AttrDict

from experiments.job_paths import data_root
from experiments.qsim.deprecated.legacy_mbr import MBRSpectrumExperiment
from experiments.qsim.notebook_helpers.mbr_loading import (
    job_id_generator,
    load_encoding_spectroscopy,
)
from experiments.qsim.notebook_helpers import mbr_n3_reprocess as n3
from experiments.qsim.notebook_helpers import mbr_disorder_h5 as h5only
from experiments.qsim.notebook_helpers import mbr_disorder_preview as dpreview
from experiments.qsim.notebook_helpers import mbr_spectral_validation as sv

# %% [markdown]
# # Global-only Matrix Pencil (source cell 188)
#
# Uses no Hamiltonian energies and no FFT peak positions. Needs the N=3
# reprocessed dataset, which this cell builds rather than inheriting.

# %%
calibration_expt, spectroscopy_expt = load_encoding_spectroscopy(
    MBRSpectrumExperiment,
    job_id_generator([20260722, 20260723], [683, 1], [712, 40]),
    job_id_generator(20260723, [48, 87], [85, 149], step=[1, 2]),
)
encspec_reprocessed = n3.reprocess_n3_spectroscopy(
    calibration_expt=calibration_expt,
    spectroscopy_expt=spectroscopy_expt,
    cycle_branches={},
    legacy=True,
    manual_kerr_MHz=-19.756e-3,
)

# %%
sv.matrix_pencil_global_diagnostic(encspec_reprocessed)

# %% [markdown]
# # Load the disorder dataset the remaining methods analyze
#
# Everything below works on the saved disorder realizations, loaded from HDF5
# through the same path `mbr_disorder.py` uses.

# %%
dataset_to_postproc = 'Sep10 K3.6 g29.2'
# The project subdirectory is part of the dataset; the root it sits under is
# per-machine, so it comes from $MULTIMODE_DATA_ROOT or the repo-root .env.
data_base_directory = data_root() / '260818_qsim_spectroscopy' / 'data'

dataset_dumps, hardware_by_job_id = h5only.build_dataset_manifest()
(
    calibration_job_ids,
    job_ids_by_realization,
    selected_ids,
) = h5only.select_dataset(dataset_dumps, dataset_to_postproc)

loaded = h5only.load_saved_spectroscopy(
    dataset_to_postproc=dataset_to_postproc,
    data_base_directory=data_base_directory,
    manifest_directory=None,
    calibration_job_ids=calibration_job_ids,
    job_ids_by_realization=job_ids_by_realization,
    selected_ids=selected_ids,
    hardware_by_job_id=hardware_by_job_id,
    hardware_override=None,
    load_shots=True,  # the shot-halves methods below need the raw shots
    reload_data=True,
)
loaded_spectroscopy = loaded["loaded_spectroscopy"]
loaded_dataset_name = loaded["loaded_dataset_name"]
phase_calibration = loaded["phase_calibration"]
partial_realizations = loaded["partial_realizations"]

# %%
(
    spectroscopy_mpm_options,
    target_level_count,
    match_tolerance_bins,
    show_mpm_figures,
) = h5only.mpm_settings()

spectroscopy_records = h5only.run_mpm(
    loaded=loaded,
    spectroscopy_mpm_options=spectroscopy_mpm_options,
    target_level_count=target_level_count,
)

# %% [markdown]
# # Joint block-Hankel shared-pole test (cells 252-255)
#
# Fits one shared pole set across occupations at fixed rank.

# %%
diag_stats_records, diag_stats_failures = dpreview.analyze_every_realization(
    diag_preview_job_ids_by_realization=job_ids_by_realization,
)

# %%
sv.report_block_hankel_matches(diag_stats_records)

# %% [markdown]
# # Alternative selection: row-normalized component singular values (cells 277-280)

# %%
row_singular_records = sv.collect_row_singular_records(spectroscopy_records)

# %%
sv.report_row_singular_statistics(
    match_levels=h5only.match_levels,
    match_tolerance_bins=match_tolerance_bins,
    partial_realizations=partial_realizations,
    row_singular_records=row_singular_records,
    spectroscopy_records=spectroscopy_records,
)

# %% [markdown]
# # Can the measured level statistics support a conclusion? (cell 282)
#
# The workspace's own honesty check, and the reason the other selections are
# kept side by side rather than merged into one answer.

# %%
sv.assess_statistical_power(spectroscopy_records)

# %% [markdown]
# # Data-only shared-frequency refinement (cells 283-291)
#
# ## 1. Build corrected traces and disjoint shot halves
#
# Needs the raw shots, which is why `load_shots=True` above.

# %%
joint_inputs = sv.build_corrected_traces_and_halves(
    load_shots=True,
    loaded_dataset_name=loaded_dataset_name,
    loaded_spectroscopy=loaded_spectroscopy,
    parameter_key=h5only.parameter_key,
    phase_calibration=phase_calibration,
    read_shots=h5only.read_shots,
    saved_floquet_timing=h5only.saved_floquet_timing,
    saved_parameters=h5only.saved_parameters,
)

# 35 for N=3 with four leaves, as source cell 286 computed it.
joint_count = math.comb(3 + 4, 3)

# %% [markdown]
# ## Optional: test the 1 kHz scale with independent measured shot halves

# %%
sv.test_kHz_scale_with_shot_halves(
    joint_inputs=joint_inputs,
    match_levels=h5only.match_levels,
    partial_realizations=partial_realizations,
    spectroscopy_records=spectroscopy_records,
)

# %% [markdown]
# # Theory-free pole recovery and its reproducibility (cells 292-297)

# %%
precision_inputs = sv.prepare_precision_inputs(joint_count=joint_count)

# %%
# The precision search's own knobs. This is the step that needs threadpoolctl
# -- it will raise with an explanatory message if that is still missing.
precision_count = joint_count
precision_target_kHz = 1.0
precision_maxiter = 200
precision_exact_maxiter = 50
precision_probe_pairs = 8
precision_repeat = 3

precision_results = sv.run_precision_search(
    joint_inputs=joint_inputs,
    loaded_spectroscopy=loaded_spectroscopy,
    partial_realizations=partial_realizations,
    precision_count=precision_count,
    precision_exact_maxiter=precision_exact_maxiter,
    precision_maxiter=precision_maxiter,
    precision_probe_pairs=precision_probe_pairs,
    precision_repeat=precision_repeat,
    precision_target_kHz=precision_target_kHz,
)

# %%
sv.report_precision_statistics(
    precision_count=precision_count,
    precision_results=precision_results,
    precision_target_kHz=precision_target_kHz,
)

# %% [markdown]
# # Fit each half independently, then refit all shots (cells 298-301)

# %%
joint_amplitude_bound = 4.0
joint_compare_count = joint_count
joint_maxiter = 200
joint_swap_attempts = 8

joint_results = sv.fit_halves_independently(
    joint_amplitude_bound=joint_amplitude_bound,
    joint_compare_count=joint_compare_count,
    joint_count=joint_count,
    joint_inputs=joint_inputs,
    joint_maxiter=joint_maxiter,
    joint_swap_attempts=joint_swap_attempts,
)

# %%
sv.report_half_reproducibility(
    joint_count=joint_count,
    joint_results=joint_results,
)

# %% [markdown]
# ## Configured-H comparison — posthoc only
#
# This compares against the configured Hamiltonian *after* the theory-free
# fit, so it does not feed back into the pole recovery.

# %%
sv.compare_configured_hamiltonian(
    joint_results=joint_results,
    loaded_spectroscopy=loaded_spectroscopy,
)

# %% [markdown]
# # Alternative row-pooled target-35 diagnostic (cells 304-306)

# %%
row_pool_records = sv.select_row_pooled_levels(
    match_levels=h5only.match_levels,
    match_tolerance_bins=match_tolerance_bins,
    spectroscopy_records=spectroscopy_records,
)

# %%
sv.report_row_pooled_level_statistics(
    pool_edge_fraction=0.10,
    pool_merge_tolerance_kHz=0.1,
    row_pool_records=row_pool_records,
    spectroscopy_records=spectroscopy_records,
)
