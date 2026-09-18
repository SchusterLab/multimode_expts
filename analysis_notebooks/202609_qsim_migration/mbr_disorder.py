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
# Split out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
# 241-251 and 256-276 by the stage-2 notebook decomposition. Acquisition is
# `measurement_notebooks/202609_qsim_migration/mbr_disorder.py`.
#
# Two independent workspaces, which the source kept as two sections:
#
# 1. **Preview from job IDs** (P241-251). Loads realizations through the job
#    server, analyzes every completed one, and pools the adjacent-gap-ratio
#    level statistics.
# 2. **Reprocess from HDF5 only** (P256-276). The same physics with nothing
#    read from the job queue.
#
# ## The parallel loading layer
#
# Source cell 261 is a 248-line **notebook-local reimplementation of the HDF5
# loading layer**: `SavedSpectroscopyExperiment`, `read_h5_header`, `load_h5`,
# `read_shots`, `saved_parameters`, `saved_floquet_timing`,
# `common_grid_jobs`, `make_plain`. It is a parallel path to the library's own
# `EncodingHamiltonianSpectroscopyExperiment.from_h5file` and to
# `experiments/floquet_timing.resolve_floquet_timing`.
#
# It is preserved as found in
# `experiments/qsim/notebook_helpers/mbr_disorder_h5.py`, with a TODO, and
# **not** reconciled with the library — the stage-2 instructions put that
# decision in a later pass. It is the most substantial duplication this split
# found. Worth noting before anyone deletes either copy:
# `experiments/floquet_timing.py` exists precisely because asking a live
# station for historical timing silently substitutes today's calibration, so
# the notebook-local copy may be carrying a real correctness fix, or may be
# the thing that needed one.
#
# ## Settings and failures
#
# The `diag_preview_*`, `diag_stats_*`, `diag_level_*` and `mpm_*` prefixes
# are function arguments now. They were not collapsed into config dataclasses
# the way `d72_*` and `diag_disorder_*` were on the measurement side, because
# each has only a handful of knobs and they are per-step rather than shared.
#
# The source caught per-realization failures and left the exception bound in
# the notebook namespace for a later cell to look at. Each function returns
# its failures explicitly instead.
#
# The dataset manifest stays in this notebook: which jobs belong to which
# realization is a dataset choice.
#
# Its neighbours: `mbr.py`, `mbr_sampling.py`,
# `mbr_spectral_validation.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import experiments as meas
from slab import AttrDict
from job_server import JobClient

from experiments.job_paths import data_root
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.notebook_helpers import mbr_disorder_preview as preview
from experiments.qsim.notebook_helpers import mbr_disorder_h5 as h5only

client = JobClient()

# %% [markdown]
# # 1. Preview the current diagonal-disorder campaign from job IDs

# %%
(
    diag_preview_calibration_job_ids,
    diag_preview_job_ids_by_realization,
    diag_preview_branch_overrides,
) = preview.diag_preview_job_ids(branch_overrides={})

print("calibration jobs:", len(diag_preview_calibration_job_ids))
print("realizations:", len(diag_preview_job_ids_by_realization))

# %%
diag_preview = preview.load_preview_realization(
    diag_preview_calibration_job_ids=diag_preview_calibration_job_ids,
    diag_preview_job_ids_by_realization=diag_preview_job_ids_by_realization,
    diag_preview_branch_overrides=diag_preview_branch_overrides,
    client=client,
    realization=0,
)

# %%
preview.report_preview_realization(diag_preview)

# %% [markdown]
# ## Analyze every completed disorder realization

# %%
diag_stats_records, diag_stats_failures = preview.analyze_every_realization(
    diag_preview_job_ids_by_realization=diag_preview_job_ids_by_realization,
    client=client,
    edge_fraction=0.10,
    gap_ratio_bins=15,
    excluded_occupations=[],
)
if diag_stats_failures:
    print(f"{len(diag_stats_failures)} realizations failed:")
    for entry in diag_stats_failures:
        print("  ", entry)

# %% [markdown]
# ## Adjacent-gap-ratio level statistics

# %%
diag_pooled = preview.pool_level_statistics(
    diag_stats_records=diag_stats_records,
    diag_stats_edge_fraction=0.10,
)

# %%
preview.plot_pooled_level_statistics(
    pooled=diag_pooled,
    diag_stats_gap_ratio_bins=15,
)

# %% [markdown]
# ## Experimental and theoretical levels for every realization

# %%
preview.plot_levels_per_realization(
    diag_stats_records=diag_stats_records,
    realization=0,
    comparison_xlim_kHz=None,
    match_tolerance_bins=1.5,
)

# %% [markdown]
# # 2. Reprocess saved disorder spectroscopy — HDF5 only
#
# Nothing below reads the job queue.

# %%
# 1. Choose a dataset; its explicit job lists are in the manifest below.
dataset_to_postproc =  'Sep10 K3.6 g29.2'           # Sep05 K3.6 g30'
# The project subdirectory is part of the dataset; the root it sits under is
# per-machine, so it comes from $MULTIMODE_DATA_ROOT or the repo-root .env.
data_base_directory = data_root() / '260818_qsim_spectroscopy' / 'data'

load_shots = False  # FFT/MPM use means; shot checks read shots on demand.
reload_data = False  # True rereads selected H5 arrays; no server requests.
hardware_override = None  # Archived timing for a new dataset lacking an H5 snapshot.

manifest_directory = 'postprocess_manifests'  # None disables the optional JSON report.

dataset_dumps, hardware_by_job_id = h5only.build_dataset_manifest()
print("datasets in the manifest:", sorted(dataset_dumps))

# %%
(
    calibration_job_ids,
    job_ids_by_realization,
    selected_ids,
) = h5only.select_dataset(dataset_dumps, dataset_to_postproc)

print("calibration jobs:", len(calibration_job_ids))
print("realizations:", len(job_ids_by_realization))

# %% [markdown]
# ## Load saved HDF5 files (no MPM here)

# %%
loaded = h5only.load_saved_spectroscopy(
    dataset_to_postproc=dataset_to_postproc,
    data_base_directory=data_base_directory,
    manifest_directory=manifest_directory,
    calibration_job_ids=calibration_job_ids,
    job_ids_by_realization=job_ids_by_realization,
    selected_ids=selected_ids,
    hardware_by_job_id=hardware_by_job_id,
    hardware_override=hardware_override,
    load_shots=load_shots,
    reload_data=reload_data,
)
partial_realizations = loaded["partial_realizations"]

# %% [markdown]
# ## Preview the saved calibration and disorder traces — no MPM

# %%
reconfigured_expts, preview_failures = h5only.preview_saved_traces(loaded)
if preview_failures:
    print(f"{len(preview_failures)} previews failed:")
    for entry in preview_failures:
        print("  ", entry)

# %%
# Source cell 267: inspect one reconfigured experiment's occupation pair.
idx2disp = 3
occ_idx_2_disp = -1
init_occ = reconfigured_expts[idx2disp].data.reconstruction.occupations
fin_occ = reconfigured_expts[idx2disp].data.reconstruction.final_occupations

print("initial_occupations \n" f"{init_occ} \n" "final_occupations \n" f"{fin_occ}")

reconfigured_expts[idx2disp].display_occupations(occupations=init_occ[occ_idx_2_disp])

# %% [markdown]
# ## MPM settings

# %%
(
    spectroscopy_mpm_options,
    target_level_count,
    match_tolerance_bins,
    show_mpm_figures,
) = h5only.mpm_settings(
    target_level_count=35,
    track_frequency_tolerance_bins=1.0,
    minimum_consecutive_ranks=3,
    minimum_supporting_rows=1,
    merge_frequency_tolerance="calibration",
    calibration_sigma_multiplier=3.0,
    frequency_tolerance_floor_kHz=0.1,
    match_tolerance_bins=1.5,
    show_mpm_figures=False,
)

# %% [markdown]
# ## Run MPM on the loaded spectroscopy

# %%
spectroscopy_records = h5only.run_mpm(
    loaded=loaded,
    spectroscopy_mpm_options=spectroscopy_mpm_options,
    target_level_count=target_level_count,
)

# %% [markdown]
# ## Match the recovered poles to all 35 exact theory levels

# %%
h5only.report_theory_matches(
    spectroscopy_records=spectroscopy_records,
    match_tolerance_bins=match_tolerance_bins,
    ncols=3,
)

# %% [markdown]
# ## Level statistics of the recovered spectra

# %%
h5only.report_level_statistics(
    spectroscopy_records=spectroscopy_records,
    partial_realizations=partial_realizations,
    include_partial_acquisitions=False,
    edge_fraction=0.10,
    level_stat_bins=15,
)
