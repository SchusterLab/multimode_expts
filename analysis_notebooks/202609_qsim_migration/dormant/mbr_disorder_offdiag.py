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
# # Off-diagonal disorder analysis (D72 HDF5 reprocessing, saved pair batch) -- DORMANT
#
# Moved on 2026-09-24 (MBR redesign step 7a), without changes except the import
# lines:
# - section 2 of `analysis_notebooks/202609_qsim_migration/mbr_disorder.py`
#   (D72 HDF5 reprocessing; source `data_postprocess.ipynb` P256-276);
# - the "Reanalyze a saved spectroscopy batch" cell of
#   `analysis_notebooks/202609_qsim_migration/mbr.py` (off-diagonal pair jobs
#   `JOB-20260823-00005..08`).
#
# See `docs/qsim/mbr_step7_plan.md`, decision 2: off-diagonal time traces
# (init != final) have no valid Stark-shift phase calibration.
#
# Not maintained; may break when live code changes. If it breaks, add a note
# here and do not fix it.

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

from experiments.job_paths import data_root
from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.deprecated.legacy_mbr import MBRSpectrumExperiment
from experiments.qsim.deprecated.legacy_mbr import (
    MBRSpectrumExperiment as LegacySpectrumExperiment,
)
from experiments.qsim.notebook_helpers import mbr_disorder_preview as preview
from experiments.qsim.deprecated import mbr_disorder_h5 as h5only

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

# %% [markdown]
# ## Reanalyze a saved spectroscopy batch
#
# Source cell 314. Loads four known-good interleaved off-diagonal jobs without
# acquiring new data. Replace the job list to analyze another saved batch.
#
# Loads from HDF5: no station, and so no risk of picking up today's
# calibration in place of the one these jobs ran under.
#
# Step 7: these are old off-diagonal pair jobs, which the migration cannot
# convert yet, so this cell stays on the old class.

# %%
saved_spectroscopy_job_ids = [
    f"JOB-20260823-{job:05d}" for job in range(5, 9)
]
saved_spectroscopy_expt = LegacySpectrumExperiment.from_job_ids(
    saved_spectroscopy_job_ids,
)
saved_spectroscopy_expt.analyze(
    cycle_branches={(0, 0, 2, 0, 0): 0},
)
saved_spectroscopy_expt.display(
    occupation=[0, 0, 1, 1, 0],
)
plt.show()
