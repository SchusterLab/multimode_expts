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
# 2. **Reprocess from HDF5 only** (P256-276). *Moved to
#    `dormant/mbr_disorder_offdiag.py` in MBR redesign step 7a (2026-09-24); see
#    `docs/qsim/mbr_step7_plan.md`.* The same physics with nothing
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
# Its neighbours: `mbr.py`.

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
from experiments.qsim.notebook_helpers import mbr_disorder_preview as preview

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
    realization=0,
)

# %%
preview.report_preview_realization(diag_preview)

# %% [markdown]
# ## Analyze every completed disorder realization

# %%
diag_stats_records, diag_stats_failures = preview.analyze_every_realization(
    diag_preview_job_ids_by_realization=diag_preview_job_ids_by_realization,
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
