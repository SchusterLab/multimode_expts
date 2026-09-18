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
# # Shot and sampling studies
#
# Split out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
# 193-203 by the stage-2 notebook decomposition. On the surface map this is
# one of the two "measurement and inference studies" workspaces.
#
# Everything here **replays already-acquired shots**. Nothing acquires data.
# The question is what the spectrum would have looked like with fewer shots,
# or with the occupation drawn at random rather than scanned.
#
# 1. **Saved-shot subsampling** (P193-197). Resample the stored shots at
#    several counts, repeat with different seeds, and track how the FFT and
#    the rowwise Matrix-Pencil spectra degrade.
# 2. **Randomized-occupation replay** (P198-203). Draw the occupation per
#    shot instead of sweeping it, under several protocols, and compare against
#    the full scan.
#
# ## The handoff this split made explicit
#
# The source read `encspec_reprocessed`, `spectroscopy_expt` and
# `calibration_expt` out of the live kernel — the N=3 reprocessing section had
# to have been run first, in the same notebook. This notebook builds them
# itself by calling `mbr_n3_reprocess`, so it does not depend on cell order in
# another file.
#
# The four closures in source cell 199 stay nested inside
# `build_random_occupation_pool` and are handed back in its result. They read
# that cell's own locals, and promoting them to module level would have meant
# inventing a parameter list for each and rewriting every call site — more
# than a structural move.
#
# Neither sampling method was promoted into `fitting/`. The surface map says
# library promotion for this workspace can be decided later, and the two
# protocols are deliberately kept distinct.
#
# Its neighbours: `mbr.py`, `mbr_disorder.py`,
# `mbr_spectral_validation.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import experiments as meas
from slab import AttrDict

from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.notebook_helpers.mbr_loading import (
    job_id_generator,
    load_encoding_spectroscopy,
)
from experiments.qsim.notebook_helpers import mbr_n3_reprocess as n3
from experiments.qsim.notebook_helpers import mbr_sampling as sampling

# %% [markdown]
# ## Load the N=3 dataset these studies replay
#
# Dataset choice: the same complete N=3 calibration and spectroscopy job
# ranges the reprocessing notebook uses.

# %%
calibration_job_ids = job_id_generator(
    [20260722, 20260723], [683, 1], [712, 40]
)
spectroscopy_job_ids = job_id_generator(
    20260723, [48, 87], [85, 149], step=[1, 2]
)

calibration_expt, spectroscopy_expt = load_encoding_spectroscopy(
    MBRSpectrumExperiment,
    calibration_job_ids,
    spectroscopy_job_ids,
)

# %%
encspec_cycle_branches = {}
encspec_legacy = True
encspec_manual_kerr_MHz = -19.756e-3

encspec_reprocessed = n3.reprocess_n3_spectroscopy(
    calibration_expt=calibration_expt,
    spectroscopy_expt=spectroscopy_expt,
    cycle_branches=encspec_cycle_branches,
    legacy=encspec_legacy,
    manual_kerr_MHz=encspec_manual_kerr_MHz,
)

# %% [markdown]
# # 1. N=3 saved-shot subsampling
#
# Resample the stored shots at several counts, several times each.

# %%
shots = sampling.run_shot_subsampling(
    spectroscopy_expt=spectroscopy_expt,
    calibration_expt=calibration_expt,
    encspec_reprocessed=encspec_reprocessed,
    encspec_cycle_branches=encspec_cycle_branches,
    encspec_legacy=encspec_legacy,
    encspec_manual_kerr_MHz=encspec_manual_kerr_MHz,
    repeats=3,
    base_seed=20260901,
    axis_scale="log",
    rowwise_mpm_kwargs={},
)
failures = shots.get("encspec_N3_shot_failures", [])
if failures:
    print(f"{len(failures)} resamplings failed:")
    for entry in failures:
        print("  ", entry)

# %%
shot_summary = sampling.summarize_shot_subsampling(shots)

# %%
sampling.plot_rowwise_pole_counts(shots, shot_summary)

# %%
sampling.inspect_example_shots(
    shots,
    example_shots=None,
    example_repeat=0,
)

# %% [markdown]
# # 2. N=3 randomized-occupation replay
#
# Build the shot pool and the reference spectra the protocols are compared
# against.

# %%
occ_pool = sampling.build_random_occupation_pool(
    spectroscopy_expt=spectroscopy_expt,
    encspec_reprocessed=encspec_reprocessed,
    pool_seed=20260902,
)

# %% [markdown]
# ## Run the single-shot-average sweep

# %%
occ_sweep = sampling.run_random_occupation_sweep(
    spectroscopy_expt=spectroscopy_expt,
    pool=occ_pool,
    repeats=3,
    base_seed=20260903,
    protocols=None,
)
occ_failures = occ_sweep.get("encspec_N3_random_occ_failures", [])
if occ_failures:
    print(f"{len(occ_failures)} replays failed:")
    for entry in occ_failures:
        print("  ", entry)

# %% [markdown]
# ## Inspect the FFT, MPM frequencies, and SFF
#
# Figure controls only; edit these rather than the module.

# %%
sampling.plot_random_occupation_results(
    pool=occ_pool,
    sweep=occ_sweep,
    overview_figsize=(16, 14),
    summary_figsize=(18, 5),
    fig_dpi=None,  # None uses matplotlib.rcParams['figure.dpi'].
    legend_fontsize=8,
    protocol_styles=None,
)
