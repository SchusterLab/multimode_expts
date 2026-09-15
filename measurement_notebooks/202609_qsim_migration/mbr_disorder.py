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
# # Disorder campaign acquisition
#
# Split out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
# 318-349 by the stage-2 notebook decomposition. This is the **acquisition**
# half; the reports are
# `analysis_notebooks/202609_qsim_migration/mbr_disorder.py`.
#
# The split follows the source headings' own annotations, which already
# labelled each step "— no jobs" or "— submits jobs".
#
# Section 7 of the source ran three campaigns, kept as three code paths
# because they select channels and match theory differently:
#
# | | source cells | what it does |
# |---|---|---|
# | intro | 319-320 | a pairwise-detuning disorder preview and its batch |
# | 7-1 | 323-334 | diagonal disorder |
# | 7-2 | 338-348 | occupation-constrained disorder (`d72`) |
#
# ## The settings prefixes
#
# This theme is the one the stage-2 instructions had in mind. Cell 338 set
# **thirty** `d72_*` names at notebook scope and cells 340/342/344/346 read
# them as globals; cell 323 set sixteen `diag_disorder_*` names the same way.
# Those are now `D72Config` and `DiagDisorderConfig` in
# `experiments/qsim/notebook_helpers/mbr_disorder_campaign.py`, with the
# source's values and its own explanatory comments preserved on the fields.
#
# Each function unpacks its config into the local names its moved body already
# used, so none of those long bodies needed renaming.
#
# ## One cross-theme input
#
# Cell 325 read `best_self_kerr_kHz` from the acquisition notebook's section
# 3-1. That now lives in
# `mbr_n3_reprocess.fit_self_kerr_from_peak_overlap`, and is an explicit
# argument here. Passing None falls back the way the source did: to the signed
# Kerr saved with the phase-calibration jobs.
#
# Its neighbours: `mbr.py`, `mbr_tomography.py`, `mbr_sff.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy
from itertools import combinations_with_replacement, product

import experiments as meas
from slab import AttrDict
from experiments import CharacterizationRunner, SweepRunner

from experiments.qsim.notebook_helpers.qsim_session import (
    open_session,
    ACTIVE_RESET_DEFAULTS as active_reset_default_dict,
    FLOQUET_DEFAULTS as floquet_default_dict,
    MEASUREMENT_CONFIG_DEFAULTS as measurement_config_default_dict,
)
from experiments.qsim.notebook_helpers.mbr_campaign import (
    build_campaign,
    ensure_calibration,
)
from experiments.qsim.notebook_helpers.mbr_disorder_campaign import (
    D72Config,
    DiagDisorderConfig,
    analyze_d72,
    analyze_diag_disorder,
    build_d72_plans,
    build_diag_disorder_plans,
    build_pairwise_plan,
    plot_diag_level_statistics,
    preview_d72_jobs,
    submit_d72,
    submit_diag_disorder,
    submit_pairwise,
)

# %%
config_dict = {
    "hardware_config": "CFG-HW-20260904-00019",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260904-00014",
    "floquet_storage_swap": "CFG-FL-20260904-00042",
}

session = open_session(
    user="jonginn",
    experiment_name="260818_qsim_spectroscopy",
    project="EncSpec",
    config_dict=config_dict,
)
station = session.station
client = session.client

# %%
# Dataset choice: the phase-calibration jobs for the sector being measured.
encspec_calibration_job_ids = {
    # 3: [f"JOB-20260826-{job:05d}" for job in range(349, 419)],
}

campaign = build_campaign(
    station=station,
    client=client,
    floquet_settings=floquet_default_dict,
    active_reset_settings=active_reset_default_dict,
    measurement_settings=measurement_config_default_dict,
    modes=[1, 2, 3, 4],
    calibration_job_ids=encspec_calibration_job_ids,
    reps=1000,
)
ensure_calibration(campaign, 3, station)

# %% [markdown]
# ## 7. Disorder-resolved spectroscopy
#
# A pairwise-detuning preview first (source cells 319-320).

# %%
pairwise_plan = build_pairwise_plan(
    campaign=campaign,
    station=station,
    N=3,
    strength_kHz=50.0,
    seed=20260815,
    pair_count=10,
    reps=300,
    batch_size=1,
)

# %%
submit_pairwise(
    campaign=campaign,
    station=station,
    client=client,
    plan=pairwise_plan,
)

# %% [markdown]
# ### 7-1. Diagonal disorder spectroscopy
#
# #### 7-1a. Settings — no jobs

# %%
diag_config = DiagDisorderConfig(
    N=3,
    realization_count=20,
    strength_kHz=50.0,
    master_seed=20260816,
    selected_states=10,
    max_cycle=200,
    min_time_points=100,
    nyquist_margin=1.35,
    reps=1200,
    batch_size=2,
    edge_fraction=0.10,
    gap_ratio_bins=15,
    match_tolerance_bins=1.5,
    require_complete_match=False,
    # None uses best_self_kerr_kHz from the analysis notebook's Kerr fit when
    # given below, then falls back to the Kerr saved with the calibration.
    self_kerr_kHz=None,
    # Add only occupations whose calibration phase needs a nonzero 180-deg
    # branch.
    branch_overrides={},
)

# %% [markdown]
# #### 7-1b. Build the theory-selected plan and check time — no jobs

# %%
# The fitted M1 self-Kerr from the analysis notebook's section 3-1, if you
# have run it. None falls back to the calibration's saved value.
best_self_kerr_kHz = None

diag_plan = build_diag_disorder_plans(
    campaign=campaign,
    station=station,
    config=diag_config,
    best_self_kerr_kHz=best_self_kerr_kHz,
)

# %% [markdown]
# #### 7-1c. Run the batch — submits jobs

# %%
diag_disorder_records = submit_diag_disorder(
    campaign=campaign,
    station=station,
    client=client,
    plan=diag_plan,
    config=diag_config,
)

# %% [markdown]
# #### 7-1d. Matrix-Pencil analysis and theory matching — no jobs
#
# Immediate post-job analysis, which the stage-2 instructions allow to stay
# on the measurement side. The standalone reports are in the analysis
# notebook.

# %%
diag_disorder_records = analyze_diag_disorder(
    plan=diag_plan,
    config=diag_config,
    diag_analysis_error="raise",
)

# %% [markdown]
# #### 7-1e. Plot pooled level statistics — no jobs

# %%
plot_diag_level_statistics(plan=diag_plan, config=diag_config)

# %% [markdown]
# ### 7-2. Occupation-constrained disorder spectroscopy
#
# #### 7-2a. Settings and channel-selection helper — no jobs
#
# The thirty knobs of source cell 338. Their comments are preserved on the
# `D72Config` fields; the values below are the source's.

# %%
d72_config = D72Config(
    # Physics and disorder ensemble.
    N=3,
    realization_count=2,
    disorder_strength_kHz=50.0,
    master_seed=20260903,
    # State/channel constraints. None removes the cap; 1 keeps only hard-core
    # states; 2 excludes |3> states.
    max_occupation=3,
    forbidden_states=[
        # [1, 0, 1, 0, 1],
    ],
    channel_count=15,
    required_support=1,
    allow_diagonal=True,
    allow_offdiagonal=True,
    # A diagnostic/submit guard after calibration access is normalized.
    min_acceptable_visibility=1e-4,
    # Coherence-limited sampling and acquisition.
    max_time_us=80.0,
    min_time_points=50,
    nyquist_margin=1.35,
    step_autocalculate=False,
    cycle_step=2,  # used when step_autocalculate is False
    cycle_chunk_points=200,
    reps=1000,
    batch_size=2,
    # None uses the signed Kerr in the current hardware configuration.
    self_kerr_kHz=None,
    branch_overrides={},
    # None uses the active common calibration, then the common file map.
    calibration_job_ids=None,
    # Diagnostics and Matrix-Pencil settings.
    theory_plot_realizations=1,
    match_tolerance_bins=1.5,
    mpm_minimum_consecutive_ranks=3,
    mpm_minimum_supporting_rows=1,
    mpm_merge_frequency_tolerance="calibration",
    mpm_calibration_sigma_multiplier=3.0,
    mpm_frequency_tolerance_floor_kHz=0.1,
    mpm_dedup_frequency_tolerance_kHz=0.1,
)

# %% [markdown]
# #### 7-2b. Build constrained theory plans and select channels — no jobs

# %%
d72_plan = build_d72_plans(
    campaign=campaign,
    station=station,
    config=d72_config,
)

# %% [markdown]
# #### 7-2c. Choose the coherence-limited cycle grid, preview jobs, and plot theory — no jobs
#
# This is the guard before the batches go in. Read its output before running
# 7-2d.

# %%
d72_records = preview_d72_jobs(
    campaign=campaign,
    station=station,
    plan=d72_plan,
    config=d72_config,
)

# %% [markdown]
# #### 7-2d. Run the constrained disorder batches — submits jobs

# %%
d72_records = submit_d72(
    campaign=campaign,
    station=station,
    client=client,
    plan=d72_plan,
    d72_records=d72_records,
    config=d72_config,
)

# %% [markdown]
# #### 7-2e. Matrix-Pencil rank stability, calibration merge, and theory matching — no jobs

# %%
d72_records = analyze_d72(
    plan=d72_plan,
    d72_records=d72_records,
    config=d72_config,
    d72_analysis_error="raise",
)

# %%
station.update_all_station_snapshots()
