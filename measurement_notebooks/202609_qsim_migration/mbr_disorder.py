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
# **MBR redesign step 7a (2026-09-24):** the pairwise preview and 7-2 (D72)
# moved to `dormant/mbr_disorder_offdiag.py`; only 7-1 is left here. See
# `docs/qsim/mbr_step7_plan.md`. The table below describes the notebook before
# the split.
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
# Its neighbours: `mbr.py`, `mbr_tomography.py`.

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
from experiments import CharacterizationRunner, SweepRunner, MultimodeStation

from job_server import JobClient
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS as active_reset_default_dict,
    FLOQUET_DEFAULTS as floquet_default_dict,
    MEASUREMENT_CONFIG_DEFAULTS as measurement_config_default_dict,
)
from experiments.qsim.notebook_helpers.run_mode import run_settings

# Set by tools/run_qsim_suite.py. Unset: through the queue in the main
# checkout, directly on this kernel in a worktree (see run_mode.py).
RUN = run_settings()
from experiments.qsim.notebook_helpers.mbr_campaign import (
    acquire_calibration,
    build_campaign,
    ensure_calibration,
)
from experiments.qsim.deprecated.legacy_mbr import MBRSpectrumExperiment
from experiments.qsim.notebook_helpers.mbr_disorder_campaign import (
    DiagDisorderConfig,
    analyze_diag_disorder,
    build_diag_disorder_plans,
    build_diag_realization_batch,
    plot_diag_level_statistics,
)

# %%
config_dict = {
    "hardware_config": "CFG-HW-20260904-00019",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260904-00014",
    "floquet_storage_swap": "CFG-FL-20260904-00042",
}

station = MultimodeStation(
    user="jonginn",
    experiment_name="260818_qsim_spectroscopy",
    project="EncSpec",
    log_measurements=not RUN.smoke,
    **RUN.station_configs(config_dict),
)
client = JobClient()

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
if RUN.smoke:
    # Smoke run: no calibration job IDs to load, so acquire a calibration
    # on the same cycle grid as mbr.py's "Run a new calibration" cell.
    acquire_calibration(
        campaign, station, client, N=3,
        use_queue=RUN.use_queue,
        cycle_pairs=np.arange(0, 65, dtype=int),
        reps=RUN.pick(1000, smoke=100),
    )
else:
    ensure_calibration(campaign, 3, station)

# %% [markdown]
# ## 7. Disorder-resolved spectroscopy
#
# ### 7-1. Diagonal disorder spectroscopy
#
# #### 7-1a. Settings — no jobs

# %%
diag_config = DiagDisorderConfig(
    N=3,
    realization_count=RUN.pick(20, smoke=1),
    strength_kHz=50.0,
    master_seed=20260816,
    selected_states=RUN.pick(10, smoke=3),
    max_cycle=200,
    min_time_points=100,
    nyquist_margin=1.35,
    reps=RUN.pick(1200, smoke=100),
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
# Acquire every realization first. Each aggregate contains only one
# disorder H, so this is one batch per realization.
diag_disorder_records = diag_plan["diag_disorder_records"]

for realization_plan in diag_plan["diag_disorder_plans"]:
    diag_realization = realization_plan["realization"]
    if diag_realization in diag_disorder_records:
        print(f"skip r={diag_realization}: already completed in this kernel")
        continue

    diag_batch, diag_runner, diag_branches = build_diag_realization_batch(
        campaign=campaign,
        station=station,
        client=client,
        use_queue=RUN.use_queue,
        plan=diag_plan,
        config=diag_config,
        realization_plan=realization_plan,
    )
    try:
        diag_expt = MBRSpectrumExperiment._from_expts(diag_runner.execute(
            overrides=diag_batch.configs,
            batch_size=diag_config.batch_size,
            log=True,
            show=False,
        ), job_ids=diag_runner.last_job_ids, station=diag_runner.station)
    except BaseException:
        # An interrupted submission still has to name the jobs it sent.
        print(f"r={diag_realization} submitted before interruption:",
              list(map(str, getattr(diag_runner, "last_job_ids", []))))
        raise

    diag_disorder_records[diag_realization] = dict(
        plan=realization_plan,
        expt=diag_expt,
        job_ids=list(map(str, diag_expt.batch_job_ids)),
        cycle_branches=diag_branches,
        data=None,
    )
    print(f"finished acquisition r={diag_realization}:",
          diag_disorder_records[diag_realization]["job_ids"])

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

# %%
# Source cells 328 and 329: look at one realization, then list every
# realization's job IDs. Short interactive pokes, left as plain cells.
idx_to_plot = 0
diag_disorder_records[idx_to_plot]['expt'].display()
diag_disorder_records[idx_to_plot]['expt'].display_occupations(occupations = [[0, 1, 0, 0, 2],
                                                                              [1, 2, 0, 0, 0],
                                                                              [1, 0, 1, 0, 1]])

# %%
for each in diag_disorder_records.keys():
    print(diag_disorder_records[each]["job_ids"])

# %% [markdown]
# #### 7-1e. Plot pooled level statistics — no jobs

# %%
plot_diag_level_statistics(plan=diag_plan, config=diag_config)


# %%
station.update_all_station_snapshots()
