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
# # Off-diagonal disorder acquisition (pairwise, D72) -- DORMANT
#
# Moved from `measurement_notebooks/202609_qsim_migration/mbr_disorder.py` on
# 2026-09-24 (MBR redesign step 7a), without changes except the import lines.
# See `docs/qsim/mbr_step7_plan.md`, decision 2: off-diagonal time traces
# (init != final) have no valid Stark-shift phase calibration. The helpers are
# in `experiments/qsim/deprecated/mbr_disorder_offdiag.py`.
#
# Not maintained; may break when live code changes. If it breaks, add a note
# here and do not fix it.

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
from experiments.qsim.deprecated.mbr_campaign_legacy import (
    acquire_calibration,
    build_campaign,
    ensure_calibration,
)
from experiments.qsim.deprecated.legacy_mbr import MBRSpectrumExperiment
from experiments.qsim.deprecated.mbr_disorder_offdiag import (
    D72Config,
    analyze_d72,
    build_d72_plans,
    build_pairwise_batch,
    build_pairwise_plan,
    check_d72_visibility,
    preview_d72_jobs,
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
# A pairwise-detuning preview first (source cells 319-320).

# %%
pairwise_plan = build_pairwise_plan(
    campaign=campaign,
    station=station,
    N=3,
    strength_kHz=50.0,
    seed=20260815,
    pair_count=RUN.pick(10, smoke=2),
    reps=RUN.pick(300, smoke=100),
    batch_size=1,
)

# %%
disorder_batch, disorder_runner, disorder_cycle_branches = build_pairwise_batch(
    campaign=campaign,
    station=station,
    client=client,
    use_queue=RUN.use_queue,
    plan=pairwise_plan,
)

disorder_expt = MBRSpectrumExperiment._from_expts(disorder_runner.execute(
    overrides=disorder_batch.configs,
    batch_size=pairwise_plan["disorder_batch_size"],
    log=True,
    show=False,
), job_ids=disorder_runner.last_job_ids, station=disorder_runner.station)
disorder_expt.analyze(
    cycle_branches=disorder_cycle_branches,
    spectrum_method="mpm",
    mpm_requested_max_modes=len(pairwise_plan["disorder_theory"].energies_MHz),
    mpm_match_decay=False,
)
disorder_expt.display(spectrum_method="fft")
disorder_expt.display(spectrum_method="mpm")
plt.show()

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
    realization_count=RUN.pick(2, smoke=1),
    disorder_strength_kHz=50.0,
    master_seed=20260903,
    # State/channel constraints. None removes the cap; 1 keeps only hard-core
    # states; 2 excludes |3> states.
    max_occupation=3,
    forbidden_states=[
        # [1, 0, 1, 0, 1],
    ],
    channel_count=RUN.pick(15, smoke=3),
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
    reps=RUN.pick(1000, smoke=100),
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
check_d72_visibility(d72_plan)

for realization_plan in d72_plan["d72_plans"]:
    if realization_plan.realization in d72_records:
        print(f"skip r={realization_plan.realization}: already completed "
              "in this 7-2 campaign")
        continue

    # build_d72_plans already built the batch for each realization.
    d72_runner = CharacterizationRunner(
        station=station,
        ExptClass=campaign.EncSpec,
        ExptProgram=realization_plan.batch.program,
        default_expt_cfg=realization_plan.batch.default_expt_cfg,
        job_client=client,
        use_queue=RUN.use_queue,
        show=False,
    )
    try:
        d72_expt = MBRSpectrumExperiment._from_expts(d72_runner.execute(
            overrides=realization_plan.batch.configs,
            batch_size=d72_config.batch_size,
            log=True,
            show=False,
        ), job_ids=d72_runner.last_job_ids, station=d72_runner.station)
    except BaseException:
        print(f"r={realization_plan.realization} submitted before "
              "interruption:",
              list(map(str, getattr(d72_runner, "last_job_ids", []))))
        raise

    d72_records[realization_plan.realization] = AttrDict(dict(
        plan=realization_plan,
        expt=d72_expt,
        job_ids=list(map(str, d72_expt.batch_job_ids)),
        cycle_branches=realization_plan.cycle_branches,
        data=None,
    ))
    print(f"finished 7-2 acquisition r={realization_plan.realization}:",
          d72_records[realization_plan.realization].job_ids)

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
