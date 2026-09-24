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
# # Disorder-averaged spectral form factor
#
# Split out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
# 359-369 by the stage-2 notebook decomposition. One of the adjacent
# extensions on the surface map, kept as its own recipe.
#
# This measures the exact fixed-$N$ trace, so every one of the 35 diagonal
# occupations is acquired for each realization. `total_reps_per_realization=4`
# means two shots in each of the two independent replicas, and the full SFF
# uses the cross product of those replicas, so the disconnected part is
# estimated without a same-shot bias.
#
# **Run the steps in order.** 9-2 measures the depth-zero visibility once and
# every realization reuses it, so it has to precede 9-3.
#
# ## This one submits a lot of jobs
#
# 9-3 submits the 2000-realization ensemble. `build_sff_plan` prints the
# workload preview first and submits nothing, which is the point of keeping
# 9-1 as its own step -- check the estimated hours before running 9-3.
#
# Cell 361's thirteen `sff_*` notebook globals are now one `SFFConfig`, per
# the stage-2 instruction about grouping a repeated settings prefix. Its
# defaults are cell 361's values.
#
# The helper module is `mbr_sff_campaign.py`, not `mbr_sff.py`: the library
# already has `experiments/qsim/mbr_sff.py`, which is where
# `DisorderSFFExperiment` and its `batch`/`analyze_ensemble` live. The helper
# is only the notebook's planning and reporting around that.
#
# Its neighbours: `mbr.py`, `mbr_disorder.py`, `mbr_tomography.py`,
# `floquet_displacement_kerr.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy
from itertools import product

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
from experiments.qsim.notebook_helpers.mbr_sff_campaign import (
    SFFConfig,
    analyze_and_plot_sff,
    build_sff_plan,
    compare_effective_hamiltonian_ensemble,
    plot_sff_visibility,
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
# The campaign base. The N=3 calibration must exist before planning -- fill
# in the job IDs, a dataset choice.
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
# ### 9-1. Settings and workload preview — no jobs
#
# Read the printed estimate before running 9-3.

# %%
sff_config = SFFConfig(
    N=3,
    realization_count=RUN.pick(2000, smoke=10),
    # 4 = two shots in each of the two independent replicas.
    total_reps_per_realization=4,
    disorder_strength_kHz=50.0,
    master_seed=20260901,
    # The coherence-limited grid of section 7-2, copied rather than imported
    # so this theme does not depend on the disorder campaign.
    max_time_us=80.0,
    cycle_step=2,
    visibility_reps=RUN.pick(1000, smoke=100),
    realizations_per_job=5,
    batch_size=14,
    bootstrap_samples=RUN.pick(300, smoke=20),
    branch_overrides={},
)

sff = build_sff_plan(
    campaign=campaign,
    station=station,
    client=client,
    config=sff_config,
)

# %% [markdown]
# ### 9-2. Measure depth-zero visibility — submits one job
#
# Run this before the full ensemble. It plots all 35 encoder/decoder returns,
# and the ensemble analysis divides by it.

# %%
sff_visibility_runner = CharacterizationRunner(
    station=station,
    ExptClass=sff["SFFExperiment"],
    ExptProgram=sff["plan"].program,
    default_expt_cfg=sff["plan"].default_expt_cfg,
    job_client=client,
    use_queue=RUN.use_queue,
    show=False,
)

# configs[:1] is the depth-zero visibility configuration.
sff_visibility_expts = sff_visibility_runner.execute(
    configs=sff["plan"].configs[:1],
    batch_size=1,
    log=True,
    show=False,
)
print("visibility job:", sff_visibility_runner.last_job_ids)

sff_visibility_expt = plot_sff_visibility(sff_visibility_expts, sff)

# %% [markdown]
# ### 9-3. Run the 2000-realization ensemble — submits jobs
#
# This submits only the positive-depth disorder jobs; the visibility
# configuration was already measured above.

# %%
sff_disorder_runner = CharacterizationRunner(
    station=station,
    ExptClass=sff["SFFExperiment"],
    ExptProgram=sff["plan"].program,
    default_expt_cfg=sff["plan"].default_expt_cfg,
    job_client=client,
    use_queue=RUN.use_queue,
    show=False,
)

# configs[1:] skips the visibility configuration measured in 9-2. The bare
# except is the source's: an interrupted submission still has to name the jobs
# it already sent.
try:
    sff_disorder_batch = sff_disorder_runner.execute(
        configs=sff["plan"].configs[1:],
        batch_size=sff["batch_size"],
        log=True,
        show=False,
    )
except BaseException:
    print("jobs submitted before the stop/failure:",
          sff_disorder_runner.last_job_ids)
    raise

print("completed disorder jobs:", len(sff_disorder_batch))
if sff_disorder_runner.last_job_ids:  # local runs have no queue job IDs
    print("first job:", sff_disorder_runner.last_job_ids[0])
    print("last job:", sff_disorder_runner.last_job_ids[-1])

# %% [markdown]
# ### 9-4. Analyze and plot the measured SFF — no jobs

# %%
sff_measured = analyze_and_plot_sff(
    sff=sff,
    sff_disorder_batch=sff_disorder_batch,
    sff_visibility_expt=sff_visibility_expt,
)
sff_time_us = sff_measured["time_us"]
sff_full = sff_measured["full"]
sff_connected = sff_measured["connected"]
sff_standard_error = sff_measured["standard_error"]

# %% [markdown]
# ### 9-5. Compare with the effective-Hamiltonian ensemble — no jobs
#
# This diagonalizes the same 35-dimensional effective Hamiltonian for every
# realization's detunings, so it is a theory curve for the exact ensemble that
# was measured -- not a random-matrix surrogate.

# %%
compare_effective_hamiltonian_ensemble(
    campaign=campaign,
    sff=sff,
    measured=sff_measured,
)
