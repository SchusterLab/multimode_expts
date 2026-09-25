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
# The 7-1 diagonal-disorder campaign of `measurement_notebooks/jonginn/
# qsim_experiments.ipynb` (cells 323-334). The reports are
# `analysis_notebooks/202609_qsim_migration/mbr_disorder.py`.
#
# **MBR redesign step 7c (2026-09-24).** On the new classes: the phase
# calibration is an `MBRCalibrationSetExperiment`, each realization one
# `MBRSpectrumExperiment` (diagonal `MBRTimeTraceExperiment` jobs with the
# realization's detunings), and the campaign one
# `MBRDisorderEnsembleExperiment` built from the saved realizations. The
# planning is `notebook_helpers/mbr_disorder_campaign.py`; the old-class
# version is in `experiments/qsim/deprecated/`. The pairwise preview and 7-2
# (D72) are in `dormant/mbr_disorder_offdiag.py`
# (`docs/qsim/mbr_step7_plan.md`, decision 2).
#
# Each realization is saved as soon as it is acquired, so an interrupted
# campaign keeps what it finished; the ensemble can be built again from the
# saved parts.
#
# Its neighbours: `mbr.py`, `mbr_tomography.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt

from experiments import MultimodeStation

from job_server import JobClient
from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment
from experiments.qsim.mbr_stark_cal import MBRStarkCalExperiment
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
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
    build_campaign,
    campaign_runner,
    fixed_n_occupations,
)
from experiments.qsim.notebook_helpers.mbr_disorder_campaign import (
    DiagDisorderConfig,
    analyze_diagonal_disorder,
    plan_diagonal_disorder,
    realization_spectrum,
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
campaign = build_campaign(
    floquet_settings=floquet_default_dict,
    active_reset_settings=active_reset_default_dict,
    measurement_settings=measurement_config_default_dict,
    modes=[1, 2, 3, 4],
    reps=1000,
)


def runner_for(JobClass):
    """The runner for one job class, over the campaign defaults."""
    return campaign_runner(campaign, station, client, JobClass, use_queue=RUN.use_queue)


# %% [markdown]
# ## Phase calibration
#
# Load a saved calibration set, or acquire a new one. Either way the later
# cells use `calibration`, which must be saved: each job records its manifest.

# %% tags=["suite-skip"]
# Suite: skipped. Needs a saved calibration manifest, and the next cell is
# its alternative. The manifest path is a dataset choice.
calibration_manifest = None  # r"C:\experiments\260818_qsim_spectroscopy\assembled_data\<stem>.yaml"
calibration = MBRCalibrationSetExperiment.from_manifest(calibration_manifest)
calibration.analyze()
calibration.display()
plt.show()

# %%
# Run this cell instead of the load cell when a new calibration is required.
calibration = MBRCalibrationSetExperiment(
    fixed_n_occupations(3, len(campaign.mode_labels)),
    cycle_pairs=np.arange(0, 65, dtype=int),  # physical cycles: 0, 2, ..., 128
    swap_stors=campaign.modes,
    sync_cycles=campaign.sync_cycles,
    reps=RUN.pick(1000, smoke=100),
)
calibration.acquire(runner_for(MBRStarkCalExperiment), batch_size=10, log=True, show=False)
calibration.analyze()
print("calibration manifest:", calibration.save())

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
    # None uses best_self_kerr_kHz below when given, then the Kerr saved
    # with the calibration jobs.
    self_kerr_kHz=None,
    # Add only occupations whose calibration phase needs a nonzero 180-deg
    # branch.
    branch_overrides={},
)

# %% [markdown]
# #### 7-1b. Build the theory-selected plan and check time — no jobs

# %%
# A fitted M1 self-Kerr (for example `mbr_n3_reprocess.fit_self_kerr` in the
# analysis notebook), if you have one. None falls back to the calibration's.
best_self_kerr_kHz = None

diag_plan = plan_diagonal_disorder(calibration, campaign.modes, config=diag_config,
                                   best_self_kerr_kHz=best_self_kerr_kHz)

# %% [markdown]
# #### 7-1c. Acquire every realization — submits jobs
#
# One Spectrum per realization, saved when it is done. Rerunning the cell
# skips the realizations already in `diag_parts`.

# %%
diag_parts = {}

# %%
diag_runner = runner_for(MBRTimeTraceExperiment)
for record in diag_plan.realizations:
    realization = record["realization"]
    if realization in diag_parts:
        print(f"skip r={realization}: already acquired in this kernel")
        continue
    spectrum = realization_spectrum(diag_plan, record, calibration, campaign, diag_config)
    try:
        spectrum.acquire(diag_runner, batch_size=diag_config.batch_size, log=True, show=False)
    except BaseException:
        # An interrupted submission still has to name the jobs it sent.
        print(f"r={realization} submitted before interruption:",
              list(map(str, getattr(diag_runner, "last_job_ids", []))))
        raise
    spectrum.analyze()
    print(f"r={realization}: {spectrum.save()}")
    diag_parts[realization] = spectrum

# %% [markdown]
# #### 7-1d. Matrix-Pencil analysis and theory matching — no jobs
#
# Immediate post-job analysis. The standalone reports are in the analysis
# notebook.

# %%
done = [record for record in diag_plan.realizations if record["realization"] in diag_parts]
diag_ensemble = MBRDisorderEnsembleExperiment.from_parts(
    [diag_parts[record["realization"]] for record in done],
    realizations=done, calibration=calibration,
    notes="7-1 diagonal disorder campaign")

# %%
diag_data = analyze_diagonal_disorder(diag_ensemble, diag_plan, diag_config)

# %%
# Look at one realization.
idx_to_plot = 0
diag_ensemble.part(idx_to_plot).display(spectrum_method="fft")
diag_ensemble.part(idx_to_plot).display(spectrum_method="matrix_pencil")
plt.show()

# %%
for realization, part in diag_parts.items():
    print(realization, part.job_ids)

# %% [markdown]
# #### 7-1e. Pooled level statistics, and save — no jobs

# %%
diag_ensemble.display(gap_ratio_bins=diag_config.gap_ratio_bins)
plt.show()
print("ensemble manifest:", diag_ensemble.save())

# %%
station.update_all_station_snapshots()
