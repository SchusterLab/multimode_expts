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
# # Hamiltonian tomography
#
# Split out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
# 350-358 by the stage-2 notebook decomposition. One of the adjacent
# extensions on the surface map, kept as its own recipe because its
# measurement and analysis identity is distinct from the four core MBR
# products.
#
# This $N=1$ pilot measures $q=[0,s,2s]$. Besides the ordinary generalized
# eigenproblem it fits one shared $s$-cycle transfer matrix $F$ to both
# $M_s\approx F M_0$ and $M_{2s}\approx F M_s$. Comparing that fit for raw
# versus phase-corrected matrices tests whether the fixed-depth phase
# correction preserves a time-homogeneous matrix model. The reported
# $E/h=-\arg(\lambda)/(2\pi sT)$ is in the principal Floquet zone and is
# defined modulo $1/(sT)$.
#
# **Run 8-1 through 8-4 in order.** A data set without all three depths
# cannot be passed to 8-3.
#
# Each depth is one `MBROrthogonalityExperiment` (the matrix $M_q$), acquired
# and saved on its own; `MBRHamTomoExperiment.from_parts` combines them
# (docs/qsim/mbr_redesign.md). The campaign base is an explicit
# `build_campaign` call, so this notebook stands alone -- though it needs a
# saved N=1 `MBRCalibrationSetExperiment`, and says so rather than guessing
# if none is given.
#
# `fit_shared_step` was a `def` nested in cell 356 closing over four notebook
# names; it is a module-level function in
# `experiments/qsim/notebook_helpers/mbr_tomography.py` now.
#
# Its neighbours: `mbr.py`, `mbr_disorder.py`, `mbr_sff.py`,
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
from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
from experiments.qsim.mbr_ham_tomo import MBRHamTomoExperiment
from experiments.qsim.mbr_stark_cal import MBRStarkCalExperiment
from experiments.qsim.notebook_helpers.mbr_campaign import (
    build_campaign,
    campaign_runner,
    fixed_n_occupations,
)
from experiments.qsim.notebook_helpers.mbr_tomography import (
    analyze_tomography,
    build_tomography_plan,
    fit_shared_step,
    plot_tomography_diagnostics,
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
# The campaign base, and the N=1 calibration this pilot needs. The
# calibration is a dataset choice: the manifest YAML of a saved
# MBRCalibrationSetExperiment (under <experiment>/assembled_data/).
hamtom_calibration_manifest = None

campaign = build_campaign(
    station=station,
    client=client,
    floquet_settings=floquet_default_dict,
    active_reset_settings=active_reset_default_dict,
    measurement_settings=measurement_config_default_dict,
    modes=[1, 2, 3, 4],
    reps=1000,
)

hamtom_N = 1
if RUN.smoke:
    # Smoke run: no saved calibration to load, so acquire one on the same
    # cycle grid as mbr.py's "Run a new calibration" cell.
    hamtom_calibration = MBRCalibrationSetExperiment(
        fixed_n_occupations(hamtom_N, len(campaign.mode_labels)),
        cycle_pairs=np.arange(0, 65, dtype=int), swap_stors=campaign.modes,
        sync_cycles=campaign.sync_cycles, reps=RUN.pick(1000, smoke=100),
    )
    hamtom_calibration.acquire(
        campaign_runner(campaign, station, client, MBRStarkCalExperiment,
                        use_queue=RUN.use_queue),
        batch_size=10, log=True, show=False,
    )
    hamtom_calibration.analyze()
    hamtom_calibration.save()
elif hamtom_calibration_manifest is None:
    raise RuntimeError("set hamtom_calibration_manifest to a saved N=1 calibration set")
else:
    hamtom_calibration = MBRCalibrationSetExperiment.from_manifest(
        hamtom_calibration_manifest)

# %% [markdown]
# ### 8-1. Build the $q=[0,s,2s]$ plan and check acquisition size — no jobs

# %%
hamtom_plan = build_tomography_plan(
    campaign=campaign,
    station=station,
    client=client,
    calibration=hamtom_calibration,
    use_queue=RUN.use_queue,
    N=hamtom_N,
    step=10,
    reps=RUN.pick(1000, smoke=100),
    batch_size=5,
)

# %% [markdown]
# ### 8-2. Acquire the three $5	imes5$ matrices — submits fifteen jobs
#
# One saved `MBROrthogonalityExperiment` per depth, then the tomography that
# combines them. Each part's manifest is saved as soon as it is acquired.

# %%
for hamtom_part in hamtom_plan["parts"]:
    hamtom_part.acquire(hamtom_plan["runner"], batch_size=hamtom_plan["batch_size"],
                        log=True, show=False)
    hamtom_part.analyze()
    hamtom_part.save()
    print(f"q={hamtom_part.cycle}: jobs {hamtom_part.job_ids}")

hamtom_expt = MBRHamTomoExperiment.from_parts(
    hamtom_plan["parts"], calibration=hamtom_calibration)

# %% [markdown]
# ### 8-3. Analyze three depths and fit one shared transfer matrix — no jobs

# %%
(
    hamtom_data,
    hamtom_raw_fit,
    hamtom_corrected_fit,
    hamtom_theory_frequencies_MHz,
) = analyze_tomography(hamtom_expt, hamtom_plan)
hamtom_expt.save()

# %% [markdown]
# ### 8-4. Three-depth diagnostics — no jobs

# %%
plot_tomography_diagnostics(
    hamtom_data=hamtom_data,
    hamtom_raw_fit=hamtom_raw_fit,
    hamtom_corrected_fit=hamtom_corrected_fit,
    hamtom_theory_frequencies_MHz=hamtom_theory_frequencies_MHz,
    plan=hamtom_plan,
)
