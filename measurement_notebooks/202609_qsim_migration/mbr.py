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
# # MBR core acquisition
#
# Split out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
# 286-307 and 315-317 by the stage-2 notebook decomposition. This is the
# **acquisition** entry point for the four core MBR products the surface map
# names: phase calibration, orthogonality, spectroscopy, and the propagator.
# Saved-data analysis is the separate
# `analysis_notebooks/202609_qsim_migration/mbr.py`.
#
# ## This section was the hub, so its handoffs are now explicit
#
# Cells 289-293 bound `EncSpec`, `BatchRunner`, `encspec_modes`,
# `encspec_mode_labels`, `encspec_sync_cycles`, `encspec_defaults`,
# `encspec_calibration_job_ids/files` and `encspec_calibrations` -- and the
# disorder, tomography and SFF sections then read those names straight out of
# the live kernel. Splitting them apart means that dependency has to be stated,
# so it is now one call:
#
# ```python
# campaign = build_campaign(station, client, ...)
# ```
#
# `mbr_disorder.py`, `mbr_tomography.py` and `mbr_sff.py` each open by building
# the same campaign base, so none of them needs this notebook to have run.
#
# ## Two cells deleted as confirmed duplicates
#
# Source cells 294 and 295 are byte-identical to cells 306 and 307, and sat
# *before* cell 305, which is what binds the `spectroscopy_expt`,
# `spectroscopy_occupations` and `cycle_branches` they read. They could only
# ever have worked as a stray re-run of the later pair. The surviving copies
# are cells 306/307, below under "3. Diagonal and off-diagonal spectroscopy".
#
# Long cells moved to `experiments/qsim/notebook_helpers/mbr_campaign.py`;
# cell 305 is split at its own substeps (select occupations, then phase-correct
# and build the batch) so that `execute()` stays a visible separate step.
# `ensure_calibration` there is the eight lines cells 305 and 316 shared.
#
# Setup is `experiments/qsim/notebook_helpers/qsim_session.py` (cells 2-6).
#
# Its neighbours: `mbr_disorder.py`, `mbr_tomography.py`, `mbr_sff.py`,
# `multiphoton_calibration.py`, `floquet_calibration.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy

import experiments as meas
from slab import AttrDict
from experiments import CharacterizationRunner, SweepRunner

# Imported under the names the relocated cells already use, so their bodies
# did not have to be edited. Definitions are in qsim_session.py.
from experiments.qsim.notebook_helpers.qsim_session import (
    open_session,
    ACTIVE_RESET_DEFAULTS as active_reset_default_dict,
    FLOQUET_DEFAULTS as floquet_default_dict,
    MEASUREMENT_CONFIG_DEFAULTS as measurement_config_default_dict,
)
from experiments.qsim.notebook_helpers.run_mode import run_settings

# Unset: a science run through the queue. The suite driver,
# tools/run_qsim_suite.py, sets mock or sandbox mode and the smoke profile.
RUN = run_settings()

# The four aggregate MBR stages. `EncodingHamiltonianSpectroscopyExperiment`
# is still the loading layer and the shared numerics, and is still the class
# every job here was acquired under -- so it stays, and these four sit beside
# it. See analysis_notebooks/guan/MBR_analysis.py for the worked example.
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
from experiments.qsim.mbr_propagator import MBRPropagatorExperiment

from experiments.qsim.notebook_helpers.floquet_calibration import (
    floquet_cycle_list_gen,
)
from experiments.qsim.notebook_helpers.mbr_campaign import (
    build_campaign,
    build_propagator_batch,
    build_spectroscopy_batch,
    ensure_calibration,
    fixed_n_occupations,
    merge_replacement_calibration,
    plot_propagator_matrices,
    select_spectroscopy_occupations,
    validate_recalibration_occupations,
)

# %%
# The config versions this campaign ran against. A scientific choice, so it
# stays written down here rather than defaulting inside open_session.
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
    run=RUN,
)
station = session.station
client = session.client
db = session.db
config_manager = session.config_manager

# %% [markdown]
# # N-photon Hamiltonian spectroscopy
#
# The diagonal and off-diagonal measurements use one analysis path. Set the decoder occupation equal to the prepared occupation for a diagonal return, or set it to a different state for an off-diagonal return.

# %%
# The campaign base. Source cells 289-293 did this inline; it is one call now
# so the other MBR notebooks can build the same base without this one.
#
# The calibration job IDs are a dataset choice, so they stay written out here.
# Both entries were commented out in the source, which is preserved -- fill in
# the sector you are working on, or acquire a fresh calibration below.
encspec_calibration_job_ids = {
    # 2: [f"JOB-20260824-{job:05d}" for job in range(64, 94)],
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

# The names the relocated cells below already use.
EncSpec = campaign.EncSpec
BatchRunner = campaign.BatchRunner
floquet_dark_mode_readout = campaign.floquet_dark_mode_readout
encspec_modes = campaign.modes
encspec_mode_labels = campaign.mode_labels
encspec_sync_cycles = campaign.sync_cycles
encspec_defaults = campaign.defaults
encspec_calibration_files = campaign.calibration_files
encspec_calibrations = campaign.calibrations

print("mode order:", encspec_mode_labels)
print("registered calibration sectors:", sorted(encspec_calibration_job_ids))

# %% [markdown]
# ## 1. Phase calibration
#
# The first option reads the calibration already stored by the job HDF5 files. The second option runs a fresh calibration through the same job/HDF5 path; no separate calibration save format is used.

# %% [markdown]
# ### Load the calibration already stored in job HDF5
#
# Change only `encspec_N` to select the fixed-photon-number sector.

# %% tags=["suite-skip"]
# Suite: skipped. Needs registered calibration job IDs, and the next cell is
# its alternative.
encspec_N = 3
calibration_expt = ensure_calibration(campaign, encspec_N, station)
calibration_expt.display()
plt.show()

# %% [markdown]
# ### Run a new calibration
#
# Run this cell instead of the HDF5-load cell when a new calibration is required. The resulting jobs are already saved as the usual job HDF5 files.

# %%
encspec_N = 3
encspec_calibration_occupations = fixed_n_occupations(
    encspec_N, len(encspec_mode_labels)
)
encspec_cycle_pairs = np.arange(0, 65, dtype=int)  # physical cycles: 0, 2, ..., 128

calibration_batch = MBRPhaseCorrectionExperiment.calibration_batch(
    encspec_defaults, encspec_modes, encspec_calibration_occupations,
    encspec_cycle_pairs, sync_cycles=encspec_sync_cycles, repeats=1,
    reps=RUN.pick(1000, smoke=100),
)
calibration_runner = BatchRunner(
    station=station,
    ExptClass=EncSpec,
    ExptProgram=floquet_dark_mode_readout.EntireFloquetCyclePhaseCalibrationProgram,
    default_expt_cfg=calibration_batch.default_expt_cfg,
    job_client=client,
    show=False,
)
calibration_expt = MBRPhaseCorrectionExperiment.from_batch(calibration_runner.execute(
    calibration_batch.configs, batch_size=10, log=True, show=False,
))
calibration_expt.analyze()
encspec_calibrations[encspec_N] = calibration_expt
calibration_expt.display()
plt.show()
calibration_expt.batch_job_ids

# %% [markdown]
# ### Replace selected calibration occupations
#
# Edit `recalibration_occupations` and run the first cell to acquire and inspect only those rows. If the fits are acceptable, run the following cell to replace those rows in the active in-memory calibration. The previous H5 files remain unchanged.

# %%
encspec_calibration_occupations = fixed_n_occupations(
    encspec_N, len(encspec_mode_labels), descending=False
)

# recalibration_occupations = [
#     [0, 0, 3, 0, 0],
#     [0, 0, 2, 1, 0],
# ]

recalibration_occupations = [entry for entry in encspec_calibration_occupations if entry[1] > 0]
recalibration_occupations = RUN.pick(recalibration_occupations, smoke=recalibration_occupations[:2])
recalibration_reps = RUN.pick(1000, smoke=100)
recalibration_batch_size = 2
recalibration_repeats = 1
recalibration_N = int(encspec_N)

base_calibration_expt, recalibration_keys = validate_recalibration_occupations(
    campaign, recalibration_N, recalibration_occupations
)

# Inherit the cycle grid and unwrap mode from the calibration being patched,
# so the replacement rows are measured the same way.
reference_cfg = base_calibration_expt.batch_expts[0].cfg.expt
recalibration_cycle_pairs = np.asarray(
    reference_cfg.n_cycle_pairs, dtype=int
)
recalibration_unwrap_mode = str(
    reference_cfg.get("phase_unwrap_mode", "pair")
)
recalibration_sync_cycles = int(
    reference_cfg.get("scramble_sync_cycles", encspec_sync_cycles)
)

recalibration_batch = MBRPhaseCorrectionExperiment.calibration_batch(
    encspec_defaults,
    encspec_modes,
    recalibration_occupations,
    recalibration_cycle_pairs,
    sync_cycles=recalibration_sync_cycles,
    repeats=recalibration_repeats,
    reps=recalibration_reps,
    unwrap_mode=recalibration_unwrap_mode,
)
recalibration_runner = BatchRunner(
    station=station,
    ExptClass=EncSpec,
    ExptProgram=(
        floquet_dark_mode_readout
        .EntireFloquetCyclePhaseCalibrationProgram
    ),
    default_expt_cfg=recalibration_batch.default_expt_cfg,
    job_client=client,
    show=False,
)
replacement_calibration_expt = MBRPhaseCorrectionExperiment.from_batch(recalibration_runner.execute(
    recalibration_batch.configs,
    batch_size=recalibration_batch_size,
    log=True,
    show=False,
))
replacement_calibration_expt.analyze(
    occupations=recalibration_occupations,
    cycle_pairs=recalibration_cycle_pairs,
    repeats=recalibration_repeats,
)
replacement_calibration_expt.display()
plt.show()
print("replacement jobs:", replacement_calibration_expt.batch_job_ids)

# %%
# Accept the inspected jobs and replace only these calibration rows. Refuses
# to mix jobs whose Floquet hardware differs, and verifies afterwards that no
# non-target row moved.
(
    updated_calibration_expt,
    old_phase_by_occupation,
    new_phase_by_occupation,
) = merge_replacement_calibration(
    campaign=campaign,
    station=station,
    N=recalibration_N,
    replacement_calibration_expt=replacement_calibration_expt,
    recalibration_keys=recalibration_keys,
    recalibration_cycle_pairs=recalibration_cycle_pairs,
)
calibration_expt = updated_calibration_expt

# %%
encspec_calibrations[encspec_N].display()

# %% [markdown]
# ## 2. Zero-cycle encoder/decoder orthogonality
#
# Rows are decoder occupations and columns are encoder occupations. This coherent cross-return check uses no Floquet phase calibration and does not submit jobs until the following cell is run.

# %%
orthogonality_N = int(encspec_N)
orthogonality_reps = RUN.pick(1000, smoke=100)
orthogonality_occupations = fixed_n_occupations(
    orthogonality_N, len(encspec_mode_labels), descending=False
)

orthogonality_occupations = [
    [3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0],
    [1, 2, 0, 0, 0],
    [2, 1, 0, 0, 0],
    # [0, 0, 0, 3, 0],
    # [1, 0, 0, 2, 0],
    # [2, 0, 0, 1, 0],
]


orthogonality_occupations.sort(reverse=True)

orthogonality_batch = MBROrthogonalityExperiment.orthogonality_batch(
    encspec_defaults,
    encspec_modes,
    orthogonality_occupations,
    sync_cycles=encspec_sync_cycles,
    reps=orthogonality_reps,
)

print("mode order:", encspec_mode_labels)
print("occupation count:", len(orthogonality_occupations))
print(
    f"{len(orthogonality_batch.configs)} jobs, "
    f"{orthogonality_batch.points_per_job} points/job, "
    f"{orthogonality_batch.total_points} total Ramsey points"
)
print("setup validated; run the next cell to submit jobs")


# %%
orthogonality_runner = BatchRunner(
    station=station,
    ExptClass=EncSpec,
    ExptProgram=(
        floquet_dark_mode_readout
        .EncodingOrthogonalityProgram
    ),
    default_expt_cfg=orthogonality_batch.default_expt_cfg,
    job_client=client,
    show=False,
)
orthogonality_expt = MBROrthogonalityExperiment.from_batch(orthogonality_runner.execute(
    orthogonality_batch.configs,
    batch_size=1,
    log=True,
    show=False
))


# %%
orthogonality_data = orthogonality_expt.analyze(
    occupations=orthogonality_occupations,
)
orthogonality_expt.display_orthogonality(figsize=(30, 8))
plt.show()

orthogonality_offdiagonal = (
    orthogonality_data.offdiagonal_normalized_power.copy()
)
np.fill_diagonal(orthogonality_offdiagonal, np.nan)
print("job ids:", orthogonality_expt.batch_job_ids)
print("mode order:", orthogonality_data.mode_labels)
print("matrix orientation:", orthogonality_data.matrix_orientation)
print("raw diagonal |M_ii|:", orthogonality_data.diagonal_amplitude)
print(
    "normalized leakage proxy by encoder column:",
    orthogonality_data.column_leakage,
)
print(
    "max normalized off-diagonal power:",
    np.nanmax(orthogonality_offdiagonal),
)


# %% [markdown]
# ## 3. Diagonal and off-diagonal spectroscopy
#
# Leave `batch_encspec_N = None` and edit `occupation` plus `decoder_occupation` to measure one matrix element. Set `batch_encspec_N = N` to ignore those two states, acquire every diagonal occupation in the fixed-N sector, and plot the complete-basis DOS.

# %%
# Set one value only: None for one matrix element, or N for all diagonal rows.
batch_encspec_N = RUN.pick(3, smoke=None) # int or None;
if batch_encspec_N is None:
    occupation = [
        [1, 0, 1, 1, 0],
        # [0, 0, 1, 1, 1],
        # [3, 0, 0, 0, 0],
    ]
    decoder_occupation = [
        [1, 0, 1, 1, 0],
        # [0, 0, 1, 1, 1],
        # [3, 0, 0, 0, 0],
    ]
else:
    occupation = None
    decoder_occupation = None

encspec_cycle_chunks = floquet_cycle_list_gen(
    0, RUN.pick(300, smoke=60), RUN.pick(300, smoke=60), 2
)
encspec_reps = RUN.pick(500, smoke=100) #1200
encspec_batch_size = 1 if batch_encspec_N is None else 10
encspec_batch_size = 2

spectroscopy_plan = select_spectroscopy_occupations(
    batch_encspec_N=batch_encspec_N,
    mode_labels=encspec_mode_labels,
    occupation=occupation,
    decoder_occupation=decoder_occupation,
)
spectroscopy_occupations = spectroscopy_plan.occupations
spectroscopy_final_occupations = spectroscopy_plan.final_occupations
spectroscopy_display_occupation = spectroscopy_plan.display_occupation
spectroscopy_N = spectroscopy_plan.N

spectroscopy_batch, spectroscopy_runner, calibration_expt, cycle_branches = (
    build_spectroscopy_batch(
        campaign=campaign,
        station=station,
        client=client,
        plan=spectroscopy_plan,
        cycle_chunks=encspec_cycle_chunks,
        reps=encspec_reps,
    )
)

# %%
# Submit. Kept separate from the setup above so the job-submitting step is
# always its own cell.
spectroscopy_expt = MBRSpectrumExperiment.from_batch(spectroscopy_runner.execute(
    spectroscopy_batch.configs,
    batch_size=encspec_batch_size,
    log=True,
    show=False,
))
spectroscopy_data = spectroscopy_expt.analyze(
    occupations=spectroscopy_occupations,
    cycle_branches=cycle_branches,
)
if batch_encspec_N is None:
    spectroscopy_expt.display(
        occupation=spectroscopy_display_occupation,
    )
else:
    if not spectroscopy_data.spectrum.complete_basis:
        raise RuntimeError(
            "the acquired rows are not the complete fixed-N diagonal basis"
        )
    spectroscopy_expt.display(level_statistics=False)
plt.show()

# %%
# Immediate post-job analysis, which the stage-2 instructions allow to stay in
# the measurement notebook. Source cells 306/307; cells 294/295 were an
# identical stray copy of this pair and are gone.
spectroscopy_data = spectroscopy_expt.analyze(
    occupations=spectroscopy_occupations,
    cycle_branches=cycle_branches,
    spectrum_method = 'mpm'
)

spectroscopy_expt.display(method = 'mpm')
print(spectroscopy_expt.batch_job_ids)

# %%
spectroscopy_expt.display_occupations(occupations = occupation)

# %% [markdown]
# ## 6. Propagator
#
# The setup uses the same occupation-keyed calibration. The next cell performs execute, analyze, and a direct matrix display.
#
# Sections 4 and 5 of the source (Matrix Pencil analysis and reanalysis of
# saved spectroscopy HDF5, cells 308-314) are saved-data work and moved to
# `analysis_notebooks/202609_qsim_migration/mbr.py`.

# %%
propagator_occupations = [
    [0, 0, 0, 0, 3],
    [1, 0, 0, 0, 2],
    [0, 1, 1, 1, 0],
]
propagator_cycles = [0, 4, 8]

propagator_batch, propagator_runner, calibration_expt = build_propagator_batch(
    campaign=campaign,
    station=station,
    client=client,
    propagator_occupations=propagator_occupations,
    propagator_cycles=propagator_cycles,
    reps=RUN.pick(1000, smoke=100),
)

# %%
propagator_expt = MBRPropagatorExperiment.from_batch(propagator_runner.execute(
    propagator_batch.configs,
    batch_size=1,
    log=True,
    show=False,
))
propagator_data = propagator_expt.analyze(
    occupations=propagator_occupations,
)

fig, axes = plot_propagator_matrices(propagator_data)
