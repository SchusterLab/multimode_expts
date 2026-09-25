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
# 286-307 and 315-317 by the stage-2 notebook decomposition, and moved to the
# redesigned MBR classes (docs/qsim/mbr_redesign.md). This is the
# **acquisition** entry point for the four core MBR products: phase
# calibration, orthogonality, spectroscopy, and the propagator. Saved-data
# analysis is the separate `analysis_notebooks/202609_qsim_migration/mbr.py`.
#
# ## How acquisition works now
#
# Each product is an *assembled* class that holds one job per occupation and
# never goes to the worker itself:
#
# | Product | Assembled class | Job class |
# |---|---|---|
# | phase calibration | `MBRCalibrationSetExperiment` | `MBRStarkCalExperiment` |
# | spectroscopy | `MBRSpectrumExperiment` | `MBRTimeTraceExperiment` |
# | orthogonality, $M_q$ | `MBROrthogonalityExperiment` | `MBROrthoColumnExperiment` |
# | propagator over $q$ | `MBRHamTomoExperiment` (`from_parts`) | -- |
#
# The pattern is always the same:
#
# ```python
# product = MBRxxxExperiment(occupations, ...)
# product.acquire(campaign_runner(campaign, station, client, JobClass))
# product.analyze(); product.display()
# product.save()          # manifest YAML + assembled HDF5 in <experiment>/assembled_data/
# ```
#
# A saved calibration set is the input of later products: each job records
# its manifest path beside the correction it played. Load one in a later
# session with `MBRCalibrationSetExperiment.from_manifest(path)`.
#
# `campaign = build_campaign(...)` is the shared base (defaults, modes); the
# other MBR notebooks build the same base, so none of them needs this one to
# have run. The defaults of cells 2-6 are in
# `experiments/qsim/notebook_helpers/defaults.py`.
#
# Its neighbours: `mbr_disorder.py`, `mbr_tomography.py`,
# `multiphoton_calibration.py`, `floquet_calibration.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt

from experiments import MultimodeStation

from job_server import JobClient
# Imported under the names the relocated cells already use.
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
from experiments.qsim.mbr_ortho_column import MBROrthoColumnExperiment
from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_stark_cal import MBRStarkCalExperiment
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
from experiments.qsim.notebook_helpers.mbr_campaign import (
    build_campaign,
    campaign_runner,
    fixed_n_occupations,
)

# %%
# The config versions this campaign ran against.
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
    station=station,
    client=client,
    floquet_settings=floquet_default_dict,
    active_reset_settings=active_reset_default_dict,
    measurement_settings=measurement_config_default_dict,
    modes=[1, 2, 3, 4],
    reps=1000,
)


def runner_for(JobClass):
    """The runner for one job class, over the campaign defaults."""
    return campaign_runner(campaign, station, client, JobClass, use_queue=RUN.use_queue)


print("mode order:", campaign.mode_labels)

# %% [markdown]
# ## 1. Phase calibration
#
# Either load a saved calibration set, or acquire a new one. Either way the
# later sections use `calibration`.

# %% [markdown]
# ### Load a saved calibration set
#
# The manifest paths are a dataset choice. Change only `encspec_N` to select
# the fixed-photon-number sector.

# %% tags=["suite-skip"]
# Suite: skipped. Needs a saved calibration manifest, and the next cell is
# its alternative.
calibration_manifests = {
    # 3: r"C:\experiments\260818_qsim_spectroscopy\assembled_data\<stem>.yaml",
}
encspec_N = 3
calibration = MBRCalibrationSetExperiment.from_manifest(calibration_manifests[encspec_N])
calibration.analyze()
calibration.display()
plt.show()

# %% [markdown]
# ### Run a new calibration
#
# Run this cell instead of the load cell when a new calibration is required.
# The jobs are saved as the usual job HDF5 files, and the set as a manifest.

# %%
encspec_N = 3
calibration = MBRCalibrationSetExperiment(
    fixed_n_occupations(encspec_N, len(campaign.mode_labels)),
    cycle_pairs=np.arange(0, 65, dtype=int),  # physical cycles: 0, 2, ..., 128
    swap_stors=campaign.modes,
    sync_cycles=campaign.sync_cycles,
    reps=RUN.pick(1000, smoke=100),
)
calibration.acquire(runner_for(MBRStarkCalExperiment), batch_size=10, log=True, show=False)
calibration.analyze()
calibration.display()
plt.show()
print("manifest:", calibration.save())

# %% [markdown]
# ### Replace selected calibration occupations
#
# Edit `recalibration_occupations` and run the first cell to acquire and
# inspect only those rows. If the fits are acceptable, run the following cell
# to make a new calibration set with those rows replaced. The previous set and
# its files stay unchanged.

# %%
# recalibration_occupations = [
#     [0, 0, 3, 0, 0],
#     [0, 0, 2, 1, 0],
# ]
recalibration_occupations = [
    occupation for occupation in fixed_n_occupations(
        encspec_N, len(campaign.mode_labels), descending=False)
    if occupation[1] > 0
]
recalibration_occupations = RUN.pick(recalibration_occupations,
                                     smoke=recalibration_occupations[:2])

# Same cycle grid and settings as the set being patched, so the replacement
# rows are measured the same way.
replacement_calibration = MBRCalibrationSetExperiment(
    recalibration_occupations,
    cycle_pairs=calibration.cycle_pairs,
    swap_stors=calibration.swap_stors,
    sync_cycles=calibration.sync_cycles,
    reps=RUN.pick(1000, smoke=100),
)
replacement_calibration.acquire(runner_for(MBRStarkCalExperiment), batch_size=2,
                                log=True, show=False)
replacement_calibration.analyze()
replacement_calibration.display()
plt.show()
print("replacement jobs:", replacement_calibration.job_ids)

# %%
# Accept the inspected rows. Refuses to mix jobs whose Floquet hardware
# differs; every other row keeps its job, so its phase cannot move.
old_phase_by_occupation = dict(zip(calibration.occupations, calibration.analyze().phase_mod180))
calibration = calibration.with_replacements(replacement_calibration)
calibration.analyze()
for occupation, phase in zip(calibration.occupations, calibration.data.phase_mod180):
    if occupation in replacement_calibration.occupations:
        print(occupation, f"{old_phase_by_occupation[occupation]:+.6f} -> {phase:+.6f} deg / cycle")
print("manifest:", calibration.save())

# %%
calibration.display()

# %% [markdown]
# ## 2. Zero-cycle encoder/decoder orthogonality
#
# Rows are decoder occupations and columns are encoder occupations. This
# coherent cross-return check uses no Floquet phase calibration and does not
# submit jobs until the following cell is run.

# %%
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

orthogonality = MBROrthogonalityExperiment(
    orthogonality_occupations,
    campaign.modes,
    sync_cycles=campaign.sync_cycles,
    reps=RUN.pick(1000, smoke=100),
)
print("mode order:", campaign.mode_labels)
print(f"{len(orthogonality_occupations)} jobs, "
      f"{4 * len(orthogonality_occupations)} points/job")
print("setup validated; run the next cell to submit jobs")

# %%
orthogonality.acquire(runner_for(MBROrthoColumnExperiment), batch_size=1,
                      log=True, show=False)
orthogonality_data = orthogonality.analyze()
orthogonality.display(figsize=(30, 8))
plt.show()
print("manifest:", orthogonality.save())

orthogonality_offdiagonal = orthogonality_data.offdiagonal_normalized_power.copy()
np.fill_diagonal(orthogonality_offdiagonal, np.nan)
print("job ids:", orthogonality.job_ids)
print("mode order:", orthogonality_data.mode_labels)
print("matrix orientation:", orthogonality_data.matrix_orientation)
print("raw diagonal |M_ii|:", orthogonality_data.diagonal_amplitude)
print("normalized leakage proxy by encoder column:", orthogonality_data.column_leakage)
print("max normalized off-diagonal power:", np.nanmax(orthogonality_offdiagonal))

# %% [markdown]
# ## 3. Diagonal spectroscopy of a fixed-N sector
#
# Every diagonal occupation in the fixed-N sector, with the calibration's
# correction played on the pulse, and the complete-basis DOS.

# %%
spectroscopy_N = RUN.pick(3, smoke=1)
spectroscopy_cycles = np.arange(0, RUN.pick(300, smoke=60), 2)
spectroscopy_cycle_branches = {}  # {occupation: branch}; unlisted use branch 0

spectrum = MBRSpectrumExperiment(
    fixed_n_occupations(spectroscopy_N, len(campaign.mode_labels)),
    spectroscopy_cycles,
    campaign.modes,
    calibration=calibration if spectroscopy_N == encspec_N else None,
    cycle_branches=spectroscopy_cycle_branches,
    sync_cycles=campaign.sync_cycles,
    reps=RUN.pick(500, smoke=100),
)
print("N:", spectroscopy_N, "occupations:", len(spectrum.occupations))

# %%
# Submit. Kept separate from the setup above so the job-submitting step is
# always its own cell.
spectrum.acquire(runner_for(MBRTimeTraceExperiment), batch_size=2, log=True, show=False)
spectroscopy_data = spectrum.analyze(cycle_branches=spectroscopy_cycle_branches)
if not spectroscopy_data.spectrum.complete_basis:
    raise RuntimeError("the acquired rows are not the complete fixed-N diagonal basis")
spectrum.display(level_statistics=False)
plt.show()
print("manifest:", spectrum.save())

# %%
# Immediate post-job analysis: Matrix Pencil beside the FFT.
spectroscopy_data = spectrum.analyze(
    cycle_branches=spectroscopy_cycle_branches,
    spectrum_method="mpm",
)
spectrum.display(spectrum_method="mpm")
print(spectrum.job_ids)

# %%
spectrum.display_occupations(occupations=spectrum.occupations[:3])

# %% [markdown]
# ## 4. Single matrix elements
#
# One `MBRTimeTraceExperiment` per (initial, final) pair; final != initial is
# an off-diagonal element. The analyzer correction is the calibration's value
# for the *final* occupation. Off-diagonal traces have no assembled class yet
# (disorder phase), so these are plain jobs.

# %%
element_pairs = [
    # (initial, final)
    ([1, 0, 1, 1, 0], [1, 0, 1, 1, 0]),
    # ([0, 0, 1, 1, 1], [1, 0, 1, 1, 0]),
]
element_cycles = np.arange(0, RUN.pick(300, smoke=60), 2)

element_overrides = [
    MBRTimeTraceExperiment.job_config(
        initial, final, element_cycles, campaign.modes,
        phase_per_cycle_deg=calibration.phase_for(final),
        calibration_manifest=calibration.manifest_path,
        sync_cycles=campaign.sync_cycles,
        reps=RUN.pick(500, smoke=100),
    )
    for initial, final in element_pairs
]
element_runner = runner_for(MBRTimeTraceExperiment)
element_traces = element_runner.execute(overrides=element_overrides, batch_size=2,
                                        log=True, show=False)
for trace in element_traces:
    trace.display()
plt.show()
print("job ids:", element_runner.last_job_ids)

# %% [markdown]
# ## 5. Propagator
#
# $M_q$ at several depths $q$: one `MBROrthogonalityExperiment` per depth,
# with the calibration's per-decoder correction on the pulse, combined by
# `MBRHamTomoExperiment`. Without a calibration set passed to `from_parts`
# this only stacks the matrices; the full tomography (complete basis plus
# calibration) is `mbr_tomography.py`.
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

propagator_parts = [
    MBROrthogonalityExperiment(
        propagator_occupations, campaign.modes, cycle=cycle, calibration=calibration,
        sync_cycles=campaign.sync_cycles, reps=RUN.pick(1000, smoke=100))
    for cycle in propagator_cycles
]

# %%
for part in propagator_parts:
    part.acquire(runner_for(MBROrthoColumnExperiment), batch_size=1, log=True, show=False)
    part.analyze()
    part.save()
propagator = MBRHamTomoExperiment.from_parts(propagator_parts)
propagator_data = propagator.analyze()
propagator.display()
plt.show()
print("manifest:", propagator.save())
