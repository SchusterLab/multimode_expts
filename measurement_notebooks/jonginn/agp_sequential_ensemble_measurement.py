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
#     display_name: Multimode (direct remote)
#     language: python
#     name: multimode-direct
# ---

# %% [markdown]
# # Heralded sequential readout and disorder-ensemble AGP
#
# Follow the calibration, walk, AGP, save and reload cells in the reference notebook.
# Every ensemble shot uses one saved disorder realization and reads M1 first,
# followed by the storages. All readout modes share the preparation-herald cut.
# The same ordered disorder patterns are reused across initial Fock states,
# evolution times and disorder strengths, including W=0.
#
# The Experiment classes assemble measured correlations and kernel integrals.
# Hamiltonian predictions are separate comparisons, without a theory correction.
# Metric entries retain the cross correlations needed for the K/W parameter plane.

# %% tags=["imports"]
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.qsim.agp_sequential_ensemble import CentralBosonAGPSequentialEnsembleDisorderScan
from experiments.qsim.fock_population_sequential import FockPopulationSequentialReadoutExperiment
from experiments.qsim.fock_population_sequential_ensemble import (
    FockPopulationSequentialEnsembleExperiment,
    FockPopulationSequentialHeraldExperiment,
)
from experiments.qsim.multiparity_confusion_matrix_sequential import (
    HeraldedMultiParityConfusionMatrixSequentialExperiment,
    MultiParityConfusionMatrixSequentialExperiment,
)
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS
from job_server import JobClient
from theory.qsim.central_boson import fixed_n_hamiltonian, fock_number_traces

# %% [markdown]
# ## 1. Station and worker
#
# Configuration versions and defaults follow agp_ensemble_measurement.py.
# Constructing the runners does not submit measurements.

# %% tags=["station"]
user = "jonginn"
config_dict = {'hardware_config': 'CFG-HW-20261113-00052',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20261113-00028',
 'floquet_storage_swap': 'CFG-FL-20261113-00028'}

client = JobClient()
health = client.health_check()
print(f"Server status: {health['status']}")
client.print_queue()

station = MultimodeStation(
    user=user,
    experiment_name="agp_sequential_disorder_ensemble",
    project="AGP",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
FLOQUET_DEFAULTS['scramble_sync_cycles'] = 2

# %% tags=["defaults"]
# The calibration and population JOBs use the same preparation-herald condition.
HERALD = True

STORAGE_MODES = [1, 2, 4, 5]
RO_STORS = [0, *STORAGE_MODES]
common_defaults = AttrDict(dict(
    expts=1, reps=1, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=STORAGE_MODES, pre_relax_delay=100, relax_delay=200,
    postpulse=True, init_fock=False, init_stor=0, ro_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    swap_stors=STORAGE_MODES, update_phases=True,
    detunings=[0.0] * len(STORAGE_MODES),
))
common_defaults.update(ACTIVE_RESET_DEFAULTS)
common_defaults.update(FLOQUET_DEFAULTS)

population_defaults = AttrDict(dict(
    common_defaults, prepulse=False, occupations=[3] + [0] * len(STORAGE_MODES),
    ro_stors=RO_STORS, floquet_cycles=[0], swept_params=["floquet_cycle"],
    prep_herald="end" if HERALD else "none",
    multiparity_confusion_matrix=None, multiparity_confusion_matrices=None,
))
calibration_defaults = AttrDict(dict(
    population_defaults, background_occupations=[0] * len(RO_STORS),
))

# %% tags=["runners"]
confusion_matrix_runner = CharacterizationRunner(
    station=station,
    ExptClass=(HeraldedMultiParityConfusionMatrixSequentialExperiment
               if HERALD else MultiParityConfusionMatrixSequentialExperiment),
    default_expt_cfg=calibration_defaults,
    job_client=client,
    use_queue=True,
    show=False,
)

population_runner = CharacterizationRunner(
    station=station,
    ExptClass=(FockPopulationSequentialHeraldExperiment
               if HERALD else FockPopulationSequentialReadoutExperiment),
    default_expt_cfg=population_defaults,
    job_client=client,
    use_queue=True,
    show=False,
)

ensemble_runner = CharacterizationRunner(
    station=station,
    ExptClass=FockPopulationSequentialEnsembleExperiment,
    default_expt_cfg=population_defaults,
    job_client=client,
    use_queue=True,
    show=False,
)

# %% [markdown]
# ## 2. Fock 0..3 calibration for every sequential readout step
#
# One JOB sweeps four prepared numbers per readout step with the full readout
# order. With HERALD, vacuum also receives a dummy preparation herald.
# The saved matrix has axes (prepared number, measured label, readout step).
# Pass its HDF5 filename as multiparity_confusion_matrices for both the walk
# and the ensemble JOBs.

# %% tags=["calibration"]
calibration_expt = confusion_matrix_runner.execute(
    reps=6000, postprocess=False,
    go_kwargs=dict(analyze=True, save=True),
)
print(calibration_expt.fname)

# %% tags=["calibration_display"]
calibration_expt.display()
condition_numbers = np.linalg.cond(
    np.moveaxis(calibration_expt.data["multiparity_confusion_matrices"], -1, 0))
for mode, condition_number in zip(RO_STORS, condition_numbers):
    print(f"Mode {mode}: confusion matrix condition number = {condition_number:.3f}")

# %% tags=["calibration_reference"]
CONFUSION_MATRIX_FNAME = str(calibration_expt.fname)
# To reuse a saved calibration, replace the assignment above with its full HDF5 filename.

# %% [markdown]
# ## 3. Population evolution from one initial occupation
#
# This is an ordinary zero-detuning walk, without an ensemble table.
# A single JOB reads all final modes after each preparation and evolution.
# It uses the same sequential calibration and herald condition as the batch.
# For total photon number 3, number modulo four agrees with photon number.

# %% tags=["walk_settings"]
swap_stors = list(STORAGE_MODES)
meas_stors = list(RO_STORS)
occupations = [3] + [0] * len(STORAGE_MODES)
detunings = [0.0] * len(STORAGE_MODES)  # MHz
floquet_cycles = list(range(0, 100, 4))

# %% tags=["walk_acquire"]
walk_expt = population_runner.execute(
    reps=1000,
    occupations=occupations,
    swap_stors=swap_stors,
    storage_reset=swap_stors,
    ro_stors=meas_stors,
    detunings=detunings,
    floquet_cycles=floquet_cycles,
    multiparity_confusion_matrices=CONFUSION_MATRIX_FNAME,
    postprocess=False, show=False, log=True,
    go_kwargs=dict(analyze=True, save=True),
)
print(walk_expt.fname)

# %% tags=["walk_display"]
walk_expt.display(raw=True)
walk_fig, walk_axes = walk_expt.display()
plt.show()

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
print(walk_expt.fname)

# %% [markdown]
# ### Optional: compare the walk with the fixed-N Hamiltonian
#
# Use the recorded coupling and Floquet timing with the supplied physical Kerr.
# The theory functions calculate the closed-system traces; no prediction changes
# the measured population or AGP estimate.

# %% tags=["theory_settings"]
KERR_KHZ = -30.0  # Signed physical Kerr for the independent theory comparison.

# %% tags=["walk_theory"]
settings = walk_expt.cfg.expt
timing = walk_expt.recorded_derived_params()
storages = list(settings.swap_stors)
couplings_mhz = np.array([timing["couplings_MHz"][storage - 1] for storage in storages])
if settings.get("zero_floquet_gain", False):
    couplings_mhz[:] = 0.0
model = fixed_n_hamiltonian(
    photon_number=sum(settings.occupations), mode_count=1 + len(storages),
    detunings=settings.detunings, couplings_MHz=couplings_mhz,
    physical_kerr_MHz=KERR_KHZ / 1000,
)
cycles = np.asarray(walk_expt.data["xpts"], dtype=float)
theory_cycles = np.linspace(cycles.min(), cycles.max(), max(1000, len(cycles)))
time_us = theory_cycles * timing["floquet_cycle_us"]
walk_fig, walk_axes = walk_expt.display()
for step, mode in enumerate(walk_expt.data["ro_stors"]):
    mode_index = [0, *storages].index(mode)
    theory_mean_number = fock_number_traces(
        model, [settings.occupations], mode_index, time_us)[0]
    walk_axes.plot(theory_cycles, theory_mean_number,
                   color=walk_axes.lines[step].get_color(), alpha=0.45,
                   linewidth=3, label=f"Theory, Mode {mode}")
walk_axes.legend(ncol=2)
walk_fig.tight_layout()
display(walk_fig)

# %% [markdown]
# ## 4. Central-boson population JOBs, correlations and metric
#
# Every Fock JOB uses the same ordered realization table, with one shot per row.
# Readout labels retain (time, realization, mode); rejected shots retain their
# original realization IDs. All directions share one union of initial states.
# For N=3 and five modes, all five directions require 35 JOBs per W.
#
# The original normalized patterns are saved even at W=0. The K/W metric applies
# each realization's pattern weights before averaging and uses one common kernel.
# Its physical units and the separately scaled AGP values are stored explicitly.

# %% tags=["agp_settings"]
BASE_FILE_NAME = 'agp_sequential_disorder_ensemble' + ('_herald' if HERALD else '')

TOTAL_PHOTONS = 3
AGP_DIRECTIONS = [0, *STORAGE_MODES]
DISORDER_RMS_KHZ = [0, 10, 20, 30]
NUMBER_OF_REALIZATIONS = 500  # One hardware shot per realization, time and Fock state.
DISORDER_SEED = 20261116
AGP_CYCLES = list(range(0, 800, 4))
AGP_BATCH_SIZE = 1
AGP_MU_KHZ = None  # None: g L/D; set a number or list of cutoffs in cyclic kHz.
BOOTSTRAP_SAMPLES = 300  # Resampling for error bars; no additional measurements.
SCAN_NAME = f"{BASE_FILE_NAME}_seed{DISORDER_SEED}"  # Use a fresh name for each run.

# %% tags=["disorder_patterns"]
# Match occupation_chaos_mechanism: center each Gaussian storage vector and
# normalize its RMS to one. The hub is excluded. Reuse these rows everywhere.
random_generator = np.random.default_rng(DISORDER_SEED)
disorder_patterns = random_generator.normal(
    size=(NUMBER_OF_REALIZATIONS, len(STORAGE_MODES)))
disorder_patterns -= disorder_patterns.mean(axis=1, keepdims=True)
disorder_patterns /= np.sqrt((disorder_patterns**2).mean(axis=1, keepdims=True))

# %% tags=["agp_plan"]
scan = CentralBosonAGPSequentialEnsembleDisorderScan.from_patterns(
    disorder_khz=DISORDER_RMS_KHZ, patterns=disorder_patterns,
    photon_sector_number=TOTAL_PHOTONS, direction=AGP_DIRECTIONS,
    swap_stors=STORAGE_MODES, cycles=AGP_CYCLES,
    calibration=CONFUSION_MATRIX_FNAME, disorder_seed=DISORDER_SEED,
    notes="Centered, unit-RMS Gaussian storage patterns; heralded sequential readout; one shot per realization, time and Fock state.",
)
jobs_per_strength = len(scan.children[0].entries)
print(f"Readout order: {RO_STORS}")
print(f"Initial Fock states / JOBs per W: {jobs_per_strength}")
print(f"Total population JOBs: {jobs_per_strength * len(scan.children)}")
print(f"Time points per JOB: {len(AGP_CYCLES)}; shots per point: {NUMBER_OF_REALIZATIONS}")

# %% tags=["agp_acquire"]
manifest_path = scan.acquire(
    ensemble_runner, name=SCAN_NAME, batch_size=AGP_BATCH_SIZE,
    analyze_kwargs=dict(mu=AGP_MU_KHZ, bootstrap_samples=BOOTSTRAP_SAMPLES,
                        kerr_mhz=KERR_KHZ / 1000),
    storage_reset=STORAGE_MODES, postprocess=False, show=False, log=True,
    go_kwargs=dict(analyze=True, save=True),
)
print(manifest_path)

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
# Change mu and reanalyze saved data if needed.
REANALYZE = False
if REANALYZE:
    scan.analyze(mu=AGP_MU_KHZ, bootstrap_samples=BOOTSTRAP_SAMPLES,
                 kerr_mhz=KERR_KHZ / 1000)

# %% tags=["agp_display"]
for part in scan.children:
    display(part.display(raw=False))

# %% tags=["agp_integral_display"]
display(scan.display()[0])

# %% [markdown]
# ### K/W metric
#
# This plots the measured metric components along the fixed-K disorder scan.
# A two-dimensional flow field additionally needs measurements at multiple Kerr
# settings. The saved eigenvectors diagonalize the ensemble-averaged metric.

# %% tags=["metric_display"]
display(scan.display_metric()[0])

# %% [markdown]
# ## 5. Save the assembled layers
#
# acquire() already saves each completed W and the scan. The manifest references
# the original population JOB HDF5s; raw readouts and realization tables stay there.

# %% tags=["assemble"]
print(manifest_path)

# %% tags=["disorder_display"]
display(scan.display_correlation(raw=False)[0])

# %% [markdown]
# ## 6. Reload and change the cutoff
#
# Reanalyze the original saved JOBs through their manifests. No new measurements
# are submitted. Supplying mu here recalculates all metric components together.

# %% tags=["reload"]
reloaded_scan = CentralBosonAGPSequentialEnsembleDisorderScan.from_manifest(manifest_path)
reloaded_scan.analyze(mu=[2, 5.915, 10], bootstrap_samples=BOOTSTRAP_SAMPLES,
                      kerr_mhz=KERR_KHZ / 1000)
display(reloaded_scan.display()[0])
display(reloaded_scan.display_metric()[0])
