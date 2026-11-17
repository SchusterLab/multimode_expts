# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # AGP from a disorder ensemble
# Each realization contributes one shot at each time and initial Fock state.
# The same ordered disorder table is reused in every Fock JOB. The RFSoC reads
# the table inside its repetition loop; Python only sweeps evolution time.

# %%
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.qsim.agp_ensemble import CentralBosonAGPEnsembleDisorderScan
from experiments.qsim.fock_population import FockPopulationExperiment
from experiments.qsim.fock_population_ensemble import FockPopulationEnsembleExperiment
from experiments.qsim.fock_prep_herald import FockPrepHeraldProgram, HeraldedConfusionMatrixExperiment
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS
from fitting import plot_style
from theory.qsim.central_boson import fixed_n_hamiltonian, fock_number_traces
from job_server import JobClient

# %% [markdown]
# ## 1. Station and acquisition settings
# Configuration IDs below match the current ordinary AGP notebook. Check them
# before submitting a new measurement. Constructing the runners submits no JOBs.

# %%
config_versions = dict(
    hardware_config="CFG-HW-20261113-00052",
    multiphoton_config="CFG-MP-20260121-00001",
    man1_storage_swap="CFG-M1-20261113-00028",
    floquet_storage_swap="CFG-FL-20261113-00028",
)
station = MultimodeStation(user="jonginn", project="AGP",
                          experiment_name="agp_disorder_ensemble",
                          log_measurements=True,
                          hardware_config=config_versions["hardware_config"],
                          multiphoton_config=config_versions["multiphoton_config"],
                          storage_man_file=config_versions["man1_storage_swap"],
                          floquet_file=config_versions["floquet_storage_swap"])
client = JobClient()

STORAGE_MODES = [1, 2, 4, 5]
TOTAL_PHOTONS = 3
AGP_DIRECTION = STORAGE_MODES[-1]  # 0: M1 Kerr; otherwise a storage ID.
DISORDER_RMS_KHZ = [0, 10, 20, 30]
NUMBER_OF_REALIZATIONS = 500
DISORDER_SEED = 20261116
FLOQUET_CYCLES = list(range(0, 800, 4))
MU_KHZ = None  # None: g L/D; a number or list specifies cutoffs in kHz.
HERALD = False
SCAN_NAME = f"n{TOTAL_PHOTONS}_direction{AGP_DIRECTION}_seed{DISORDER_SEED}"

common_defaults = AttrDict(dict(
    expts=1, reps=1, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=STORAGE_MODES, pre_relax_delay=100, relax_delay=200,
    postpulse=True, init_fock=False, init_stor=0, ro_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    swap_stors=STORAGE_MODES, update_phases=True,
))
common_defaults.update(ACTIVE_RESET_DEFAULTS)
common_defaults.update(FLOQUET_DEFAULTS)
common_defaults.scramble_sync_cycles = 2
population_defaults = AttrDict(dict(common_defaults, prepulse=False))
calibration_defaults = AttrDict(dict(common_defaults, prepulse=not HERALD))
if HERALD:
    population_defaults.prep_herald = "end"
    calibration_defaults.prep_herald = "end"

confusion_runner = CharacterizationRunner(
    station=station,
    ExptClass=HeraldedConfusionMatrixExperiment if HERALD else MultiParityConfusionMatrixExperiment,
    default_expt_cfg=calibration_defaults, job_client=client, use_queue=True, show=False,
)
ensemble_runner = CharacterizationRunner(
    station=station, ExptClass=FockPopulationEnsembleExperiment,
    default_expt_cfg=population_defaults, job_client=client, use_queue=True, show=False,
)
population_runner = CharacterizationRunner(
    station=station, ExptClass=FockPopulationExperiment,
    ExptProgram=FockPrepHeraldProgram if HERALD else None,
    default_expt_cfg=population_defaults, job_client=client, use_queue=True, show=False,
)

# %% [markdown]
# ## 2. Readout calibration
# Run this cell to measure a calibration, or set CALIBRATION_FILE to an existing
# matching calibration HDF5 instead.

# %% tags=["calibration"]
calibration = confusion_runner.execute(reps=6000, postprocess=False)
CALIBRATION_FILE = calibration.fname

# %% tags=["calibration_display"]
calibration.display()
print(CALIBRATION_FILE)
print("Confusion matrix condition number:",
      np.linalg.cond(calibration.data["multiparity_confusion_matrix"]))

# %% [markdown]
# ## 3. Verify the quantum walk before submitting the AGP batch
# Each mode uses a fresh preparation and a separate JOB, as in agp_measurement.
# Use the existing population runner with no detuning, as in agp_measurement.
# The same confusion calibration and herald setting are used for the walk
# and the subsequent AGP scan.
# For at most three photons, number and number modulo four agree.

# %% tags=["walk_settings"]
WALK_OCCUPATIONS = [TOTAL_PHOTONS] + [0] * len(STORAGE_MODES)
WALK_DETUNINGS_MHZ = [0.0] * len(STORAGE_MODES)
WALK_CYCLES = list(range(0, 100, 4))
WALK_REPS = 1000
WALK_KERR_KHZ = -30.0  # Independent physical Kerr; match the current setup.

# %% tags=["walk_acquire"]
walk_expts = []
for readout_mode in [0] + STORAGE_MODES:
    walk_expt = population_runner.execute(
        reps=WALK_REPS, occupations=WALK_OCCUPATIONS, ro_stor=readout_mode,
        swap_stors=STORAGE_MODES, storage_reset=STORAGE_MODES,
        floquet_cycles=WALK_CYCLES, detunings=WALK_DETUNINGS_MHZ,
        multiparity_confusion_matrix=CALIBRATION_FILE,
        postprocess=False, show=False, log=True,
    )
    walk_expts.append(walk_expt)
    print(walk_expt.fname)

# %% tags=["walk_display"]
with plot_style.context():
    walk_figure, walk_axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7),
                                         constrained_layout=True)
    for walk_expt in walk_expts:
        data = walk_expt.data
        mode = walk_expt.cfg.expt.ro_stor
        name = "M1" if mode == 0 else f"S{mode}"
        color = plot_style.MODE_COLORS[name]
        walk_axes[0].plot(data["xpts"], data["nmod4_mean"], ".-", color=color, label=name)
        walk_axes[1].errorbar(data["xpts"], data["nmod4_mean_corrected"],
                              yerr=data["n_err"], fmt=".-", color=color, label=name)
    for axis, title in zip(walk_axes, ("Raw", "Confusion corrected")):
        axis.set(title=title, xlabel="Floquet cycles", ylabel="Mean n mod 4")
        axis.legend()
display(walk_figure)

# %% tags=["walk_theory"]
for line in list(walk_axes[1].lines):
    if line.get_label().startswith("Theory,"):
        line.remove()
for walk_expt in walk_expts:
    settings = walk_expt.cfg.expt
    storages = list(settings.swap_stors)
    mode = settings.ro_stor
    name = "M1" if mode == 0 else f"S{mode}"
    timing = walk_expt.recorded_derived_params()
    couplings_mhz = np.asarray([timing["couplings_MHz"][storage - 1] for storage in storages])
    if settings.get("zero_floquet_gain", False):
        couplings_mhz[:] = 0.
    model = fixed_n_hamiltonian(
        photon_number=sum(settings.occupations), mode_count=1 + len(storages),
        detunings=settings.detunings, couplings_MHz=couplings_mhz,
        physical_kerr_MHz=WALK_KERR_KHZ / 1000)
    cycles = np.asarray(walk_expt.data["xpts"])
    theory_cycles = np.linspace(cycles.min(), cycles.max(), 1000)
    theory_mean = fock_number_traces(
        model, [settings.occupations], [0, *storages].index(mode),
        theory_cycles * timing["floquet_cycle_us"])[0]
    walk_axes[1].plot(theory_cycles, theory_mean, "--", color=plot_style.MODE_COLORS[name],
                      label=f"Theory, {name}")
walk_axes[1].legend(ncol=2)
display(walk_figure)

# %% [markdown]
# Inspect the raw and corrected walks against the theory overlay above before
# running the AGP acquisition cell below. Theory does not correct measured data.

# %% [markdown]
# ## 4. Generate the realization table once
# Match occupation_chaos_mechanism: draw Gaussian storage vectors, subtract
# each row's mean, then normalize each row to unit RMS. The hub is excluded.
# Scaling a row by W gives storage detunings with zero mean and RMS W.
# Every Fock state, evolution time and W uses these same ordered patterns.

# %%
random_generator = np.random.default_rng(DISORDER_SEED)
disorder_patterns = random_generator.normal(
    size=(NUMBER_OF_REALIZATIONS, len(STORAGE_MODES)))
disorder_patterns -= disorder_patterns.mean(axis=1, keepdims=True)
disorder_patterns /= np.sqrt((disorder_patterns**2).mean(axis=1, keepdims=True))

scan = CentralBosonAGPEnsembleDisorderScan.from_patterns(
    disorder_khz=DISORDER_RMS_KHZ, patterns=disorder_patterns,
    photon_sector_number=TOTAL_PHOTONS, direction=AGP_DIRECTION,
    swap_stors=STORAGE_MODES, cycles=FLOQUET_CYCLES,
    calibration=CALIBRATION_FILE, disorder_seed=DISORDER_SEED,
    notes="Gaussian draws centered and normalized per storage row; one shot per realization, time and Fock state.",
)

# %% [markdown]
# ## 5. Acquire and save
# batch_size limits queued Fock JOBs, not the number of disorder realizations.
# reps is set from the table length; all Fock JOBs receive the same full table.
# Each completed W is saved before proceeding. Raw JOB HDF5s retain all IQ,
# decoded labels, herald masks, requested detunings and applied frequency words.

# %% tags=["agp_acquire"]
manifest_path = scan.acquire(
    ensemble_runner, name=SCAN_NAME, batch_size=1,
    analyze_kwargs=dict(mu=MU_KHZ, bootstrap_samples=300),
)
display(scan.display()[0])

# %% [markdown]
# ## 6. Reload and change the cutoff
# Reanalysis uses saved decoded shots and the exact realization tables. No
# theory-derived subtraction, fitted background or unmeasured tail is added.

# %%
reloaded_scan = CentralBosonAGPEnsembleDisorderScan.from_manifest(manifest_path)
reloaded_scan.analyze(mu=[2, 5.915, 10])
display(reloaded_scan.display()[0])
