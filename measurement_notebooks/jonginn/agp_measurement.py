# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Number correlations and regularized AGP at fixed Kerr
#
# Run this file as a Jupytext notebook. The measurement order is:
# multiparity calibration, Floquet population dynamics, a five-input Kerr
# pilot, storage-transfer checks, then the five-probe disorder scan.
# Kerr stays at the already calibrated operating point, |K| = 44 kHz.
# Disorder changes storage detunings only; the hub detuning stays zero.
#
# Each shot gives a number modulo four. For a state supported on 0, 1, 2, 3
# photons, this is the photon number. Average `number` for a detuning probe,
# or `number * (number - 1) / 2` for the Kerr probe. No eigenstate preparation
# or vacuum-reference Ramsey encoder is required.
#
# With V diagonal in the input Fock basis, the correlation is
# `C_V(t) = sum_b V(b) * <V(t)>_b / D`. Here N = 3 and D = 35.
# Matrix Pencil fits frequencies and weights of this correlation. Applying
# the regularized AGP spectral weight gives an inferred closed-system norm;
# removing fitted decay is a model assumption, not a measured correction.
# Finite-time integration of the measured correlation is saved separately.
#
# Station and worker setup follow Jonginn's `Autocalibrate.ipynb` and
# `qsim_experiments_highkerr_untracked_refactored.ipynb`.
# Analysis reads raw HDF5 files, so it can run again without a station or job DB.
#
# To install manually on the measurement PC, copy these three files to the same
# relative paths under `C:\python\multimode_expts`:
# - `experiments/qsim/floquet_dark_mode_readout.py` (existing file)
# - `fitting/qsim/agp.py` (new analysis file)
# - `measurement_notebooks/jonginn/agp_measurement.py` (this notebook)
# Copy between jobs and rerun this notebook's imports and station/runner setup.
# The setup refreshes these modules in an existing kernel. The current worker
# also reloads experiment/fitting code before each job; no worker code is changed.
# Preserve any separate remote edits when replacing the existing Program file.
# The former standalone measurement module `fock_population.py` is no longer used.

# %%
from copy import deepcopy
from datetime import datetime
from importlib import import_module, invalidate_caches, reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Pick up a manually copied analysis file in an already open notebook kernel.
invalidate_caches()
reload(import_module("fitting.qsim.agp"))
from fitting.qsim.agp import (
    build_correlation,
    calibrate_readout,
    discover_population_jobs,
    finite_time_agp,
    fit_correlation_mpm,
    fock_basis,
    load_analysis_h5,
    load_population_job,
    population_moments,
    probe_weights,
    regularized_agp_from_fit,
    save_analysis_h5,
)

# %% [markdown]
# ## 1. Session settings
#
# Measurements always go through the job server and worker, as in Autocalibrate.
# Choose the sections to acquire explicitly; an empty set submits no measurements.
# Set your username and config versions in the station cell in section 2.
#
# For analysis of saved HDF5 files, leave ACQUIRE empty and set RAW_DATA_DIR
# or SOURCE_FILES below. Skip the three cells marked "Measurement setup" in
# section 2, then run the raw-data helpers and the later analysis cells.

# %%
ACQUIRE = set()                       # e.g. {"readout", "walk", "kerr_pilot"}
CAMPAIGN_ID = "agp_k44_first_scan"     # keep this unchanged when reloading completed sections
RAW_DATA_DIR = None                   # folder containing this campaign's raw HDF5
SOURCE_FILES = []                     # alternatively list specific raw HDF5 files
OUTPUT_DIR = Path("agp_processed")

STORAGE_MODES = [1, 2, 3, 4]          # physical storage IDs; occupation order follows this
TOTAL_PHOTONS = 3
MODE_COUNT = 1 + len(STORAGE_MODES)
OCCUPATIONS = fock_basis(TOTAL_PHOTONS, MODE_COUNT)
DIMENSION = len(OCCUPATIONS)

DISORDER_PATTERN = np.array([-0.774342, -1.061424, 1.460449, 0.375317])
DISORDER_PATTERN = DISORDER_PATTERN - np.mean(DISORDER_PATTERN)
DISORDER_PATTERN = DISORDER_PATTERN / np.sqrt(np.mean(DISORDER_PATTERN ** 2))
DISORDER_RMS_KHZ = [0.0, 40.0, 100.0]
PILOT_DISORDER_KHZ = 40.0
TARGET_DURATION_US = 200.0
TARGET_SAMPLE_INTERVAL_US = 0.5
FLOQUET_CYCLES = [0]                 # resolved from compiled timing below
REPS = 1000                         # pilot setting; increase after inspecting errors
CALIBRATION_REPS = 6000
USE_MULTIPHOTON_PREP_SWAP = True
CUTOFF_KHZ = None                     # None: mean compiled g * MODE_COUNT / DIMENSION
MPM_MAX_MODES = 16
NUMERICAL_GAP_RATIO = None            # optional averaged numerical r, in DISORDER_RMS_KHZ order

# One fixed terminal pulse sequence for every unknown occupation 0..3.
# These are candidate routes to test, not an assertion of readout fidelity.
READOUT_SWAP_PULSES = {
    storage: [["storage", f"M1-S{storage}", "pi", 0.0]]
    for storage in STORAGE_MODES
}
STORAGE_ROUTE = "dump_then_swap"      # alternative: "direct_swap"
VALIDATED_STORAGE_MODES = []          # fill after reviewing the transfer checks

PROBES = [("Kerr", "kerr", 0)]
PROBES += [(f"Storage {storage}", "number", index + 1)
           for index, storage in enumerate(STORAGE_MODES)]

print(f"Fixed N = {TOTAL_PHOTONS}, modes = {MODE_COUNT}, dimension = {DIMENSION}")
for name, probe, mode in PROBES:
    weights = probe_weights(OCCUPATIONS, probe=probe, mode=mode)
    print(f"{name}: {np.count_nonzero(weights)} input/readout traces per disorder")

# %% [markdown]
# ## 2. Station, worker, and measurement defaults
#
# Edit `user` and `config_dict` together in the next cell. A version ID pins
# a config, as in Autocalibrate. None selects the config database's registered
# main version (not the Git branch). Select the versions calibrated at 44 kHz.
# Rerunning this cell creates a fresh station from these settings; it never
# reuses a station left in the kernel. The selected configs are saved with jobs.
# The station reads device configuration; acquisition is submitted to the worker.
#
# The Program prepares each occupied storage through man, then prepares
# man last. Each added photon uses the existing selective GE, EF, and sideband
# pulses. The broadband GE pulses that shelve the vacuum Ramsey reference are
# absent. The ordinary MBR Ramsey encoder is unchanged.
#
# Initial reset empties every participating storage before preparation.
# After evolution, a man readout needs no swap. A storage readout uses the
# explicitly selected terminal sequence, optionally after dumping only man.
# A full active reset must never occur between evolution and readout.

# %% tags=["measurement-setup"]
# Measurement setup: worker client and a fresh station.
from slab import AttrDict
from experiments import CharacterizationRunner, MultimodeStation
from job_server import JobClient

reload(import_module("experiments.qsim.floquet_dark_mode_readout"))
from experiments.qsim.floquet_dark_mode_readout import FockPopulationProgram
from experiments.qsim.qsim_base import QsimExperiment
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS,
)

user = "jonginn"
config_dict = {
    "hardware_config": None,        # e.g. "CFG-HW-..."; None = registered main
    "multiphoton_config": None,     # e.g. "CFG-MP-..."; None = registered main
    "man1_storage_swap": None,      # e.g. "CFG-M1-..."; None = registered main
    "floquet_storage_swap": None,   # e.g. "CFG-FL-..."; None = registered main
}

client = JobClient()
health = client.health_check()
print(f"Server status: {health['status']}")
client.print_queue()

station = MultimodeStation(
    user=user,
    experiment_name=CAMPAIGN_ID,
    project="AGP",
    log_measurements=False,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)
RAW_DATA_DIR = Path(station.data_path)

# %% tags=["measurement-setup"]
# Measurement setup: defaults, preprocessor, and worker runner.
population_defaults = AttrDict(dict(
    expts=1, reps=REPS, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=STORAGE_MODES, pre_relax_delay=100, relax_delay=200,
    prepulse=False, postpulse=False, init_fock=False, init_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    occupations=[3, 0, 0, 0, 0], swap_stors=STORAGE_MODES,
    use_multiphoton_swap=USE_MULTIPHOTON_PREP_SWAP,
    storage_pulse_wait_us=0.2, readout_route="hub", ro_stor=0,
    readout_swap_pulses=[], readout_dump_mode=2,
    update_phases=True, detunings=[0.0] * len(STORAGE_MODES),
    floquet_cycles=FLOQUET_CYCLES, swept_params=["floquet_cycle"],
    agp_campaign=CAMPAIGN_ID, agp_total_photons=TOTAL_PHOTONS,
    agp_kerr_magnitude_khz=44.0, agp_disorder_pattern=DISORDER_PATTERN.tolist(),
    agp_mock=False,
))
population_defaults.update(ACTIVE_RESET_DEFAULTS)
population_defaults.update(FLOQUET_DEFAULTS)


def population_preprocessor(station, defaults, **overrides):
    """Keep every physical input in the raw job's saved experiment config."""
    settings = deepcopy(defaults)
    settings.update(overrides)
    return settings


population_runner = CharacterizationRunner(
    station=station,
    ExptClass=QsimExperiment,
    ExptProgram=FockPopulationProgram,
    default_expt_cfg=population_defaults,
    preprocessor=population_preprocessor,
    job_client=client,
    use_queue=True,
    show=False,
)

# %% tags=["measurement-setup"]
# Measurement setup: compile a short preview and choose the time grid.
# Compilation emits instructions locally; it does not send them to the FPGA.
from experiments.qsim.utils import ensure_list_in_cfg

preview_config = AttrDict(deepcopy(station.hardware_cfg))
preview_config.expt = deepcopy(population_defaults)
preview_config.expt.floquet_cycle = 2
ensure_list_in_cfg(preview_config)
preview_program = FockPopulationProgram(station.soccfg, preview_config)
print("Preparation gates:")
for pulse in preview_program.preparation_pulses:
    print(pulse)
print("Readout gates:", preview_program.readout_pulses)
cycle_us = preview_program.calculate_floquet_cycle_us()
cycle_step = max(1, round(TARGET_SAMPLE_INTERVAL_US / cycle_us))
last_cycle = int(TARGET_DURATION_US / cycle_us)
FLOQUET_CYCLES = np.arange(0, last_cycle + 1, cycle_step).tolist()
population_defaults.floquet_cycles = FLOQUET_CYCLES
population_runner.default_expt_cfg.floquet_cycles = FLOQUET_CYCLES
print("Compiled Floquet cycle (us):", cycle_us)
print("Recorded time grid (us):", cycle_us * np.array(FLOQUET_CYCLES)[[0, -1]])
print("Sample interval (us):", cycle_us * cycle_step)
trace_floor_s = REPS * len(FLOQUET_CYCLES) * (200 + TARGET_DURATION_US / 2) / 1e6
print(f"Per-trace duration >= {trace_floor_s:.0f} s, excluding preparation/reset/readout overhead.")

# %% [markdown]
# ### Raw-data helpers
#
# Run this cell for both acquisition and saved-data analysis. Defining these
# functions does not connect to hardware or submit a job. `campaign_jobs`
# reads HDF5 files without a station. `acquire_population` submits to the worker.

# %%
def readout_settings(mode, route=None):
    """mode is the occupation-column index; ro_stor is a physical storage ID."""
    if mode == 0:
        return dict(readout_route="hub", ro_stor=0, readout_swap_pulses=[])
    storage = STORAGE_MODES[mode - 1]
    return dict(
        readout_route=route or STORAGE_ROUTE, ro_stor=storage,
        readout_swap_pulses=deepcopy(READOUT_SWAP_PULSES[storage]),
    )


def acquire_population(occupation, stage, disorder_khz=0.0, mode=0, **overrides):
    """One raw HDF5 contains one input state, one measured mode, and all times."""
    settings = readout_settings(mode, overrides.pop("readout_route", None))
    settings.update(
        occupations=list(occupation), agp_stage=stage, agp_mode_index=mode,
        agp_disorder_rms_khz=float(disorder_khz),
        detunings=(0.001 * disorder_khz * DISORDER_PATTERN).tolist(),
    )
    settings.update(overrides)
    experiment = population_runner.execute(
        use_queue=True, **settings, postprocess=False, show=False, log=False,
    )
    filename = str(Path(experiment.fname).resolve())
    SOURCE_FILES.append(filename)
    print(filename)
    return filename


def campaign_jobs(stage):
    """Reload raw files; no dependence on in-memory Experiment objects."""
    filenames = list(SOURCE_FILES)
    if RAW_DATA_DIR is not None:
        filenames += discover_population_jobs(
            RAW_DATA_DIR, agp_campaign=CAMPAIGN_ID, agp_stage=stage,
        )
    jobs = []
    for filename in sorted(set(str(path) for path in filenames)):
        job = load_population_job(filename)
        settings = job["config"]["expt"]
        if settings.get("agp_campaign") == CAMPAIGN_ID and settings.get("agp_stage") == stage:
            jobs.append(job)
    return jobs

# %% [markdown]
# ## 3. Multiparity response for prepared hub numbers 0, 1, 2, 3
#
# Set `ACQUIRE = {"readout"}` and run this section first. The two final readout
# lanes use the source notebook's `cond_sec_phase=-90` convention.
# The response matrix has rows = prepared number and columns = measured number:
# `p_measured = confusion.T @ p_true`.
# Its finite calibration statistics enter the later correlation uncertainties.
# It also includes preparation errors; a four-state calibration alone does not
# separate preparation error from readout error.

# %%
if "readout" in ACQUIRE:
    for number in range(4):
        acquire_population(
            [number] + [0] * len(STORAGE_MODES), "readout",
            floquet_cycles=[0], agp_prepared_number=number, reps=CALIBRATION_REPS,
        )

# %%
readout_jobs = campaign_jobs("readout")
calibration = None
if readout_jobs:
    prepared_numbers = [job["config"]["expt"]["agp_prepared_number"] for job in readout_jobs]
    readout_counts = np.zeros((4, 4))
    for job, number in zip(readout_jobs, prepared_numbers):
        readout_counts[number] += job["counts"][0]
    readout_response = readout_counts / np.maximum(readout_counts.sum(axis=1, keepdims=True), 1)
    if not any(job["config"]["expt"].get("agp_mock") for job in readout_jobs):
        calibration = calibrate_readout(readout_jobs, prepared_numbers)
        print("Readout matrix condition number:", calibration["condition_number"])
    else:
        print("Mock readout response: display only; no calibration inversion.")
    fig, axis = plt.subplots(figsize=(4, 3))
    plot = axis.imshow(readout_response, vmin=0, vmax=1)
    axis.set(xlabel="Measured number", ylabel="Prepared number", xticks=range(4), yticks=range(4))
    fig.colorbar(plot, ax=axis, label="Probability")
    plt.show()

# %% [markdown]
# ## 4. Floquet population dynamics (mod-4 random walk)
#
# Start with the hub readout. Each input sees the same deterministic Floquet
# sequence. Here "random walk" means coherent population mixing, not a fresh
# random pulse sequence in each shot. Storage readouts can be added after the
# transfer section establishes their response.

# %%
if "walk" in ACQUIRE:
    walk_occupation = [1] + [0] * len(STORAGE_MODES)
    walk_modes = [0] + [STORAGE_MODES.index(storage) + 1 for storage in VALIDATED_STORAGE_MODES]
    for mode in walk_modes:
        acquire_population(walk_occupation, "walk", PILOT_DISORDER_KHZ, mode=mode)

# %%
walk_jobs = campaign_jobs("walk")
for job in walk_jobs:
    trace = population_moments(job)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    image = axes[0].pcolormesh(job["time_us"], np.arange(4), job["probabilities"].T, shading="nearest")
    axes[0].set(xlabel="Evolution time (us)", ylabel="Measured number modulo 4")
    fig.colorbar(image, ax=axes[0], label="Raw probability")
    axes[1].plot(job["time_us"], trace["number"]["mean"], label="Raw mean number")
    axes[1].set(xlabel="Evolution time (us)", ylabel="Mean number")
    axes[1].legend()
    fig.suptitle(f"Readout mode: {job['config']['expt']['ro_stor']}")
    plt.show()

# %% [markdown]
# ## 5. Five-input Kerr pilot
#
# Only input states with at least two hub photons have nonzero Kerr weight.
# For N = 3, these are (3,0,0,0,0) and the four states with two hub photons and
# one storage photon. Read the hub directly; no terminal storage swap is needed.
#
# Write q_b(t) = P_b(2,t) + 3 P_b(3,t). Then
# `C_K(t) = (3*q_30000(t) + sum_s q_2,1s(t)) / 35` and `C_K(0) = 13/35`.
# The denominator stays 35 even though only five input states contribute.

# %%
kerr_weights = probe_weights(OCCUPATIONS, probe="kerr", mode=0)
if "kerr_pilot" in ACQUIRE:
    for occupation, weight in zip(OCCUPATIONS, kerr_weights):
        if weight > 0:
            acquire_population(occupation, "kerr_pilot", PILOT_DISORDER_KHZ, agp_probe="Kerr")

# %% [markdown]
# ## 6. Test the storage readout channel
#
# Compare the same fixed transfer sequence with and without a selective hub dump.
# Test storage occupations 0..3 with an empty hub, then occupied-hub cases at
# total occupation <= 3. Compare full outcome probabilities and mean number with
# the prepared storage occupation. These population checks are necessary; if
# phases/coherences change the transfer, add an evolved-state or phase-sensitive
# check before treating this channel as an ideal SWAP.
#
# The candidate `M1-Sj` pulse below is explicit. A known-N preparation pulse
# `M1-Sj@N3` is not automatically a readout pulse for an unknown final number.
# A selective hub dump can preserve the storage state in the ideal channel,
# but adds its own duration and experimental errors. No outcome is postselected
# to force total photon number to remain three.

# %%
if "transfer" in ACQUIRE:
    transfer_inputs = [(0, number) for number in range(4)] + [(1, 1), (2, 1), (1, 2)]
    for mode, storage in enumerate(STORAGE_MODES, start=1):
        for route in ["direct_swap", "dump_then_swap"]:
            for hub_number, storage_number in transfer_inputs:
                occupation = [0] * MODE_COUNT
                occupation[0], occupation[mode] = hub_number, storage_number
                acquire_population(
                    occupation, "transfer", mode=mode, readout_route=route,
                    floquet_cycles=[0], agp_expected_number=storage_number,
                )

# %%
transfer_jobs = campaign_jobs("transfer")
if transfer_jobs:
    fig, axes = plt.subplots(1, len(STORAGE_MODES), figsize=(4 * len(STORAGE_MODES), 3), squeeze=False)
    for job in transfer_jobs:
        settings = job["config"]["expt"]
        mode = settings["agp_mode_index"]
        moments = population_moments(job, calibration)
        axis = axes[0, mode - 1]
        marker = "o" if settings["readout_route"] == "direct_swap" else "x"
        axis.errorbar(settings["agp_expected_number"], moments["number"]["mean"][0],
                      yerr=moments["number"]["standard_error"][0], marker=marker)
        axis.set(title=f"Storage {STORAGE_MODES[mode - 1]}", xlabel="Prepared storage number", ylabel="Readout mean")
    for axis in axes[0]:
        axis.plot([0, 3], [0, 3], color="black", linestyle="--")
    fig.suptitle("Circle: direct swap; cross: dump then swap; varying initial hub occupation")
    plt.show()

# %% [markdown]
# ## 7. Disorder scan: four storage probes and Kerr
#
# Choose the validated route and fill `VALIDATED_STORAGE_MODES` in the settings.
# Select `"scan"` to acquire those storage probes plus Kerr at every W.
# With all four storages, this is 65 input/readout traces per W, using 35 unique
# Fock inputs. A Kerr-only scan requires five traces per W and can start earlier.
# One normalized disorder pattern is held fixed; no experimental disorder
# average is required by the definition. A numerical disorder-averaged gap ratio
# is a separate reference, not a measured quantity or a guaranteed correlate.

# %%
if "scan" in ACQUIRE:
    for disorder_khz in DISORDER_RMS_KHZ:
        for name, probe, mode in PROBES:
            if mode > 0 and STORAGE_MODES[mode - 1] not in VALIDATED_STORAGE_MODES:
                continue
            weights = probe_weights(OCCUPATIONS, probe=probe, mode=mode)
            for occupation, weight in zip(OCCUPATIONS, weights):
                if weight > 0:
                    acquire_population(
                        occupation, "scan", disorder_khz, mode=mode, agp_probe=name,
                    )

# %% [markdown]
# ## 8. Reconstruct correlations from saved data
#
# For saved data, set a raw-data folder or explicit source filenames and leave
# ACQUIRE empty. Run imports, session settings, raw-data helpers, and the analysis
# cells; skip the three Measurement setup cells. Repeated inputs are reported as
# duplicates rather than silently selecting one. Use `SOURCE_FILES` and set
# `RAW_DATA_DIR=None` to select the intended data after a retry.
#
# Readout-corrected probabilities may be negative with finite statistics.
# Inspect them and the calibration condition number; do not silently clip them.
# Uncertainty includes shot noise and the common finite calibration sample.
# It does not include drift, unknown preparation error, or transfer-model error.

# %%
def correlate_jobs(jobs, probe, mode):
    traces = [population_moments(job, calibration) for job in jobs]
    inputs = [job["config"]["expt"]["occupations"] for job in jobs]
    return build_correlation(
        traces, inputs, total_photons=TOTAL_PHOTONS, mode_count=MODE_COUNT,
        probe=probe, mode=mode,
    )


correlations = {}
pilot_jobs = campaign_jobs("kerr_pilot")
if pilot_jobs and calibration is not None and not any(job["config"]["expt"].get("agp_mock") for job in pilot_jobs):
    correlations["pilot_Kerr"] = correlate_jobs(pilot_jobs, "kerr", 0)

scan_jobs = campaign_jobs("scan")
for disorder_khz in DISORDER_RMS_KHZ:
    for name, probe, mode in PROBES:
        jobs = [job for job in scan_jobs
                if job["config"]["expt"].get("agp_probe") == name
                and np.isclose(job["config"]["expt"]["agp_disorder_rms_khz"], disorder_khz)]
        if jobs and calibration is not None and not any(job["config"]["expt"].get("agp_mock") for job in jobs):
            key = f"W{disorder_khz:g}_{name.replace(' ', '_')}"
            correlations[key] = correlate_jobs(jobs, probe, mode)

for name, result in correlations.items():
    plt.errorbar(result["time_us"], result["correlation"], yerr=result["standard_error"], label=name)
    print(name, "measured C(0):", result["correlation"][0], "ideal C(0):", result["expected_t0"])
if correlations:
    plt.xlabel("Evolution time (us)")
    plt.ylabel("Calibrated correlation")
    plt.legend()
    plt.show()

# %% [markdown]
# ## 9. Spectral weights, cutoff, and the AGP estimate
#
# Frequencies from Matrix Pencil are cyclic MHz; time is in microseconds.
# For angular frequency omega = 2*pi*f and mu = 2*pi*cutoff_mhz, each signed
# spectral line contributes `weight * omega**2 / (omega**2 + mu**2)**2`.
# Keep both positive and negative frequencies, and remove the zero-frequency
# contribution through this filter. Do not divide a cosine's weight twice.
# The unnormalized norm has units us^2 for these dimensionless probe operators.
#
# `cutoff = mean(g) * L / D` is an analysis convention, not a coherence rate.
# L = 5 and D = 35 here. The default uses the compiled coupling metadata of the
# actual pulses. A smaller cutoff emphasizes longer times and unresolved gaps.
# Inspect MPM residuals, rank dependence, negative/complex spectral weights,
# and variation with fit window before interpreting the inferred norm.
#
# Floquet measurements also require a numerical check that the pulse train
# represents the intended static Hamiltonian in the chosen range. The measured
# correlation is well-defined even when that approximation fails.

# %%
def compiled_reference_mhz(job):
    params = job["metadata"]["derived_params"]
    couplings = np.asarray(params["couplings_MHz"], dtype=float)
    return float(np.mean([couplings[storage - 1] for storage in STORAGE_MODES]))


agp_results = {}
fit_results = {}
if correlations:
    reference_job = (pilot_jobs + scan_jobs)[0]
    reference_mhz = compiled_reference_mhz(reference_job)
    cutoff_mhz = 0.001 * CUTOFF_KHZ if CUTOFF_KHZ is not None else reference_mhz * MODE_COUNT / DIMENSION
    print(f"g reference = {1000 * reference_mhz:.4g} kHz; cutoff = {1000 * cutoff_mhz:.4g} kHz")
    for name, result in correlations.items():
        direct = finite_time_agp(result["time_us"], result["correlation"], cutoff_mhz,
                                 covariance=result["covariance"])
        agp_results[name] = dict(finite_time=direct, cutoff_mhz=cutoff_mhz)
        try:
            fit = fit_correlation_mpm(result["time_us"], result["correlation"], max_modes=MPM_MAX_MODES)
        except (ValueError, np.linalg.LinAlgError) as error:
            fit_results[name] = dict(error=str(error))
            print(name, "MPM failed:", error)
            continue
        inferred = regularized_agp_from_fit(fit, cutoff_mhz, reference_mhz=reference_mhz)
        fit_results[name] = fit
        agp_results[name]["inferred"] = inferred
        plt.plot(result["time_us"], result["correlation"], ".", label="Measured correlation")
        plt.plot(result["time_us"], np.real(fit["fitted"][0]), label="Damped MPM fit")
        plt.title(name)
        plt.xlabel("Evolution time (us)")
        plt.legend()
        plt.show()
        print(name, inferred)
        print("MPM diagnostics:", fit["diagnostics"])

# %% [markdown]
# ## 10. Five disorder curves, their mean, and fit stability
#
# All five curves use the same cutoff and the same angular coupling to make
# the norms dimensionless. Their arithmetic mean uses exactly these five probe
# operators; it is not an invariant under changing parameter coordinates.
# The mean is shown only where all five probes have been acquired.
# `NUMERICAL_GAP_RATIO`, if supplied, is a numerical disorder-averaged reference.
# It is never extracted from the population measurement.

# %%
agp_by_disorder = np.full((len(PROBES), len(DISORDER_RMS_KHZ)), np.nan)
for probe_index, (name, probe, mode) in enumerate(PROBES):
    for disorder_index, disorder_khz in enumerate(DISORDER_RMS_KHZ):
        key = f"W{disorder_khz:g}_{name.replace(' ', '_')}"
        if key in agp_results and "inferred" in agp_results[key]:
            agp_by_disorder[probe_index, disorder_index] = agp_results[key]["inferred"]["values"][0]

if np.any(np.isfinite(agp_by_disorder)):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for index, (name, probe, mode) in enumerate(PROBES):
        axes.flat[index].plot(DISORDER_RMS_KHZ, agp_by_disorder[index], "o-")
        axes.flat[index].set(title=name, xlabel="Disorder RMS (kHz)", ylabel="Inferred g^2 ||A||^2")
    five_mean = np.mean(agp_by_disorder, axis=0)
    axes.flat[5].plot(DISORDER_RMS_KHZ, five_mean, "o-", color="black")
    axes.flat[5].set(title="Five-probe mean", xlabel="Disorder RMS (kHz)", ylabel="Mean inferred norm")
    if NUMERICAL_GAP_RATIO is not None:
        if len(NUMERICAL_GAP_RATIO) != len(DISORDER_RMS_KHZ):
            raise ValueError("Supply one numerical mean gap ratio per disorder strength.")
        reference_axis = axes.flat[5].twinx()
        reference_axis.plot(DISORDER_RMS_KHZ, NUMERICAL_GAP_RATIO, "s--", color="tab:red")
        reference_axis.set_ylabel("Numerical mean gap ratio", color="tab:red")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# A small rank/window comparison for one measured correlation. Stable fitted
# decay alone does not establish that the inferred weights are the unitary
# spectral weights. Compare this spread with shot errors and inspect residuals.
# The following spread is a diagnostic, not a confidence interval.

# %%
stability_results = {}
if correlations:
    stability_name = "pilot_Kerr" if "pilot_Kerr" in correlations else next(iter(correlations))
    result = correlations[stability_name]
    for modes in [8, 12, 16]:
        for fraction in [0.75, 1.0]:
            points = max(5, int(len(result["time_us"]) * fraction))
            key = f"modes{modes}_window{fraction:g}"
            try:
                fit = fit_correlation_mpm(result["time_us"][:points], result["correlation"][:points], max_modes=modes)
            except (ValueError, np.linalg.LinAlgError) as error:
                stability_results[key] = dict(target=stability_name, error=str(error))
                print(key, "MPM failed:", error)
                continue
            inferred = regularized_agp_from_fit(fit, cutoff_mhz, reference_mhz=reference_mhz)
            stability_results[key] = dict(target=stability_name, estimate=inferred, diagnostics=fit["diagnostics"])
            print(key, inferred["values"], "relative residual:", fit["diagnostics"]["relative_residual"])

# %% [markdown]
# ## 11. Save processed results separately from raw data
#
# The output contains calibration, correlations and their errors, fit parameters,
# cutoff conventions, the fixed disorder pattern, and raw source filenames.
# It preserves the observed finite-time estimate separately from the inferred
# zero-decay MPM value. The raw HDF5 files remain the canonical shot records.

# %%
if readout_jobs or correlations:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{CAMPAIGN_ID}_{datetime.now():%Y%m%d_%H%M%S_%f}.h5"
    processed = dict(
        calibration=calibration, correlations=correlations,
        mpm_fits=fit_results, agp=agp_results, stability=stability_results,
        agp_by_disorder=agp_by_disorder, disorder_rms_khz=DISORDER_RMS_KHZ,
        numerical_gap_ratio=NUMERICAL_GAP_RATIO,
        campaign=CAMPAIGN_ID, storage_modes=STORAGE_MODES,
        total_photons=TOTAL_PHOTONS, disorder_pattern=DISORDER_PATTERN,
        source_files=[job["source"] for job in readout_jobs + walk_jobs + transfer_jobs + pilot_jobs + scan_jobs],
    )
    save_analysis_h5(output_path, processed, sources=processed["source_files"],
                     metadata=dict(probe_order=[name for name, probe, mode in PROBES]))
    print("Processed data:", output_path)
    reloaded = load_analysis_h5(output_path)

# %% [markdown]
# ## Next comparison
#
# At fixed Kerr, plot the five inferred AGP norms against disorder strength and
# their predefined mean in the same operator convention. A K-versus-W map is a
# numerical reference until measurements at other calibrated Kerr points exist.
# Compare a fixed experimental disorder pattern with its own numerical model
# as well as the disorder-averaged gap-ratio reference; keep those references
# distinct. Correlation with level statistics is a question for the data.
