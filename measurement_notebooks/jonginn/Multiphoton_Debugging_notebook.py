# ---
# jupyter:
#   jupytext:
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
# # Debugging notebook: Fock 2 to Fock 3
#
# Select the station configs for the calibration you want to investigate. No new
# QICK Program or Experiment class is defined here: endpoint IQ uses
# HistogramPrepulseExperiment; tomography uses WignerTomography1ModeExperiment.
# Both receive the same resolved preparation pulses.
#
# ```text
# Fock 2 ---- A: |g,2> ---- ge2 ---- B: |e,2> ---- ef2 ---- C: |f,2> ---- f2-g3 ---- D: |g,3>
#              | IQ / Wigner          | IQ                 | IQ                     | IQ / Wigner
# ```
#
# Every endpoint, displacement, and parity-phase branch starts with a fresh
# preparation. There is no measurement, reset, or postselection in the middle
# of the preparation. Initial active-reset readouts are retained in raw data.
#
# Run the measurement cells manually, one at a time. There is no acquisition
# selection dictionary. USE_QUEUE defaults to True.
# Skip the optional calibration cells when current calibrations are valid.
# After changing readout phase, acquire fresh g/e/f references. Keep calibration
# fixed while comparing the target states. Never train the classifier on the
# unknown target states themselves.

# %% tags=["imports"]
# %load_ext autoreload
# %autoreload 2

from copy import deepcopy
from pathlib import Path
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import experiments as meas
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.MM_base import MM_base, prepulse_creator2
from experiments.qubit_cavity.single_mode_wigner_tomography import (
    WignerTomography1ModeExperiment,
)
from experiments.single_qubit.single_shot import HistogramExperiment
from experiments.single_qubit.single_shot_prepulse import HistogramPrepulseExperiment
from fitting import fitting as fitter
from fitting.fit_display_classes import LengthRabiFitting, Spectroscopy
from fitting.wigner import OptimalDisplacementGeneration
from job_server import JobClient

# %% [markdown]
# ## 1. Station and measurement settings
#
# Set the four config entries below, as in the other measurement notebooks.
# Each accepts a config version ID or a file path; None loads its main version.
# Run this cell before calibration. Rerunning it reloads the selected configs.

# %% tags=["client"]
client = JobClient()
health = client.health_check()
client.print_queue()

# %% tags=["station"]
user = "jonginn"
config_dict = {'hardware_config': 'CFG-HW-20261028-00035',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20261027-00052',
 'floquet_storage_swap': 'CFG-FL-20261027-00074'}

station = MultimodeStation(
    user=user,
    experiment_name="261028_fock_ladder_debug",
    project="Fock preparation",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)

# %% tags=["settings"]
USE_QUEUE = True

MAN_MODE = 1
PARITY_FAST = False
ACTIVE_RESET = True
STORAGE_RESET = [1, 3, 5, 7]
WAIT_AFTER_RESET_US = 100.0 if ACTIVE_RESET else 0.0
RELAX_DELAY_US = 200.0 if ACTIVE_RESET else 2500.0
REFERENCE_SHOTS = 5000
IQ_SHOTS = 5000
LENGTH_SHOTS = 1500
WIGNER_SHOTS = 300
FOCK_DIMENSION = 8

# Set this to an existing complex-alpha .npy file. None creates a regular grid.
DISPLACEMENT_PATH = None
GRID_EXTENT = 2.5
GRID_POINTS_PER_AXIS = 21
GENERATE_OPTIMIZED_DISPLACEMENTS = True
OPTIMIZED_POINT_COUNT = 65

output_directory = Path(station.experiment_path) / "debugging_notebook"
output_directory.mkdir(parents=True, exist_ok=True)
stages = ["fock2", "fock2_ge", "fock2_ge_ef", "fock3"]
state_names = ["g", "e", "f"]

# Preserve previously acquired objects when rerunning setup cells.
if "endpoint_jobs" not in globals():
    endpoint_jobs = {}
if "length_jobs" not in globals():
    length_jobs = []
if "wigner_jobs" not in globals():
    wigner_jobs = {}
if "calibration_jobs" not in globals():
    calibration_jobs = {}

# %% [markdown]
# ## 2. Shared preparation and runner defaults
#
# `ge2` and `ef2` mean the photon-dependent multiphoton table entries, not the
# bare-qubit aliases. The gate list comes from the same definite-Fock builder
# used by FockPopulationProgram. No station pulse parameter is changed by a
# duration scan: only the copied pulse sent to that job is modified.

# %% tags=["preparation_helpers"]
mm_base_calib = MM_base(station.hardware_cfg, station.soccfg)


def stage_gates(stage):
    """Return the existing Fock ladder, truncated at the requested endpoint."""
    if stage == "vacuum":
        return []
    if stage == "fock1":
        return mm_base_calib.prep_man_fock_state(MAN_MODE, "1")
    if stage == "fock3":
        return mm_base_calib.prep_man_fock_state(MAN_MODE, "3")
    pulses = mm_base_calib.prep_man_fock_state(MAN_MODE, "2")
    if stage in ("fock2_ge", "fock2_ge_ef", "f2g3_scan"):
        pulses.append(["multiphoton", "g2-e2", "pi", 0])
    if stage in ("fock2_ge_ef", "f2g3_scan"):
        pulses.append(["multiphoton", "e2-f2", "pi", 0])
    if stage == "f2g3_scan":
        pulses.append(["multiphoton", "f2-g3", "pi", 0])
    if stage not in ("fock2", "fock2_ge", "fock2_ge_ef", "f2g3_scan"):
        raise ValueError(f"Unknown preparation stage: {stage}")
    return pulses


def resolve_preparation(station, stage, wait_us=0.0, fg_length_us=None, fg_gain=None):
    """Resolve one logical prefix using the current station's actual table."""
    gates = stage_gates(stage)
    if fg_length_us is not None:
        if stage != "f2g3_scan" or fg_length_us < 0:
            raise ValueError("A nonnegative length override requires stage='f2g3_scan'.")
        if fg_length_us == 0:
            gates = gates[:-1]  # No sideband ramps either: this is truly pulse-off.
    if wait_us > 0:
        gates = [["wait", float(wait_us)]] + gates
    cfg = deepcopy(station.hardware_cfg)
    cfg.device.storage._ds_storage = station.ds_storage
    cfg.device.storage._ds_floquet = station.ds_floquet
    creator = prepulse_creator2(cfg)
    for gate in gates:
        getattr(creator, gate[0])(gate[1:])
    pulses = creator.pulse.tolist()
    if stage == "f2g3_scan" and fg_length_us is not None and fg_length_us > 0:
        pulses[2][-1] = float(fg_length_us)
    if stage == "f2g3_scan" and fg_gain is not None and fg_length_us != 0:
        pulses[1][-1] = int(fg_gain)
    return gates, pulses


def preparation_preproc(station, default_expt_cfg, **kwargs):
    cfg = deepcopy(default_expt_cfg)
    cfg.update(kwargs)
    gates, pulses = resolve_preparation(
        station, cfg.stage, cfg.wait_after_reset_us,
        cfg.get("fg_length_us"), cfg.get("fg_gain"),
    )
    cfg.preparation_gates = gates
    cfg.pre_sweep_pulse = pulses
    cfg.prepulse = bool(gates)
    cfg.gate_based = False
    return cfg


common_defaults = AttrDict(dict(
    qubits=[0],
    qubit=0,
    expts=1,
    reps=IQ_SHOTS,
    rounds=1,
    active_reset=ACTIVE_RESET,
    man_reset=True,
    storage_reset=STORAGE_RESET,
    ef_reset=True,
    use_qubit_man_reset=False,
    pre_selection_reset=False,
    pre_selection_parity=False,
    reset_dump_mode=2,
    dump_reset_iter_num=1,
    relax_delay=RELAX_DELAY_US,
    normalize=False,
    prepulse=True,
    gate_based=False,
    postpulse=False,
    stage="fock2",
    wait_after_reset_us=WAIT_AFTER_RESET_US,
    man_mode_no=MAN_MODE,
    parity_fast=PARITY_FAST,
    sigma_z_mode="off",
    measure_parity=False,
    parity_shot=False,
    post_select_pre_pulse=False,
))

if MAN_MODE != 1:
    raise ValueError("This notebook investigates the calibrated M1 ladder.")

# %% tags=["calibration_inventory"]
m = station.hardware_cfg.device.manipulate
parity_delay_key = "revival_time_fast" if PARITY_FAST else "revival_time"
print("Current M1 displacement / parity settings")
for name in ("f_ge", "displace_sigma", "gain_to_alpha", "alpha_scale", parity_delay_key):
    print(f"  {name}: {m[name][0]}")
print("Parity hpi source:", "multiphoton.hpi.gn-en[0]" if PARITY_FAST else "qubit.pulses.hpi_ge")
print("Final-readout lane layout:", MM_base.lane_layout(AttrDict({"expt": common_defaults})))
for stage in stages:
    gates, pulses = resolve_preparation(station, stage, WAIT_AFTER_RESET_US)
    print(f"\n{stage}: {gates}")
    for frequency, gain, length, phase, channel, shape, sigma in zip(*pulses):
        print(f"  ch={channel}, {shape}, f={frequency} MHz, gain={gain}, "
              f"plateau={length} us, sigma={sigma} us, phase={phase} deg")

# %% [markdown]
# ## 3. Optional calibration before the target measurements
#
# Required inputs are fresh readout references, valid parity hpi pulses and
# delay, the M1 carrier frequency, and gain_to_alpha / alpha_scale calibrated
# using the same displacement sigma as tomography. Parity spectroscopy finds
# the M1 displacement frequency; it does not calibrate the parity delay.
#
# Apply accepted calibrations before collecting the final g/e/f references.
# Do not apply the phase reported by check_f=True: that histogram rotates on
# the e/f axis while still returning a two-state g/e confusion matrix.
# Run the reference defaults and runner setup cells even if calibration is
# already valid. The optional execute/apply cells can then be skipped.

# %% tags=["readout_calibration"]
reference_defaults = AttrDict(dict(
    common_defaults,
    reps=REFERENCE_SHOTS,
    stage="vacuum",
    check_e=True,
    check_f=True,
    pulse_e=False,
    pulse_f=False,
))

# %% tags=["reference_runner"]
# Execute: configure the reference runner before calibrating or measuring IQ.
# The wait-only vacuum prefix matches target reset-settling conditions.
reference_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.HistogramExperiment,
    default_expt_cfg=reference_defaults,
    preprocessor=preparation_preproc,
    postprocessor=None,
    job_client=client,
    use_queue=USE_QUEUE,
    show=False,
)

# %% tags=["readout_calibration_execute"]
# Optional: run only when the GE readout calibration needs updating.
ge_calibration = reference_runner.execute(
    check_f=False,
    active_reset=False,
    relax_delay=2500,
    wait_after_reset_us=0.0,
)
ge_calibration.analyze(station=station, subdir=station.autocalib_path)
station.hardware_cfg.device.readout.update(ge_calibration.readout_cfg)
calibration_jobs["readout"] = ge_calibration
# Acquire fresh g/e/f references below after applying this calibration.

# %% tags=["parity_delay_calibration"]
parity_delay_defaults = AttrDict(dict(
    common_defaults,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    relax_delay=2500,
    stage="fock1",
    wait_after_reset_us=0.0,
    start=0.01,
    step=0.02,
    expts=200,
    reps=200,
    param_to_calib="length",
    shift_sign=1,
))

# %% tags=["parity_delay_execute"]
# Execute: optional parity-delay calibration.
parity_delay_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityDelayExperiment,
    default_expt_cfg=parity_delay_defaults,
    preprocessor=preparation_preproc,
    postprocessor=None,
    job_client=client,
    use_queue=USE_QUEUE,
    show=False,
)
parity_delay_job = parity_delay_runner.execute()
# The existing Experiment's default fit is exponential. Refit the raw
# oscillation explicitly; its phase inversion differs for fast/slow parity.
delay_fit = LengthRabiFitting(parity_delay_job.data, config=parity_delay_job.cfg)
delay_fit.analyze()
delay_parameters = delay_fit.results["fit_avgi"]
parity_delay_candidate = 1.0 / (2.0 * abs(delay_parameters[1]))
t = np.asarray(parity_delay_job.data["xpts"])
plt.figure()
plt.plot(t, parity_delay_job.data["avgi"], ".", label="Measured I")
plt.plot(t, fitter.decaysin(t, *delay_parameters), label="Damped-sine fit")
plt.axvline(parity_delay_candidate, color="k", ls="--", label="Half-period candidate")
plt.xlabel("Parity delay (us)")
plt.ylabel("I (ADC)")
plt.legend()
print("Candidate parity delay (us):", parity_delay_candidate)
calibration_jobs["parity_delay"] = parity_delay_job

# %% tags=["apply_parity_delay"]
# Review the scan first. Override this value if finite-pulse phase offsets
# require a different delay; vacuum / Fock parity must agree afterward.
accepted_parity_delay_us = float(parity_delay_candidate)
if not np.isfinite(accepted_parity_delay_us) or accepted_parity_delay_us <= 0:
    raise ValueError("Invalid parity delay.")
station.hardware_cfg.device.manipulate[parity_delay_key][0] = accepted_parity_delay_us

# %% tags=["parity_spectroscopy"]
parity_spectroscopy_defaults = AttrDict(dict(
    start=float(m.f_ge[0]) - 30,
    stop=float(m.f_ge[0]) + 30,
    step=0.6,
    reps=300,
    rounds=1,
    qubits=[0],
    normalize=False,
    active_reset=False,
    single_shot=False,
    singleshot_reps=10000,
    span=20,
    man_mode_no=MAN_MODE,
    parity_fast=PARITY_FAST,
    displace=[True, float(m.displace_sigma[0]), 1000],
    const_pulse=[False, 1],
    f0g1_cavity=0,
    prepulse=False,
    pre_sweep_pulse=[],
    relax_delay=2500,
))

# %% tags=["parity_spectroscopy_execute"]
# Execute: optional M1 displacement-frequency calibration.
parity_spectroscopy_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityFreqExperiment,
    default_expt_cfg=parity_spectroscopy_defaults,
    preprocessor=None,
    postprocessor=None,
    job_client=client,
    use_queue=USE_QUEUE,
    show=False,
)
parity_frequency_job = parity_spectroscopy_runner.execute()
frequency_fit = Spectroscopy(parity_frequency_job.data, config=parity_frequency_job.cfg)
frequency_fit.analyze()
frequency_fit.display()
m1_frequency_candidate = float(parity_frequency_job.data["fit_avgi"][2])
print("Candidate M1 frequency (MHz):", m1_frequency_candidate)
calibration_jobs["parity_spectroscopy"] = parity_frequency_job

# %% tags=["apply_m1_frequency"]
accepted_m1_frequency_mhz = float(m1_frequency_candidate)
if not parity_spectroscopy_defaults.start < accepted_m1_frequency_mhz < parity_spectroscopy_defaults.stop:
    raise ValueError("The fitted M1 frequency is outside the measured range.")
station.hardware_cfg.device.manipulate.f_ge[0] = accepted_m1_frequency_mhz

# %% tags=["gain_to_alpha_calibration"]
gain_max = min(30000, int(1.2 / float(m.gain_to_alpha[0])))
gain_step = max(1, gain_max // 40)
gain_to_alpha_defaults = AttrDict(dict(
    start=0,
    step=gain_step,
    expts=41,
    reps=2000,
    rounds=1,
    qubit=0,
    qubits=[0],
    normalize=False,
    single_shot=False,
    singleshot_reps=1000,
    singleshot_active_reset=False,
    singleshot_man_reset=False,
    singleshot_storage_reset=False,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    span=1000,
    prep_e=False,
    manipulate=MAN_MODE,
    displace=[True, float(m.displace_sigma[0])],
    const_pulse=[False, 1],
    f0g1_cavity=0,
    prepulse=False,
    pre_sweep_pulse=[],
    relax_delay=2500,
    pulse_correction=True,
    parity_fast=PARITY_FAST,
    scaling_amplitude=True,
    plot_IQ=False,
    man_mode_no=MAN_MODE,
))

# %% tags=["gain_to_alpha_execute"]
# Execute: optional displacement conversion and parity-contrast calibration.
gain_to_alpha_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityGainExperiment,
    default_expt_cfg=gain_to_alpha_defaults,
    preprocessor=None,
    postprocessor=None,
    job_client=client,
    use_queue=USE_QUEUE,
    show=False,
)
gain_to_alpha_job = gain_to_alpha_runner.execute()
gain_to_alpha_job.analyze(scale=True, plot=True)
print("Candidate gain_to_alpha:", gain_to_alpha_job.data["gain_to_alpha"])
print("Candidate parity contrast:", gain_to_alpha_job.data["scale"])
calibration_jobs["gain_to_alpha"] = gain_to_alpha_job

# %% tags=["apply_displacement_calibration"]
accepted_gain_to_alpha = float(gain_to_alpha_job.data["gain_to_alpha"])
accepted_alpha_scale = float(gain_to_alpha_job.data["scale"])
if not np.isfinite([accepted_gain_to_alpha, accepted_alpha_scale]).all() or min(accepted_gain_to_alpha, accepted_alpha_scale) <= 0:
    raise ValueError("Invalid displacement conversion or parity contrast.")
m.gain_to_alpha[0] = accepted_gain_to_alpha
m.alpha_scale[0] = accepted_alpha_scale

# %% [markdown]
# ## 4. Displacement file: load, generate, and inspect
#
# The file contains a one-dimensional complex alpha array, not DAC gains.
# Use the same coordinates for vacuum, Fock 2 and Fock 3. Optimized coordinates
# need not form a rectangular grid. The existing Wigner code handles the
# conjugation and drive-phase convention; do not conjugate the file here.
# Coordinates are also sent inline so a queue worker does not need access to
# a notebook-local path. The source path remains in the saved config.

# %% tags=["displacement_file"]
if GENERATE_OPTIMIZED_DISPLACEMENTS:
    generator = OptimalDisplacementGeneration(
        FD=FOCK_DIMENSION, n_disps=OPTIMIZED_POINT_COUNT,
    )
    generated = generator.optimize(save_dir=str(output_directory))
    DISPLACEMENT_PATH = generated["path"]

if DISPLACEMENT_PATH is None:
    DISPLACEMENT_PATH = output_directory / f"debugging_grid_{GRID_POINTS_PER_AXIS}_extent_{GRID_EXTENT:g}.npy"
    if not Path(DISPLACEMENT_PATH).exists():
        axis = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_POINTS_PER_AXIS)
        alpha_real, alpha_imag = np.meshgrid(axis, axis)
        np.save(DISPLACEMENT_PATH, (alpha_real + 1j * alpha_imag).ravel(), allow_pickle=False)

displacement_path = Path(DISPLACEMENT_PATH).expanduser().resolve()
alphas = np.asarray(np.load(displacement_path, allow_pickle=False), dtype=complex)
if alphas.ndim != 1 or alphas.size == 0 or not np.isfinite(alphas).all():
    raise ValueError("The displacement file must contain a finite, nonempty 1D alpha array.")
required_gains = np.abs(alphas) / float(m.gain_to_alpha[0])
if np.max(required_gains) > 32767:
    raise ValueError("The loaded alpha array exceeds the displacement DAC gain range.")
print("Displacement file:", displacement_path)
print("Points:", len(alphas), "origin included:", bool(np.any(np.isclose(alphas, 0))))
print("Maximum |alpha|:", np.max(np.abs(alphas)), "maximum DAC gain:", int(np.max(required_gains)))
plt.figure(figsize=(5, 5))
plt.scatter(alphas.real, alphas.imag, s=12)
plt.xlabel("Re(alpha)")
plt.ylabel("Im(alpha)")
plt.title("Actual displacement coordinates from file")
plt.axis("equal")

wigner_file_settings = dict(
    displacement_path=str(displacement_path),
    alpha_list=np.column_stack([alphas.real, alphas.imag]).tolist(),
    display_mode_state_num=FOCK_DIMENSION, display_rotate=False,
)

# %% [markdown]
# ## 5. Three-state IQ analysis
#
# Fit one labelled 2D Gaussian per reference state, using only training shots.
# Evaluate a 3x3 assignment matrix on held-out reference shots. All endpoint
# shots remain in the analysis.
#
# Report both raw and confusion-corrected populations without clipping. Error
# bars below include target shot noise with the reference model/matrix fixed;
# they do not include reference-preparation errors or calibration uncertainty.

# %% tags=["gef_analysis_helpers"]
def terminal_iq(expt, i_key="I", q_key="Q"):
    layout = MM_base.lane_layout(expt.cfg)
    i = np.asarray(expt.data[i_key]).reshape(-1, layout["read_num"])
    q = np.asarray(expt.data[q_key]).reshape(-1, layout["read_num"])
    return np.column_stack([i[:, layout["idx_final"]], q[:, layout["idx_final"]]])


def classify_iq(iq, model):
    log_likelihood = []
    for mean, covariance in zip(model["means"], model["covariances"]):
        residual = iq - mean
        inverse = np.linalg.inv(covariance)
        distance = np.einsum("ni,ij,nj->n", residual, inverse, residual)
        log_likelihood.append(-0.5 * (distance + np.linalg.slogdet(covariance)[1]))
    return np.argmax(np.column_stack(log_likelihood), axis=1)


def fit_gef_references(reference, seed=20261028):
    rng = np.random.default_rng(seed)
    means, covariances, held_out = [], [], []
    for state in state_names:
        iq = terminal_iq(reference, "I" + state, "Q" + state)
        if len(iq) < 100:
            raise ValueError("At least 100 reference shots per state are needed.")
        order = rng.permutation(len(iq))
        cut = int(0.7 * len(iq))
        train, validation = iq[order[:cut]], iq[order[cut:]]
        mean = np.mean(train, axis=0)
        covariance = np.cov(train, rowvar=False)
        # A small diagonal regularizer avoids a singular reference cloud.
        covariance += np.eye(2) * max(np.trace(covariance) * 0.005, 1e-9)
        means.append(mean)
        covariances.append(covariance)
        held_out.append(validation)
    model = dict(means=np.asarray(means), covariances=np.asarray(covariances))
    confusion = np.empty((3, 3))  # rows=prepared state, columns=assigned state
    for state_index, iq in enumerate(held_out):
        assigned = classify_iq(iq, model)
        confusion[state_index] = np.bincount(assigned, minlength=3) / len(iq)
    if np.linalg.cond(confusion) > 100:
        raise ValueError("The three reference states are not sufficiently distinguishable.")
    model["confusion"] = confusion
    model["reference_file"] = str(reference.fname)
    return model


def population_from_iq(iq, model):
    assigned = classify_iq(iq, model)
    raw = np.bincount(assigned, minlength=3) / len(iq)
    covariance = (np.diag(raw) - np.outer(raw, raw)) / len(iq)
    inverse = np.linalg.inv(model["confusion"].T)
    corrected = inverse @ raw
    corrected_covariance = inverse @ covariance @ inverse.T
    return dict(
        raw=raw, corrected=corrected,
        error=np.sqrt(np.maximum(np.diag(corrected_covariance), 0)),
        shots=len(iq),
    )


def plot_iq_clouds(iq, model, axis, title):
    assigned = classify_iq(iq, model)
    for index, state in enumerate(state_names):
        points = iq[assigned == index]
        axis.scatter(points[::max(1, len(points) // 2000), 0],
                     points[::max(1, len(points) // 2000), 1], s=3, alpha=0.35, label=state)
        covariance = model["covariances"][index]
        values, vectors = np.linalg.eigh(covariance)
        theta = np.linspace(0, 2 * np.pi, 120)
        contour = (vectors @ (2 * np.sqrt(values)[:, None] * np.array([np.cos(theta), np.sin(theta)]))).T
        contour += model["means"][index]
        axis.plot(contour[:, 0], contour[:, 1], color="k", lw=0.8)
    axis.set(title=title, xlabel="I (ADC)", ylabel="Q (ADC)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(markerscale=3)


def update_matching_ge_matrix(station, reference):
    """Update only the g/e assignment matrix, at the current phase/threshold."""
    threshold = float(station.hardware_cfg.device.readout.threshold[0])
    ig = terminal_iq(reference, "Ig", "Qg")[:, 0]
    ie = terminal_iq(reference, "Ie", "Qe")[:, 0]
    pgg, peg = np.mean(ig < threshold), np.mean(ie < threshold)
    name = "confusion_matrix_with_active_reset" if reference.cfg.expt.active_reset else "confusion_matrix_without_reset"
    station.hardware_cfg.device.readout[name] = [pgg, 1 - pgg, peg, 1 - peg]
    print("Updated", name, "without changing the IQ phase or threshold.")

# %% tags=["reference_acquisition"]
reference_job = reference_runner.execute(
    reps=REFERENCE_SHOTS,
)
gef_model = fit_gef_references(reference_job)
update_matching_ge_matrix(station, reference_job)
print("Held-out reference matrix C[prepared, assigned]:\n", gef_model["confusion"])
print("Matrix condition number:", np.linalg.cond(gef_model["confusion"]))
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for state, axis in zip(state_names, axes):
    plot_iq_clouds(terminal_iq(reference_job, "I" + state, "Q" + state), gef_model, axis, "Reference " + state)
fig.tight_layout()

# %% [markdown]
# ## 6. Four preparation endpoints: requests 1, 3, 4, and 5
#
# The reference model is fixed for all four stages. Expected ancilla states
# are g, e, f, g. A correct ancilla state alone does not prove the M1 photon
# number, which is why independent Wigner measurements follow.

# %% tags=["iq_defaults"]
iq_defaults = AttrDict(dict(common_defaults, reps=IQ_SHOTS))

# %% tags=["iq_runner"]
# Execute: use the same runner for each independent endpoint below.
iq_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.HistogramPrepulseExperiment,
    default_expt_cfg=iq_defaults,
    preprocessor=preparation_preproc,
    postprocessor=None,
    job_client=client,
    use_queue=USE_QUEUE,
    show=False,
)

# %% tags=["fock2_iq"]
# Request 1: Fock 2 -> terminal g/e/f IQ. Expected ancilla: g.
fock2_iq = iq_runner.execute(
    stage="fock2",
    reps=IQ_SHOTS,
)
endpoint_jobs["fock2"] = fock2_iq

# %% tags=["fock2_ge_iq"]
# Request 3: Fock 2 -> ge2 -> terminal g/e/f IQ. Expected ancilla: e.
fock2_ge_iq = iq_runner.execute(
    stage="fock2_ge",
    reps=IQ_SHOTS,
)
endpoint_jobs["fock2_ge"] = fock2_ge_iq

# %% tags=["fock2_ge_ef_iq"]
# Request 4: Fock 2 -> ge2 -> ef2 -> terminal g/e/f IQ. Expected ancilla: f.
fock2_ge_ef_iq = iq_runner.execute(
    stage="fock2_ge_ef",
    reps=IQ_SHOTS,
)
endpoint_jobs["fock2_ge_ef"] = fock2_ge_ef_iq

# %% tags=["fock3_iq"]
# Request 5: Fock 3 -> terminal g/e/f IQ. Expected ancilla: g.
fock3_iq = iq_runner.execute(
    stage="fock3",
    reps=IQ_SHOTS,
)
endpoint_jobs["fock3"] = fock3_iq

# %% tags=["endpoint_comparison"]
if endpoint_jobs and "gef_model" in globals():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    endpoint_results = {}
    for stage, axis in zip(stages, axes):
        if stage not in endpoint_jobs:
            continue
        iq = terminal_iq(endpoint_jobs[stage])
        endpoint_results[stage] = population_from_iq(iq, gef_model)
        plot_iq_clouds(iq, gef_model, axis, stage)
    fig.tight_layout()
    fig, axis = plt.subplots(figsize=(9, 4))
    for state_index, state in enumerate(state_names):
        positions = np.arange(len(endpoint_results)) + (state_index - 1) * 0.23
        values = [r["corrected"][state_index] for r in endpoint_results.values()]
        errors = [r["error"][state_index] for r in endpoint_results.values()]
        axis.bar(positions, values, width=0.23, yerr=errors, capsize=3, label=state)
    axis.set_xticks(np.arange(len(endpoint_results)), list(endpoint_results))
    axis.set_ylabel("Corrected ancilla population")
    axis.legend()
    fig.tight_layout()
    for stage, result in endpoint_results.items():
        print(stage, "raw:", result["raw"], "corrected:", result["corrected"])

# %% [markdown]
# ## 7. Final-sideband duration scan
#
# Hold gain/frequency fixed. Prepare |f,2>, then vary only the f2-g3 plateau.
# A successful transfer exchanges Pf and Pg, with little Pe. The zero-duration
# point omits the whole sideband, including ramps. The gain-zero control has
# the nominal duration and ramps, separating driven transfer from waiting.

# %% tags=["length_defaults"]
nominal_pi_us = float(station.hardware_cfg.device.multiphoton["pi"]["fn-gn+1"]["length"][2])
lengths_us = np.linspace(0, 2 * nominal_pi_us, 31)

# %% tags=["length_scan"]
length_jobs = []
for length_us in lengths_us:
    job = iq_runner.execute(
        stage="f2g3_scan",
        fg_length_us=float(length_us),
        reps=LENGTH_SHOTS,
    )
    length_jobs.append(job)

# %% tags=["gain_zero_control"]
# Optional: match the nominal pulse duration with its drive gain set to zero.
wait_control_job = iq_runner.execute(
    stage="f2g3_scan",
    fg_length_us=nominal_pi_us,
    fg_gain=0,
    reps=LENGTH_SHOTS,
)

# %% tags=["length_scan_analysis"]
if length_jobs and "gef_model" in globals():
    measured_lengths = np.array([job.cfg.expt.fg_length_us for job in length_jobs])
    length_results = [population_from_iq(terminal_iq(job), gef_model) for job in length_jobs]
    fig, axis = plt.subplots(figsize=(9, 4))
    for index, state in enumerate(state_names):
        axis.errorbar(measured_lengths, [r["corrected"][index] for r in length_results],
                      yerr=[r["error"][index] for r in length_results], fmt=".-", label=state)
    axis.axvline(nominal_pi_us, color="k", ls="--", label="Current pi plateau")
    axis.set(xlabel="Final f2-g3 plateau (us)", ylabel="Ancilla population")
    axis.legend()
    fig.tight_layout()
    if "wait_control_job" in globals():
        print("Gain-zero control:", population_from_iq(terminal_iq(wait_control_job), gef_model))

# %% [markdown]
# ## 8. Vacuum and Fock-state Wigner: requests 2 and 6
#
# ```text
# fresh preparation -> D(alpha from file) -> hpi -> parity delay -> hpi -> readout
# ```
#
# Each alpha uses independent second-pulse phases 180 and 0. The existing
# analysis combines them and applies the matching g/e assignment matrix and
# alpha_scale. No post-preparation herald/reset is enabled. Preserve raw
# response and parity even if ancilla leakage makes a cavity fidelity unreliable.

# %% tags=["wigner_defaults"]
wigner_defaults = AttrDict(dict(
    common_defaults,
    reps=WIGNER_SHOTS,
    pulse_correction=True,
))

# %% tags=["wigner_runner"]
wigner_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.WignerTomography1ModeExperiment,
    default_expt_cfg=wigner_defaults,
    preprocessor=preparation_preproc,
    postprocessor=None,
    job_client=client,
    use_queue=USE_QUEUE,
    show=False,
)

# %% tags=["vacuum_wigner"]
# Optional reference: vacuum Wigner with the same coordinates and parity pulses.
vacuum_wigner = wigner_runner.execute(
    stage="vacuum",
    reps=WIGNER_SHOTS,
    **wigner_file_settings,
)
vacuum_wigner.analyze()
wigner_jobs["vacuum"] = vacuum_wigner

# %% tags=["fock2_wigner"]
# Request 2: Fock 2 Wigner, using exactly the Fock 2 IQ preparation.
fock2_wigner = wigner_runner.execute(
    stage="fock2",
    reps=WIGNER_SHOTS,
    **wigner_file_settings,
)
fock2_wigner.analyze()
wigner_jobs["fock2"] = fock2_wigner

# %% tags=["fock3_wigner"]
# Request 6: Fock 3 Wigner, using exactly the Fock 3 IQ preparation.
fock3_wigner = wigner_runner.execute(
    stage="fock3",
    reps=WIGNER_SHOTS,
    **wigner_file_settings,
)
fock3_wigner.analyze()
wigner_jobs["fock3"] = fock3_wigner

# %% tags=["wigner_raw_comparison"]
if wigner_jobs:
    fig, axes = plt.subplots(2, len(wigner_jobs), figsize=(5 * len(wigner_jobs), 8), squeeze=False)
    for column, (stage, job) in enumerate(wigner_jobs.items()):
        alpha = np.asarray(job.data["alpha"])
        raw_i = np.asarray(job.data["i0"])
        read_num = MM_base.lane_layout(job.cfg)["read_num"]
        last_i = raw_i.reshape(len(alpha), 2, -1, read_num)[..., -1]
        pe = np.mean(last_i > job.cfg.device.readout.threshold[0], axis=-1)
        raw_parity = ((1 - 2 * pe[:, 0]) - (1 - 2 * pe[:, 1])) / 2
        for row, values, limit, label in (
            (0, raw_parity, 1, "raw paired parity"),
            (1, 2 / np.pi * np.asarray(job.data["parity"]), 2 / np.pi, "corrected Wigner W(alpha)"),
        ):
            axis = axes[row, column]
            points = axis.scatter(-alpha.real, -alpha.imag, c=values,
                                  cmap="RdBu_r", vmin=-limit, vmax=limit, s=35)
            axis.set(title=stage + ": " + label, xlabel="Re(alpha)", ylabel="Im(alpha)")
            axis.set_aspect("equal")
            fig.colorbar(points, ax=axis)
    fig.tight_layout()

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
wigner_jobs['fock2'].data['rho_rotated']

# %% tags=["wigner_reconstruction"]
# Reconstruction uses the existing implementation. Review the endpoint IQ
# and vacuum map first: a physical density-matrix fit cannot diagnose a bad
# parity measurement by itself. Reconstructing does not overwrite raw files.
for stage, job in wigner_jobs.items():
    target_n = {"vacuum": 0, "fock2": 2, "fock3": 3}[stage]
    job.display(
        initial_state=qt.fock(FOCK_DIMENSION, target_n),
        mode_state_num=FOCK_DIMENSION,
        rotate=False,
        state_label=stage,
    )

# %% tags=["end_reference_check"]
end_reference_job = reference_runner.execute(
    reps=REFERENCE_SHOTS,
)
for index, state in enumerate(state_names):
    iq = terminal_iq(end_reference_job, "I" + state, "Q" + state)
    print("End reference", state, "mean IQ shift:", np.mean(iq, axis=0) - gef_model["means"][index])
# Keep the original reference model for the target comparison.

# %% [markdown]
# ## 9. Save a processed HDF5 and reload without measuring
#
# Raw IQ stays in the original JOB HDF5s. The summary below stores the classifier,
# populations and links to those files in a new HDF5; it never rewrites them.
# A manifest records the files needed to repeat the analysis in a fresh kernel.

# %% tags=["persistence_helpers"]
def save_debugging_results(directory, model, reference, endpoints, length_scan, wigners, calibrations):
    directory = Path(directory)
    manifest = dict(
        references=str(reference.fname),
        endpoints={name: str(job.fname) for name, job in endpoints.items()},
        length_scan=[str(job.fname) for job in length_scan],
        wigner={name: str(job.fname) for name, job in wigners.items()},
        calibrations={name: str(job.fname) for name, job in calibrations.items()},
        displacement_path=str(displacement_path),
    )
    if "wait_control_job" in globals():
        manifest["wait_control"] = str(wait_control_job.fname)
    if "end_reference_job" in globals():
        manifest["end_references"] = str(end_reference_job.fname)
    stem = Path(reference.fname).stem + "_debugging_analysis"
    summary_path = directory / (stem + ".h5")
    suffix = 1
    while summary_path.exists():
        summary_path = directory / f"{stem}_{suffix}.h5"
        suffix += 1
    with h5py.File(summary_path, "w") as handle:
        handle.attrs["manifest"] = json.dumps(manifest)
        handle.attrs["population_error_scope"] = "Target shot noise; reference classifier and matrix held fixed"
        handle.create_dataset("alpha", data=alphas)
        group = handle.create_group("gef_reference")
        for key in ("means", "covariances", "confusion"):
            group.create_dataset(key, data=model[key])
        for name, job in endpoints.items():
            result = population_from_iq(terminal_iq(job), model)
            group = handle.create_group("endpoints/" + name)
            group.attrs["source"] = str(job.fname)
            for key, value in result.items():
                group.create_dataset(key, data=value)
        if length_scan:
            group = handle.create_group("length_scan")
            group.create_dataset("plateau_us", data=[job.cfg.expt.fg_length_us for job in length_scan])
            results = [population_from_iq(terminal_iq(job), model) for job in length_scan]
            for key in results[0]:
                group.create_dataset(key, data=[result[key] for result in results])
        if "wait_control_job" in globals():
            group = handle.create_group("wait_control")
            result = population_from_iq(terminal_iq(wait_control_job), model)
            for key, value in result.items():
                group.create_dataset(key, data=value)
        for name, job in wigners.items():
            group = handle.create_group("wigner/" + name)
            group.attrs["source"] = str(job.fname)
            for key in ("alpha", "parity", "parity_plus", "parity_minus", "pe_plus", "pe_minus",
                        "rho", "rho_linear", "populations", "fidelity", "linear_fidelity"):
                if key in job.data:
                    group.create_dataset(key, data=job.data[key])
    manifest["summary_hdf5"] = str(summary_path)
    manifest_path = summary_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary_path, manifest_path


def reload_debugging_run(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    with h5py.File(manifest["summary_hdf5"], "r") as handle:
        model = {key: handle["gef_reference/" + key][()] for key in ("means", "covariances", "confusion")}
    model["reference_file"] = manifest["references"]
    reference = HistogramExperiment.from_h5file(manifest["references"])
    endpoints = {name: HistogramPrepulseExperiment.from_h5file(path) for name, path in manifest["endpoints"].items()}
    scan = [HistogramPrepulseExperiment.from_h5file(path) for path in manifest["length_scan"]]
    wigners = {name: WignerTomography1ModeExperiment.from_h5file(path) for name, path in manifest["wigner"].items()}
    return model, reference, endpoints, scan, wigners

# %% tags=["save_analysis"]
if "gef_model" in globals() and "reference_job" in globals():
    summary_path, manifest_path = save_debugging_results(
        output_directory, gef_model, reference_job, endpoint_jobs,
        length_jobs, wigner_jobs, calibration_jobs,
    )
    print("Processed HDF5:", summary_path)
    print("Reload manifest:", manifest_path)

# %% tags=["reload_only"]
# Set a saved manifest to analyze existing data without submitting jobs.
RELOAD_MANIFEST = None
if RELOAD_MANIFEST is not None:
    gef_model, reference_job, endpoint_jobs, length_jobs, wigner_jobs = reload_debugging_run(RELOAD_MANIFEST)
    # Rerun the comparison/plot cells above; skip cells containing execute().

# %% [markdown]
# ## Interpreting the comparison
#
# - Fock 2 already has incorrect ancilla IQ: investigate the first two rungs.
# - The change starts at ge2 or ef2: investigate that conditional qubit pulse.
# - |f,2> looks good but Fock 3 retains f or gains e: inspect the final sideband.
# - Ancilla g looks good but Wigner is wrong: ancilla ground population alone
#   does not establish the cavity state; compare vacuum/parity calibration too.
# - Both preparation diagnostics look good: reconnect the same preparation to
#   the original multiparity readout next, before adding Floquet evolution.
#
# A cloud outside the reference templates is not proof of f leakage. Check
# reference stability and photon-dependent readout response before interpreting
# a precise population or tomography fidelity.
