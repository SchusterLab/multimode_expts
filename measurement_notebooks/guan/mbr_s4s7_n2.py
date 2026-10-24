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
# # MBR spectroscopy, S4-S7, N = 2 (guan + Claude, from 2026-10-20)
#
# Driven cell by cell into the shared `guan-meas` kernel (main checkout, jobs through
# the queue, interleaved with seb). Jonginn's set point: coupler 0.25 mA, his 10-19
# config versions. Checks first (measure, compare with the config, no change), then
# the MBR campaign of `measurement_notebooks/202609_qsim_migration/mbr.py` with
# `modes=[4, 5, 6, 7]` (sites M1, S4..S7).
#
# Config: in-memory changes are fine; snapshots only with `update_main=False`.
# Plots: each check cell saves its figures as PNGs (`save_figs`) for the CC Board.

# %% Cell 1: station on jonginn's 10-19 configs, single shot without active reset
# (single_qubit_autocalibrate_v2 cells 18-20). Ran 2026-10-20 15:14, JOB-20261020-00076,
# at 0.30 mA (avoid_yoko): fid 0.910.
import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import experiments as meas
from slab import AttrDict
from experiments import MultimodeStation, CharacterizationRunner
from job_server import JobClient

AVOID_YOKO = True  # cell 2 ramps the coupler once with avoid_yoko=False

# Started on jonginn's 10-19 versions: HW CFG-HW-20261019-00001, M1 CFG-M1-20261019-00001
# (MP and FL as below). Since 2026-10-22: the checked snapshots (cells 7 and 12b).
config_dict = {
    "hardware_config": "CFG-HW-20261021-00009",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261020-00016",
    "floquet_storage_swap": "CFG-FL-20261017-00034",
}

client = JobClient()
client.print_queue()

station = MultimodeStation(
    user="guan",
    experiment_name="261020_mbr_s4s7_n2",
    project="EncSpec",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)

ss_reset = dict(active_reset=False, ef_reset=True, storage_reset=False, coupler_reset=False,
                pre_selection_reset=True, man_reset=False, use_qubit_man_reset=True,
                pre_selection_parity=False)

singleshot_defaults = AttrDict(dict(
    reps=5000, relax_delay=500, check_f=False, active_reset=False, man_reset=False,
    storage_reset=False, qubit=0, pulse_manipulate=False, cavity_freq=4984.373226159381,
    cavity_gain=400, cavity_length=2, prepulse=False, pre_sweep_pulse=None, gate_based=True,
    qubits=[0], avoid_yoko=AVOID_YOKO,
))


def singleshot_postproc(station, expt):
    expt.analyze(plot=False, station=station, subdir=station.autocalib_path)
    d = expt.data
    ro = station.hardware_cfg.device.readout
    print("old: phase", ro.phase[0], "threshold", ro.threshold)
    ro.phase = [ro.phase[0] + d["angle"]]
    ro.threshold = d["thresholds"]
    ro.threshold_list = [d["thresholds"]]
    ro.Ie = [np.median(d["Ie_rot"])]
    ro.Ig = [np.median(d["Ig_rot"])]
    if expt.cfg.expt.active_reset:
        ro.confusion_matrix_with_active_reset = d["confusion_matrix"]
    else:
        ro.confusion_matrix_without_reset = d["confusion_matrix"]
    print("new: phase", ro.phase[0], "threshold", ro.threshold)
    print("fid", d["fids"], "angle", d["angle"], "confusion", np.round(d["confusion_matrix"], 4))


ss_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.HistogramExperiment,
    default_expt_cfg=singleshot_defaults,
    postprocessor=singleshot_postproc,
    job_client=client,
)

SCRATCH = r"C:\Users\slab\AppData\Local\Temp\claude\C--python-multimode-expts\d9311a48-5940-4f04-83ff-c0bbd5ee4156\scratchpad"


def save_figs(tag):
    """Save the open figures as PNGs in the scratchpad (for the CC Board)."""
    paths = []
    for k, num in enumerate(plt.get_fignums()):
        path = os.path.join(SCRATCH, f"{tag}_{k}.png")
        plt.figure(num).savefig(path, dpi=80)
        paths.append(path)
    plt.close("all")
    print("saved", paths)


# %% Cell 1b: run the single shot
ss = ss_runner.execute(go_kwargs=dict(analyze=False, display=False),
                       check_f=False, relax_delay=2500, reps=5000, **ss_reset)
ss.display(station)
save_figs("ss_01")

# %% Cell 2: ramp the coupler to the config value (0.25 mA) with a single shot, then check
# the ge and ef frequencies with jonginn's Ramsey settings (his Autocalibrate cells 19-20,
# 38-39). The Ramseys only measure: postprocess=False, the config is not changed.
print("coupler in config:", station.hardware_cfg.hw.yoko_coupler.current)
ss = ss_runner.execute(go_kwargs=dict(analyze=False, display=False), check_f=False,
                       relax_delay=2500, reps=5000, avoid_yoko=False, **ss_reset)
ss.display(station)
save_figs("ss_02")

geramsey_defaults = AttrDict(dict(
    ramsey_freq=0.2, start=0.01, step=0.5, expts=101, reps=200, rounds=1,
    pre_sweep_pulse=None, post_sweep_pulse=None, if_ef=False, ef_init=True, qubits=[0],
    user_defined_freq=[False, 3568.2038290468167, 5304, 0.035], f0g1_cavity=0,
    normalize=False, active_reset=False, man_reset=False, storage_reset=False,
    prepulse=None, postpulse=None, pre_active_reset_pulse=False, gate_based=False,
    advance_phase=0, echoes=[False, 0], relax_delay=200,
))


def geramsey_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    expt_cfg.checkEF = bool(expt_cfg.if_ef)
    expt_cfg.qubit_ge_init = bool(expt_cfg.if_ef and expt_cfg.ef_init)
    expt_cfg.qubit_ge_after = bool(expt_cfg.if_ef and expt_cfg.ef_init)
    expt_cfg.prepulse = expt_cfg.pre_sweep_pulse is not None
    expt_cfg.postpulse = expt_cfg.post_sweep_pulse is not None
    return expt_cfg


eframsey_defaults = AttrDict(deepcopy(geramsey_defaults))
eframsey_defaults.update(if_ef=True, ef_init=True, ramsey_freq=3)

geramsey_runner = CharacterizationRunner(station=station, ExptClass=meas.RamseyExperiment,
                                         default_expt_cfg=geramsey_defaults,
                                         preprocessor=geramsey_preproc, job_client=client)
eframsey_runner = CharacterizationRunner(station=station, ExptClass=meas.RamseyExperiment,
                                         default_expt_cfg=eframsey_defaults,
                                         preprocessor=geramsey_preproc, job_client=client)

qcfg = station.hardware_cfg.device.qubit
geramsey = geramsey_runner.execute(ramsey_freq=0.2, step=0.4, active_reset=False, relax_delay=2500,
                                   if_ef=False, postprocess=False)
geramsey.display()
save_figs("geramsey_02")
print("f_ge config", qcfg.f_ge[0], "| f_adjust (MHz)", geramsey.data.get("f_adjust_ramsey_avgi"),
      "| fit", geramsey.data.get("fit_avgi"))

eframsey = eframsey_runner.execute(ramsey_freq=0.2, step=0.2, relax_delay=2500, active_reset=True,
                                   postprocess=False)
eframsey.display()
save_figs("eframsey_02")
print("f_ef config", qcfg.f_ef[0], "| f_adjust (MHz)", eframsey.data.get("f_adjust_ramsey_avgi"),
      "| fit", eframsey.data.get("fit_avgi"))

# %% Cell 3: sideband check, f0-g1 and f1-g2 frequency by error amplification (jonginn's
# Autocalibrate cell 140: band 0.3 MHz, 50 points, 7 pulses). Measure only.
error_amp_defaults = AttrDict(dict(
    reps=100, rounds=1, qubit=0, n_pulses=10, active_reset=False, man_reset=True,
    storage_reset=True, reset_dump_mode=2, dump_reset_iter_num=1, relax_delay=2500,
    start=0, expts=10, step=100, parameter_to_test="gain", pulse_type=["qubit", "ge", "pi", 0],
    man_no=1, pulse_name="pi",
))


def error_amp_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    expt_cfg.qubits = [expt_cfg.qubit]
    return expt_cfg


error_amp_runner = CharacterizationRunner(station=station, ExptClass=meas.ErrorAmplificationExperiment,
                                          default_expt_cfg=error_amp_defaults,
                                          preprocessor=error_amp_preproc, job_client=client)


def fit_optimum(expt, tag):
    """Fit as jonginn's postprocessors do (state_fin='e'); return the optimum."""
    expt.analyze(data=expt.data, state_fin="e")
    expt.display(data=expt.data, state_fin="e")
    save_figs(tag)
    return expt.data["fit_avgi"][2]


mp_pi = station.hardware_cfg.device.multiphoton.pi["fn-gn+1"]
sideband_check = {}
for n, sideband in enumerate(["f0-g1", "f1-g2"]):
    band, expts = 0.3, 50
    config_freq = mp_pi["frequency"][n]
    expt = error_amp_runner.execute(
        transition=sideband, parameter_to_test="frequency", start=config_freq - band,
        step=2 * band / expts, expts=expts, n_pulses=7,
        pulse_type=["multiphoton", sideband, "pi", 0],
        go_kwargs=dict(analyze=False, display=False), postprocess=False,
    )
    optimum = fit_optimum(expt, f"ea_{sideband}_03")
    sideband_check[sideband] = (config_freq, optimum)
    print(f"{sideband}: config {config_freq:.4f} MHz, error amp. {optimum:.4f} MHz, "
          f"difference {1e3 * (optimum - config_freq):+.1f} kHz")

# %% Cell 3b: apply the sideband results in memory (as jonginn's postprocessor: multiphoton
# pi frequencies and the ds M1 row). 2026-10-20 21:40: f0-g1 -126 kHz (JOB-20261020-00148),
# f1-g2 -158 kHz (JOB-20261020-00153). Frequencies only; the pi lengths stay.
mp_pi = station.hardware_cfg.device.multiphoton.pi["fn-gn+1"]
for n, sideband in enumerate(["f0-g1", "f1-g2"]):
    old, new = sideband_check[sideband]
    mp_pi["frequency"][n] = float(new)
    print(f"{sideband}: {old:.4f} -> {new:.4f} MHz")
station.ds_storage.update_freq("M1", float(sideband_check["f0-g1"][1]))
print("ds M1 freq:", station.ds_storage.get_freq("M1"))

# %% Cell 4: M1-S4..S7 swap frequency by error amplification (jonginn's Autocalibrate
# cells 106-108, frequency part: 100 points, 5 pulses, active reset). Measure only.
error_amp_stor_defaults = AttrDict(dict(
    reps=50, rounds=1, qubits=[0], man_mode_no=1, stor_is_dump=False, active_reset=False,
    man_reset=True, storage_reset=True, reset_dump_mode=2, dump_reset_iter_num=1,
    relax_delay=2500, expts=30, n_start=0, n_step=1, n_pulses=5,
))
error_amp_stor_runner = CharacterizationRunner(station=station, ExptClass=meas.ErrorAmplificationExperiment,
                                               default_expt_cfg=error_amp_stor_defaults,
                                               job_client=client)
stor_span = {4: 0.3, 5: 0.3, 6: 0.3, 7: 0.2}  # jonginn's span_list for S4..S7
swap_check = {}
for stor in [4, 5, 6, 7]:
    name = f"M1-S{stor}"
    config_freq = station.ds_storage.get_freq(name)
    span, expts = stor_span[stor], 100
    expt = error_amp_stor_runner.execute(
        stor_mode_no=stor, parameter_to_test="frequency", pulse_type=["storage", name, "pi", 0],
        start=config_freq - span / 2, step=span / (expts - 1), expts=expts,
        active_reset=True, man_reset=True, storage_reset=[stor], relax_delay=1000,
        go_kwargs=dict(analyze=False, display=False), postprocess=False,
    )
    optimum = fit_optimum(expt, f"ea_{name}_04")
    swap_check[name] = (config_freq, optimum)
    print(f"{name}: config {config_freq:.4f} MHz, error amp. {optimum:.4f} MHz, "
          f"difference {1e3 * (optimum - config_freq):+.1f} kHz")

# %% Cell 5: Trotter (Floquet) pulse frequency, S4..S7, by error amplification (jonginn's
# Autocalibrate cells 118-123, fine: span 0.05 MHz, 50 points, 7 pulses). Measure only.
floquet_settings = dict(include_10cycles_buffer=True, include_10cycles_buffer_in_pi_half=True,
                        floquet_hardware_loop=True, scramble_sync_cycles=1, palindrome_scramble=False)
error_amp_floquet_defaults = AttrDict(dict(
    reps=100, rounds=1, qubits=[0], active_reset=False, man_mode_no=1, stor_is_dump=False,
    man_reset=True, storage_reset=True, relax_delay=2500, expts=50, n_pulses=7,
    qubit_start_storage="g", floquet_sync_delay=1, reset_dump_mode=2, dump_reset_iter_num=1,
    **floquet_settings,
))
error_amp_floquet_runner = CharacterizationRunner(station=station, ExptClass=meas.ErrorAmplificationExperiment,
                                                  default_expt_cfg=error_amp_floquet_defaults,
                                                  job_client=client)
floquet_check = {}
for stor in [4, 5, 6, 7]:
    name = f"M1-S{stor}"
    config_freq = station.ds_floquet.get_freq(name)
    pi_frac = station.ds_floquet.get_pi_frac(name)
    span, expts = 0.05, 50
    expt = error_amp_floquet_runner.execute(
        stor_mode_no=stor, parameter_to_test="frequency",
        pulse_type=["floquet", name, f"pi/{pi_frac}", 0],
        start=config_freq - span / 2, step=span / (expts - 1), expts=expts,
        active_reset=True, man_reset=True, storage_reset=[stor], relax_delay=200,
        go_kwargs=dict(analyze=False, display=False), postprocess=False,
    )
    optimum = fit_optimum(expt, f"ea_floquet_{name}_05")
    floquet_check[name] = (config_freq, optimum)
    print(f"Trotter {name}: config {config_freq:.4f} MHz, error amp. {optimum:.4f} MHz, "
          f"difference {1e3 * (optimum - config_freq):+.1f} kHz")

# %% Cell 6a: pi_ge_broadband frequency check. The config still has the pre-warm-up value
# (3566.20 MHz; f_ge moved to 3561.5 on 10-16). Wide error-amp. frequency scan; measure only.
bb = station.hardware_cfg.device.qubit.pulses.pi_ge_broadband
print("broadband config: freq", bb.frequency[0], "gain", bb.gain[0], "sigma", bb.sigma[0],
      "| f_ge", station.hardware_cfg.device.qubit.f_ge[0])
bb_pulse = ["qubit", "ge_broadband", "pi", 0]
bb_freq = error_amp_runner.execute(
    parameter_to_test="frequency", start=3559.5, step=0.2, expts=41, n_pulses=10, reps=100,
    pulse_type=bb_pulse, go_kwargs=dict(analyze=False, display=False), postprocess=False)
bb_freq.analyze(data=bb_freq.data)
bb_freq.display(data=bb_freq.data)
save_figs("ea_bb_freq_06")
bb_f_opt = bb_freq.data["fit_avgi"][2]
print(f"broadband frequency: config {bb.frequency[0]:.3f}, error amp. {bb_f_opt:.3f} MHz")

# %% Cell 6a2: set the broadband frequency in memory from 6a. 2026-10-20: JOB-20261020-00174
# gave 3563.862 MHz (config 3566.204; Ramsey f_ge 3561.496).
bb.frequency = [float(bb_f_opt)]
print("broadband freq now", bb.frequency[0])

# %% Cell 6b: pi_ge_broadband gain check at the frequency set in memory (run after 6a and
# after bb.frequency is set). Measure only.
gain0 = bb.gain[0]
print("broadband: freq", bb.frequency[0], "gain", gain0)
bb_gain = error_amp_runner.execute(
    parameter_to_test="gain", start=int(gain0 * 0.7), step=int(gain0 * 0.6 / 30), expts=31,
    n_pulses=10, reps=100, pulse_type=bb_pulse,
    go_kwargs=dict(analyze=False, display=False), postprocess=False)
bb_gain.analyze(data=bb_gain.data)
bb_gain.display(data=bb_gain.data)
save_figs("ea_bb_gain_06")
print(f"broadband gain: config {gain0}, error amp. {bb_gain.data['fit_avgi'][2]:.0f}")

# %% Cell 6c: broadband gain in memory from 6b (JOB-20261020-00179: 18447, config 18699).
bb.gain = [int(round(bb_gain.data["fit_avgi"][2]))]
print("broadband now: freq", bb.frequency[0], "gain", bb.gain[0])

# %% Cell 7: snapshot the checked config as new versions (update_main=False; main stays).
# Contents vs jonginn's 10-19 versions: readout from the 0.25 mA single shot (JOB-20261020-00142),
# f0-g1 / f1-g2 frequencies (Cell 3b, also the ds M1 row), pi_ge_broadband freq and gain (6a2, 6c).
hw_version = station.snapshot_hardware_config(update_main=False)
m1_version = station.snapshot_man1_storage_swap(update_main=False)
print("new versions:", hw_version, m1_version)

# %% Cell 8: quantum-walk setup and the multiparity readout calibration (jonginn's
# agp_measurement.py sections 1-2): Fock 0..3 in M1, read M1 by multiparity -> C[prepared, measured].
from experiments.qsim.fock_population import FockPopulationProgram
from experiments.qsim.qsim_base import QsimExperiment
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS

WALK_STORS = [4, 5, 6, 7]
population_defaults = AttrDict(dict(
    expts=1, reps=1, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=WALK_STORS, pre_relax_delay=100, relax_delay=200,
    prepulse=False, postpulse=True, init_fock=False, init_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    occupations=[1, 0, 0, 0, 0], swap_stors=WALK_STORS,
    use_multiphoton_swap=False, storage_pulse_wait_us=0.2, ro_stor=0,
    update_phases=True, detunings=[0.0] * len(WALK_STORS),
    floquet_cycles=[0], swept_params=["floquet_cycle"],
))
population_defaults.update(ACTIVE_RESET_DEFAULTS)
population_defaults.update(FLOQUET_DEFAULTS)


def population_preprocessor(station, defaults, **overrides):
    settings = deepcopy(defaults)
    settings.update(overrides)
    return settings


population_runner = CharacterizationRunner(
    station=station, ExptClass=QsimExperiment, ExptProgram=FockPopulationProgram,
    default_expt_cfg=population_defaults, preprocessor=population_preprocessor,
    job_client=client, use_queue=True, show=False,
)


# %% Cell 8b: multiparity readout confusion, Fock 0..3 in M1, in one job. Qsim's prepulse
# plays the same pulses as a zero-cycle population job (2026-10-23 compile and device checks,
# agp_layers_hw_check.py cell 2). qubit + M with active reset: relax_delay=500 us.
# 2026-10-21 ran it as 4 population jobs (JOB-20261021-00010/12/14/16); cell R loads those.
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment

POPULATION_ONLY = ("expts", "occupations", "use_multiphoton_swap", "storage_pulse_wait_us",
                   "floquet_cycles", "swept_params")
calibration_defaults = AttrDict({key: value for key, value in population_defaults.items()
                                 if key not in POPULATION_ONLY})
calibration_defaults.update(prepulse=True)
calibration_runner = CharacterizationRunner(
    station=station, ExptClass=MultiParityConfusionMatrixExperiment,
    default_expt_cfg=calibration_defaults, job_client=client, use_queue=True, show=False,
)
readout_calibration = calibration_runner.execute(reps=6000, relax_delay=500, show=False, log=False)
readout_confusion = np.asarray(readout_calibration.data["multiparity_confusion_matrix"])
print("C[prepared, measured]:\n", np.round(readout_confusion, 3))
print("condition number:", np.linalg.cond(readout_confusion))
fig, axis = plt.subplots(figsize=(4, 3))
image = axis.imshow(readout_confusion, vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        axis.text(j, i, f"{readout_confusion[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
axis.set(xlabel="Measured number", ylabel="Prepared number", xticks=range(4), yticks=range(4))
fig.colorbar(image, ax=axis, label="Probability")
save_figs("confusion_08")


# %% Cell 9: quantum walk from WALK_OCCUPATION, read out every mode (jonginn's agp_measurement
# section 3). One job per read-out mode; cycles 0, 2, ..., 98.
WALK_OCCUPATION = [1, 0, 0, 0, 0]
walk_cycles = list(range(0, 100, 2))
walk_expts = []
for meas_stor in [0] + WALK_STORS:
    walk_expts.append(population_runner.execute(
        reps=300, occupations=WALK_OCCUPATION, ro_stor=meas_stor, floquet_cycles=walk_cycles,
        postprocess=False, show=False, log=False))
    print(meas_stor, walk_expts[-1].fname)


# %% Cell 9b: walk from [2,0,0,0,0] (M1 self-Kerr). Keep the [1,0,0,0,0] walk
# (2026-10-21: JOB-20261021-00018/23/25/27/29, clean) as walk_expts_10000.
walk_expts_10000 = walk_expts
WALK_OCCUPATION = [2, 0, 0, 0, 0]
walk_expts = []
for meas_stor in [0] + WALK_STORS:
    walk_expts.append(population_runner.execute(
        reps=600, occupations=WALK_OCCUPATION, ro_stor=meas_stor, floquet_cycles=walk_cycles,
        postprocess=False, show=False, log=False))
    print(meas_stor, walk_expts[-1].fname)


# %% Cell 10: plot the walk, raw and corrected by the readout confusion (p_measured = C.T @ p_true),
# over the fixed-N model with the couplings and cycle time saved in each job (derived_params).
import json
import h5py
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian

KERR_KHZ = -27.0  # jonginn's value for K / (2 pi), with sign


def walk_theory(walk_expt, cycles, kerr_khz=KERR_KHZ):
    settings = walk_expt.cfg.expt
    storages = list(settings.swap_stors)
    occupation = tuple(settings.occupations)
    with h5py.File(walk_expt.fname, "r") as data_file:
        timing = json.loads(data_file.attrs["derived_params"])
    cycle_us = float(timing["floquet_cycle_us"])
    couplings_mhz = np.array([timing["couplings_MHz"][s - 1] for s in storages])
    model = fixed_n_hamiltonian(photon_number=sum(occupation), mode_count=1 + len(storages),
                                detunings=np.zeros(len(storages)), couplings_MHz=couplings_mhz,
                                physical_kerr_MHz=kerr_khz / 1000)
    time_us = np.asarray(cycles, dtype=float) * cycle_us
    amplitudes = model.states[model.fock_index[occupation], :].conj()
    state = model.states @ (amplitudes[:, None] * np.exp(-2j * np.pi * model.energies_MHz[:, None] * time_us[None, :]))
    readout_index = ([0] + storages).index(settings.ro_stor)
    return np.asarray(model.fock_basis)[:, readout_index] @ (np.abs(state) ** 2), cycle_us, couplings_mhz


fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for k, walk_expt in enumerate(walk_expts):
    multiparity = walk_expt.analyze_multiparity()
    raw = np.column_stack([multiparity[f"p_mod{n}"] for n in range(4)])
    corrected = np.linalg.solve(readout_confusion.T, raw.T).T
    cycles = np.asarray(multiparity["xpts"], dtype=float)
    mode = walk_expt.cfg.expt.ro_stor
    label = "M1" if mode == 0 else f"S{mode}"
    axes[0].plot(cycles, raw @ np.arange(4), "o-", ms=3, color=f"C{k}", label=label)
    axes[1].plot(cycles, corrected @ np.arange(4), "o", ms=3, color=f"C{k}", label=label)
    theory_cycles = np.linspace(cycles.min(), cycles.max(), 1000)
    theory, cycle_us, couplings_mhz = walk_theory(walk_expt, theory_cycles)
    axes[1].plot(theory_cycles, theory, color=f"C{k}", alpha=0.45, lw=3)
print(f"cycle {cycle_us:.6f} us, g = {np.round(1e3 * couplings_mhz, 3)} kHz, K = {KERR_KHZ} kHz")
axes[0].set(ylabel="Mean number (raw)", title=f"Quantum walk from {WALK_OCCUPATION}")
axes[1].set(xlabel="Floquet cycles", ylabel="Mean number (corrected)")
axes[0].legend(ncol=5)
fig.tight_layout()
save_figs(f"walk_{''.join(map(str, WALK_OCCUPATION))}_10")


# %% Cell 11: model fit of the M1 Kerr from the [2,0,0,0,0] walk. Only K changes; the couplings
# and cycle time stay at the saved values. Least squares on the corrected mean number of all modes.
# The walk on the star graph at zero detuning gives the same populations for +K and -K, so it
# gives |K| only.
def walk_corrected(walk_expt):
    multiparity = walk_expt.analyze_multiparity()
    raw = np.column_stack([multiparity[f"p_mod{n}"] for n in range(4)])
    corrected = np.linalg.solve(readout_confusion.T, raw.T).T
    return np.asarray(multiparity["xpts"], dtype=float), corrected @ np.arange(4)


walk_data = [walk_corrected(walk_expt) for walk_expt in walk_expts]
kerr_scan_khz = np.arange(0.0, 81.0, 1.0)
cost = []
for kerr_khz in kerr_scan_khz:
    residuals = [mean - walk_theory(walk_expt, cycles, kerr_khz=-kerr_khz)[0]
                 for walk_expt, (cycles, mean) in zip(walk_expts, walk_data)]
    cost.append(np.sum(np.concatenate(residuals) ** 2))
cost = np.asarray(cost)
best_khz = kerr_scan_khz[np.argmin(cost)]
point_count = sum(len(cycles) for cycles, _ in walk_data)
sigma2 = cost.min() / (point_count - 1)
inside = kerr_scan_khz[cost - cost.min() < sigma2]  # delta chi2 < 1 with sigma from the residuals
print(f"|K| best {best_khz:.0f} kHz, 1-sigma range {inside.min():.0f}..{inside.max():.0f} kHz, "
      f"rms residual {np.sqrt(sigma2):.3f}; cost at 27 kHz / best = {cost[27] / cost.min():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(kerr_scan_khz, cost, "o-", ms=3)
axes[0].axvline(27, color="gray", ls="--", label="jonginn 27 kHz")
axes[0].set(xlabel="|K| / 2pi (kHz)", ylabel="sum of squared residuals", title="Kerr scan, [2,0,0,0,0] walk")
axes[0].legend()
cycles, mean = walk_data[0]
fine = np.linspace(cycles.min(), cycles.max(), 1000)
axes[1].plot(cycles, mean, "o", ms=3, color="C0", label="M1 data")
for kerr_khz, style in [(27, "--"), (best_khz, "-")]:
    axes[1].plot(fine, walk_theory(walk_expts[0], fine, kerr_khz=-kerr_khz)[0], style,
                 label=f"|K| = {kerr_khz:.0f} kHz")
axes[1].set(xlabel="Floquet cycles", ylabel="M1 mean number (corrected)")
axes[1].legend()
fig.tight_layout()
save_figs("walk_20000_kerr_11")


# %% Cell 12: longer ge Ramsey for the drifting qubit: virtual frequency 0.03 MHz over 400 us
# (T2 was about 630 us in cell 2). Measures only; cell 12b applies the result.
geramsey_long = geramsey_runner.execute(ramsey_freq=0.03, start=0.01, step=4.0, expts=101,
                                        reps=200, active_reset=False, relax_delay=2500,
                                        if_ef=False, postprocess=False)
geramsey_long.display()
save_figs("geramsey_long_12")
print("f_ge config", qcfg.f_ge[0], "| f_adjust (MHz)", geramsey_long.data.get("f_adjust_ramsey_avgi"),
      "| fit", geramsey_long.data.get("fit_avgi"))


# %% Cell 12b: apply the long Ramsey (2026-10-21, f_adjust -29.5 kHz, T2 440 us; cell 2 gave
# -29.6 kHz at 0.2 MHz virtual frequency, so the sign is certain) and snapshot. f_ef stays.
qcfg.f_ge[0] = float(qcfg.f_ge[0] + geramsey_long.data["f_adjust_ramsey_avgi"][0])
print("f_ge new", qcfg.f_ge[0])
hw_version = station.snapshot_hardware_config(update_main=False)
print("new version:", hw_version)


# %% Cell 13: MBR section 0 setup (202609_qsim_migration/mbr.py), no jobs: the campaign and its
# runners. Mode order M1, S4, S5, S6, S7. [2,0,0,0,0] is the main interest (M1 self-Kerr);
# [0,1,1,0,0] is the sanity check. batch_size=2 in 13b/14 sends both jobs together.
from experiments.qsim.notebook_helpers.defaults import MEASUREMENT_CONFIG_DEFAULTS
from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_stark_cal import MBRStarkCalExperiment
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
from experiments.qsim.mbr_campaign import build_campaign, campaign_runner, fixed_n_occupations

campaign = build_campaign(floquet_settings=FLOQUET_DEFAULTS,
                          active_reset_settings=ACTIVE_RESET_DEFAULTS,
                          measurement_settings=MEASUREMENT_CONFIG_DEFAULTS,
                          modes=WALK_STORS, reps=1000)


def runner_for(JobClass):
    """The runner for one MBR job class, over the campaign defaults."""
    return campaign_runner(campaign, station, client, JobClass, use_queue=True)


check_occupations = [[2, 0, 0, 0, 0], [0, 1, 1, 0, 0]]


# %% Cell 13b: StarkCal of the check occupations (2 jobs). 2026-10-21: JOB-20261021-00050/51,
# manifest 261021_041134; cell R loads it again.
print("mode order:", campaign.mode_labels, "occupations:", check_occupations)
check_calibration = MBRCalibrationSetExperiment(
    check_occupations,
    cycle_pairs=np.arange(0, 65, dtype=int),  # physical cycles: 0, 2, ..., 128
    swap_stors=campaign.modes,
    sync_cycles=campaign.sync_cycles,
    reps=1000,
)
check_calibration.acquire(runner_for(MBRStarkCalExperiment), batch_size=2, log=True, show=False)
check_calibration.analyze()
check_calibration.display()
save_figs("mbr_check_cal_13")
print("job ids:", check_calibration.job_ids)
print("phase deg/cycle:", {o: check_calibration.phase_for(o) for o in check_calibration.occupations})
print("manifest:", check_calibration.save())


# %% Cell 14: MBR section 0, time traces of the check occupations with the cell-13 calibration
# played on the pulse. Cycles 0, 2, ..., 298 (about 92 us).
check_trace = MBRSpectrumExperiment(
    check_occupations,
    np.arange(0, 300, 2),
    campaign.modes,
    calibration=check_calibration,
    sync_cycles=campaign.sync_cycles,
    reps=500,
)
check_trace.acquire(runner_for(MBRTimeTraceExperiment), batch_size=2, log=True, show=False)
check_trace.analyze()
check_trace.display_occupations(occupations=check_trace.occupations)
save_figs("mbr_check_trace_14")
print("job ids:", check_trace.job_ids)


# %% Cell 15: M1 self-Kerr from the [2,0,0,0,0] trace, model-light. Fit the return as three damped
# complex exponentials, then compare the two peak spacings (frame-free) with the fixed-N model
# at the saved couplings. The energy sign is the one of the cell-14 FFT plot.
from scipy.optimize import least_squares

kerr_row = [tuple(o) for o in check_trace.data.reconstruction.occupations].index((2, 0, 0, 0, 0))
kerr_return = np.asarray(check_trace.data.reconstruction.A[kerr_row])
with h5py.File(check_trace.children[kerr_row].fname, "r") as data_file:
    timing = json.loads(data_file.attrs["derived_params"])
trace_cycle_us = float(timing["floquet_cycle_us"])
trace_couplings_mhz = np.array([timing["couplings_MHz"][s - 1] for s in campaign.modes])
trace_time_us = np.asarray(check_trace.data.reconstruction.cycles, dtype=float) * trace_cycle_us
fft_energy = np.asarray(check_trace.data.spectrum.energy_MHz)
fft_local = np.asarray(check_trace.data.spectrum.measured_local[kerr_row])
fft_sign = -1.0  # the spectrum code plots A ~ exp(-2 pi i E t); a +1 fit gave mirrored lines


def three_lines(params, time_us):
    out = np.zeros_like(time_us, dtype=complex)
    for k in range(3):
        f, gamma, re, im = params[4 * k:4 * k + 4]
        out += (re + 1j * im) * np.exp((fft_sign * 2j * np.pi * f - gamma) * time_us)
    return out


start = []
for f0 in (-0.09, -0.01, 0.075):  # cell-14 FFT peaks
    start += [f0, 0.01, 0.2, 0.0]
fit = least_squares(lambda p: np.concatenate([(three_lines(p, trace_time_us) - kerr_return).real,
                                              (three_lines(p, trace_time_us) - kerr_return).imag]),
                    start)
jac = fit.jac
cov = np.linalg.pinv(jac.T @ jac) * np.sum(fit.fun ** 2) / (len(fit.fun) - len(fit.x))
line_f = fit.x[0::4]
line_df = np.sqrt(np.diag(cov))[0::4]
order = np.argsort(line_f)
line_f, line_df = line_f[order], line_df[order]
line_w = np.abs(fit.x[2::4] + 1j * fit.x[3::4])[order]
print("lines (MHz):", np.round(line_f, 5), "+/-", np.round(line_df, 5), "| |c|:", np.round(line_w, 3),
      "| gamma (1/us):", np.round(fit.x[1::4][order], 4))
spacing = np.diff(line_f)
spacing_err = np.sqrt(line_df[1:] ** 2 + line_df[:-1] ** 2)


def model_lines(kerr_khz, couplings_mhz=trace_couplings_mhz):
    model = fixed_n_hamiltonian(photon_number=2, mode_count=1 + len(campaign.modes),
                                detunings=np.zeros(len(campaign.modes)), couplings_MHz=couplings_mhz,
                                physical_kerr_MHz=kerr_khz / 1000)
    weight = np.abs(model.states[model.fock_index[(2, 0, 0, 0, 0)], :]) ** 2
    keep = weight > 0.02
    return np.sort(model.energies_MHz[keep]), weight[keep][np.argsort(model.energies_MHz[keep])]


kerr_grid = np.arange(-80.0, 80.01, 0.1)
chi2 = []
for kerr_khz in kerr_grid:
    energies, _ = model_lines(kerr_khz)
    chi2.append(np.sum(((np.diff(energies) - spacing) / spacing_err) ** 2) if len(energies) == 3 else np.nan)
chi2 = np.asarray(chi2)
best = kerr_grid[np.nanargmin(chi2)]
inside = kerr_grid[chi2 - np.nanmin(chi2) < 1]
print(f"spacings (kHz): {np.round(1e3 * spacing, 2)} +/- {np.round(1e3 * spacing_err, 2)}")
print(f"K best {best:.1f} kHz, delta-chi2<1: {inside.min():.1f}..{inside.max():.1f} kHz, chi2_min {np.nanmin(chi2):.2f} (1 dof)")
energies_best, weights_best = model_lines(best)
print("model lines at best K (MHz):", np.round(energies_best, 5), "weights", np.round(weights_best, 3),
      "| frame offset measured - model (kHz):", np.round(1e3 * (line_f - energies_best), 2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(trace_time_us, kerr_return.real, ".", ms=3, label="Re A")
axes[0].plot(trace_time_us, kerr_return.imag, ".", ms=3, label="Im A")
fine_t = np.linspace(0, trace_time_us.max(), 1500)
axes[0].plot(fine_t, three_lines(fit.x, fine_t).real, "C0-", lw=1)
axes[0].plot(fine_t, three_lines(fit.x, fine_t).imag, "C1-", lw=1)
axes[0].set(xlabel="time (us)", ylabel="return", title="[2,0,0,0,0], 3-line fit")
axes[0].legend()
axes[1].plot(fft_energy, np.abs(fft_local), "k-", lw=1, label="measured FFT")
for f in line_f:
    axes[1].axvline(f, color="C3", ls=":")
for f in energies_best:
    axes[1].axvline(f, color="C0", alpha=0.4, lw=3)
axes[1].set(xlim=(-0.2, 0.2), xlabel="energy (MHz)", title=f"lines (red) and model at K={best:.1f} kHz (blue)")
axes[2].plot(kerr_grid, chi2, "-")
axes[2].axvline(-27.1, color="gray", ls="--", label="config -27.1 kHz")
axes[2].set(xlabel="K / 2pi (kHz)", ylabel="chi2 of spacings", yscale="log", title="Kerr from line spacings")
axes[2].legend()
fig.tight_layout()
save_figs("mbr_kerr_15")


# %% Cell 15b: the same spacings with g and K both free (two spacings, two parameters: an exact
# solve, no dof left).
def spacing_residual(params):
    kerr_khz, g_scale = params
    energies, _ = model_lines(kerr_khz, trace_couplings_mhz * g_scale)
    return (np.diff(energies) - spacing) / spacing_err


joint = least_squares(spacing_residual, [best, 1.0])
joint_cov = np.linalg.pinv(joint.jac.T @ joint.jac)
print(f"g and K free: K = {joint.x[0]:.2f} +/- {np.sqrt(joint_cov[0, 0]):.2f} kHz, "
      f"g = {1e3 * trace_couplings_mhz[0] * joint.x[1]:.3f} +/- "
      f"{1e3 * trace_couplings_mhz[0] * np.sqrt(joint_cov[1, 1]):.3f} kHz "
      f"(saved {1e3 * trace_couplings_mhz[0]:.3f}), residual {np.round(joint.fun, 3)}")
energies_joint, weights_joint = model_lines(joint.x[0], trace_couplings_mhz * joint.x[1])
print("model lines (MHz):", np.round(energies_joint, 5), "weights", np.round(weights_joint, 3),
      "| measured", np.round(line_f, 5), "|c| normalized", np.round(line_w / line_w.sum(), 3))


# %% Cell R: restore after a lost kernel, no jobs. Run cells 1, 8 and 13 first (definitions only,
# station on the snapshots), then this cell: it loads from disk what cells 8b, 13b and 14 made.
# Checked 2026-10-22 against the values before the loss (same C, phases and traces).
import json
import h5py
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian  # cells 10 and 15 use these
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment

RUN_DATA = r"C:\experiments\261020_mbr_s4s7_n2\data"
CHECK_CAL_MANIFEST = (r"C:\experiments\261020_mbr_s4s7_n2\assembled_data"
                      r"\261021_041134_MBRCalibrationSetExperiment.yaml")
CHECK_TRACE_JOBS = ["JOB-20261021-00056", "JOB-20261021-00057"]

readout_expts = [QsimExperiment.from_h5file(rf"{RUN_DATA}\JOB-20261021-{job}_QsimExperiment.h5")
                 for job in ["00010", "00012", "00014", "00016"]]
readout_confusion = np.zeros((4, 4))
for readout_expt in readout_expts:
    multiparity = readout_expt.analyze_multiparity()
    readout_confusion[readout_expt.cfg.expt.occupations[0]] = [multiparity[f"p_mod{n}"][0] for n in range(4)]
print("C[prepared, measured]:\n", np.round(readout_confusion, 3))

check_calibration = MBRCalibrationSetExperiment.from_manifest(CHECK_CAL_MANIFEST)
check_calibration.analyze()
print("phase deg/cycle:", {o: check_calibration.phase_for(o) for o in check_calibration.occupations})

check_trace = MBRSpectrumExperiment.from_children(
    [MBRTimeTraceExperiment.from_h5file(rf"{RUN_DATA}\{job}_MBRTimeTraceExperiment.h5", load_shots=False)
     for job in CHECK_TRACE_JOBS],
    job_ids=CHECK_TRACE_JOBS, calibration=check_calibration)
check_trace.analyze()
print("check trace occupations:", check_trace.occupations)


# %% Cell 1c: daily single shot at the set point (avoid_yoko=False: the worker ramps to 0.25 mA).
# The postprocessor updates the readout in memory.
ss = ss_runner.execute(go_kwargs=dict(analyze=False, display=False), check_f=False,
                       relax_delay=2500, reps=5000, avoid_yoko=False, **ss_reset)
ss.display(station)
save_figs("ss_1c")


# %% Cell 16: [1,0,0,0,0] MBR (StarkCal + trace). N=1 has no Kerr: the two lines give g (span
# sqrt(4G^2 + d^2), G = 2g) and a common storage detuning d (weight asymmetry d / span).
one_photon = [[1, 0, 0, 0, 0]]
one_calibration = MBRCalibrationSetExperiment(
    one_photon, cycle_pairs=np.arange(0, 65, dtype=int), swap_stors=campaign.modes,
    sync_cycles=campaign.sync_cycles, reps=1000)
one_calibration.acquire(runner_for(MBRStarkCalExperiment), batch_size=1, log=True, show=False)
one_calibration.analyze()
one_calibration.display()
save_figs("mbr_one_cal_16")
print("StarkCal job:", one_calibration.job_ids, "phase deg/cycle:", one_calibration.phase_for((1, 0, 0, 0, 0)))
print("calibration manifest:", one_calibration.save())

one_trace = MBRSpectrumExperiment(
    one_photon, np.arange(0, 300, 2), campaign.modes, calibration=one_calibration,
    sync_cycles=campaign.sync_cycles, reps=500)
one_trace.acquire(runner_for(MBRTimeTraceExperiment), batch_size=1, log=True, show=False)
one_trace.analyze()
one_trace.display_occupations(occupations=one_trace.occupations)
save_figs("mbr_one_trace_16")
print("trace job:", one_trace.job_ids, "| trace manifest:", one_trace.save())


# %% Cell 16b: two-line fit of the [1,0,0,0,0] return (plot frame, as cell 15), then the 2-level
# model H = [[0, G], [G, d]]: span = sqrt(4G^2 + d^2), M1 weight of the upper line (1 - d/span)/2.
def damped_lines(params, time_us, count):
    out = np.zeros_like(time_us, dtype=complex)
    for k in range(count):
        f, gamma, re, im = params[4 * k:4 * k + 4]
        out += (re + 1j * im) * np.exp((-2j * np.pi * f - gamma) * time_us)
    return out


from scipy.optimize import least_squares

with h5py.File(one_trace.children[0].fname, "r") as data_file:
    one_timing = json.loads(data_file.attrs["derived_params"])
trace_cycle_us = float(one_timing["floquet_cycle_us"])
trace_couplings_mhz = np.array([one_timing["couplings_MHz"][s - 1] for s in campaign.modes])
one_return = np.asarray(one_trace.data.reconstruction.A[0])
one_time_us = np.asarray(one_trace.data.reconstruction.cycles, dtype=float) * trace_cycle_us
one_fit = least_squares(
    lambda p: np.concatenate([(damped_lines(p, one_time_us, 2) - one_return).real,
                              (damped_lines(p, one_time_us, 2) - one_return).imag]),
    [-0.04, 0.01, 0.3, 0.0, 0.04, 0.01, 0.3, 0.0])
one_cov = np.linalg.pinv(one_fit.jac.T @ one_fit.jac) * np.sum(one_fit.fun ** 2) / (len(one_fit.fun) - 8)
rng = np.random.default_rng(0)
draws = rng.multivariate_normal(one_fit.x, one_cov, size=4000)


def two_level(params):
    f = params[0::4]
    w = np.abs(params[2::4] + 1j * params[3::4])
    lower, upper = np.argsort(f)
    span = f[upper] - f[lower]
    w_lower, w_upper = w[lower] / w.sum(), w[upper] / w.sum()
    d = span * (w_lower - w_upper)
    g = np.sqrt(span ** 2 - d ** 2) / 4
    return np.array([f[lower], f[upper], w_lower, w_upper, 1e3 * span, 1e3 * d, 1e3 * g])


best_values = two_level(one_fit.x)
errors = np.std([two_level(p) for p in draws], axis=0)
names = ["f_lower MHz", "f_upper MHz", "w_lower", "w_upper", "span kHz", "d kHz", "g kHz"]
for name, value, error in zip(names, best_values, errors):
    print(f"{name:12s} {value:+.5f} +/- {error:.5f}")
print(f"saved g {1e3 * trace_couplings_mhz[0]:.3f} kHz | gamma (1/us): {np.round(one_fit.x[1::4], 4)}")
print("2-photon predictions: (g fixed, d free) d = +3.8 kHz; (g free, d = 0) g = 20.530 kHz")
