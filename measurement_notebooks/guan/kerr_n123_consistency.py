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
# # M1 Kerr at N = 1, 2, 3 and the detuning sign, one config (guan + Claude, from 2026-10-23)
#
# Lab book of one overnight campaign. Central boson: M1 in a star with S4..S7, equal g,
# M1 self-Kerr only. Coupler 0.25 mA.
#
# Why: the 10-21 MBR of [2,0,0,0,0] gave K = -20 kHz (`mbr_s4s7_n2.py` cell 15). The
# 10-17 quantum walks gave a different K at N = 3 (the -28 kHz of the walks came from
# fitting N = 2 and N = 3 together). The readout is new since then (750.1 MHz, gain 2300,
# 1.5 us). Here all data use one config snapshot:
# 1. checks: single shot, qubit Ramsey, sidebands, M1-S swaps, Floquet, broadband pi;
# 2. multiparity readout confusion;
# 3. quantum walks from [n,0,0,0,0], n = 1, 2, 3, every mode read;
# 4. MBR (StarkCal + trace) from [n,0,0,0,0], n = 1, 2, 3;
# 5. detuning sign: walks with one disorder pattern at +W and -W (K < 0 fixes the sign);
# 6. undriven M1 Kerr and higher order from the cavity Ramsey gain sweep.
#
# Driven cell by cell into a separate kernel (connection file claude-kerr.json, not
# guan-meas). Jobs go through the queue, interleaved with jonginn and seb. Config changes
# stay in memory; snapshots only with update_main=False. Notes go into the vault entries
# (guan / KerrCheck).

# %% Cell 1: station, helpers, runners. No jobs.
# Base: jonginn's 10-23 versions at 0.25 mA (his N=3 AGP jobs of 10-23 evening use them):
# newest Floquet (10-21 20:20) and M1-S swap (10-21 19:59) calibrations, the new readout.
# His pi_ge_broadband is still the pre-warm-up value (3566.204 MHz, gain 18699), which MBR
# needs. Start from guan's checked value (CFG-HW-20261023-00039: 3563.862 MHz, 18447);
# cell 7 checks it again.
import glob
import os
import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import experiments as meas
from slab import AttrDict
from experiments import CharacterizationRunner, MultimodeStation
from job_server import JobClient

config_dict = {
    "hardware_config": "CFG-HW-20261023-00038",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261023-00003",
    "floquet_storage_swap": "CFG-FL-20261023-00002",
}

client = JobClient()
client.print_queue()

station = MultimodeStation(
    user="guan",
    experiment_name="261023_kerr_n123",
    project="KerrCheck",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)
qcfg = station.hardware_cfg.device.qubit
bb = qcfg.pulses.pi_ge_broadband
bb.frequency = [3563.862277402883]
bb.gain = [18447]
ro = station.hardware_cfg.device.readout
print("readout:", ro.frequency, ro.gain, ro.readout_length, "| coupler (A):",
      station.hardware_cfg.hw.yoko_coupler.current, "| broadband:", bb.frequency, bb.gain)

FIGURES = r"C:\experiments\261023_kerr_n123\claude_work\figures"
os.makedirs(FIGURES, exist_ok=True)


def save_figs(tag):
    """Save the open figures as PNGs (for the CC Board) and close them."""
    paths = []
    for k, num in enumerate(plt.get_fignums()):
        path = os.path.join(FIGURES, f"{tag}_{k}.png")
        plt.figure(num).savefig(path, dpi=80, bbox_inches="tight")
        paths.append(path)
    plt.close("all")
    print("saved", paths)
    return paths


VAULT_DIR = os.path.join(str(station.vault_root), "Lab", station.user, station.project)
PLACEHOLDER = "_add interpretation here_"


def vault_note(key, text):
    """Write `text` under the Notes of the vault entry whose parameters contain `key`
    (a job ID or a file name). Replaces the placeholder, or appends to earlier notes."""
    for path in sorted(glob.glob(os.path.join(VAULT_DIR, "*", "*", "*.md")), reverse=True):
        content = open(path, encoding="utf-8").read()
        at = content.find(key)
        if at < 0:
            continue
        notes = content.find("### Notes\n\n", at)
        end = content.find("\n---\n", notes)
        if notes < 0 or end < 0:
            continue
        body = content[notes + len("### Notes\n\n"):end]
        stamp = f"*Claude, {np.datetime64('now')}*: "
        new_body = (stamp + text.strip() + "\n") if body.strip() == PLACEHOLDER else (
            body.rstrip() + "\n\n" + stamp + text.strip() + "\n")
        content = content[:notes + len("### Notes\n\n")] + new_body + content[end:]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        print("vault note ->", path)
        return path
    print("vault note: no entry has", key)
    return None


def vault_entry(title, notes, fig=None):
    """A separate vault entry (analysis or plan), with figures if given."""
    return station.log_measurement(title=title, notes=notes, fig=fig, parameters={})


def job_id(expt):
    match = re.search(r"JOB-\d{8}-\d{5}", str(getattr(expt, "fname", "")))
    return match.group(0) if match else None


# Single shot (as mbr_s4s7_n2.py cells 1, 1c). The worker ramps the coupler to 0.25 mA.
ss_reset = dict(active_reset=False, ef_reset=True, storage_reset=False, coupler_reset=False,
                pre_selection_reset=True, man_reset=False, use_qubit_man_reset=True,
                pre_selection_parity=False)
singleshot_defaults = AttrDict(dict(
    reps=5000, relax_delay=2500, check_f=False, active_reset=False, man_reset=False,
    storage_reset=False, qubit=0, pulse_manipulate=False, cavity_freq=4984.373226159381,
    cavity_gain=400, cavity_length=2, prepulse=False, pre_sweep_pulse=None, gate_based=True,
    qubits=[0], avoid_yoko=False,
))


def singleshot_postproc(station, expt):
    expt.analyze(station=station, subdir=station.autocalib_path)
    station.hardware_cfg.device.readout.update(expt.readout_cfg)


ss_runner = CharacterizationRunner(
    station=station, ExptClass=meas.HistogramExperiment, default_expt_cfg=singleshot_defaults,
    postprocessor=singleshot_postproc, job_client=client,
)

# %% Cell 2: single shot at 0.25 mA; the postprocessor sets phase, threshold, Ig, Ie in memory.
ss = ss_runner.execute(go_kwargs=dict(analyze=False, display=False), **ss_reset)
ss.display(station)
save_figs("ss_02")
print("job:", job_id(ss), ss.fname)
# 2026-10-23 23:09: JOB-20261023-00524, fidelity 0.953, angle -2.49 deg (phase 467.71 -> 465.22).


# %% Cell 3: runners of the checks (as mbr_s4s7_n2.py cells 2-6), then submit all 14 checks
# at once so they wait in the queue together. Measure only; cell 4 fits, cell 5 applies.
geramsey_defaults = AttrDict(dict(
    ramsey_freq=0.2, start=0.01, step=0.4, expts=101, reps=200, rounds=1,
    pre_sweep_pulse=None, post_sweep_pulse=None, if_ef=False, ef_init=True, qubits=[0],
    user_defined_freq=[False, 3568.2038290468167, 5304, 0.035], f0g1_cavity=0,
    normalize=False, active_reset=False, man_reset=False, storage_reset=False,
    prepulse=None, postpulse=None, pre_active_reset_pulse=False, gate_based=False,
    advance_phase=0, echoes=[False, 0], relax_delay=2500,
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


ramsey_runner = CharacterizationRunner(station=station, ExptClass=meas.RamseyExperiment,
                                       default_expt_cfg=geramsey_defaults,
                                       preprocessor=geramsey_preproc, job_client=client)

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
error_amp_stor_runner = CharacterizationRunner(
    station=station, ExptClass=meas.ErrorAmplificationExperiment, job_client=client,
    default_expt_cfg=AttrDict(dict(
        reps=50, rounds=1, qubits=[0], man_mode_no=1, stor_is_dump=False, active_reset=False,
        man_reset=True, storage_reset=True, reset_dump_mode=2, dump_reset_iter_num=1,
        relax_delay=2500, expts=30, n_start=0, n_step=1, n_pulses=5)))
error_amp_floquet_runner = CharacterizationRunner(
    station=station, ExptClass=meas.ErrorAmplificationExperiment, job_client=client,
    default_expt_cfg=AttrDict(dict(
        reps=100, rounds=1, qubits=[0], active_reset=False, man_mode_no=1, stor_is_dump=False,
        man_reset=True, storage_reset=True, relax_delay=2500, expts=50, n_pulses=7,
        qubit_start_storage="g", floquet_sync_delay=1, reset_dump_mode=2, dump_reset_iter_num=1,
        include_10cycles_buffer=True, include_10cycles_buffer_in_pi_half=True,
        floquet_hardware_loop=True, scramble_sync_cycles=1, palindrome_scramble=False)))

STORS = [4, 5, 6, 7]
mp_pi = station.hardware_cfg.device.multiphoton.pi["fn-gn+1"]
checks = {}  # name -> (runner, config value, job ID)


def submit(name, runner, config_value, **kwargs):
    checks[name] = (runner, config_value, runner._submit(**kwargs))
    print(name, checks[name][2])


submit("ge Ramsey", ramsey_runner, qcfg.f_ge[0], if_ef=False)
submit("ef Ramsey", ramsey_runner, qcfg.f_ef[0], if_ef=True, ef_init=True, step=0.2, active_reset=True)
for n, sideband in enumerate(["f0-g1", "f1-g2", "f2-g3"]):
    band, expts = 0.3, 50
    submit(sideband, error_amp_runner, mp_pi["frequency"][n], transition=sideband,
           parameter_to_test="frequency", start=mp_pi["frequency"][n] - band, step=2 * band / expts,
           expts=expts, n_pulses=7, pulse_type=["multiphoton", sideband, "pi", 0])
for stor, span in zip(STORS, [0.3, 0.3, 0.3, 0.2]):
    name, expts = f"M1-S{stor}", 100
    freq = station.ds_storage.get_freq(name)
    submit(f"swap {name}", error_amp_stor_runner, freq, stor_mode_no=stor,
           parameter_to_test="frequency", pulse_type=["storage", name, "pi", 0],
           start=freq - span / 2, step=span / (expts - 1), expts=expts, active_reset=True,
           man_reset=True, storage_reset=[stor], relax_delay=1000)
for stor in STORS:
    name, span, expts = f"M1-S{stor}", 0.05, 50
    freq = station.ds_floquet.get_freq(name)
    submit(f"Floquet {name}", error_amp_floquet_runner, freq, stor_mode_no=stor,
           parameter_to_test="frequency",
           pulse_type=["floquet", name, f"pi/{station.ds_floquet.get_pi_frac(name)}", 0],
           start=freq - span / 2, step=span / (expts - 1), expts=expts, active_reset=True,
           man_reset=True, storage_reset=[stor], relax_delay=200)
submit("broadband freq", error_amp_runner, bb.frequency[0], parameter_to_test="frequency",
       start=3559.5, step=0.2, expts=41, n_pulses=10, reps=100,
       pulse_type=["qubit", "ge_broadband", "pi", 0])


# %% Cell 4: collect the checks and fit them (jonginn's fits: state_fin='e' for error
# amplification; the Ramsey f_adjust). Prints config value, fit, difference.
check_results = {}
for name, (runner, config_value, job) in checks.items():
    expt = runner._collect(job, postprocess=False, show=False)
    if "Ramsey" in name:
        fitted = config_value + expt.data["f_adjust_ramsey_avgi"][0]
        extra = f"T2 fit {expt.data['fit_avgi'][3]:.1f} us"
    else:
        state_fin = None if name.startswith("broadband") else "e"
        kwargs = dict(state_fin=state_fin) if state_fin else {}
        expt.analyze(data=expt.data, **kwargs)
        expt.display(data=expt.data, **kwargs)
        fitted = expt.data["fit_avgi"][2]
        extra = ""
    save_figs(f"check_{name.replace(' ', '_')}")
    check_results[name] = (config_value, float(fitted), job)
    print(f"{name:18s} {job}: config {config_value:.4f}, fit {fitted:.4f} MHz, "
          f"difference {1e3 * (fitted - config_value):+.1f} kHz {extra}")
# 2026-10-23 23:47 - 10-24 00:06, JOB-20261023-00530..00543 (kHz, fit - config):
# ge Ramsey +3.3 (T2 460 us), ef Ramsey -3.9; f0-g1 -57.0, f1-g2 -37.4, f2-g3 -4.5;
# swaps S4..S7 -5.8, -1.7, -4.8, +0.7; Floquet S4..S7 +1.0, +0.2, -0.0, +0.8;
# broadband frequency -165 (less than the 200 kHz scan step: not significant).


# %% Cell 5: apply in memory: f_ge, f_ef, the three sidebands (and the ds M1 row), the four
# swaps. Kept: Floquet (all within 1 kHz) and the broadband frequency (shift below the scan
# step). Then the broadband gain check (as mbr_s4s7_n2.py cell 6b) and the snapshot.
qcfg.f_ge[0] = check_results["ge Ramsey"][1]
qcfg.f_ef[0] = check_results["ef Ramsey"][1]
for n, sideband in enumerate(["f0-g1", "f1-g2", "f2-g3"]):
    mp_pi["frequency"][n] = check_results[sideband][1]
station.ds_storage.update_freq("M1", check_results["f0-g1"][1])
for stor in STORS:
    station.ds_storage.update_freq(f"M1-S{stor}", check_results[f"swap M1-S{stor}"][1])
print("f_ge", qcfg.f_ge[0], "f_ef", qcfg.f_ef[0], "| sidebands", mp_pi["frequency"][:3],
      "| ds M1", station.ds_storage.get_freq("M1"),
      "| swaps", [station.ds_storage.get_freq(f"M1-S{stor}") for stor in STORS])

gain0 = bb.gain[0]
bb_gain = error_amp_runner.execute(
    parameter_to_test="gain", start=int(gain0 * 0.7), step=int(gain0 * 0.6 / 30), expts=31,
    n_pulses=10, reps=100, pulse_type=["qubit", "ge_broadband", "pi", 0], postprocess=False)
bb_gain.analyze(data=bb_gain.data)
bb_gain.display(data=bb_gain.data)
save_figs("check_broadband_gain_05")
bb.gain = [int(round(bb_gain.data["fit_avgi"][2]))]
print(f"broadband gain: config {gain0}, error amp. {bb.gain[0]} ({job_id(bb_gain)})")
snapshot_versions = dict(
    hardware_config=station.snapshot_hardware_config(update_main=False),
    man1_storage_swap=station.snapshot_man1_storage_swap(update_main=False),
    multiphoton_config=config_dict["multiphoton_config"],
    floquet_storage_swap=config_dict["floquet_storage_swap"],
)
print("snapshot:", snapshot_versions)


# 2026-10-24 00:37: broadband gain 18447 -> 18387 (JOB-20261024-00001). Snapshot of the checked
# config: HW CFG-HW-20261024-00002, M1 CFG-M1-20261024-00001, FL CFG-FL-20261023-00002,
# MP CFG-MP-20260121-00001. All data below use it.


# %% Cell 6: population and readout-calibration runners (as agp_layers_hw_check.py cells 1-2),
# then the multiparity readout confusion in one job: Fock 0..3 in M1, read M1 (qubit + M,
# active reset: relax_delay=500 us). C[prepared, measured]; the population jobs carry it.
from experiments.qsim.fock_population import FockPopulationExperiment, FockPopulationSetExperiment
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS

population_defaults = AttrDict(dict(
    expts=1, reps=100, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=STORS, pre_relax_delay=100, relax_delay=200,
    prepulse=False, postpulse=True, init_fock=False, init_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    occupations=[1, 0, 0, 0, 0], swap_stors=STORS,
    use_multiphoton_swap=False, storage_pulse_wait_us=0.2, ro_stor=0,
    update_phases=True, detunings=[0.0] * len(STORS),
    floquet_cycles=[0], swept_params=["floquet_cycle"],
))
population_defaults.update(ACTIVE_RESET_DEFAULTS)
population_defaults.update(FLOQUET_DEFAULTS)
population_runner = CharacterizationRunner(
    station=station, ExptClass=FockPopulationExperiment, default_expt_cfg=population_defaults,
    job_client=client, use_queue=True, show=False,
)
POPULATION_ONLY = ("expts", "occupations", "use_multiphoton_swap", "storage_pulse_wait_us",
                   "floquet_cycles", "swept_params")
calibration_defaults = AttrDict({key: value for key, value in population_defaults.items()
                                 if key not in POPULATION_ONLY})
calibration_defaults.update(prepulse=True)
calibration_runner = CharacterizationRunner(
    station=station, ExptClass=MultiParityConfusionMatrixExperiment,
    default_expt_cfg=calibration_defaults, job_client=client, use_queue=True, show=False,
)
readout_calibration = calibration_runner.execute(reps=6000, relax_delay=500)
readout_confusion = np.asarray(readout_calibration.data["multiparity_confusion_matrix"])
CONFUSION_FNAME = str(readout_calibration.fname)
print("job:", job_id(readout_calibration), CONFUSION_FNAME)
print("C[prepared, measured]:\n", np.round(readout_confusion, 3),
      "\ncondition number", np.linalg.cond(readout_confusion))
readout_calibration.display()
save_figs("confusion_06")


# %% Cell 7: quantum walks from [n,0,0,0,0], n = 1, 2, 3, every mode read, zero detunings.
# Cycles 0..398 by 2 (about 122 us, as the 10-17 walks), 600 reps. All 15 jobs in one batch
# (fewer gaps for other users' jobs to fall in), then one population set per n.
WALK_CYCLES = list(range(0, 400, 2))
READ_MODES = [0] + STORS


def acquire_walk_sets(plans, reps=600):
    """plans: {key: (occupation, detunings, notes)} -> {key: analyzed FockPopulationSetExperiment}.
    One batch for all jobs; each set is then assembled from its own jobs and saved."""
    sets = {key: FockPopulationSetExperiment([(occupation, mode) for mode in READ_MODES], WALK_CYCLES,
                                             STORS, detunings, calibration=CONFUSION_FNAME, notes=notes)
            for key, (occupation, detunings, notes) in plans.items()}
    overrides = [override for walk in sets.values() for override in walk.job_overrides()]
    children = population_runner.execute(overrides=overrides, batch_size=len(overrides), reps=reps,
                                         show=False)
    job_ids = list(population_runner.last_job_ids)
    out = {}
    for k, (key, walk) in enumerate(sets.items()):
        part = slice(k * len(READ_MODES), (k + 1) * len(READ_MODES))
        assembled = FockPopulationSetExperiment.from_children(children[part], job_ids=job_ids[part],
                                                              notes=walk.notes)
        assembled.analyze()
        print(key, "jobs", assembled.job_ids, "| manifest", assembled.save())
        assembled.display()
        save_figs(f"walk_{key}")
        out[key] = assembled
    return out


walks = acquire_walk_sets({
    f"n{photons}": ([photons, 0, 0, 0, 0], [0.0] * 4, f"walk N={photons} from [{photons},0,0,0,0], all modes read")
    for photons in (1, 2, 3)})


# 2026-10-24 01:39-02:49 (cells 6, 7): confusion JOB-20261024-00008, diagonal 0.932 / 0.870 /
# 0.847 / 0.828, condition number 1.33. Walks JOB-20261024-00012..26, manifests
# assembled_data/261024_024939_FockPopulationSetExperiment{,_2,_3}.yaml (n = 1, 2, 3).


# %% Cell 8: MBR from [n,0,0,0,0], n = 1, 2, 3 (as mbr_s4s7_n2.py cells 13-14): StarkCal set
# (3 jobs, cycle pairs 0..64), then the time traces with that calibration (3 jobs, cycles
# 0..298 by 2, about 92 us).
from experiments.qsim.notebook_helpers.defaults import MEASUREMENT_CONFIG_DEFAULTS
from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_stark_cal import MBRStarkCalExperiment
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
from experiments.qsim.mbr_campaign import build_campaign, campaign_runner

campaign = build_campaign(floquet_settings=FLOQUET_DEFAULTS, active_reset_settings=ACTIVE_RESET_DEFAULTS,
                          measurement_settings=MEASUREMENT_CONFIG_DEFAULTS, modes=STORS, reps=1000)


def runner_for(JobClass):
    return campaign_runner(campaign, station, client, JobClass, use_queue=True)


mbr_occupations = [[1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]
mbr_calibration = MBRCalibrationSetExperiment(
    mbr_occupations, cycle_pairs=np.arange(0, 65, dtype=int), swap_stors=campaign.modes,
    sync_cycles=campaign.sync_cycles, reps=1000)
mbr_calibration.acquire(runner_for(MBRStarkCalExperiment), batch_size=3, show=False)
mbr_calibration.analyze()
mbr_calibration.display()
save_figs("mbr_cal_08")
print("StarkCal jobs:", mbr_calibration.job_ids, "| manifest:", mbr_calibration.save())
print("phase deg/cycle:", {o: mbr_calibration.phase_for(o) for o in mbr_calibration.occupations})

mbr_trace = MBRSpectrumExperiment(
    mbr_occupations, np.arange(0, 300, 2), campaign.modes, calibration=mbr_calibration,
    sync_cycles=campaign.sync_cycles, reps=500)
mbr_trace.acquire(runner_for(MBRTimeTraceExperiment), batch_size=3, show=False)
print("trace jobs:", mbr_trace.job_ids)
# 2026-10-24 03:40-04:47: StarkCal JOB-20261024-00030..32 (manifest 261024_034032), traces
# JOB-20261024-00036..38. The acquisition is one batch, but MBRSpectrumExperiment.analyze()
# takes one photon-number sector only (it raised a ValueError here): one spectrum per n.
mbr_traces = {}
for occupation, child, job in zip(mbr_occupations, mbr_trace.children, mbr_trace.job_ids):
    n = sum(occupation)
    mbr_traces[n] = MBRSpectrumExperiment.from_children([child], job_ids=[job], calibration=mbr_calibration)
    mbr_traces[n].analyze()
    mbr_traces[n].display_occupations(occupations=mbr_traces[n].occupations)
    print(f"N={n}: manifest {mbr_traces[n].save()}")
save_figs("mbr_trace_08")


# %% Cell 9: detuning sign. Walks from [2,0,0,0,0] with the 10-17 disorder pattern (unit RMS)
# at W = +30 kHz and W = -30 kHz, and from [3,0,0,0,0] at +30 kHz; every mode read; one batch.
# Populations are the same for (d, K) and (-d, -K), and K < 0: the sign convention under
# which the fitted K is negative is the right one. Model (10-17 values): the two conventions
# differ by 0.1-0.6 in <n> per mode here; shot noise is about 0.03.
PATTERN = np.array([-0.70675335, 0.19463085, -1.03457499, 1.54669749])
PATTERN = (PATTERN - PATTERN.mean()) / np.sqrt(np.mean((PATTERN - PATTERN.mean()) ** 2))
SIGN_W_KHZ = 30.0
sign_walks = acquire_walk_sets({
    f"n{photons}_w{w_khz:+.0f}": ([photons, 0, 0, 0, 0], list(w_khz / 1000 * PATTERN),
                                  f"sign walk N={photons}, W={w_khz:+.0f} kHz x 10-17 pattern, all modes read")
    for photons, w_khz in [(2, SIGN_W_KHZ), (2, -SIGN_W_KHZ), (3, SIGN_W_KHZ)]})
print("detunings (kHz) at W=+30:", np.round(SIGN_W_KHZ * PATTERN, 2))


# 2026-10-24 05:57: sign walks JOB-20261024-00042..56; manifests
# 261024_055732_FockPopulationSetExperiment.yaml (n2 +30), 261024_055733_...yaml (n2 -30),
# 261024_055733_..._2.yaml (n3 +30). Detunings at W = +30 kHz: [-21.2, 5.84, -31.04, 46.4] kHz.


# %% Cell 10: systematic check of the relax delay: the [2,0,0,0,0] walk read in M1 with
# relax_delay = 2500 us instead of 200 us (residual storage photons would show here).
relax_check = population_runner.execute(
    occupations=[2, 0, 0, 0, 0], ro_stor=0, floquet_cycles=WALK_CYCLES, swap_stors=STORS,
    detunings=[0.0] * 4, multiparity_confusion_matrix=CONFUSION_FNAME, reps=600, relax_delay=2500)
print("job:", job_id(relax_check))


# 2026-10-24 07:41: JOB-20261024-00065 vs 00017 (200 us): mean difference +0.014 photons, rms 0.071
# (0.057 from shots). No sign of residual photons at 200 us (claude_work/relax_compare.py).


# %% Cell 11: walk model with decay (Lindblad, all photon numbers <= N; M1 Kerr
# K/2 n(n-1) + K3/6 n(n-1)(n-2); storage onsite energy sign * detuning, where
# fixed_n_hamiltonian assumes sign = -1). Copy of claude_work/walk_model.py, checked against
# fixed_n_hamiltonian to 1e-13 without decay.
import json
from itertools import product

import h5py

from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import expm_multiply


class WalkModel:
    def __init__(self, max_photons, storage_count):
        self.mode_count = 1 + storage_count
        self.states = [s for s in product(range(max_photons + 1), repeat=self.mode_count)
                       if sum(s) <= max_photons]
        self.index = {s: k for k, s in enumerate(self.states)}
        self.dim = len(self.states)
        self.a = []
        for mode in range(self.mode_count):
            op = np.zeros((self.dim, self.dim))
            for k, s in enumerate(self.states):
                if s[mode] > 0:
                    t = list(s)
                    t[mode] -= 1
                    op[self.index[tuple(t)], k] = np.sqrt(s[mode])
            self.a.append(op)
        self.numbers = np.asarray(self.states, dtype=float)
        totals = self.numbers.sum(axis=1)
        # vec(rho) column-stacked, element (i, j) at j * dim + i; keep the blocks of equal
        # photon number on both sides (closed under H and loss; a Fock start lives there).
        self.block_index = np.array([j * self.dim + i for j in range(self.dim)
                                     for i in range(self.dim) if totals[i] == totals[j]])
        self.block_position = {v: p for p, v in enumerate(self.block_index)}
        self.diag_positions = np.array([self.block_position[i * self.dim + i] for i in range(self.dim)])

    def hamiltonian(self, couplings_mhz, detunings_mhz, kerr_mhz, kerr3_mhz=0.0, sign=-1.0):
        n_m = self.numbers[:, 0]
        diag = kerr_mhz / 2 * n_m * (n_m - 1) + kerr3_mhz / 6 * n_m * (n_m - 1) * (n_m - 2)
        diag = diag + self.numbers[:, 1:] @ (sign * np.asarray(detunings_mhz, dtype=float))
        h = np.diag(diag)
        for i, g in enumerate(couplings_mhz):
            hop = self.a[0].T @ self.a[1 + i]
            h = h + g * (hop + hop.T)
        return h

    def mean_numbers_many(self, occupations, time_us, couplings_mhz, detunings_mhz, kerr_mhz,
                          kerr3_mhz=0.0, kappa_m=0.0, kappa_s=0.0, sign=-1.0, gamma_phi=0.0):
        """<n_mode>(t), shape (states, times, modes); time_us evenly spaced from 0."""
        h = sparse.csr_matrix(2 * np.pi * self.hamiltonian(couplings_mhz, detunings_mhz, kerr_mhz,
                                                            kerr3_mhz, sign))
        eye = sparse.identity(self.dim, format="csr")
        liouville = -1j * (sparse.kron(eye, h) - sparse.kron(h.T, eye))  # vec(A rho B) = (B^T x A) vec(rho)
        for rate, op in zip([kappa_m] + [kappa_s] * (self.mode_count - 1), self.a):
            if rate:
                op = sparse.csr_matrix(op)
                odo = op.T @ op
                liouville = liouville + rate * (sparse.kron(op, op) - 0.5 * sparse.kron(eye, odo)
                                                - 0.5 * sparse.kron(odo.T, eye))
        if gamma_phi:  # M1 pure dephasing, collapse operator sqrt(gamma_phi) n_M1
            n_op = sparse.diags(self.numbers[:, 0])
            liouville = liouville + gamma_phi * (sparse.kron(n_op, n_op) - 0.5 * sparse.kron(eye, n_op @ n_op)
                                                 - 0.5 * sparse.kron(n_op @ n_op, eye))
        keep = self.block_index
        liouville = sparse.csr_matrix(liouville)[keep][:, keep].tocsc()
        vecs = np.zeros((len(keep), len(occupations)), dtype=complex)
        for column, occupation in enumerate(occupations):
            k = self.index[tuple(occupation)]
            vecs[self.block_position[k * self.dim + k], column] = 1.0
        times = np.asarray(time_us, dtype=float)
        states = expm_multiply(liouville, vecs, start=times[0], stop=times[-1], num=len(times),
                               endpoint=True)
        return np.einsum("tds,dm->stm", np.real(states[:, self.diag_positions, :]), self.numbers)


MODELS = {n: WalkModel(n, 4) for n in (1, 2, 3)}
KAPPA_S = 1e-3  # 1/us, storage loss (T1 about 1 ms); M1 loss is fitted


def walk_arrays(walk):
    """(occupation, time_us, read-mode columns, corrected <n>, n_err, detunings, couplings)."""
    data = walk.data
    child = walk.children[0]
    derived = json.loads(h5py.File(child.fname, "r").attrs["derived_params"])
    couplings = np.array([derived["couplings_MHz"][s - 1] for s in data.swap_stors])
    order = [0] + list(data.swap_stors)
    return dict(occupation=tuple(data.occupations[0]), time=np.asarray(data.time_us),
                cols=[order.index(m) for m in data.ro_stor], mean=np.asarray(data.n_mean_corrected),
                err=np.maximum(np.asarray(data.n_err), 0.01), detunings=np.asarray(data.detunings),
                couplings=couplings)


def fit_walks(sets, start, fixed=None, sign=-1.0):
    """Least squares over walk sets. Parameters: K, K3 (kHz), g_scale, kappa_m, gamma_phi
    (1/us; M1 loss and M1 pure dephasing). Returns ({name: (value, error)}, reduced chi2,
    best predictions)."""
    fixed = dict(dict(K=0.0, K3=0.0, g_scale=1.0, kappa_m=0.005, gamma_phi=0.0), **(fixed or {}))
    names = list(start)

    def predictions(values):
        p = dict(fixed, **dict(zip(names, values)))
        out = []
        for s in sets:
            n = sum(s["occupation"])
            pred = MODELS[n].mean_numbers_many(
                [s["occupation"]], s["time"], s["couplings"] * p["g_scale"], s["detunings"],
                p["K"] / 1e3, p["K3"] / 1e3, p["kappa_m"], KAPPA_S, sign, p["gamma_phi"])[0]
            out.append(pred[:, s["cols"]].T)
        return out

    def residuals(values):
        return np.concatenate([((pred - s["mean"]) / s["err"]).ravel()
                               for pred, s in zip(predictions(values), sets)])

    result = least_squares(residuals, list(start.values()),
                           x_scale=[abs(v) if v else 1.0 for v in start.values()])
    dof = len(result.fun) - len(names)
    chi2 = np.sum(result.fun ** 2) / dof
    errors = np.sqrt(np.diag(np.linalg.pinv(result.jac.T @ result.jac) * chi2))
    return dict(zip(names, zip(result.x, errors))), chi2, predictions(result.x)


def show(params):
    return ", ".join(f"{k} = {v:.4g} +/- {e:.2g}" for k, (v, e) in params.items())


# %% Cell 12: MBR lines. For each [n,0,0,0,0] return A(t): fit n+1 damped lines (start values:
# the largest FFT peaks), then the line spacings against fixed_n_hamiltonian
# (mbr_s4s7_n2.py cells 15, 15b). Spacings do not depend on the frame (StarkCal phase).
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian

FFT_SIGN = -1.0  # the spectrum code plots A ~ exp(-2 pi i E t) (mbr_s4s7_n2.py cell 15)


def damped_lines(params, time_us):
    out = np.zeros_like(time_us, dtype=complex)
    for k in range(len(params) // 4):
        f, gamma, re, im = params[4 * k:4 * k + 4]
        out += (re + 1j * im) * np.exp((FFT_SIGN * 2j * np.pi * f - gamma) * time_us)
    return out


def fit_lines(trace, occupation, count=None, extra_starts=()):
    """count: lines started at the largest FFT peaks; extra_starts: more start
    frequencies (MHz), for a weak line below the noise peaks."""
    row = [tuple(o) for o in trace.data.reconstruction.occupations].index(tuple(occupation))
    values = np.asarray(trace.data.reconstruction.A[row])
    derived = json.loads(h5py.File(trace.children[row].fname, "r").attrs["derived_params"])
    time_us = np.asarray(trace.data.reconstruction.cycles, dtype=float) * float(derived["floquet_cycle_us"])
    couplings = np.array([derived["couplings_MHz"][s - 1] for s in campaign.modes])
    energy = np.asarray(trace.data.spectrum.energy_MHz)
    power = np.abs(np.asarray(trace.data.spectrum.measured_local[row]))
    count = count or sum(occupation) + 1  # lines to fit; the weakest N=3 line has weight ~0.05
    peaks = [k for k in range(1, len(power) - 1) if power[k] >= power[k - 1] and power[k] >= power[k + 1]]
    peaks = sorted(peaks, key=lambda k: -power[k])[:count]
    start = []
    for f0 in [energy[k] for k in peaks] + list(extra_starts):
        start += [f0, 0.01, 0.3, 0.0]
    fit = least_squares(lambda p: np.concatenate([(damped_lines(p, time_us) - values).real,
                                                  (damped_lines(p, time_us) - values).imag]), start)
    cov = np.linalg.pinv(fit.jac.T @ fit.jac) * np.sum(fit.fun ** 2) / (len(fit.fun) - len(fit.x))
    order = np.argsort(fit.x[0::4])
    return dict(occupation=tuple(occupation), time=time_us, values=values, params=fit.x,
                f=fit.x[0::4][order], df=np.sqrt(np.diag(cov))[0::4][order],
                weight=np.abs(fit.x[2::4] + 1j * fit.x[3::4])[order],
                gamma=fit.x[1::4][order], couplings=couplings, energy=energy, power=power)


def model_lines(photons, kerr_khz, couplings_mhz, kerr3_khz=0.0, keep_weight=0.02):
    model = fixed_n_hamiltonian(photon_number=photons, mode_count=1 + len(couplings_mhz),
                                detunings=np.zeros(len(couplings_mhz)), couplings_MHz=couplings_mhz,
                                physical_kerr_MHz=kerr_khz / 1000)
    h = np.asarray(model.hamiltonian_MHz).copy()
    n_m = np.array([state[0] for state in model.fock_basis], dtype=float)
    h += np.diag(kerr3_khz / 1000 / 6 * n_m * (n_m - 1) * (n_m - 2))
    energies, states = np.linalg.eigh(h)
    weight = np.abs(states[model.fock_index[(photons,) + (0,) * len(couplings_mhz)], :]) ** 2
    keep = weight > keep_weight
    return energies[keep], weight[keep]


def fit_spacings(line_sets, start, fixed=None):
    """Fit line spacings of several occupations. Parameters K, K3 (kHz), g_scale."""
    fixed = dict(dict(K=0.0, K3=0.0, g_scale=1.0), **(fixed or {}))
    names = list(start)

    def residuals(values):
        p = dict(fixed, **dict(zip(names, values)))
        out = []
        for lines in line_sets:
            energies, weights = model_lines(sum(lines["occupation"]), p["K"], lines["couplings"] * p["g_scale"],
                                            p["K3"], keep_weight=0.0)
            # Compare with the same number of strongest model lines (a weak line may not be fitted).
            energies = np.sort(energies[np.argsort(weights)[::-1][:len(lines["f"])]])
            spacing_err = np.sqrt(lines["df"][1:] ** 2 + lines["df"][:-1] ** 2)
            out.append((np.diff(energies) - np.diff(lines["f"])) / spacing_err)
        return np.concatenate(out)

    result = least_squares(residuals, list(start.values()), x_scale=[abs(v) if v else 1.0 for v in start.values()])
    dof = max(len(result.fun) - len(names), 1)
    errors = np.sqrt(np.diag(np.linalg.pinv(result.jac.T @ result.jac)))
    return dict(zip(names, zip(result.x, errors))), np.sum(result.fun ** 2), dof


# %% Cell 13: fit tonight's walks (cell 7) with the cell-11 model: loss + M1 dephasing.
# Per N: g fixed at the saved couplings, and g free; then N = 2 and 3 together with K3 free.
# Same fits as for the 10-17 walks (claude_work/refit_1017_dephasing.py), so they compare.
walk_sets = {n: walk_arrays(walks[f"n{n}"]) for n in (1, 2, 3)}
walk_fits = {}
walk_fits["n1"] = fit_walks([walk_sets[1]], dict(g_scale=1.0, kappa_m=0.005, gamma_phi=0.01))
for n in (2, 3):
    walk_fits[f"n{n} g fixed"] = fit_walks([walk_sets[n]], dict(K=-20.0, kappa_m=0.005, gamma_phi=0.01))
    walk_fits[f"n{n} g free"] = fit_walks([walk_sets[n]], dict(K=-20.0, g_scale=1.0, kappa_m=0.005, gamma_phi=0.01))
walk_fits["n2+3 K3"] = fit_walks([walk_sets[2], walk_sets[3]],
                                 dict(K=-20.0, K3=0.5, g_scale=1.0, kappa_m=0.005, gamma_phi=0.01))
for name, (params, chi2, _) in walk_fits.items():
    print(f"{name:10s}: {show(params)}, chi2/dof {chi2:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for axis, n in zip(axes, (1, 2, 3)):
    s = walk_sets[n]
    _, _, (pred,) = walk_fits["n1" if n == 1 else f"n{n} g free"]
    for row, col in enumerate(s["cols"]):
        axis.plot(s["time"], s["mean"][row], ".", ms=2, color=f"C{col}")
        axis.plot(s["time"], pred[row], "-", lw=1, color=f"C{col}", label=["M1", "S4", "S5", "S6", "S7"][col])
    axis.set(xlabel="time (us)", ylabel="<n>, readout corrected", title=f"walk from [{n},0,0,0,0], g free")
axes[0].legend(ncol=5, fontsize=8)
fig.tight_layout()
walk_fit_figures = save_figs("walk_fits_13")


# 2026-10-24 (run from the manifests, claude_work/analyze_walks_tonight.py): g/g_saved 0.991 /
# 1.012 / 1.033 (N = 1 / 2 / 3); K, g free: -20.8 +/- 0.3 (N=2), -20.1 +/- 0.5 kHz (N=3);
# g fixed: -21.9 / -18.6 (worse chi2); N=2+3 one g: K -22.2, K3 0.8 +/- 0.6 kHz.


# %% Cell 14: MBR lines of tonight's traces (cell 8) with the cell-12 functions. Per N: the
# n+1 line positions; then the spacings with g fixed, and with g free (N = 2: exact, no dof;
# N = 3: one dof left). Last: N = 2 and 3 together with one g and K3 free.
# N = 3: the weakest model line (weight ~0.04, near +110 kHz in the plot frame) is below the
# FFT noise peaks, so a 4-peak start took noise at -208 kHz (2026-10-24 04:49). Fit the three
# strong lines; the 4-line fit with a start at +0.110 MHz is a cross-check ("n3 4 lines").
mbr_lines = {n: fit_lines(mbr_traces[n], [n, 0, 0, 0, 0], count=min(n + 1, 3)) for n in (1, 2, 3)}
mbr_lines_n3_4 = fit_lines(mbr_traces[3], [3, 0, 0, 0, 0], count=3, extra_starts=[0.110])
for n, lines in mbr_lines.items():
    print(f"N={n}: lines {np.round(lines['f'], 5)} +/- {np.round(lines['df'], 5)} MHz, "
          f"weights {np.round(lines['weight'] / lines['weight'].sum(), 3)}, "
          f"spacings {np.round(1e3 * np.diff(lines['f']), 2)} kHz")
lines1 = mbr_lines[1]
g1_khz = 1e3 * np.diff(lines1["f"])[0] / 4  # N = 1, d = 0: two lines 2 * (2 g) apart (4 storages)
print(f"N=1: g = {g1_khz:.3f} kHz (saved {1e3 * lines1['couplings'][0]:.3f}), "
      f"g/g_saved = {g1_khz / (1e3 * lines1['couplings'][0]):.4f}")
mbr_fits = {}
for n in (2, 3):
    mbr_fits[f"n{n} g fixed"] = fit_spacings([mbr_lines[n]], dict(K=-20.0))
    mbr_fits[f"n{n} g free"] = fit_spacings([mbr_lines[n]], dict(K=-20.0, g_scale=1.0))
mbr_fits["n3 g at N=1"] = fit_spacings([mbr_lines[3]], dict(K=-20.0),
                                       dict(g_scale=g1_khz / (1e3 * lines1["couplings"][0])))
mbr_fits["n2+3 K3"] = fit_spacings([mbr_lines[2], mbr_lines[3]], dict(K=-20.0, K3=0.5, g_scale=1.0))
mbr_fits["n3 4 lines g free"] = fit_spacings([mbr_lines_n3_4], dict(K=-20.0, g_scale=1.0))
mbr_fits["n3 4 lines K3"] = fit_spacings([mbr_lines_n3_4], dict(K=-20.0, K3=0.5, g_scale=1.0))
print("N=3, 4 lines:", np.round(mbr_lines_n3_4["f"], 5), "+/-", np.round(mbr_lines_n3_4["df"], 5),
      "weights", np.round(mbr_lines_n3_4["weight"] / mbr_lines_n3_4["weight"].sum(), 3))
for name, (params, chi2, dof) in mbr_fits.items():
    print(f"{name:12s}: {show(params)}, chi2 {chi2:.2f} ({dof} dof)")

fig, axes = plt.subplots(2, 3, figsize=(17, 7))
for col, n in enumerate((1, 2, 3)):
    lines = mbr_lines[n]
    fine = np.linspace(0, lines["time"].max(), 1500)
    axes[0, col].plot(lines["time"], lines["values"].real, ".", ms=3, label="Re A")
    axes[0, col].plot(lines["time"], lines["values"].imag, ".", ms=3, label="Im A")
    axes[0, col].plot(fine, damped_lines(lines["params"], fine).real, "C0-", lw=1)
    axes[0, col].plot(fine, damped_lines(lines["params"], fine).imag, "C1-", lw=1)
    axes[0, col].set(xlabel="time (us)", title=f"MBR [{n},0,0,0,0], {n + 1}-line fit")
    axes[1, col].plot(lines["energy"], lines["power"], "k-", lw=1)
    for f in lines["f"]:
        axes[1, col].axvline(f, color="C3", ls=":")
    axes[1, col].set(xlim=(-0.25, 0.25), xlabel="energy (MHz)", ylabel="|FFT|")
axes[0, 0].legend()
fig.tight_layout()
mbr_fit_figures = save_figs("mbr_fits_14")


# 2026-10-24 (claude_work/analyze_mbr_tonight.py): N=1 g = 20.18 kHz (g/g_saved 0.991).
# N=2 spacings 78.4 / 87.6 kHz: K = -19.1 +/- 0.6, g/g_saved 1.011. N=3, 4 lines (weak 4th at
# +0.106 MHz), spacings 79.1 / 86.7 / 94.0 kHz: g free K = -16.6 +/- 0.9, g/g_saved 1.040,
# chi2 0.02 (1 dof); g fixed: K = -30.0, chi2 12.6 (2 dof), + K3: chi2 12 (claude_work/
# mbr_n3_4lines_test.py). So: K about -17 to -19 kHz with a coupling that grows with N.


# %% Cell 15: detuning sign from the cell-9 walks. For each set fit K, g_scale, M1 loss and
# dephasing under sign -1 (storage onsite energy = -detuning, fixed_n_hamiltonian) and sign +1.
# Populations are the same for (d, K) and (-d, -K): the right sign gives K < 0. The W = +30
# and W = -30 sets at N = 2 must give the same K under the right sign.
sign_sets = {key: walk_arrays(walk) for key, walk in sign_walks.items()}
sign_fits = {}
for key, s in sign_sets.items():
    for sign in (-1.0, +1.0):
        best = None
        for k0 in (-20.0, 20.0):
            result = fit_walks([s], dict(K=k0, g_scale=1.0, kappa_m=0.005, gamma_phi=0.005), sign=sign)
            if best is None or result[1] < best[1]:
                best = result
        sign_fits[(key, sign)] = best
        print(f"{key:9s} sign {sign:+.0f}: {show(best[0])}, chi2/dof {best[1]:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for axis, key in zip(axes, sign_sets):
    s = sign_sets[key]
    _, _, (pred,) = sign_fits[(key, -1.0)]
    for row, col in enumerate(s["cols"]):
        axis.plot(s["time"], s["mean"][row], ".", ms=2, color=f"C{col}")
        axis.plot(s["time"], pred[row], "-", lw=1, color=f"C{col}", label=["M1", "S4", "S5", "S6", "S7"][col])
    axis.set(xlabel="time (us)", ylabel="<n>, readout corrected", title=f"sign walk {key}, fit with sign -1")
axes[0].legend(ncol=5, fontsize=8)
fig.tight_layout()
sign_figures = save_figs("sign_fits_15")
# 2026-10-24 (claude_work/analyze_sign_tonight.py, sign_fixedK_tonight.py): K free, sign -1:
# K = -20.7 / -34.3 / -25.5 kHz (n2 +30 / n2 -30 / n3 +30), all < 0. K = -19 and g fixed (MBR):
# chi2/dof sign -1 vs +1: 4.3 vs 14.4, 10.0 vs 16.8, 6.4 vs 36.2. The sign in the code is right.
# Open: the fitted M1 loss is negative in these detuned sets.


# %% Cell 16: repeat of the [3,0,0,0,0] MBR trace, 1000 reps (cell 8 had 500), same StarkCal.
# Why: the g-fixed vs g-free test (cell 14) rests on the weak 4th N=3 line (about 3.4 sigma).
repeat_n3 = MBRSpectrumExperiment(
    [[3, 0, 0, 0, 0]], np.arange(0, 300, 2), campaign.modes, calibration=mbr_calibration,
    sync_cycles=campaign.sync_cycles, reps=1000)
repeat_n3.acquire(runner_for(MBRTimeTraceExperiment), batch_size=1, show=False)
repeat_n3.analyze()
print("repeat N=3 job:", repeat_n3.job_ids, "| manifest:", repeat_n3.save())
# 2026-10-24 08:42: JOB-20261024-00073 (manifest 261024_084204). 4-line test (claude_work/
# mbr_n3_4lines_repeat.py): spacings 78.2 / 86.3 / 94.6 kHz; g saved: chi2 4.3 (2 dof); g free:
# K = -19.2 +/- 1.5 kHz, g/g_saved 1.029, chi2 0.09. Same direction as cell 8 (chi2 12.6 there),
# weaker; the two g-free fits differ by 2.8 sigma (1.040 vs 1.029).
