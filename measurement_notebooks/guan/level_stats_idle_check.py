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
# # Level statistics: can an idle per Trotter pulse halve g at fixed Kerr? (Claude, 2026-10-25 night)
#
# Part of the level-statistics verdict (vault: guan/LevelStatistics/LevelStatistics - overview.md).
# The chaotic region at N = 3 needs K/g <= -1.8 at W/g ~ 1.7; the Kerr at the calibrated 0.25 mA set
# point is -17..-19 kHz with g = 20.4 kHz (K/g ~ -0.9). g = 1/(4 n T_cycle): an idle of
# ``scramble_sync_cycles`` tProc cycles after each swap pulse lengthens T_cycle with the same pulses.
# sync 34 should double T_cycle (0.307 -> 0.614 us) and halve g, while the Kerr (continuous) stays.
#
# Check, both settings on one config, zero disorder: [1,0,0,0,0] (two lines: g and a common
# detuning) and [2,0,0,0,0] (three lines: K/g), StarkCal + trace each. 8 jobs, about 70 min.
#
# Config: guan's checked 10-24 snapshot at 0.25 mA (kerr_n123_consistency.py cell 5), with seb's
# readout block of CFG-HW-20261025-00020 (same readout frequency; single shot below updates only
# phase/threshold/Ig/Ie in memory). Run as a script: ``pixi run python <this file> [mock]``.

# %% Cell 1: station, readout, campaigns for the two settings. No jobs.
import json
import os
import sys
from copy import deepcopy

import h5py
import numpy as np
import yaml

import experiments as meas
from slab import AttrDict
from experiments import CharacterizationRunner, MultimodeStation
from job_server import JobClient
from experiments.qsim.notebook_helpers.defaults import (ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS,
                                                       MEASUREMENT_CONFIG_DEFAULTS)
from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_stark_cal import MBRStarkCalExperiment
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
from experiments.qsim.mbr_campaign import build_campaign, campaign_runner

MOCK = len(sys.argv) > 1 and sys.argv[1] == "mock"
STORS = [4, 5, 6, 7]
SYNCS = (1, 34)
OCCUPATIONS = [[1, 0, 0, 0, 0], [2, 0, 0, 0, 0]]
OUT = r"C:\experiments\261025_level_stats\claude_work"
os.makedirs(OUT, exist_ok=True)

config_dict = {
    "hardware_config": "CFG-HW-20261024-00002",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261024-00001",
    "floquet_storage_swap": "CFG-FL-20261023-00002",
}
client = None if MOCK else JobClient()
if client:
    client.print_queue()
station = MultimodeStation(
    user="guan", experiment_name="261025_level_stats", project="LevelStatistics",
    log_measurements=not MOCK, mock=MOCK,
    hardware_config=config_dict["hardware_config"], multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"], floquet_file=config_dict["floquet_storage_swap"],
)
seb_hw = yaml.safe_load(open(os.path.join(r"C:\python\multimode_expts\configs\versions\hardware_config",
                                          "CFG-HW-20261025-00020.yml")))
ro = station.hardware_cfg.device.readout
for key, value in seb_hw["device"]["readout"].items():
    ro[key] = value
print("readout:", ro.frequency, ro.gain, ro.readout_length, "phase", ro.phase,
      "| coupler (A):", station.hardware_cfg.hw.yoko_coupler.current)

campaigns = {}
for sync in SYNCS:
    settings = dict(FLOQUET_DEFAULTS, scramble_sync_cycles=sync)
    campaigns[sync] = build_campaign(floquet_settings=settings, active_reset_settings=ACTIVE_RESET_DEFAULTS,
                                     measurement_settings=MEASUREMENT_CONFIG_DEFAULTS, modes=STORS, reps=1000)


def runner_for(sync, JobClass):
    return campaign_runner(campaigns[sync], station, client, JobClass, use_queue=not MOCK)


# %% Cell 2: single shot at 0.25 mA (the worker ramps the coupler); phase/threshold/Ig/Ie in memory.
def singleshot_postproc(station, expt):
    expt.analyze(plot=False, station=station, subdir=station.autocalib_path)
    d = expt.data
    print("single shot: old phase", ro.phase[0], "angle", d["angle"], "fid", d["fids"])
    ro.phase = [ro.phase[0] + d["angle"]]
    ro.threshold = d["thresholds"]
    ro.threshold_list = [d["thresholds"]]
    ro.Ie = [np.median(d["Ie_rot"])]
    ro.Ig = [np.median(d["Ig_rot"])]
    ro.confusion_matrix_without_reset = d["confusion_matrix"]


if not MOCK:
    ss_runner = CharacterizationRunner(
        station=station, ExptClass=meas.HistogramExperiment,
        default_expt_cfg=AttrDict(dict(
            reps=5000, relax_delay=2500, check_f=False, active_reset=False, man_reset=False,
            storage_reset=False, qubit=0, pulse_manipulate=False, cavity_freq=4984.373226159381,
            cavity_gain=400, cavity_length=2, prepulse=False, pre_sweep_pulse=None, gate_based=True,
            qubits=[0], avoid_yoko=False)),
        postprocessor=singleshot_postproc, job_client=client)
    ss = ss_runner.execute(go_kwargs=dict(analyze=False, display=False), active_reset=False, ef_reset=True,
                           storage_reset=False, coupler_reset=False, pre_selection_reset=True, man_reset=False,
                           use_qubit_man_reset=True, pre_selection_parity=False)
    print("single shot file:", ss.fname)

# %% Cell 3: StarkCal + traces, both settings. Same 150 samples every 2 cycles: 92 us (sync 1), 184 us (sync 34).
record = dict(config=config_dict, readout_from="CFG-HW-20261025-00020", runs={})
cycle_pairs = np.arange(0, 3, dtype=int) if MOCK else np.arange(0, 65, dtype=int)
cycles = np.array([0, 4]) if MOCK else np.arange(0, 300, 2)
reps_cal, reps_trace = (10, 10) if MOCK else (1000, 500)
for sync in SYNCS:
    calibration = MBRCalibrationSetExperiment(OCCUPATIONS, cycle_pairs=cycle_pairs, swap_stors=STORS,
                                              sync_cycles=sync, reps=reps_cal)
    calibration.acquire(runner_for(sync, MBRStarkCalExperiment), batch_size=2, log=not MOCK, show=False)
    if not MOCK:  # mock data have no phase to fit
        calibration.analyze()
    trace = MBRSpectrumExperiment(OCCUPATIONS, cycles, STORS, calibration=None if MOCK else calibration,
                                  sync_cycles=sync, reps=reps_trace)
    trace.acquire(runner_for(sync, MBRTimeTraceExperiment), batch_size=2, log=not MOCK, show=False)
    files = [child.fname for child in trace.children]
    with h5py.File(files[0], "r") as h5:
        timing = json.loads(h5.attrs["derived_params"])
    print(f"sync {sync}: Floquet cycle {timing['floquet_cycle_us']:.5f} us, g {1e3 * timing['couplings_MHz'][3]:.3f} kHz")
    record["runs"][sync] = dict(stark_jobs=calibration.job_ids, trace_jobs=trace.job_ids, trace_files=files,
                                cycle_us=timing["floquet_cycle_us"], g_MHz=timing["couplings_MHz"][3],
                                phase_deg_per_cycle=None if MOCK else {str(o): calibration.phase_for(o) for o in calibration.occupations},
                                calibration_manifest=None if MOCK else str(calibration.save()))
    print("sync", sync, json.dumps(record["runs"][sync], default=str))
    with open(os.path.join(OUT, "idle_check_mock.json" if MOCK else "idle_check.json"), "w") as f:
        json.dump(record, f, indent=1, default=str)
print("done")
