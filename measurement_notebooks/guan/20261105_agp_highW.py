# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (pixi env of cwd)
#     language: python
#     name: pixi-cwd
# ---

# %% [markdown]
# # AGP at high W: is S7 frozen by its large detuning? (guan + Claude, 2026-11-05)
#
# Hypothesis (guan): at high W the S7-direction traces lose their oscillations and only decay,
# while the Kerr (M1) traces oscillate at all W, for any initial Fock state. The disorder
# pattern puts S7 at the largest detuning (+1.55 W) and S5 at the smallest (+0.19 W). When the
# S7 detuning is large compared with g, S7 is frozen, but M1 still swaps photons with S5, which
# is close to resonance. Test: the S5-direction signal must decay much more slowly with W
# than the S7 one. Coarse, short time axis; compare the three directions (M1, S5, S7).
#
# Continues jonginn's 11-05 run (`measurement_notebooks/jonginn/agp_measurement.py`, scan
# name `261105_debugging_direction_of_<d>`, stopped at 13:29). Same configs, defaults and
# calibration as his run. Kernel: `guan_agp_meas` (Lab server, port 8765). Analysis:
# `analysis_notebooks/guan/agp/20261105_agp_plots.py`, kernel `guan_agp_plot`.

# %% Cell 1: station on jonginn's 11-05 configs (after his autocal at 0.25 mA, 11:06-12:28), the
# runners with the defaults of his agp_measurement.py, and his confusion matrix of 12:52.
# The readout baseline of today is from his autocal (HistogramExperiment 11:06, same set point).
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.qsim.agp import CentralBosonAGPDisorderScan, direction_sector
from experiments.qsim.fock_population import FockPopulationExperiment
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS
from job_server import JobClient

config_dict = {
    "hardware_config": "CFG-HW-20261105-00039",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261105-00013",
    "floquet_storage_swap": "CFG-FL-20261105-00029",
}

client = JobClient()
client.print_queue()

station = MultimodeStation(
    user="guan",
    experiment_name="261027AGPlowKerr",
    project="AGP",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)

STORAGE_MODES = [4, 5, 6, 7]
# jonginn's agp_measurement.py, cell "defaults" (relax_delay 200 us with dump reset of M and S).
common_defaults = AttrDict(dict(
    expts=1, reps=1, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=STORAGE_MODES, pre_relax_delay=100, relax_delay=200,
    postpulse=True, init_fock=False, init_stor=0, ro_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    swap_stors=STORAGE_MODES, update_phases=True,
    detunings=[0.0, 0.0, 0.0, 0.0],
))
common_defaults.update(ACTIVE_RESET_DEFAULTS)
common_defaults.update(FLOQUET_DEFAULTS)
calibration_defaults = AttrDict(dict(common_defaults, prepulse=True))
population_defaults = AttrDict(dict(
    common_defaults, prepulse=False, occupations=[3, 0, 0, 0, 0],
    floquet_cycles=[0], swept_params=["floquet_cycle"],
))

confusion_matrix_runner = CharacterizationRunner(
    station=station, ExptClass=MultiParityConfusionMatrixExperiment,
    default_expt_cfg=calibration_defaults, job_client=client, use_queue=True, show=False,
)
population_runner = CharacterizationRunner(
    station=station, ExptClass=FockPopulationExperiment,
    default_expt_cfg=population_defaults, job_client=client, use_queue=True, show=False,
)

DATA = station.experiment_path / "data"
CONFUSION_MATRIX_FNAME = str(DATA / "JOB-20261105-00152_MultiParityConfusionMatrixExperiment.h5")
DISORDER_PATTERN = np.array([-0.70675335, 0.19463085, -1.03457499, 1.54669749])
DISORDER_PATTERN -= np.mean(DISORDER_PATTERN)   # from_pattern() scales it to unit RMS
print("calibration:", CONFUSION_MATRIX_FNAME)

# %% Cell 2: the scan plan, one direction at a time. Coarse grid (guan, 11-05): cycles 0-256 by 4
# (65 points, about 79 us), so that one exchange period at W = 200 kHz (about 20 us in M1) fits
# several times. W = 0 and 200 kHz, 1000 reps, as in jonginn's run. Name: 261105_coarse_dir<d>.
AGP_CYCLES = list(range(0, 257, 4))
DISORDER_RMS_KHZ = [0, 200]
AGP_REPS = 1000


def plan(direction):
    return CentralBosonAGPDisorderScan.from_pattern(
        DISORDER_RMS_KHZ, DISORDER_PATTERN, photon_sector_number=3, direction=direction,
        ro_stor=direction, swap_stors=STORAGE_MODES, cycles=AGP_CYCLES,
        calibration=CONFUSION_MATRIX_FNAME, notes=f"guan 11-05: coarse grid, direction {direction}")


def print_files(scan):
    """The assembled files: one HDF5 per W, and the scan HDF5 (for the lab book)."""
    for part in scan.children:
        print(f"W = {part.disorder_khz:g} kHz: {Path(part.manifest_path).with_suffix('.h5')}")
    if scan.manifest_path is not None:
        print(f"scan: {Path(scan.manifest_path).with_suffix('.h5')}")


# %% Cell 3: mock check of one S5 job on the full grid (board resources, program length), 10 reps.
station.use_mock_instruments()
try:
    mock_override = plan(5).children[1].job_overrides()[0]
    mock_job = population_runner.execute(overrides=[mock_override], reps=10,
                                         postprocess=False, show=False)
    print("mock ok:", mock_override["occupations"], mock_override["detunings"],
          len(mock_override["floquet_cycles"]), "cycles")
finally:
    station.use_real_instruments()

# %% Cell 4: S5 direction on the device, W by W (each W is saved when its 15 jobs are in).
scan_s5 = plan(5)
scan_s5.acquire(population_runner, name="261105_coarse_dir5", batch_size=3, reps=AGP_REPS,
                storage_reset=STORAGE_MODES, postprocess=False, show=False, log=True)
print_files(scan_s5)

# %% Cell 5: shot-time profile (timing only; the data of variants B, C are not physics).
# Budget from the code (MM_base.active_reset, QsimProgram.man_reset; dump M1-D2 pi = 80 us):
#   M1 reset, chi-dressed: 3 photon shifts x 2 chis = 6 dump pulses     480 us
#   storage reset S4..S7: swap (1.2 us) + 1 dump pulse each, 4 x ~82    330 us
#   relax_delay 200 + pre_relax_delay 100                               300 us
#   qubit ge/ef reset readouts, prep, Floquet (<= 79 us), multiparity   ~50 us
# One point (0 cycles), N = 3 in M1, read M1. Jobs A1/A2 differ only in reps: they give the
# per-shot time and the fixed time per job. B: no M1 and storage reset. C: no waits.
PROFILE_POINT = dict(occupations=[3, 0, 0, 0, 0], ro_stor=0, floquet_cycles=[0],
                     swap_stors=STORAGE_MODES, detunings=[0.0] * 4,
                     multiparity_confusion_matrix=CONFUSION_MATRIX_FNAME)
PROFILE = {
    "A1 as now, 1000 reps": dict(reps=1000),
    "A2 as now, 4000 reps": dict(reps=4000),
    "B no M1/storage reset": dict(reps=4000, man_reset=False, storage_reset=[]),
    "C no relax/pre-relax waits": dict(reps=4000, relax_delay=0.1, pre_relax_delay=0),
}
profile_ids = {}
for label, change in PROFILE.items():
    population_runner.execute(overrides=[dict(PROFILE_POINT, **change)], postprocess=False,
                              show=False, log=False)
    profile_ids[label] = population_runner.last_job_ids[0]


def job_seconds(job_id):
    status = client.get_status(job_id)
    return (status.completed_at - status.started_at).total_seconds()


seconds = {label: job_seconds(job_id) for label, job_id in profile_ids.items()}
shot_ms = 1e3 * (seconds["A2 as now, 4000 reps"] - seconds["A1 as now, 1000 reps"]) / 3000
fixed_s = seconds["A1 as now, 1000 reps"] - 1e-3 * shot_ms * 1000
print(f"per shot as now: {shot_ms:.3f} ms; fixed time per one-point job: {fixed_s:.1f} s")
for label, job_id in profile_ids.items():
    reps = PROFILE[label]["reps"]
    print(f"   {label:28s} {job_id}: {seconds[label]:6.1f} s -> "
          f"{1e3 * (seconds[label] - fixed_s) / reps:.3f} ms per shot")

# %% Cell 6: chi_ge and chi_f of M1 by pulse-probe spectroscopy (guan, 11-05). The config values
# are hand-entered (chi_ef = 1.231 MHz in every version since 2026-01-15) and the reset pulses
# use them. Four jobs, no active reset, relax 2.5 ms (qubit + M): ge line with M1 in |0> and |1>;
# ef line with the qubit in e and M1 in |0> and |1>. Probe: 4 us flat top, gain 150 (about
# 0.75 pi on the ge line), 10 kHz steps. chi_ge = ge(n=1) - ge(n=0); the ef shift per photon
# (the config's chi_ef) = ef(n=1) - ef(n=0); chi_f = chi_ge + chi_ef.
import experiments as meas

spec_defaults = AttrDict(dict(
    expts=201, reps=200, rounds=1, length=4.0, gain=150, sigma=0.1, qubit=[0], qubits=[0],
    prepulse=False, pre_sweep_pulse=[], gate_based=True, relax_delay=2500,
    active_reset=False,
))
spec_runner = CharacterizationRunner(
    station=station, ExptClass=meas.PulseProbeSpectroscopyExperiment,
    default_expt_cfg=spec_defaults, job_client=client, use_queue=True, show=False,
)
F_GE = station.hardware_cfg.device.qubit.f_ge[0]
F_EF = station.hardware_cfg.device.qubit.f_ef[0]
M1_ONE = [["multiphoton", "g0-e0", "pi", 0], ["multiphoton", "e0-f0", "pi", 0],
          ["multiphoton", "f0-g1", "pi", 0]]   # MM_base.prep_man_photon(1): |g,1>
QUBIT_E = [["qubit", "ge", "pi", 0]]


def span(center, low, high, step=0.01):
    """-> start, step, expts for a sweep from center+low to center+high (MHz)."""
    expts = int(round((high - low) / step)) + 1
    return dict(start=center + low, step=step, expts=expts)


CHI_JOBS = {
    "ge, M1 |0>": dict(span(F_GE, -1.0, 0.5)),
    "ge, M1 |1>": dict(span(F_GE, -1.0, 0.5), prepulse=True, pre_sweep_pulse=M1_ONE),
    "ef, qubit e, M1 |0>": dict(span(F_EF, -1.5, 1.5), prepulse=True, pre_sweep_pulse=QUBIT_E),
    "ef, qubit e, M1 |1>": dict(span(F_EF, -1.5, 1.5), prepulse=True,
                                pre_sweep_pulse=M1_ONE + QUBIT_E),
}


def run_chi_jobs(mock):
    """Run CHI_JOBS (on the mock station with 2 reps when mock) -> {label: expt}."""
    if mock:
        station.use_mock_instruments()
    try:
        results = {}
        for label, change in CHI_JOBS.items():
            extra = dict(reps=2) if mock else {}
            results[label] = spec_runner.execute(**change, **extra, postprocess=False, show=False,
                                                 log=not mock)
            if not mock:
                print(f"{label}: {results[label].fname}")
        return results
    finally:
        if mock:
            station.use_real_instruments()


# %% Cell 7: mock check of the four spectroscopy jobs.
mock_chi = run_chi_jobs(mock=True)
print("mock ok:", list(mock_chi))

# %% Cell 8: the four spectroscopy jobs on the device (about 2 min each).
chi_expts = run_chi_jobs(mock=False)

# %% Cell 9: M1 reset check (experiments/qsim/man_reset_verification.py). Prepare M1 |1> with the
# qubit in g, e or f; run the active reset with or without the M1 dump; then map a photon left in
# M1 to qubit e (f0-g1 pi, e0-f0 pi) and read. P(e) - floor = photon left after the reset. The
# floor job prepares no photon. relax 2.5 ms, so the previous shot does not matter.
from experiments.MM_dual_rail_base import MM_dual_rail_base

gates = MM_dual_rail_base(station.hardware_cfg, soccfg=station.soccfg)


def pulses(gate_list):
    return gates.get_prepulse_creator(gate_list).pulse.tolist()


QUBIT_TO_E = [["qubit", "ge", "pi", 0]]
QUBIT_TO_F = [["qubit", "ge", "pi", 0], ["qubit", "ef", "pi", 0]]
reset_check_defaults = AttrDict(dict(
    expts=1, reps=3000, rounds=1, qubits=[0], ro_stor=0, init_fock=False, parity_fast=False,
    dummys=[0], swept_params=["dummy"], normalize=False,
    active_reset=True, man_reset=False, storage_reset=False, parity_check=False,
    prepulse=False, postpulse=True, init_alpha=0, init_stor=0, relax_delay=2500,
    reset_dump_mode=2, dump_reset_iter_num=1, prep_man_photons=1, pre_reset_pulse=[],
    post_sweep_pulse=pulses([["multiphoton", "f0-g1", "pi", 0], ["multiphoton", "e0-f0", "pi", 0]]),
))
reset_check_runner = CharacterizationRunner(
    station=station, ExptClass=meas.QsimExperiment, ExptProgram=meas.qsim.man_reset_verification.ManActiveResetVerificationProgram,
    default_expt_cfg=reset_check_defaults, job_client=client, use_queue=True, show=False,
)
RESET_CASES = {
    "floor: no photon, M1 reset": dict(prep_man_photons=0, man_reset=True),
    "g,1 no M1 reset": dict(man_reset=False),
    "g,1 M1 reset": dict(man_reset=True),
    "e,1 no M1 reset": dict(man_reset=False, pre_reset_pulse=pulses(QUBIT_TO_E)),
    "e,1 M1 reset": dict(man_reset=True, pre_reset_pulse=pulses(QUBIT_TO_E)),
    "f,1 no M1 reset": dict(man_reset=False, pre_reset_pulse=pulses(QUBIT_TO_F)),
    "f,1 M1 reset": dict(man_reset=True, pre_reset_pulse=pulses(QUBIT_TO_F)),
}


def run_reset_check(mock, tag):
    """Run RESET_CASES -> {label: job ID} (mock: 5 reps, nothing logged)."""
    if mock:
        station.use_mock_instruments()
    try:
        ids = {}
        for label, change in RESET_CASES.items():
            extra = dict(reps=5) if mock else {}
            expt = reset_check_runner.execute(**change, **extra, postprocess=False, show=False,
                                              log=not mock)
            ids[label] = None if mock else reset_check_runner.last_job_ids[0]
            if not mock:
                print(f"{tag} | {label}: {expt.fname}")
        return ids, expt
    finally:
        if mock:
            station.use_real_instruments()


# %% Cell 10: mock check of the reset cases.
_, mock_reset = run_reset_check(mock=True, tag="mock")
print("mock ok; data keys:", sorted(mock_reset.data.keys()))
print({k: np.shape(v) for k, v in mock_reset.data.items() if hasattr(v, "__len__")})

# %% Cell 11: baseline on the device, with the current (6-pulse chi-dressed) M1 reset.
baseline_reset_ids, _ = run_reset_check(mock=False, tag="baseline 6-pulse reset")

# %% Cell 12: the same check with the new 3-pulse M1 reset (MM_base.man_reset_offsets_mhz: g, e, f
# lines) and the measured chi (CFG-HW-20261105-00040: chi_ge -0.3175, chi_ef -0.2759 MHz), then
# one profile job (cell 5, A2) for the new shot time.
new_reset_ids, _ = run_reset_check(mock=False, tag="new 3-pulse reset")
population_runner.execute(overrides=[dict(PROFILE_POINT, **PROFILE["A2 as now, 4000 reps"])],
                          postprocess=False, show=False, log=False)
profile_new = population_runner.last_job_ids[0]
print(f"per shot, new reset: {1e3 * (job_seconds(profile_new) - fixed_s) / 4000:.3f} ms "
      f"({profile_new}); before {shot_ms:.3f} ms")

# %% Cell 13: trimmed reset (guan, 11-05): the 3 legacy pulses that work, offsets 0, chi_ge, 2 chi_ge
# with the old config chi_ge (-0.3025 MHz; MM_base.man_reset_offsets_mhz); dropped: 0 a second time,
# chi_ge + chi_ef, 2 (chi_ge + chi_ef). The station has the CFG-HW-20261105-00039 chi values again
# (snapshot below). Same reset check as cells 11-12, then the profile job for the shot time.
hw_trimmed = station.snapshot_hardware_config(update_main=False)
print("hardware config:", hw_trimmed, station.hardware_cfg.device.manipulate.chi_ge,
      station.hardware_cfg.device.manipulate.chi_ef)
trimmed_reset_ids, _ = run_reset_check(mock=False, tag="trimmed 3-pulse reset")
population_runner.execute(overrides=[dict(PROFILE_POINT, **PROFILE["A2 as now, 4000 reps"])],
                          postprocess=False, show=False, log=False)
profile_trimmed = population_runner.last_job_ids[0]
print(f"per shot, trimmed reset: {1e3 * (job_seconds(profile_trimmed) - fixed_s) / 4000:.3f} ms "
      f"({profile_trimmed}); before {shot_ms:.3f} ms")

# %% Cell 14: guan's guess: D2 is above M1, so the M1-D2 sideband moves by MINUS M1's shift. Trial
# reset with 3 pulses at 0, -chi_ge, -chi_f (+0.3175, +0.593 MHz with the measured chi,
# CFG-HW-20261105-00040). Same reset check and profile job.
m = station.hardware_cfg.device.manipulate
m.chi_ge, m.chi_ef = [-0.3175], [-0.2759]
hw_flipped = station.snapshot_hardware_config(update_main=False)
print("hardware config:", hw_flipped, m.chi_ge, m.chi_ef)
flipped_reset_ids, _ = run_reset_check(mock=False, tag="sign-flipped 3-pulse reset")
population_runner.execute(overrides=[dict(PROFILE_POINT, **PROFILE["A2 as now, 4000 reps"])],
                          postprocess=False, show=False, log=False)
profile_flipped = population_runner.last_job_ids[0]
print(f"per shot, sign-flipped reset: {1e3 * (job_seconds(profile_flipped) - fixed_s) / 4000:.3f} ms "
      f"({profile_flipped}); before {shot_ms:.3f} ms")

# %% Cell 15: the same sign-flipped pulses in the order of the qubit decay: f, e, g (-chi_f, -chi_ge, 0),
# so a qubit that decays f -> e -> g during the reset still meets the pulse of its new state.
reordered_reset_ids, _ = run_reset_check(mock=False, tag="sign-flipped f,e,g reset")
population_runner.execute(overrides=[dict(PROFILE_POINT, **PROFILE["A2 as now, 4000 reps"])],
                          postprocess=False, show=False, log=False)
profile_reordered = population_runner.last_job_ids[0]
print(f"per shot, f,e,g reset: {1e3 * (job_seconds(profile_reordered) - fixed_s) / 4000:.3f} ms "
      f"({profile_reordered}); before {shot_ms:.3f} ms")
