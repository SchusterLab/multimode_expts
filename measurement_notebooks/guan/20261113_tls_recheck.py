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
# # Is the ~11.96 GHz TLS still there after the cooldown? (guan + Claude, 2026-11-13)
#
# Hold |f,n> (n = 0, 1, 2 photons in M1) and read g/e/f, as on 2026-10-28/29
# (`yoko450ua_calibration.py` cells 8-11; vault report
# `Lab/guan/AGP/2026/10/2026-10-29_TLS_12GHz_report`). Now one job:
# `FStateHoldExperiment` (`experiments/qsim/f_state_hold.py`) measures the g/e/f
# reference clouds and all holds together.
#
# 10-28 at 0.45 mA: nu_1 = 11963.3, nu_2 = 11961.1 MHz; |f,1> -> g 5.4 us, |f,2> -> g 1.9 us;
# |f,0> decays only to e (~120 us). Today the qubit is +1.62 (ge) / +1.75 (ef) MHz higher, so
# nu_n is ~3.7 MHz higher. A slower decay than on 10-28 does not by itself mean that the TLS moved.
#
# No single shot first: the analysis assigns each shot to the nearest in-job reference cloud,
# so the readout phase and threshold do not enter.

# %% Cell 1: station on jonginn's 0.45 mA calibration of this cooldown, and the config branch.
import numpy as np
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.branch_manager import BranchManager
from experiments.qsim.f_state_hold import FStateHoldExperiment
from job_server import JobClient

# jonginn's vault note `20261111 AGP 0.45 mA` lists CFG-HW-20261111-00097. -00016 has the same
# multiphoton ladder, his 11-12 single shot and f_ge 3 kHz lower.
config_dict = {
    "hardware_config": "CFG-HW-20261112-00016",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261111-00132",
    "floquet_storage_swap": "CFG-FL-20261111-00111",
}

client = JobClient()
client.print_queue()

station = MultimodeStation(
    user="guan",
    experiment_name="261113_tls_recheck",
    project="AGP",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)
print("coupler set point (A):", station.hardware_cfg.hw.yoko_coupler.current)

bm = BranchManager(station)
print(bm.commit("tls_0.45mA", note="jonginn's 0.45 mA calibration of 11-11/12, for the TLS re-check"))

# %% Cell 2: hold |f,n> for n = 0, 1, 2 over the waits, in one job (24 points, ~1-1.5 min).
# No reset with photons in M1: relax_delay 2500 us, as on 10-28.
hold_defaults = AttrDict(dict(
    reps=1000, rounds=1, qubit=0, qubits=[0], relax_delay=2500,
    active_reset=False, man_reset=False, storage_reset=False,
    hold_ns=[0, 1, 2], hold_waits=[0, 2, 5, 10, 20, 40, 60],
))
hold_runner = CharacterizationRunner(
    station=station, ExptClass=FStateHoldExperiment, default_expt_cfg=hold_defaults,
    job_client=client, use_queue=True, show=True,
)
hold = hold_runner.execute()
d = hold.data
print("job", hold_runner.last_job_ids)
print("assignment A[prepared, assigned] (g, e, f):\n", d["assignment_matrix"].round(3))
print(" n   nu_n (MHz)   tau (us)        share to g   tau to g (us)")
for i, n in enumerate(d["hold_ns"]):
    tau, tau_err = 1 / d["decay_rate"][i], d["decay_rate_err"][i] / d["decay_rate"][i] ** 2
    print(f" {n}   {d['nu'][i]:9.2f}   {tau:6.1f} +- {tau_err:4.1f}   {d['share_to_g'][i]:6.2f}"
          f"       {1 / d['rate_to_g'][i]:7.1f}")

# %% Cell 3: herald, or herald + qubit reset, in the M1 Fock ladder (guan's question, 2026-11-13).
# After a Fock 3 prep the qubit is in f at readout in ~5 % of shots at 0.45 mA, and such a shot
# reads n = 3 in every mode (analysis: ../../analysis_notebooks/guan/20261113_f_postselect.py).
# Jobs: vacuum (no prep), then Fock 3 and Fock 2 in M1, each with prep_herald = none / end / each /
# each_reset (experiments/qsim/fock_prep_herald.py). Each job reads M1 and the empty storages at
# Floquet cycle 0, with the expt settings of jonginn's JOB-20261112-00178..00182.
from experiments.qsim.fock_prep_herald import HERALD_MODES, FockPrepHeraldExperiment

POPULATION_EXPT = dict(
    expts=1, rounds=1, qubits=[0], f0g1_cavity=1, normalize=False,
    active_reset=True, man_reset=True, storage_reset=[1, 2, 4, 5], pre_relax_delay=100,
    relax_delay=200, postpulse=True, init_fock=False, init_stor=0, readout="multiparity",
    parity_fast=False, cond_sec_phase=-90, phase_second_pulse=180, swap_stors=[1, 2, 4, 5],
    update_phases=True, detunings=[0.0] * 4, reset_dump_mode=2, dump_reset_iter_num=1,
    include_10cycles_buffer=True, include_10cycles_buffer_in_pi_half=True,
    floquet_waveform="preload_flattop", floquet_hardware_loop=True, scramble_sync_cycles=2,
    palindrome_scramble=False, prepulse=False, floquet_cycle=0)
herald_defaults = AttrDict(dict(POPULATION_EXPT, reps=2000, ro_stors=[0, 1, 2, 4, 5]))
HERALD_JOBS = ([dict(occupations=[0, 0, 0, 0, 0], prep_herald="none")]
               + [dict(occupations=[n, 0, 0, 0, 0], prep_herald=mode)
                  for n in (3, 2) for mode in HERALD_MODES])
herald_runner = CharacterizationRunner(
    station=station, ExptClass=FockPrepHeraldExperiment, default_expt_cfg=herald_defaults,
    job_client=client, use_queue=True, show=False,
)
# Mock first with these configs (docs/running_experiments.md item 6): compile + board resources.
station.use_mock_instruments()
try:
    herald_runner.execute(overrides=[dict(job, reps=2) for job in HERALD_JOBS],
                          go_kwargs=dict(analyze=True), log=False)
finally:
    station.use_real_instruments()
print("mock ok")
heralds = herald_runner.execute(overrides=HERALD_JOBS, batch_size=len(HERALD_JOBS))
print("jobs", herald_runner.last_job_ids)

# %% Cell 4: results of cell 3. <n> raw; post-selected on the heralds; and on no f in the parity
# readouts (f centers from the cell 2 references, same readout phase).
from fitting.qsim import f_postselect

centers = hold.data["blob_centers_i"] + 1j * hold.data["blob_centers_q"]
print("N  herald      read  kept   <n>: all  herald  no-f  herald+no-f   p(3): all  herald+no-f")
for h in heralds:
    lab = h.shot_labels(centers=centers)
    for k, ro in enumerate(h.cfg.expt.ro_stors):
        ok = np.ones(lab["n"].shape[1], bool) if lab["herald"] is None else lab["herald"][k]
        p = [f_postselect.n_distribution(lab["n"][k], keep)
             for keep in (None, ok, lab["no_f"][k], ok & lab["no_f"][k])]
        print(f"{h.cfg.expt.occupations[0]}  {h.cfg.expt.prep_herald:10s}  {'M1' if ro == 0 else f'S{ro}':4s}"
              f"  {ok.mean():.3f}  " + "  ".join(f"{f_postselect.mean_n(x):5.3f}" for x in p)
              + f"     {p[0][3]:.3f}  {p[3][3]:.3f}")

# %% [markdown]
# ## Problems found on 2026-11-13, to deal with after this run
# (record: `docs/log/2026-11-13_f-state-hold-experiment.md`; no code changed for them yet)
#
# 1. **`Experiment.go()` analyzes before it saves.** An error in `analyze()` on the worker
#    loses the data of the job. `FStateHoldExperiment.analyze()` catches and prints errors, but
#    other Experiments do not: for example `HistogramExperiment.analyze()` raises on invalid
#    readout centers.
# 2. **Mock runs skip the analysis by default** (on purpose: `docs/running_experiments.md`
#    item 6, a mock fit must not write into the configs). The worker always runs
#    `go(analyze=True)`, so a mock test of an analysis must pass `go_kwargs=dict(analyze=True)`.
#    Not a bug; easy to forget in tests.
# 3. **`singleshot_runner` defaults** (`experiments/qsim/readout_calibration.py`):
#    `avoid_yoko=True` and `use_queue=False`. Wrong for a single shot at a set point that another
#    user changed, unless both are passed.
# 4. **`SlabFile.add` cannot store strings** (`np.array('...')` has a unicode dtype that h5py
#    refuses: "No conversion path for dtype"). Analysis results in `data` must be numeric.

# %% Cell 5: does prep_herald="end" help a real population job? Fock 3 quantum walk in M1 (as
# jonginn's JOB-20261112-00178..00182, cycles 0-196 here), read M1, S1, S2, S4, S5, with and
# without the herald; FockPopulationExperiment with FockPrepHeraldProgram. Confusion rows for both:
# FockPrepHeraldExperiment, Fock 0-3 read in M1 at cycle 0 (same program as the walks).
# The walk files hold the analysis of all shots; the herald cut is done offline
# (../../analysis_notebooks/guan/20261113_f_postselect.py, cell 7).
from experiments.qsim.fock_population import FockPopulationExperiment
from experiments.qsim.fock_prep_herald import FockPrepHeraldProgram

CONFUSION_JOBS = ([dict(occupations=[0, 0, 0, 0, 0], prep_herald="none")]
                  + [dict(occupations=[n, 0, 0, 0, 0], prep_herald=mode)
                     for mode in ("none", "end") for n in (1, 2, 3)])
WALK_JOBS = [dict(ro_stor=ro, prep_herald=mode) for ro in (0, 1, 2, 4, 5) for mode in ("none", "end")]
walk_runner = CharacterizationRunner(
    station=station, ExptClass=FockPopulationExperiment, ExptProgram=FockPrepHeraldProgram,
    default_expt_cfg=AttrDict(dict(POPULATION_EXPT, reps=1000, occupations=[3, 0, 0, 0, 0],
                                   floquet_cycles=list(range(0, 200, 4)))),
    job_client=client, use_queue=True, show=False,
)
station.use_mock_instruments()
try:
    herald_runner.execute(overrides=[dict(job, reps=2, ro_stors=[0]) for job in CONFUSION_JOBS],
                          go_kwargs=dict(analyze=True), log=False)
    walk_runner.execute(overrides=[dict(job, reps=2, floquet_cycles=[0, 4]) for job in WALK_JOBS],
                        go_kwargs=dict(analyze=True), log=False)
finally:
    station.use_real_instruments()
print("mock ok")
confusion_rows = herald_runner.execute(overrides=[dict(job, reps=3000, ro_stors=[0]) for job in CONFUSION_JOBS],
                                       batch_size=len(CONFUSION_JOBS))
confusion_job_ids = list(herald_runner.last_job_ids)
walks = walk_runner.execute(overrides=WALK_JOBS, batch_size=3)
walk_job_ids = list(walk_runner.last_job_ids)
print("confusion jobs", confusion_job_ids)
print("walk jobs", walk_job_ids)
