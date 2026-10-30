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
# # Fock 1, 2, 3 preparation in M1 at 0.45 mA (guan + Claude, 2026-10-28)
#
# Aim: how well we prepare Fock 1, 2, 3 in M1. Starts from jonginn's calibration at the
# 0.45 mA coupler set point. Config branch `agp_0.45mA` (log: ./branches.jsonl).

# %% Cell 1: station on jonginn's 0.45 mA snapshots, and the config branch.
import os
from contextlib import ExitStack, contextmanager

import numpy as np
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.branch_manager import BranchManager
from job_server import JobClient

config_dict = {
    "hardware_config": "CFG-HW-20261028-00035",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261027-00052",
    "floquet_storage_swap": "CFG-FL-20261027-00074",
}

client = JobClient()
client.print_queue()

station = MultimodeStation(
    user="guan",
    experiment_name="261028_yoko450ua_calib",
    project="AGP",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)
print("coupler set point (A):", station.hardware_cfg.hw.yoko_coupler.current)

bm = BranchManager(station)
print(bm.commit("agp_0.45mA", note="jonginn's 0.45 mA calibration, start of Fock 1-3 prep check"))

# %% Cell 2: single shot g/e, the readout baseline. Writes phase, threshold, Ig, Ie and the
# confusion matrix into station RAM (as single_qubit_autocalibrate_v2.py), then commits the branch.
# No active reset: it would act on the old threshold.
import experiments as meas

singleshot_defaults = AttrDict(dict(
    reps=5000, relax_delay=2000, check_f=False,
    active_reset=False, man_reset=False, storage_reset=False,
    qubit=0, qubits=[0], pulse_manipulate=False,
    cavity_freq=4984.373226159381, cavity_gain=400, cavity_length=2,
    prepulse=False, pre_sweep_pulse=None, gate_based=True,
))


def singleshot_postproc(station, expt):
    expt.analyze(station=station, subdir=station.autocalib_path)
    station.hardware_cfg.device.readout.update(expt.readout_cfg)


def ge_direction_deg(expt):
    """Direction of median(e) - median(g) in the raw I/Q plane; near 0 when the phase is right."""
    d = expt.data
    return np.degrees(np.arctan2(np.median(d["Qe"]) - np.median(d["Qg"]),
                                 np.median(d["Ie"]) - np.median(d["Ig"])))


ss_runner = CharacterizationRunner(
    station=station, ExptClass=meas.HistogramExperiment, default_expt_cfg=singleshot_defaults,
    postprocessor=singleshot_postproc, job_client=client, use_queue=True, show=False,
)
old_phase = station.hardware_cfg.device.readout.phase[0]
ss = ss_runner.execute(go_kwargs=dict(analyze=False, display=False))
print("job", ss_runner.last_job_ids, "raw e-g direction (deg):", ge_direction_deg(ss))
print("phase", old_phase, "->", station.hardware_cfg.device.readout.phase[0])
ss.display(station=station)
print(bm.commit("agp_0.45mA", note="single shot ge baseline, no active reset"))

# %% Cell 3: single shot g/e/f, to look at the g vs f separation. Runs on the phase that cell 2
# just set. No postprocessor here (as run on 10-28); since 10-29 the analysis keeps the g-e axis
# with f shots, so singleshot_postproc would also work on this run.
gef_runner = CharacterizationRunner(
    station=station, ExptClass=meas.HistogramExperiment, default_expt_cfg=singleshot_defaults,
    job_client=client, use_queue=True, show=False,
)
ss_gef = gef_runner.execute(check_f=True)
print("job", gef_runner.last_job_ids, "raw e-g direction (deg):", ge_direction_deg(ss_gef))
ss_gef.display(station=station)

# %% Cell 4: readout confusion for Fock 0..3 in M1, read in M1 (one job). C[prepared, measured].
# Settings as the 2026-10-23 reference (agp_layers_hw_check.py cell 2, JOB-20261023-00444,
# condition number 1.36): Qsim's prepulse, multiparity readout, active reset with the threshold
# from cell 2, relax_delay=500 us (qubit + M only).
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS

STORS = [4, 5, 6, 7]
calibration_defaults = AttrDict(dict(
    reps=100, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=STORS, pre_relax_delay=100, relax_delay=500,
    prepulse=True, postpulse=True, init_fock=False, init_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    swap_stors=STORS, ro_stor=0, update_phases=True, detunings=[0.0] * len(STORS),
))
calibration_defaults.update(ACTIVE_RESET_DEFAULTS)
calibration_defaults.update(FLOQUET_DEFAULTS)
calibration_runner = CharacterizationRunner(
    station=station, ExptClass=MultiParityConfusionMatrixExperiment,
    default_expt_cfg=calibration_defaults, job_client=client, use_queue=True, show=False,
)
calibration = calibration_runner.execute(reps=600)
confusion = np.asarray(calibration.data["multiparity_confusion_matrix"])
print("job", calibration_runner.last_job_ids)
print("C[prepared, measured]:\n", np.round(confusion, 3), "\ncondition number", np.linalg.cond(confusion))
calibration.display()

# %% [markdown]
# ## Where does the Fock 2/3 infidelity come from?
# The Fock N prep plays, for i < N: `g{i}-e{i}`, `e{i}-f{i}`, `f{i}-g{i+1}`, all read from
# `device.multiphoton` (qsim_base.prep_man_fock_state, MM_base.prepulse_creator2.multiphoton).
# The Fock 3 sequence holds the Fock 1 and Fock 2 sequences as prefixes, so one set of
# truncations covers all three.

# %% Cell 5: ladder truncation. Play the first k of the 9 Fock 3 pulses, then read the qubit
# (g/e/f, one shot). Ideal: g, e, f, g, e, f, g, e, f, g. A pulse that fails leaves the qubit in
# the wrong state. Shots are assigned to the nearest g/e/f blob of cell 3 (same phase) and
# corrected with the 3x3 assignment matrix of those blobs; no fit.
# The histogram's e branch (one ge pi after the prefix) is acquired too: the analysis needs it.
def fock_prep_pulses(n_photons):
    """Same list as qsim_base.prep_man_fock_state(man_no=1, str(n))."""
    seq = []
    for i in range(n_photons):
        seq += [["multiphoton", f"g{i}-e{i}", "pi", 0],
                ["multiphoton", f"e{i}-f{i}", "pi", 0],
                ["multiphoton", f"f{i}-g{i + 1}", "pi", 0]]
    return seq


LADDER = fock_prep_pulses(3)
IDEAL = ["g"] + ["e", "f", "g"] * 3  # qubit state after each pulse
truncation_overrides = [dict(prepulse=k > 0, pre_sweep_pulse=LADDER[:k]) for k in range(len(LADDER) + 1)]

ladder_runner = CharacterizationRunner(
    station=station, ExptClass=meas.HistogramExperiment, default_expt_cfg=singleshot_defaults,
    job_client=client, use_queue=True, show=False,
)
# qubit + M1, no active reset: relax_delay 2500 us (docs/running_experiments.md item 10)
truncations = ladder_runner.execute(overrides=truncation_overrides, batch_size=3,
                                    reps=2000, relax_delay=2500)
print("jobs", ladder_runner.last_job_ids)


def blob_centers(expt):
    d = expt.data
    return {s: np.median(d["I" + s]) + 1j * np.median(d["Q" + s]) for s in "gef"}


def assign(shots, centers):
    """Index (0=g, 1=e, 2=f) of the nearest blob center for each complex shot."""
    c = np.array([centers[s] for s in "gef"])
    return np.argmin(np.abs(shots[:, None] - c[None, :]), axis=1)


def assignment_matrix(expt, centers):
    """A[prepared, assigned] from the g/e/f histogram shots."""
    d = expt.data
    return np.array([np.bincount(assign(d["I" + s] + 1j * d["Q" + s], centers), minlength=3) / len(d["I" + s])
                     for s in "gef"])


def ladder_populations(truncations, ref):
    centers = blob_centers(ref)
    A = assignment_matrix(ref, centers)
    raw, corrected = [], []
    for expt in truncations:
        d = expt.data
        p = np.bincount(assign(d["Ig"] + 1j * d["Qg"], centers), minlength=3) / len(d["Ig"])
        raw.append(p)
        corrected.append(np.linalg.solve(A.T, p))
    return np.array(raw), np.array(corrected), A


ladder_raw, ladder_pop, A_gef = ladder_populations(truncations, ss_gef)
print("assignment A[prepared, assigned] (g, e, f):\n", np.round(A_gef, 3))
print(" k  last pulse     ideal   P(g)   P(e)   P(f)   (corrected)")
for k, p in enumerate(ladder_pop):
    name = "-" if k == 0 else LADDER[k - 1][1]
    print(f"{k:2d}  {name:12s}  {IDEAL[k]:>5s}  " + "  ".join(f"{x:5.3f}" for x in p))

fig, ax = plt.subplots(figsize=(7, 3.5))
for j, s in enumerate("gef"):
    ax.plot(range(len(ladder_pop)), ladder_pop[:, j], "o-", label=f"P({s})")
ax.set_xticks(range(len(ladder_pop)), ["0"] + [p[1] for p in LADDER], rotation=45)
ax.set(ylabel="population (corrected)", title="Fock 3 ladder, truncated after k pulses")
ax.legend()
fig.tight_layout()

# %% Cell 6: photon-number distribution with a number-selective qubit probe after Fock N prep.
# Peaks at f_ge(n) = f_ge + n*chi_ge (chi_ge = -0.302 MHz: -0.30 MHz per photon; measured 2026-10-28,
# JOB-20261028-00387: the Fock 1 peak is below the vacuum peak). Flat-top probe,
# 3.3 us flat: the first spectral zero is at 1/3.3 us = 0.30 MHz, on the next peak. Gain from pi_ge
# area (gauss sigma 35 ns, gain 9061) / 3.3 us: about a pi pulse, so peak heights are relative;
# P(n) is normalized by the vacuum n=0 peak. avgi -> P(e) with the cell-2 Ig, Ie.
# A qubit left in f after prep reads near e in I, so it shows as a raised off-peak background.
F_GE0 = station.hardware_cfg.device.multiphoton.pi["gn-en"].frequency[0]
CHI = station.hardware_cfg.device.manipulate.chi_ge[0]
PROBE_LEN, PROBE_SIGMA = 3.3, 0.02
probe_gain = int(round(9061 * 0.035 * np.sqrt(2 * np.pi) / (PROBE_LEN + PROBE_SIGMA * np.sqrt(2 * np.pi))))
print("probe gain", probe_gain, "peaks at", np.round(F_GE0 + CHI * np.arange(5), 3))

spec_defaults = AttrDict(dict(
    start=F_GE0 + 5 * CHI, step=0.02, expts=95, reps=200, rounds=1,
    length=PROBE_LEN, sigma=PROBE_SIGMA, gain=probe_gain,
    qubit=0, qubits=[0], active_reset=False, man_reset=False, storage_reset=False,
    prepulse=True, gate_based=True, pre_sweep_pulse=[], relax_delay=2500,
))
spec_runner = CharacterizationRunner(
    station=station, ExptClass=meas.PulseProbeSpectroscopyExperiment, default_expt_cfg=spec_defaults,
    job_client=client, use_queue=True, show=False,
)
spec_overrides = [dict(prepulse=n > 0, pre_sweep_pulse=fock_prep_pulses(n)) for n in range(4)]
specs = spec_runner.execute(overrides=spec_overrides, batch_size=2)
print("jobs", spec_runner.last_job_ids)


def p_excited(expt, readout):
    return (np.asarray(expt.data["avgi"]) - readout.Ig[0]) / (readout.Ie[0] - readout.Ig[0])


def number_distribution(specs, readout, n_max=4):
    """P(n) for each prepared N: fit S_N(f) = b + sum_n a_n R(f - d_n), a_n >= 0 (NNLS).
    R is the measured vacuum spectrum minus its baseline: the probe's own response, side lobes
    included (this flat-top probe has lobes of ~0.3 at +-0.3 MHz, on the next photon line, so
    peak heights alone are wrong). d_n are the measured main-peak offsets of each prepared n;
    n_max is extrapolated. Model assumption: the line shape does not change with n (the fit rms
    says how well that holds)."""
    x = np.asarray(specs[0].data["xpts"])[1:-1]
    S = [p_excited(e, readout)[1:-1] for e in specs]
    R = S[0] - np.percentile(S[0], 10)
    peaks = [x[np.argmax(s)] for s in S]
    d = [p - peaks[0] for p in peaks]
    while len(d) <= n_max:
        d.append(d[-1] + (d[-1] - d[-2]))
    M = np.column_stack([np.interp(x[:, None] - np.array(d)[None, :], x, R, left=0, right=0)
                         .reshape(len(x), -1), np.ones_like(x)])
    fits = []
    for s in S:
        coef, res = nnls(M, s)
        fits.append(dict(P=coef[:-1] / coef[:-1].sum(), bg=coef[-1], rms=res / np.sqrt(len(x)), fit=M @ coef))
    return fits, np.array(d), x, S


fits, offsets, x_spec, S_spec = number_distribution(specs, station.hardware_cfg.device.readout)
print("peak offsets (MHz):", np.round(offsets, 3))
for n, f in enumerate(fits):
    print(f"prep {n}: P(n=0..4) {np.round(f['P'], 3)}  background {f['bg']:.3f}  rms {f['rms']:.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
for n, expt in enumerate(specs):
    ax.plot(np.asarray(expt.data["xpts"])[1:-1], p_excited(expt, station.hardware_cfg.device.readout)[1:-1],
            ".-", label=f"prep Fock {n}")
for n in range(5):
    ax.axvline(F_GE0 + CHI * n, color="0.7", ls=":")
ax.set(xlabel="probe frequency (MHz)", ylabel="P(e)", title="number-selective qubit spectroscopy, M1")
ax.legend()
fig.tight_layout()

# %% Cell 7: amplitude Rabi of the e_n-f_n pulse after preparing |e_n>, read g/e/f in one shot.
# Prefix: Fock n prep + g_n-e_n. Then one e_n-f_n pulse whose multiphoton gain is set for that job
# only (`multiphoton_entry`; the runner sends the station config as it is at submit time, and the
# entry is restored after). n=0 is the reference (cell 5: e0-f0 gives P(f)=1.000), n=1 is the weak
# pulse of the Fock 2 stage (cell 5: P(f)=0.933, P(g)=0.05). Ideal: e<->f Rabi, P(g)=0 at all gains.
# 2026-10-28: blob centers move <= 3 units with up to 3 photons in M1 (blob spacing 90-150,
# sigma ~30), so g/e/f assignment does not depend on the M1 state.
@contextmanager
def multiphoton_entry(station, kind, transition, n, **values):
    """Set device.multiphoton[kind][transition][key][n] = value for the block, then restore."""
    entry = station.hardware_cfg.device.multiphoton[kind][transition]
    old = {key: entry[key][n] for key in values}
    try:
        for key, value in values.items():
            entry[key][n] = value
        yield
    finally:
        for key, value in old.items():
            entry[key][n] = value


def ef_amplitude_scan(n, gains, reps=1000):
    prefix = fock_prep_pulses(n) + [["multiphoton", f"g{n}-e{n}", "pi", 0]]
    pulse = ["multiphoton", f"e{n}-f{n}", "pi", 0]
    expts = []
    for gain in gains:
        with multiphoton_entry(station, "pi", "en-fn", n, gain=int(gain)):
            expts.append(ladder_runner.execute(prepulse=True, pre_sweep_pulse=prefix + [pulse],
                                               reps=reps, relax_delay=2500))
    return expts


EF_GAINS = np.arange(0, 15001, 1500)
ef_scans = {n: ef_amplitude_scan(n, EF_GAINS) for n in (0, 1)}
for n, expts in ef_scans.items():
    _, pop, _ = ladder_populations(expts, ss_gef)
    nominal = station.hardware_cfg.device.multiphoton.pi["en-fn"].gain[n]
    print(f"e{n}-f{n} (nominal gain {nominal}): gain  P(g)  P(e)  P(f)")
    for gain, p in zip(EF_GAINS, pop):
        print(f"  {gain:6d}  " + "  ".join(f"{x:6.3f}" for x in p))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for ax, (n, expts) in zip(axs, ef_scans.items()):
    _, pop, _ = ladder_populations(expts, ss_gef)
    for j, s in enumerate("gef"):
        ax.plot(EF_GAINS, pop[:, j], "o-", label=f"P({s})")
    ax.axvline(station.hardware_cfg.device.multiphoton.pi["en-fn"].gain[n], color="0.7", ls=":")
    ax.set(title=f"e{n}-f{n} after |e{n}>", xlabel="gain")
axs[0].set_ylabel("population (corrected)")
axs[0].legend()
fig.tight_layout()

# %% Cell 8: hold |f,n> for a time t, then read g/e/f. Cell 7 (2026-10-28, JOB-20261028-00606..00731):
# e1-f1 has its pi at the nominal gain, but P(g) follows P(f) (0.064 at pi, ~0 at 0 and 2pi), and
# e0-f0 has P(g) ~ 0 everywhere; with cell 5 the share in g is ~0 / 0.06 / 0.135 for n = 0 / 1 / 2.
# If P(g) grows with t (faster for larger n), |f,n> decays toward g; if it stays flat, the leak
# happens during the e-f pulse. f -> e decay (T1_ef ~ 40 us) shows as P(e) growing with t.
# Result 2026-10-28 (JOB-20261028-00735..00762): |f,n>=1 decays to g, not e: tau ~6 us (n=1),
# ~2.4 us (n=2), from log-linear rates on the first points. |f,0> shows only f -> e (single-exp
# T1 ~120 us). So the e->g signal of cells 5 and 7 is decay of |f,n> during and after the pulses.
HOLD_US = [0, 2, 5, 10, 20, 40]


def f_state_prefix(n):
    return fock_prep_pulses(n) + [["multiphoton", f"g{n}-e{n}", "pi", 0],
                                  ["multiphoton", f"e{n}-f{n}", "pi", 0]]


hold_overrides = [dict(prepulse=True, pre_sweep_pulse=f_state_prefix(n) + ([["wait", t]] if t > 0 else []))
                  for n in (0, 1, 2) for t in HOLD_US]
holds = ladder_runner.execute(overrides=hold_overrides, batch_size=3, reps=1000, relax_delay=2500)
print("jobs", ladder_runner.last_job_ids)
_, hold_pop, _ = ladder_populations(holds, ss_gef)
hold_pop = hold_pop.reshape(3, len(HOLD_US), 3)
for n in range(3):
    print(f"|f,{n}> hold (us)  P(g)   P(e)   P(f)")
    for t, p in zip(HOLD_US, hold_pop[n]):
        print(f"   {t:5.0f}      " + "  ".join(f"{x:6.3f}" for x in p))

fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for n, ax in enumerate(axs):
    for j, s in enumerate("gef"):
        ax.plot(HOLD_US, hold_pop[n, :, j], "o-", label=f"P({s})")
    ax.set(title=f"hold |f,{n}>", xlabel="wait (us)")
axs[0].set_ylabel("population (corrected)")
axs[0].legend()
fig.tight_layout()

# %% Cell 9: which photon number does |f,n> decay into? Number-selective probe after (a) Fock 0..3
# prep (references: the measured line of each n, so no line-shape model) and (b) |f,1> held 20 us,
# |f,2> held 10 us (cell 8: > 90% decayed to g). Each test spectrum is fit as a non-negative mix
# of the references plus a flat background (qubit left in e/f is off-resonant with the ge probe).
# Probe: Gaussian, sigma 0.8 us (flat part 0.02 us), 4 sigma = 3.2 us. Envelope memory is 32768
# samples per generator IN TOTAL (qick: addr + length <= maxlen). On the qubit generator (ch 2,
# 16 samples per 399 MHz cycle) the standard pi/hpi pulses take 4992 and every qubit gate of the
# prepulse adds its own 896 (not shared), so with |f,2> (8 qubit gates) the probe gets <= 20608
# samples: sigma <= 0.80 us. sigma 1.2 / 1.15 / 1.05 us failed on the device (JOB-20261028-00775,
# -00777, -00780: "buffer length must be 32768 samples or less"); the mock does not check this.
# Lines overlap more (spectral sigma ~0.2 MHz, spacing ~0.28 MHz); the fit uses measured references.
GAUSS_SIGMA = 0.8
gauss_gain = int(round(9061 * 0.035 / GAUSS_SIGMA * 1.05))  # pi_ge area / sigma, +5% for the 4-sigma cut
sel_defaults = AttrDict({**spec_defaults, "sigma": GAUSS_SIGMA, "length": 0.02, "gain": gauss_gain,
                         "start": F_GE0 + 4.6 * CHI, "step": 0.02, "expts": 85})
# Result 2026-10-28 (JOB-20261028-00781..00786): |f,1> + 20 us -> 0.96 Fock 0; |f,2> + 10 us -> 0.92 Fock 1.
# |f,n> decays to |g,n-1>: qubit excitation and one photon lost. This is the 2->0 and 3->1 of cell 4.
sel_runner = CharacterizationRunner(
    station=station, ExptClass=meas.PulseProbeSpectroscopyExperiment, default_expt_cfg=sel_defaults,
    job_client=client, use_queue=True, show=False,
)
DECAY_TESTS = {"f1 + 20 us": f_state_prefix(1) + [["wait", 20]], "f2 + 10 us": f_state_prefix(2) + [["wait", 10]]}
sel_overrides = ([dict(prepulse=n > 0, pre_sweep_pulse=fock_prep_pulses(n)) for n in range(4)]
                 + [dict(prepulse=True, pre_sweep_pulse=seq) for seq in DECAY_TESTS.values()])
sel_specs = sel_runner.execute(overrides=sel_overrides, batch_size=2)
print("jobs", sel_runner.last_job_ids, "probe gain", gauss_gain)


def mix_of_references(spec, references, readout):
    """Non-negative weights w_n (sum-normalized) with spec ~ b + sum_n w_n R_n; R_n = reference
    spectrum minus its 10th-percentile baseline."""
    s = p_excited(spec, readout)[1:-1]
    R = [p_excited(r, readout)[1:-1] for r in references]
    M = np.column_stack([r - np.percentile(r, 10) for r in R] + [np.ones_like(s)])
    coef, res = nnls(M, s)
    w = coef[:-1]
    return dict(w=w / w.sum(), scale=w.sum(), bg=coef[-1], rms=res / np.sqrt(len(s)), fit=M @ coef)


ro = station.hardware_cfg.device.readout
refs, tests = sel_specs[:4], dict(zip(DECAY_TESTS, sel_specs[4:]))
x_sel = np.asarray(refs[0].data["xpts"])[1:-1]
print("reference peaks (MHz):", [round(x_sel[np.argmax(p_excited(r, ro)[1:-1])], 3) for r in refs])
decay_mix = {name: mix_of_references(spec, refs, ro) for name, spec in tests.items()}
for name, m in decay_mix.items():
    print(f"{name}: weights of Fock 0..3 references {np.round(m['w'], 3)}  scale {m['scale']:.2f}  "
          f"bg {m['bg']:.3f}  rms {m['rms']:.3f}")

fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
for n, r in enumerate(refs):
    axs[0].plot(x_sel, p_excited(r, ro)[1:-1], ".-", label=f"prep Fock {n}")
for name, spec in tests.items():
    axs[1].plot(x_sel, p_excited(spec, ro)[1:-1], ".", label=name)
    axs[1].plot(x_sel, decay_mix[name]["fit"], "-", color="0.4")
axs[0].set(ylabel="P(e)", title="references: Gaussian number-selective probe")
axs[1].set(ylabel="P(e)", xlabel="probe frequency (MHz)", title="|f,n> after decay, with fit to the references")
for ax in axs:
    ax.legend()
fig.tight_layout()

# %% Cell 10: the |f,n> -> g decay at 0.25 mA. On 10-23 at 0.25 mA, Fock 2/3 confusion diagonals were
# 0.835 / 0.820 (today at 0.45 mA: 0.775 / 0.748). The sideband shifts also differ: f1-g2 / f2-g3
# are -0.57 / -1.14 MHz from f0-g1 at 0.25 mA but -0.64 / -2.88 MHz at 0.45 mA.
# Second station on the 0.25 mA snapshots (guan, latest as of 2026-10-28), its own single shot
# (phase drifts daily; same readout frequency and length as 0.45 mA), then cell 8's hold scan.
config_025 = {
    "hardware_config": "CFG-HW-20261026-00006",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261023-00003",
    "floquet_storage_swap": "CFG-FL-20261023-00002",
}
station_025 = MultimodeStation(
    user="guan", experiment_name="261028_yoko450ua_calib", project="AGP", log_measurements=True,
    hardware_config=config_025["hardware_config"], multiphoton_config=config_025["multiphoton_config"],
    storage_man_file=config_025["man1_storage_swap"], floquet_file=config_025["floquet_storage_swap"],
)
print("coupler set point (A):", station_025.hardware_cfg.hw.yoko_coupler.current)
bm_025 = BranchManager(station_025)

ss_025 = CharacterizationRunner(
    station=station_025, ExptClass=meas.HistogramExperiment, default_expt_cfg=singleshot_defaults,
    postprocessor=singleshot_postproc, job_client=client, use_queue=True, show=False,
).execute(go_kwargs=dict(analyze=False, display=False))
print("phase ->", station_025.hardware_cfg.device.readout.phase[0], "raw e-g direction (deg):", ge_direction_deg(ss_025))
print(bm_025.commit("agp_0.25mA", note="guan 0.25 mA snapshots of 10-23/26, single shot ge 2026-10-28"))

ladder_runner_025 = CharacterizationRunner(
    station=station_025, ExptClass=meas.HistogramExperiment, default_expt_cfg=singleshot_defaults,
    job_client=client, use_queue=True, show=False,
)
ss_gef_025 = ladder_runner_025.execute(check_f=True)
print("raw e-g direction (deg) after the update:", ge_direction_deg(ss_gef_025))

holds_025 = ladder_runner_025.execute(
    overrides=[dict(prepulse=True, pre_sweep_pulse=f_state_prefix(n) + ([["wait", t]] if t > 0 else []))
               for n in (1, 2) for t in HOLD_US],
    batch_size=3, reps=1000, relax_delay=2500)
print("jobs", ladder_runner_025.last_job_ids)
_, hold_pop_025, A_gef_025 = ladder_populations(holds_025, ss_gef_025)
hold_pop_025 = hold_pop_025.reshape(2, len(HOLD_US), 3)
print("assignment A (g, e, f):\n", np.round(A_gef_025, 3))
for i, n in enumerate((1, 2)):
    print(f"0.25 mA |f,{n}> hold (us)  P(g)   P(e)   P(f)     | 0.45 mA P(f)")
    for t, p, p45 in zip(HOLD_US, hold_pop_025[i], hold_pop[n]):
        print(f"   {t:5.0f}      " + "  ".join(f"{x:6.3f}" for x in p) + f"    | {p45[2]:6.3f}")

# Result 2026-10-28 (JOB-20261028-00803..00816): the decay to g is ~20x slower at 0.25 mA. Rough
# log-linear rates over 0..40 us: |f,1> total tau ~74 us, ~0.64 of it to g (tau_fg ~115 us; 0.45 mA:
# ~6 us); |f,2> total ~42 us, ~0.83 to g (tau_fg ~50 us; 0.45 mA: ~2.4 us). n=2/n=1 ratio ~2.3 at both.
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for i, n in enumerate((1, 2)):
    ax = axs[i]
    ax.plot(HOLD_US, hold_pop[n, :, 2], "o-", color="#1baf7a", lw=2, label="P(f), 0.45 mA")
    ax.plot(HOLD_US, hold_pop[n, :, 0], "o-", color="#2a78d6", lw=2, label="P(g), 0.45 mA")
    ax.plot(HOLD_US, hold_pop_025[i, :, 2], "s--", color="#1baf7a", lw=2, label="P(f), 0.25 mA")
    ax.plot(HOLD_US, hold_pop_025[i, :, 0], "s--", color="#2a78d6", lw=2, label="P(g), 0.25 mA")
    ax.set(title=f"hold |f,{n}>", xlabel="wait (us)")
    ax.grid(True, color="#e4e3de")
axs[0].set_ylabel("population (corrected)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.08))
fig.suptitle("|f,n> -> |g,n-1> decay: ~20x faster at 0.45 mA than at 0.25 mA (2026-10-28)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(station.plot_path, "261028_f_decay_setpoint_compare.png"), dpi=110, bbox_inches="tight")

# %% Cell 11: the same decay check at 0.3 mA (seb, 2026-10-29) and 0.5 mA (jonginn's last ladder
# calibration of 10-28 evening, CFG-HW-20261028-00169, found from his job configs). 0.3 mA has only
# n=0,1 calibrated (g2-e2 +4.0 MHz, e2-f2 3421.97: placeholders), so for the |f,2> holds g2-e2 and
# e2-f2 are set in RAM for those jobs only: f_ge - 0.57 MHz and f_ef - 0.40 MHz, gain 7420 (the
# offsets the other set points show; the 35 ns pulses are several MHz wide). Not written to config.
def decay_check(label, configs, n_overrides=None, reps=1000):
    """Station on `configs`, ge single shot (phase, threshold; committed as branch `label`),
    g/e/f reference, then hold |f,n> for n = 1, 2 over HOLD_US. Returns a dict of the parts."""
    st = MultimodeStation(
        user="guan", experiment_name="261028_yoko450ua_calib", project="AGP", log_measurements=True,
        hardware_config=configs["hardware_config"], multiphoton_config=configs["multiphoton_config"],
        storage_man_file=configs["man1_storage_swap"], floquet_file=configs["floquet_storage_swap"],
    )
    ss = CharacterizationRunner(
        station=st, ExptClass=meas.HistogramExperiment, default_expt_cfg=singleshot_defaults,
        postprocessor=singleshot_postproc, job_client=client, use_queue=True, show=False,
    ).execute(go_kwargs=dict(analyze=False, display=False))
    ids = BranchManager(st).commit(label, note="decay check 2026-10-29, single shot ge")
    runner = CharacterizationRunner(
        station=st, ExptClass=meas.HistogramExperiment, default_expt_cfg=singleshot_defaults,
        job_client=client, use_queue=True, show=False,
    )
    ref = runner.execute(check_f=True)
    holds = []
    for n in (1, 2):
        with ExitStack() as stack:
            for (kind, transition, idx), values in (n_overrides or {}).get(n, {}).items():
                stack.enter_context(multiphoton_entry(st, kind, transition, idx, **values))
            holds += runner.execute(
                overrides=[dict(prepulse=True, pre_sweep_pulse=f_state_prefix(n) + ([["wait", t]] if t > 0 else []))
                           for t in HOLD_US],
                batch_size=3, reps=reps, relax_delay=2500)
    _, pop, A = ladder_populations(holds, ref)
    print(label, "coupler", st.hardware_cfg.hw.yoko_coupler.current, "fid", ss.data["fids"],
          "e-g direction after update", round(ge_direction_deg(ref), 1), "configs", ids)
    return dict(station=st, single_shot=ss, ref=ref, holds=holds, pop=pop.reshape(2, len(HOLD_US), 3), A=A)


config_030 = {"hardware_config": "CFG-HW-20261028-00170", "multiphoton_config": "CFG-MP-20260121-00001",
              "man1_storage_swap": "CFG-M1-20261020-00010", "floquet_storage_swap": None}
config_050 = {"hardware_config": "CFG-HW-20261028-00169", "multiphoton_config": "CFG-MP-20260121-00001",
              "man1_storage_swap": "CFG-M1-20261028-00047", "floquet_storage_swap": "CFG-FL-20261027-00074"}
f_ge_030, f_ef_030 = 3561.498, 3417.978  # CFG-HW-20261028-00170 device.qubit
overrides_030 = {2: {("pi", "gn-en", 2): {"frequency": f_ge_030 - 0.57},
                     ("pi", "en-fn", 2): {"frequency": f_ef_030 - 0.40, "gain": 7420}}}
decay = {
    "0.25 mA": dict(pop=hold_pop_025),  # cell 10
    "0.45 mA": dict(pop=hold_pop[1:3]),  # cell 8
}
decay["0.30 mA"] = decay_check("agp_0.30mA", config_030, n_overrides=overrides_030)
decay["0.50 mA"] = decay_check("agp_0.50mA", config_050)
for label in ("0.30 mA", "0.50 mA"):
    for i, n in enumerate((1, 2)):
        print(f"{label} |f,{n}> hold (us)  P(g)   P(e)   P(f)")
        for t, p in zip(HOLD_US, decay[label]["pop"][i]):
            print(f"   {t:5.0f}      " + "  ".join(f"{x:6.3f}" for x in p))

# %% Cell 12: decay rates and the transition frequency nu_n = E(f,n) - E(g,n-1) per set point.
# Rate (model, rough): fit P(f)(t) = A exp(-G t) + C over the hold scan (C: the slow part, e.g. |f,0>
# left from earlier pulses); share to g b = -slope of P(g) against P(f) over the same points;
# G_fg = G * b. nu_n = s_n + 2 * omega_M1, omega_M1 = f_ge + f_ef - s_0, with s_n the calibrated
# f{n}-g{n+1} frequency of that set point (includes the sideband drive's Stark shift: +-few MHz).
from scipy.optimize import curve_fit

SET_POINT_CONFIGS = {"0.25 mA": "CFG-HW-20261026-00006", "0.30 mA": "CFG-HW-20261028-00170",
                     "0.45 mA": "CFG-HW-20261028-00046", "0.50 mA": "CFG-HW-20261028-00169"}


def transition_freqs(hw_version):
    c = yaml.safe_load(open(os.path.join(r"C:\python\multimode_expts\configs\versions\hardware_config",
                                         hw_version + ".yml"), encoding="utf-8"))
    q, s_n = c["device"]["qubit"], c["device"]["multiphoton"]["pi"]["fn-gn+1"]["frequency"]
    omega_m = q["f_ge"][0] + q["f_ef"][0] - s_n[0]
    s_n = list(s_n[:3])
    if abs(s_n[2] - s_n[1]) > 3:  # placeholder (0.3 mA: f2-g3 = 1996.13): estimate from the n=0,1 spacing
        s_n[2] = s_n[1] + (s_n[1] - s_n[0])
    return omega_m, {n: s_n[n] + 2 * omega_m for n in (1, 2)}


def decay_rate(t, pop):
    """(G_fg [1/us], G_total, share to g, fit params) from one hold scan; pop[:, (g, e, f)]."""
    t, pf, pg = np.asarray(t, float), pop[:, 2], pop[:, 0]
    (A, G, C), _ = curve_fit(lambda t, A, G, C: A * np.exp(-G * t) + C, t, pf, p0=[pf[0] - pf[-1], 0.1, pf[-1]],
                             bounds=([0, 1e-4, -0.1], [1.2, 5, 1]))
    b = -np.polyfit(pf, pg, 1)[0]
    return G * b, G, b, (A, G, C)


rates = []
for label, d in decay.items():
    omega_m, nu = transition_freqs(SET_POINT_CONFIGS[label])
    for i, n in enumerate((1, 2)):
        g_fg, g_tot, b, _ = decay_rate(HOLD_US, d["pop"][i])
        rates.append(dict(set_point=label, n=n, nu=nu[n], omega_m1=omega_m, G_fg=g_fg, G=g_tot, share_g=b))
        print(f"{label} n={n}: nu = {nu[n]:9.2f} MHz  G_fg = {g_fg:.4f} /us (tau {1 / g_fg:6.1f} us)  "
              f"G_total {g_tot:.4f}  share to g {b:.2f}  G_fg/n {g_fg / n:.4f}")

# %% Cell 13: report figure. Rate per photon against nu_n, Lorentzian fit (model, 8 points, all on
# the high side of the peak; nu includes the sideband Stark shift). 0.45 mA n=2: its f2-g3 entry was
# calibrated at gain 10000 (all others 8000), so its s_2 has a different Stark shift; nu_2 is taken
# from the n-spacing instead (11962.7 MHz). Fit (2026-10-29): nu0 11961.4 +- 0.2 MHz, FWHM 3.5 +- 0.2 MHz,
# peak G_fg/n 0.43 /us.
def lorentz(nu, A, nu0, w):
    return A * (w / 2) ** 2 / ((nu - nu0) ** 2 + (w / 2) ** 2)


for r in rates:
    r["nu_used"] = 11962.7 if (r["set_point"] == "0.45 mA" and r["n"] == 2) else r["nu"]
x_r = np.array([r["nu_used"] for r in rates])
y_r = np.array([r["G_fg"] / r["n"] for r in rates])
lor_p, lor_cov = curve_fit(lorentz, x_r, y_r, p0=[0.5, 11961, 3], sigma=0.15 * y_r + 0.003)
print("Lorentzian: A %.3f /us, nu0 %.2f MHz, FWHM %.2f MHz" % tuple(lor_p), "+-", np.round(np.sqrt(np.diag(lor_cov)), 2))

COL = {"0.25 mA": "#2a78d6", "0.30 mA": "#eb6834", "0.45 mA": "#1baf7a", "0.50 mA": "#eda100"}
with mpl.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for label in ("0.25 mA", "0.30 mA", "0.45 mA", "0.50 mA"):
        axs[0].plot(HOLD_US, decay[label]["pop"][0, :, 2], "o-", color=COL[label], lw=2, label=label)
    axs[0].set(xlabel="wait in |f,1> (us)", ylabel="P(f) (assignment-corrected)", title="|f,1> survival vs coupler set point")
    axs[0].legend(frameon=False)
    nu_fine = np.linspace(11958, 11973, 400)
    axs[1].plot(nu_fine, lorentz(nu_fine, *lor_p), "-", color="0.55", lw=1.5,
                label=f"Lorentzian: {lor_p[1]:.1f} MHz, FWHM {lor_p[2]:.1f} MHz (model)")
    for r in rates:
        axs[1].plot(r["nu_used"], r["G_fg"] / r["n"], "o" if r["n"] == 1 else "s", ms=9, color=COL[r["set_point"]],
                    mec="white", mew=1.5)
    axs[1].plot([], [], "o", color="0.3", label="n=1"); axs[1].plot([], [], "s", color="0.3", label="n=2")
    axs[1].set_yscale("log")
    axs[1].set(xlabel="nu_n = E(f,n) - E(g,n-1)  (MHz)", ylabel="decay rate to g per photon (1/us)",
               title="|f,n> -> |g,n-1> rate vs transition frequency")
    axs[1].legend(frameon=False, fontsize=9, loc="lower left")
    for ax in axs:
        ax.grid(True, color="#e4e3de")
    fig.suptitle("Suspected TLS near 11.96 GHz: |f,n> -> |g,n-1> loss in M1 Fock prep (2026-10-28/29)", x=0.01, ha="left")
    fig.tight_layout()
    report_fig = os.path.join(station.plot_path, "261029_tls_12GHz_report.png")
    fig.savefig(report_fig, dpi=110, bbox_inches="tight", facecolor="white")
print("saved", report_fig)

# %% [markdown]
# ## Module gaps (running list): why this script has inline logic
# Each item is a helper or a block above that should move into a module, so the final notebook is
# only defaults -> pre/postprocessor -> runner.execute.
#
# 1. **Single-shot postprocessor** (cell 2): done 2026-10-29. `HistogramExperiment.analyze()`
#    sets `expt.readout_cfg`; every notebook's `singleshot_postproc` copies it into the station.
# 2. **`HistogramExperiment` with f shots** (cell 3): done 2026-10-29. The rotation stays g-e;
#    gf is read on I, ef on Q, and `data["assignment_matrix"]` is the 3x3 g/e/f matrix
#    (I against the ge threshold, then Q). The cell-5 helpers (`blob_centers`, `assign`,
#    `assignment_matrix`, nearest center) still run here; they move with item 3.
# 3. **`HistogramExperiment` has no prepared-state-only mode** (cell 5): with `check_e=False`
#    the analysis fails, so every truncation job also acquires an e branch we do not use. Missing:
#    a "read g/e/f populations after a prepulse" experiment that returns corrected populations.
# 4. **Fock-prep pulse list** (`fock_prep_pulses`, cell 5): duplicates
#    `qsim_base.prep_man_fock_state`, which is a program method and cannot be called from a
#    notebook without a program. Belongs as a plain function (e.g. in `MM_base`) that both use.
# 5. **`PulseProbeSpectroscopyExperiment`** (cell 6): returns only averaged I/Q, drops the first
#    and last points in analysis, and fits one Lorentzian. Missing: P(e) from the readout Ig/Ie,
#    and a number-resolved mode (peak heights at f_ge + n*chi, P(n) normalized to vacuum).
#    The flat-top probe has side lobes of ~0.3 on the next photon line, so heights are not P(n)
#    (`number_distribution`, cell 6, fits shifted copies of the vacuum response instead).
# 6. **Photon-number shift sign** (cell 6): the measured n-photon qubit line is at
#    f_ge + n*chi_ge (chi_ge < 0), but `multiphoton.pi['gn-en']` has g1-e1 at +0.30 MHz from g0-e0.
#    Nothing checks that the table entries agree with chi_ge. With broadband (35 ns) ladder pulses
#    this costs nothing; it breaks any number-selective use of the table.
# 7. **No per-job device override in the runner** (cell 7, `multiphoton_entry`): to sweep one
#    multiphoton entry inside a gate-based prepulse, we change the station config in RAM around each
#    submit. `ErrorAmplificationExperiment` plays only even pulse counts, and
#    `AmplitudeRabiGeneralExperiment` takes raw pulse lists and returns averaged I/Q, so neither is
#    a one-pulse Rabi with g/e/f readout after a gate-based Fock prep.
# 8. **Mock station does not check generator envelope memory** (cell 9): the 32768-sample limit is
#    the total of all envelopes on a generator. The standard qubit pulses take 4992 on ch 2, and
#    `custom_pulse` adds a new envelope for every prepulse gate (896 each, identical ones not
#    shared), so a long probe after a long prepulse passes the mock and fails only on the device.
