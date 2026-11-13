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
# # Multiparity shots with the qubit in f, heralds in the M1 ladder, and the walk floor (guan + Claude, 2026-11-13)
#
# A qubit in f at readout reads e in both parity readouts (I threshold only), so the shot decodes
# as n = 3 in every mode. Shots that read f in either parity readout are flagged with the g/e/f
# cloud centers of `JOB-20261113-00259` (FStateHoldExperiment, readout phase 273.0 deg; the jobs
# here used 271.7-274.6 deg). Each cut has its own confusion matrix, made from its own shots.
# Estimators: `fitting/qsim/f_postselect.py`; per-shot labels: `experiments/qsim/fock_prep_herald.py`.
# Summary of the day: vault `Lab/guan/AGP/2026/11/2026-11-13_f_state_floor_report.md`.
#
# Cells 4, 6, 8, 11, 12 wrote the vault entries of 2026-11-13; they run again only with
# `LOG_TO_VAULT = True`.

# %% Cell 1: imports, the f centers, loading, small table helpers.
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np

from experiments import MultimodeStation
from experiments.qsim.fock_prep_herald import labels_from_h5, select
from fitting.qsim import f_postselect
from theory.qsim.central_boson import fixed_n_hamiltonian, fock_number_traces

LOG_TO_VAULT = False  # the entries exist: ANA-20261113-122433, -122833, -142706, -145845
DATA = r"C:\experiments\261111AGPlowKerr\data"
HERALD_DIR = r"C:\experiments\261113_tls_recheck\data"
with h5py.File(rf"{HERALD_DIR}\JOB-20261113-00259_FStateHoldExperiment.h5", "r") as f:
    CENTERS = np.array(f["blob_centers_i"]) + 1j * np.array(f["blob_centers_q"])
MODE_COLORS = {"M1": "#2a78d6", "S1": "#eb6834", "S2": "#1baf7a", "S4": "#eda100", "S5": "#e87ba4"}


def load(job, kind="FockPopulationExperiment", folder=DATA):
    """-> (n mod 4 shots, no-f mask, cfg) of one job file, shape (points, shots)."""
    cfg, lab = labels_from_h5(rf"{folder}\{job}_{kind}.h5", CENTERS)
    return lab["n"], lab["no_f"], cfg


def mode_label(cfg):
    return "M1" if cfg["expt"]["ro_stor"] == 0 else f"S{cfg['expt']['ro_stor']}"


def at_m1_peaks(cycles, modes, total, window=4):
    """-> (storages, rows (cycle, M1, [storages], sum of storages, total)) at the local maxima of M1."""
    m1 = modes["M1"]
    peaks = [k for k in range(len(m1)) if m1[k] == m1[max(0, k - window):k + window + 1].max() and m1[k] > 1]
    storages = [m for m in modes if m != "M1"]
    return storages, [(cycles[k], m1[k], [modes[m][k] for m in storages], sum(modes[m][k] for m in storages),
                       total[k]) for k in peaks]


def at_m1_minima(cycles, m1, window=4):
    """-> [(cycle, M1)] at the local minima of M1 below 1.5."""
    return [(cycles[k], m1[k]) for k in range(len(m1))
            if m1[k] == m1[max(0, k - window):k + window + 1].min() and m1[k] < 1.5]


def style(ax, title, xlabel="Floquet cycle"):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# %% Cell 2: cycle 0 of the 11-11/12 jobs (Fock 2/3 in M1, read M1 or an empty storage).
SETS = [("0.45 mA, 11-11", "JOB-20261111-00630", ["00631", "00632", "00633", "00634", "00635"]
         + ["00636", "00637", "00638", "00639", "00640"]),
        ("0.45 mA, 11-12", "JOB-20261112-00176", ["00178", "00179", "00180", "00181", "00182"]),
        ("0.25 mA, 11-12", "JOB-20261112-00013", ["00014", "00015", "00016", "00017", "00018"]
         + ["00019", "00020", "00021", "00022", "00023"]),
        ("0.25 mA, 11-12", "JOB-20261112-00266", ["00267", "00268", "00269", "00270", "00271"])]
print("set              job     prep  read  f in either  <n> corr: all  no-f")
for label, cjob, jobs in SETS:
    cn, cf, _ = load(cjob, "MultiParityConfusionMatrixExperiment")
    C_all, C_nof = f_postselect.confusion_from_shots(cn), f_postselect.confusion_from_shots(cn, cf)
    for j in jobs:
        n, no_f, cfg = load(f"JOB-{cjob[4:12]}-{j}")
        print(f"{label:16s} {j}  {cfg['expt']['occupations'][0]}     {mode_label(cfg):4s}  "
              f"{1 - no_f[0].mean():.3f}      {f_postselect.corrected_mean_n(n[:1], C_all)[0]:6.3f}"
              f"  {f_postselect.corrected_mean_n(n[:1], C_nof, no_f[:1])[0]:6.3f}")

# %% Cell 3: jonginn's 0.45 mA quantum walk, Fock 3 in M1 (JOB-20261112-00178..00182), all cycles,
# corrected <n> with all shots and without the f shots. Confusion matrix JOB-20261112-00176.
WALK = ["00178", "00179", "00180", "00181", "00182"]
cn, cf, _ = load("JOB-20261112-00176", "MultiParityConfusionMatrixExperiment")
C_all, C_nof = f_postselect.confusion_from_shots(cn), f_postselect.confusion_from_shots(cn, cf)
walk = {}
for j in WALK:
    n, no_f, cfg = load(f"JOB-20261112-{j}")
    walk[mode_label(cfg)] = dict(
        cycles=np.array(cfg["expt"]["floquet_cycles"]),
        all=f_postselect.corrected_mean_n(n, C_all),
        no_f=f_postselect.corrected_mean_n(n, C_nof, no_f),
        f_fraction=1 - no_f.mean(axis=1))
total_all = sum(w["all"] for w in walk.values())
total_nof = sum(w["no_f"] for w in walk.values())
for mode, w in walk.items():
    print(f"{mode}: cycle 0 <n> {w['all'][0]:.3f} -> {w['no_f'][0]:.3f};  mean over cycles {w['all'].mean():.3f}"
          f" -> {w['no_f'].mean():.3f};  f fraction {w['f_fraction'].mean():.3f}")
print(f"sum of the 5 modes: mean {total_all.mean():.3f} -> {total_nof.mean():.3f} (prepared 3)")

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
for mode, w in walk.items():
    for ax, key in zip(axs, ("all", "no_f", "f_fraction")):
        ax.plot(w["cycles"], w[key], "-", lw=1.5, color=MODE_COLORS[mode], label=mode)
axs[0].plot(walk["M1"]["cycles"], total_all, "-", lw=2, color="#52514e", label="sum")
axs[1].plot(walk["M1"]["cycles"], total_nof, "-", lw=2, color="#52514e", label="sum")
for ax, title in zip(axs, ("corrected <n>, all shots", "corrected <n>, f shots removed", "fraction of f shots")):
    style(ax, title)
axs[0].set_ylabel("<n>", fontsize=9)
axs[1].sharey(axs[0])
axs[0].legend(frameon=False, fontsize=8, ncol=2)
fig.suptitle("Fock 3 quantum walk, 0.45 mA (JOB-20261112-00178..00182): shots with f in a parity "
             "readout removed", x=0.01, ha="left", fontsize=10)
fig.tight_layout()

# %% Cell 4: vault entry (mock station: only for the vault path; no hardware).
station = None
if LOG_TO_VAULT:
    station = MultimodeStation(mock=True, user="guan", project="AGP", experiment_name="261113_f_postselect",
                               hardware_config="CFG-HW-20261112-00016", multiphoton_config="CFG-MP-20260121-00001",
                               storage_man_file="CFG-M1-20261111-00132", floquet_file="CFG-FL-20261111-00111")
NOTE = (
    "analysis_notebooks/guan/20261113_f_postselect.py, cells 2-3. Shots with the qubit in f in either "
    "parity readout (nearest of the g/e/f centers of JOB-20261113-00259) removed; the confusion matrix "
    "(JOB-20261112-00176) is made again from its own shots with the same cut.\n"
    "**Measured:** cycle 0, empty storages after Fock 3 in M1: corrected <n> 0.26-0.40 -> 0.07-0.16; M1 "
    "3.03 -> 3.02. Sum of the 5 modes (prepared 3): cycle 0 4.3 -> 3.45; mean over cycles 0-396 3.43 -> "
    "2.81. Fraction of f shots in the storage reads: 0.065-0.085 at cycle 0, ~0.05 at late cycles (f "
    "decays to e during the walk). The same cut on the 11-11 jobs and the 0.25 mA jobs: cell 2.\n"
    "**Caveats:** the nearest-center cut also removes ~5 % of good e shots, more when the read mode "
    "holds photons (M1 read: 0.17 at cycle 0); the remade matrix has the same cut, so this is partly, "
    "not exactly, compensated. A qubit that decays from f to e before the readout reads n = 1 and is not "
    "removed. The rest of the excess (sum 3.45 at cycle 0; empty modes 0.07-0.16) is not explained by "
    "the f shots."
)
if LOG_TO_VAULT:
    print(station.log_analysis(fig, "f-shot post-selection of the Fock 3 quantum walk (0.45 mA)",
                               [f"JOB-20261112-{j}" for j in WALK] + ["JOB-20261112-00176", "JOB-20261113-00259"],
                               note=NOTE, author="Claude"))

# %% Cell 5: herald / herald + reset in the M1 ladder (JOB-20261113-00278..00286, cell 3 of
# ../../measurement_notebooks/guan/20261113_tls_recheck.py). Raw <n> (no confusion correction:
# no matrix was measured with heralds). Figure of merit: the empty-storage <n> minus the vacuum
# job's (same read mode, same cut), averaged over S1, S2, S4, S5; and p(N) of the M1 read.
HERALD_JOBS = [f"JOB-20261113-{k:05d}" for k in range(278, 287)]
CUTS = {"all": {}, "herald": dict(herald=True), "herald + no f": dict(herald=True, no_f=True)}

results = {}
for job in HERALD_JOBS:
    cfg, lab = labels_from_h5(rf"{HERALD_DIR}\{job}_FockPrepHeraldExperiment.h5", CENTERS)
    expt = cfg.expt
    storages = [k for k, ro in enumerate(expt.ro_stors) if ro != 0]
    m1 = list(expt.ro_stors).index(0)
    N = expt.occupations[0]
    row = dict(job=job, kept=select(lab, herald=True)[storages].mean())
    for cut, kwargs in CUTS.items():
        keep = select(lab, **kwargs)
        m, s = f_postselect.mean_n_with_error(lab["n"], keep)
        p_N = f_postselect.n_distribution(lab["n"][m1], keep[m1])[N % 4] if N else np.nan
        row[cut] = dict(storage=m[storages].mean(), storage_err=np.sqrt((s[storages] ** 2).sum()) / len(storages),
                        m1=m[m1], p_N=p_N)
    results[(N, expt.prep_herald)] = row
vac = results[(0, "none")]
print("N  mode        kept   empty-storage excess <n> over vacuum: all / herald / herald+no-f   M1 p(N): all / herald / herald+no-f")
for (N, mode), row in results.items():
    if N == 0:
        continue
    ref = {c: vac["all" if c != "herald + no f" else c] for c in CUTS}  # the vacuum job has no herald
    row["excess"] = [row[c]["storage"] - ref[c]["storage"] for c in CUTS]
    row["excess_err"] = [np.hypot(row[c]["storage_err"], ref[c]["storage_err"]) for c in CUTS]
    print(f"{N}  {mode:10s}  {row['kept']:.3f}  "
          + "  ".join(f"{e:+.3f}+-{s:.3f}" for e, s in zip(row["excess"], row["excess_err"]))
          + "    " + "  ".join(f"{row[c]['p_N']:.3f}" for c in CUTS))

MODES = ("none", "end", "each", "each_reset")
CUT_COLORS = dict(zip(CUTS, ("#2a78d6", "#eb6834", "#1baf7a")))
fig5, axs5 = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
x = np.arange(len(MODES))
for ax, N in zip(axs5[:2], (3, 2)):
    for j, cut in enumerate(CUTS):
        ax.errorbar(x + (j - 1) * 0.08, [results[(N, m)]["excess"][j] for m in MODES],
                    yerr=[results[(N, m)]["excess_err"][j] for m in MODES], fmt="o-", ms=6, lw=1.2,
                    capsize=3, color=CUT_COLORS[cut], label=cut)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_ylabel("excess <n>", fontsize=9)
for N, marker in ((3, "o"), (2, "s")):
    axs5[2].plot(x, [results[(N, m)]["herald + no f"]["p_N"] for m in MODES], marker + "-", ms=6, lw=1.2,
                 color="#52514e" if N == 3 else "#898781", label=f"Fock {N}: p({N}), herald + no f")
for ax, title in zip(axs5, ("Fock 3 in M1: empty-storage <n> minus vacuum (raw)",
                            "Fock 2 in M1: empty-storage <n> minus vacuum (raw)", "M1 read: p(N) (raw)")):
    style(ax, title, xlabel="")
    ax.set_xticks(x, [f"{m}\nkept {results[(3, m)]['kept']:.2f} / {results[(2, m)]['kept']:.2f}" for m in MODES],
                  fontsize=8)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8)
fig5.suptitle("Heralds in the M1 Fock ladder, 0.45 mA (JOB-20261113-00278..00286); kept: Fock 3 / Fock 2",
              x=0.01, ha="left", fontsize=10)
fig5.tight_layout()

# %% Cell 6: vault entry for cell 5.
NOTE5 = (
    "analysis_notebooks/guan/20261113_f_postselect.py, cell 5; jobs from cell 3 of "
    "measurement_notebooks/guan/20261113_tls_recheck.py (FockPrepHeraldExperiment, 0.45 mA, jonginn's "
    "settings of JOB-20261112-00178..00182 at Floquet cycle 0, 2000 reps; read M1, S1, S2, S4, S5). "
    "Raw <n> (no confusion matrix was measured with heralds); excess = empty-storage <n> minus the "
    "vacuum job's, same cut, mean of 4 storages; errors +-0.01 (shots).\n"
    "**Measured, Fock 3** (excess: no cut / herald cut / herald + no f; kept; M1 p(3) herald + no f): "
    "none +0.224 / - / +0.063 (f cut only); 1.00; 0.73. end +0.268 / +0.051 / +0.039; 0.90; 0.69. "
    "each +0.294 / +0.035 / +0.027; 0.84; 0.67. each_reset +0.178 / +0.073 / +0.021; 0.79; 0.58.\n"
    "**Measured, Fock 2:** none +0.102 / - / +0.041; 1.00; p(2) 0.82. end +0.133 / +0.012 / +0.006;"
    "0.93; 0.80. each +0.178 / +0.026 / +0.024; 0.91; 0.82. each_reset +0.133 / +0.064 / +0.009; 0.88; 0.72.\n"
    "**Reading:** one herald at the end of the ladder, post-selected, removes most of the empty-mode excess "
    "(Fock 3 +0.22 -> +0.05, Fock 2 +0.10 -> +0.01) for 7-10 % of the shots and a small drop of M1 p(N) "
    "(photons decay during the ~3.6 us readout). A herald after each step is not clearly better (Fock 3 "
    "+0.035 vs +0.051, 1.5 sigma; Fock 2 worse) and keeps fewer shots. A reset after each step is worst for "
    "the M1 photon number (Fock 3 p(3) 0.58) and without post-selection removes only part of the excess. "
    "Not explained: without post-selection a herald makes the excess larger (Fock 3 +0.22 -> +0.27 / +0.29). "
    "Next: prep_herald='end' in the population jobs, a confusion matrix with the same herald, and the f "
    "cut in the multiparity analysis."
)
if LOG_TO_VAULT:
    print(station.log_analysis(fig5, "Heralds and qubit resets in the M1 Fock ladder (0.45 mA)", HERALD_JOBS,
                               note=NOTE5, author="Claude"))

# %% Cell 7: the Fock 3 quantum walk with and without prep_herald="end" (cell 5 of
# ../../measurement_notebooks/guan/20261113_tls_recheck.py). Corrected <n> per mode and cycle, for
# four cuts; each cut has its own confusion matrix from the Fock 0-3 rows of the same session
# (row 0: the vacuum job, no herald; rows 1-3: the "none" or the "end" jobs with the same cut).
CONFUSION_IDS = [f"JOB-20261113-{k:05d}" for k in range(398, 405)]  # vacuum; Fock 1-3 "none"; Fock 1-3 "end"
WALK_IDS = [f"JOB-20261113-{k:05d}" for k in (407, 408, 409, 412, 413, 414, 417, 418, 419, 422)]  # 5 modes x (none, end)
WALK_CUTS = {"no herald, all shots": ("none", {}), "no herald, f cut": ("none", dict(no_f=True)),
             "end herald": ("end", dict(herald=True)), "end herald + f cut": ("end", dict(herald=True, no_f=True))}

conf = [labels_from_h5(rf"{HERALD_DIR}\{j}_FockPrepHeraldExperiment.h5", CENTERS) for j in CONFUSION_IDS]
walk_files = [labels_from_h5(rf"{HERALD_DIR}\{j}_FockPopulationExperiment.h5", CENTERS) for j in WALK_IDS]
walk_result = {}
for cut, (mode, kwargs) in WALK_CUTS.items():
    rows = [conf[0]] + [c for c in conf[1:] if c[0].expt.prep_herald == mode]
    C = f_postselect.confusion_from_shots([lab["n"][0] for _, lab in rows],
                                          [select(lab, **kwargs)[0] for _, lab in rows])
    per_mode = {}
    for cfg, lab in walk_files:
        if cfg.expt.prep_herald == mode:
            keep = select(lab, **kwargs)
            per_mode[mode_label(cfg)] = dict(n=f_postselect.corrected_mean_n(lab["n"], C, keep), kept=keep.mean())
    walk_result[cut] = dict(C=C, modes=per_mode, total=sum(m["n"] for m in per_mode.values()),
                            cycles=np.array(walk_files[0][0].expt.floquet_cycles))
print("cut                    kept   empty storages at cycle 0   sum: cycle 0  mean   C diagonal")
for cut, r in walk_result.items():
    storages = [m for m in r["modes"] if m != "M1"]
    print(f"{cut:22s} {np.mean([m['kept'] for m in r['modes'].values()]):.3f}   "
          + " ".join(f"{r['modes'][m]['n'][0]:.3f}" for m in storages)
          + f"     {r['total'][0]:.3f}  {r['total'].mean():.3f}   {np.round(np.diag(r['C']), 3)}")

CUT_COLORS7 = dict(zip(WALK_CUTS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
fig7, axs7 = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
for cut, r in walk_result.items():
    storages = [m for m in r["modes"] if m != "M1"]
    axs7[0].plot(r["cycles"], r["total"], "-", lw=1.5, color=CUT_COLORS7[cut], label=cut)
    axs7[1].plot(r["cycles"], r["modes"]["M1"]["n"], "-", lw=1.5, color=CUT_COLORS7[cut], label=cut)
    axs7[2].plot(r["cycles"], np.mean([r["modes"][m]["n"] for m in storages], axis=0), "-", lw=1.5,
                 color=CUT_COLORS7[cut], label=cut)
for ax, title in zip(axs7, ("sum of the 5 modes (prepared 3)", "M1", "mean of S1, S2, S4, S5")):
    style(ax, title)
axs7[0].set_ylabel("corrected <n>", fontsize=9)
axs7[0].legend(frameon=False, fontsize=8)
fig7.suptitle("Fock 3 quantum walk, 0.45 mA: no herald vs prep_herald='end', with and without the f cut",
              x=0.01, ha="left", fontsize=10)
fig7.tight_layout()

# %% Cell 8: vault entry for cell 7.
NOTE7 = (
    "analysis_notebooks/guan/20261113_f_postselect.py, cell 7; jobs from cell 5 of "
    "measurement_notebooks/guan/20261113_tls_recheck.py. Fock 3 in M1, quantum walk at 0.45 mA (settings "
    "of JOB-20261112-00178..00182, cycles 0-196, 1000 reps), read M1, S1, S2, S4, S5, each with "
    "prep_herald none and end (FockPopulationExperiment + FockPrepHeraldProgram). Each cut has its own "
    "confusion matrix from the Fock 0-3 rows of the same session (JOB-20261113-00398..00404, same "
    "program, read M1 at cycle 0).\n"
    "**Measured** (corrected <n>; kept; empty storages at cycle 0; sum of the 5 modes at cycle 0 / mean "
    "over cycles; prepared 3): no herald, all shots: 1.00; 0.26-0.33; 4.26 / 3.64. No herald, f cut: 0.92; "
    "0.05-0.10; 3.42 / 2.96. End herald: 0.90; 0.02-0.09; 3.23 / 2.88. End herald + f cut: 0.87; "
    "0.03-0.06; 3.20 / 2.75. M1 at cycle 0 is ~3.0 with every cut; the storage minima during the walk "
    "drop from ~0.3 to ~0.05-0.1.\n"
    "**Reading:** the end herald removes most of the empty-mode floor in a real walk (sum at cycle 0 "
    "4.26 -> 3.23) for 10 % of the shots; the f cut alone does about as much (3.42), and both together "
    "little more (3.20). A rest of ~0.05 per empty storage (sum +0.2) stays. Errors: about +-0.03 per "
    "corrected point, +-0.08 for the sum at one cycle; the none and end jobs ran alternately within ~15 min."
)
if LOG_TO_VAULT:
    print(station.log_analysis(fig7, "Fock 3 quantum walk with prep_herald='end' (0.45 mA)",
                               CONFUSION_IDS + WALK_IDS, note=NOTE7, author="Claude"))

# %% Cell 9: does the storage floor rise because photons stay in the storages, or because the
# readout gets worse? At each M1 peak of the walk: M1, the four storages, their sum. Today's walk
# (end herald, cell 7) and jonginn's longer walk (JOB-20261112-00178..00182, f cut, cell 3).
r = walk_result["end herald"]
today = {m: v["n"] for m, v in r["modes"].items()}
jonginn = {m: w["no_f"] for m, w in walk.items()}
for label, cycles, modes, total in (("today, end herald", r["cycles"], today, r["total"]),
                                    ("jonginn 11-12, f cut", walk["M1"]["cycles"], jonginn, total_nof)):
    storages, rows = at_m1_peaks(np.asarray(cycles), modes, total)
    print(f"{label}: cycle  M1     " + "  ".join(f"{s:5s}" for s in storages) + "  stor sum  M1+stor  total")
    for c, m1, st, ssum, tot in rows:
        print(f"{'':22s}{c:4d}  {m1:5.2f}  " + "  ".join(f"{x:5.2f}" for x in st)
              + f"   {ssum:5.2f}    {m1 + ssum:5.2f}  {tot:5.2f}")

# %% Cell 10: model (closed system, theory/qsim/central_boson.py), not a fit: the 3-photon walk with
# the job's couplings and cycle time, for M1 Kerr K = 0 and K = device.manipulate.kerr (-0.0284 MHz,
# CFG-HW-20261112-00016; a stale value, see cell 12). Same table as cell 9, at the model's M1 peaks.
with h5py.File(rf"{HERALD_DIR}\{WALK_IDS[0]}_FockPopulationExperiment.h5", "r") as f:
    derived = json.loads(f.attrs["derived_params"])
CYCLE_US, G_MHZ = derived["floquet_cycle_us"], derived["couplings_MHz"][0]
model_cycles = np.arange(0, 400, 4)


def walk_model(kerr_MHz):
    """-> {mode: <n>(cycle)} of the closed 3-photon walk from Fock 3 in M1."""
    h = fixed_n_hamiltonian(3, 5, [0.0] * 4, [G_MHZ] * 4, kerr_MHz)
    return {mode: fock_number_traces(h, [[3, 0, 0, 0, 0]], k, model_cycles * CYCLE_US)[0]
            for k, mode in enumerate(("M1", "S1", "S2", "S4", "S5"))}


model = {kerr: walk_model(kerr) for kerr in (0.0, -0.0284)}
for kerr, traces in model.items():
    _, rows = at_m1_peaks(model_cycles, traces, sum(traces.values()))
    print(f"model K = {kerr * 1e3:+.1f} kHz: cycle  M1    stor sum")
    for c, m1, st, ssum, tot in rows[:8]:
        print(f"{'':24s}{c:4d}  {m1:5.2f}  {ssum:5.2f}")
print("M1 at its minima:")
for label, cycles, m1 in (("today, end herald", r["cycles"], today["M1"]),
                          ("jonginn, f cut", walk["M1"]["cycles"], jonginn["M1"]),
                          ("model K = 0", model_cycles, model[0.0]["M1"]),
                          ("model K = -28.4 kHz", model_cycles, model[-0.0284]["M1"])):
    print(f"  {label:20s} " + "  ".join(f"{c}:{v:.2f}" for c, v in at_m1_minima(np.asarray(cycles), np.asarray(m1))))

fig10, axs10 = plt.subplots(1, 2, figsize=(14, 4.2), facecolor="#fcfcfb", sharey=True)
for ax, (label, cycles, modes) in zip(axs10, (("today, end herald (JOB-20261113-00407..00422)", r["cycles"], today),
                                              ("jonginn 11-12, f cut (JOB-20261112-00178..00182)",
                                               walk["M1"]["cycles"], jonginn))):
    storages = [m for m in modes if m != "M1"]
    ax.plot(cycles, modes["M1"], "o", ms=4, color="#2a78d6", label="M1, data")
    ax.plot(cycles, np.mean([modes[m] for m in storages], axis=0), "s", ms=4, color="#eb6834",
            label="storage mean, data")
    mask = model_cycles <= np.max(cycles)
    for kerr, linestyle in ((0.0, ":"), (-0.0284, "-")):
        tr = model[kerr]
        ax.plot(model_cycles[mask], tr["M1"][mask], linestyle, lw=1.3, color="#2a78d6",
                label=f"M1, model K = {kerr * 1e3:.1f} kHz")
        ax.plot(model_cycles[mask], np.mean([tr[m] for m in ("S1", "S2", "S4", "S5")], axis=0)[mask], linestyle,
                lw=1.3, color="#eb6834", label=f"storage mean, model K = {kerr * 1e3:.1f} kHz")
    style(ax, label)
axs10[0].set_ylabel("corrected <n>", fontsize=9)
axs10[1].legend(frameon=False, fontsize=7, loc="upper right")
fig10.suptitle("Fock 3 walk at 0.45 mA against the closed central-boson model (no loss, no dephasing, no fit)",
               x=0.01, ha="left", fontsize=10)
fig10.tight_layout()

# %% Cell 11: vault entry for cells 9-10.
NOTE10 = (
    "analysis_notebooks/guan/20261113_f_postselect.py, cells 9-10 (guan's question: why does the empty-mode "
    "floor of the Fock 3 walk rise with the cycle number?).\n"
    "**Measured** (corrected <n>; today's walk with the end herald, cycles 0-196, and jonginn's 11-12 walk "
    "with the f cut, cycles 0-396): at the M1 peaks the storage sum rises (today 0.21 -> 0.76; jonginn "
    "0.4 -> 1.1) while M1 falls (today 3.02 -> 1.96; jonginn 3.02 -> 1.39), and the total over the 5 modes "
    "falls slowly (today 3.23 -> 2.72; jonginn 3.43 -> 2.48). The four storages rise about equally. M1 "
    "still empties almost fully at its minima (today 0.02 -> 0.18, jonginn 0.08 -> 0.36).\n"
    "**Reading:** not a readout error that grows with time (that would raise the total); photons stay in "
    "the storages when they should be back in M1.\n"
    "**Model** (theory/qsim/central_boson.py, closed, no loss, no fit; job couplings 4 x 19.76 kHz, "
    "0.3162 us per cycle): K = 0 returns fully. K = device.manipulate.kerr = -28.4 kHz "
    "(CFG-HW-20261112-00016) keeps photons in M1: its M1 minima are 0.52 -> 1.33, far above the data. So "
    "that Kerr value does not describe these walks, and an M1 Kerr is not the main cause of the floor. "
    "The pattern (M1 empties, does not refill, storages rise equally) is that of photons going into the "
    "storage combinations that do not couple to M1: unequal storage frequencies in the Floquet frame "
    "(swap calibration, phase tracking) or dephasing. Not tested. Tests: a Fock 1 walk (no Kerr effect at "
    "all); single-storage M1-Sk walks for the frequency error and the dephasing of each pair."
)
if LOG_TO_VAULT:
    print(station.log_analysis(fig10, "Why the empty-mode floor of the Fock 3 walk rises (data vs central-boson model)",
                               WALK_IDS + [f"JOB-20261112-{j}" for j in WALK], note=NOTE10, author="Claude"))

# %% Cell 12: the same closed model for an M1 Kerr of a few kHz. guan (2026-11-13): at 0.45 mA the
# Kerr is a few kHz; the -28.4 kHz in device.manipulate.kerr is a stale value, not yet calibrated.
for kerr in (-0.002, -0.003, -0.005):
    model[kerr] = walk_model(kerr)
print("storage sum at the M1 peaks / M1 at its minima")
for label, cycles, modes in (("today, end herald", r["cycles"], today), ("jonginn, f cut", walk["M1"]["cycles"], jonginn)):
    _, rows = at_m1_peaks(np.asarray(cycles), modes, sum(modes.values()))
    print(f"  {label:20s} peaks " + " ".join(f"{c}:{s:.2f}" for c, _, _, s, _ in rows))
    print(f"  {'':20s} minima " + " ".join(f"{c}:{v:.2f}" for c, v in at_m1_minima(np.asarray(cycles), np.asarray(modes['M1']))))
for kerr in (0.0, -0.002, -0.003, -0.005):
    tr = model[kerr]
    _, rows = at_m1_peaks(model_cycles, tr, sum(tr.values()))
    print(f"  model K = {kerr * 1e3:+.0f} kHz    peaks " + " ".join(f"{c}:{s:.2f}" for c, _, _, s, _ in rows))
    print(f"  {'':20s} minima " + " ".join(f"{c}:{v:.2f}" for c, v in at_m1_minima(model_cycles, tr["M1"])))
NOTE12 = (
    "Update to this entry (cell 12 of analysis_notebooks/guan/20261113_f_postselect.py). guan: at 0.45 mA the "
    "M1 Kerr is a few kHz; the -28.4 kHz in device.manipulate.kerr is stale (not yet calibrated). So the "
    "-28.4 kHz model above is not relevant. **Model** (closed, no fit) for K = -3 / -5 kHz: M1 minima at "
    "cycles 20-180: 0.01-0.11 / 0.02-0.28; storage sum at the M1 peaks at cycles 80, 160, ~200: 0.02, 0.08, "
    "0.13 / 0.06, 0.22, 0.33. **Measured:** M1 minima 0.02-0.18 (today); storage-sum rise at the M1 peaks "
    "over its first value: 0.24, 0.39, 0.55 (today), 0.42, 0.51, 0.59 (jonginn). **Reading:** a Kerr of "
    "about -3 to -5 kHz fits the slow rise of the M1 minima, and adds to the floor at late cycles, but it "
    "rises too slowly at the start (quadratic): by cycle 80 the measured floor is 4-7x what it gives. So a "
    "second effect fills the storages early: storage combinations that do not couple to M1 (unequal "
    "storage frequencies in the Floquet frame, or dephasing). Not tested; the Fock 1 walk (no Kerr effect) "
    "shows the second effect alone."
)
if LOG_TO_VAULT:
    print(station.add_note("ANA-20261113-145845", NOTE12, author="Claude"))
