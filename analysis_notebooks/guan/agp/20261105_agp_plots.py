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

# %% [markdown]# # AGP at high W: photon-number traces by direction (guan + Claude, 2026-11-05)
#
# Analysis side of `measurement_notebooks/guan/20261105_agp_highW.py` (hypothesis there:
# at high W, S7 is frozen by its large detuning, while M1 still swaps with S5). Kernel:
# `guan_agp_plot`. Baseline: the data sets named on 11-03 (`analysis_notebooks/datasets/agp_scans.yaml`)
# and the jobs of jonginn's stopped 11-05 run.

# %% Cell 1: load. Old scans by registry key; jonginn's 11-05 jobs as population sets.
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.agp import load_scan
from experiments.qsim.fock_population import FockPopulationExperiment, FockPopulationSetExperiment

warnings.filterwarnings("ignore", message="Calibration file not found")
DATA_1105 = Path(r"C:\experiments\261027AGPlowKerr\data")


def load_jobs(first, last, skip=()):
    """-> a FockPopulationSetExperiment of JOB-20261105-<first..last>, analyzed."""
    ids = [f"JOB-20261105-{n:05d}" for n in range(first, last + 1) if n not in skip]
    jobs = [FockPopulationExperiment.from_h5file(DATA_1105 / f"{i}_FockPopulationExperiment.h5",
                                                 load_shots=False) for i in ids]
    population_set = FockPopulationSetExperiment.from_children(jobs, job_ids=ids)
    population_set.analyze()
    return population_set


scans = {key: load_scan(key) for key in ["s7_fine", "s7_high_w", "kerr", "kerr_w0"]}
for scan in scans.values():
    for part in scan.children:
        part.analyze()


def part_at(key, w_khz):
    return next(part for part in scans[key].children if part.disorder_khz == w_khz)


# Rows: direction; columns: W = 0, W = 200 kHz. Each value: (population set, label) or None.
TRACES = {
    "S7": {0: (part_at("s7_fine", 0.), "s7_fine, 10-17..19"),
           200: (part_at("s7_high_w", 200.), "s7_high_w, 10-29..30")},
    "M1 (Kerr)": {0: (part_at("kerr_w0", 0.), "kerr_w0 (f0-g1 57 kHz off)"),
                  200: (part_at("kerr", 200.), "kerr, 10-24")},
    "S5": {0: (load_jobs(165, 171), "11-05 jobs 165-171: 7 of 15 states"), 200: None},
}
TODAY_M1_W0 = load_jobs(158, 162)   # 11-05, all 5 states, cycles 0-59
g_khz = 1e3 * part_at("kerr", 200.).recorded_couplings_mhz()
print("g (kHz):", np.round(g_khz, 1))
print("detunings at W = 200 kHz (kHz), S4..S7:", np.round(1e3 * np.asarray(part_at("kerr", 200.).detunings), 1))


def data_file(population_set):
    """-> the assembled HDF5 of a saved set, else its job files (not assembled)."""
    if population_set.manifest_path is not None:
        return str(Path(population_set.manifest_path).with_suffix(".h5"))
    return f"not assembled: {population_set.job_ids[0]} .. {population_set.job_ids[-1]} in {DATA_1105}"


def print_files(traces):
    for direction, by_w in traces.items():
        for w_khz, entry in by_w.items():
            if entry is not None:
                print(f"{direction}, W = {w_khz} kHz: {data_file(entry[0])}")


print_files(TRACES)

# %% Cell 2: baseline figure. Each thin line: <n>(t) of the read mode (readout-corrected) for one
# initial Fock state; color: the initial photon number in the read mode. Black: mean over the
# measured states. The x range is the first 80 us of the 245 us traces.
T_MAX_US = 80.
SHADES = {1: plt.cm.Blues(0.45), 2: plt.cm.Blues(0.7), 3: plt.cm.Blues(0.95)}


def plot_set(ax, population_set, label):
    data = population_set.data
    mode_index = [0, *population_set.swap_stors].index(int(data["ro_stor"][0]))
    time_us = np.asarray(data["time_us"])
    traces = np.asarray(data["n_mean_corrected"])
    for occupation, trace in zip(data["occupations"], traces):
        ax.plot(time_us, trace, lw=0.8, color=SHADES[int(occupation[mode_index])])
    ax.plot(time_us, traces.mean(axis=0), lw=2, color="k")
    ax.set_title(label, fontsize=8)


handles = [plt.Line2D([], [], color=SHADES[n], lw=1.5, label=f"n(0) = {n}") for n in (1, 2, 3)]
handles.append(plt.Line2D([], [], color="k", lw=2, label="mean over states"))


def plot_baseline(traces):
    """Rows: read direction; columns: W = 0, 200 kHz. Prints the file of each panel."""
    fig, axes = plt.subplots(len(traces), 2, figsize=(11, 8), sharex=True, sharey="row",
                             constrained_layout=True)
    for row, (direction, by_w) in zip(axes, traces.items()):
        for ax, (w_khz, entry) in zip(row, by_w.items()):
            if entry is None:
                ax.text(0.5, 0.5, "not measured yet", transform=ax.transAxes, ha="center",
                        color="0.5")
            else:
                plot_set(ax, *entry)
            ax.set_xlim(0, T_MAX_US)
            ax.grid(alpha=0.3)
        row[0].set_ylabel(f"read {direction}\n<n> (corrected)")
    for ax, w_khz in zip(axes[0], (0, 200)):
        ax.set_title(f"W = {w_khz} kHz\n" + ax.get_title(), fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("time (us)")
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=8)
    plt.show()
    print_files(traces)
    return fig, axes


fig, axes = plot_baseline(TRACES)

# %% Cell 3: closed-system model (theory.qsim.central_boson.fixed_n_hamiltonian), same layout as
# Cell 2: <n>(t) of the read mode for each initial state of the direction sector, at the recorded
# couplings (20.4 kHz) and Kerr (-27.1 kHz), the 11-05 pattern. MODEL, not data: no loss, no
# readout error.
from experiments.qsim.agp import direction_sector
from theory.qsim.central_boson import fixed_n_hamiltonian, fock_number_traces

REFERENCE = part_at("kerr", 200.)
COUPLINGS_MHZ = REFERENCE.recorded_couplings_mhz()
KERR_MHZ = REFERENCE.recorded_kerr_mhz()
PATTERN = np.asarray(REFERENCE.detunings) / (1e-3 * REFERENCE.disorder_khz)   # unit RMS
STORS = list(REFERENCE.swap_stors)
MODEL_TIME_US = np.linspace(0., T_MAX_US, 801)


MODEL_DIRECTIONS = {"S7": 7, "M1 (Kerr)": 0, "S5": 5}
fig, axes = plt.subplots(len(MODEL_DIRECTIONS), 2, figsize=(11, 8), sharex=True, sharey="row",
                         constrained_layout=True)
for row, (label, direction) in zip(axes, MODEL_DIRECTIONS.items()):
    mode_index = [0, *STORS].index(direction)
    states = direction_sector(3, STORS, direction)
    for ax, w_khz in zip(row, (0, 200)):
        model = fixed_n_hamiltonian(3, 1 + len(STORS), 1e-3 * w_khz * PATTERN, COUPLINGS_MHZ, KERR_MHZ)
        traces = fock_number_traces(model, states, mode_index, MODEL_TIME_US)
        for state, trace in zip(states, traces):
            ax.plot(MODEL_TIME_US, trace, lw=0.8, color=SHADES[state[mode_index]])
        ax.plot(MODEL_TIME_US, traces.mean(axis=0), lw=2, color="k")
        ax.set_xlim(0, T_MAX_US)
        ax.grid(alpha=0.3)
    row[0].set_ylabel(f"read {label}\n<n> (model)")
for ax, w_khz in zip(axes[0], (0, 200)):
    ax.set_title(f"W = {w_khz} kHz, closed-system model", fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("time (us)")
fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=8)
plt.show()
print(f"model: g = {1e3 * COUPLINGS_MHZ[0]:.1f} kHz, Kerr = {1e3 * KERR_MHZ:.1f} kHz, "
      f"pattern from {data_file(REFERENCE)}")

# %% Cell 4: the baseline with the S5 row from the coarse-grid scan of 11-05 (guan's run,
# measurement_notebooks/guan/20261105_agp_highW.py, name 261105_coarse_dir5), the W that are saved.
from experiments.qsim.agp import CentralBosonAGPDisorderScan

s5_scan = load_scan(DATA_1105.parent / "assembled_data" / "CentralBosonAGPDisorderScan_261105_coarse_dir5.yaml")
for part in s5_scan.children:
    part.analyze()
TRACES_COARSE = dict(TRACES)
TRACES_COARSE["S5"] = {w: None for w in (0, 200)}
for part in s5_scan.children:
    TRACES_COARSE["S5"][int(part.disorder_khz)] = (part, "261105_coarse_dir5, 11-05, cycles 0-256 by 4")
fig, axes = plot_baseline(TRACES_COARSE)

# %% Cell 5: measured AGP at mu = 10 kHz, with its shot + calibration error, per data set (for the
# plan: is the error small compared with the change over W?). The S5 window is 79 us, the
# others 122-245 us, so the window values are not the same quantity; compare each with its own
# agp_theory_window (closed system, same kernel and window).
MU_PLAN_KHZ = 10.
AGP_SETS = {"kerr": scans["kerr"], "kerr_w0": scans["kerr_w0"], "s7_fine": scans["s7_fine"],
            "s7_high_w": scans["s7_high_w"], "261105_coarse_dir5": s5_scan}
for key, scan in AGP_SETS.items():
    scan.analyze(mu=[MU_PLAN_KHZ])
    part = scan.children[0]
    reps = int(part.children[0].cfg.expt.reps)
    window_us = float(part.data["time_us"][-1])
    print(f"{key}: direction {part.direction}, {reps} reps, window {window_us:.0f} us, "
          f"{len(part.cycles)} points; {data_file(scan)}")
    for row, w_khz in enumerate(scan.data["disorder_khz"]):
        agp, err = scan.data["agp"][row, 0], scan.data["agp_err"][row, 0]
        window = scan.data["agp_theory_window"][row, 0] if "agp_theory_window" in scan.data else np.nan
        print(f"   W = {w_khz:5.0f} kHz: agp = {agp:.4f} +- {err:.4f} ({100 * err / abs(agp):.1f} %), "
              f"closed-system window value {window:.4f}")

# %% Cell 6: closed-system MODEL, for the plan. Infinite-time AGP at mu = 10 kHz against W, for
# each direction: our pattern, and 200 random patterns (Gaussian, mean removed, unit RMS), at the
# recorded couplings and Kerr. Question for option 3: is one pattern typical, and does the mean
# over S4..S7 of one pattern act like 4 independent patterns?
from theory.qsim.agp import probe_values, spectral_agp, spectral_weights

W_PLAN_KHZ = np.arange(0., 201., 20.)
G_MHZ = float(np.mean(COUPLINGS_MHZ))
PLAN_DIRECTIONS = {"M1 (Kerr)": (0, "kerr"), "S4": (4, "number"), "S5": (5, "number"),
                   "S6": (6, "number"), "S7": (7, "number")}
rng = np.random.default_rng(20261105)
random_patterns = rng.normal(size=(200, len(STORS)))
random_patterns -= random_patterns.mean(axis=1, keepdims=True)
random_patterns /= np.sqrt(np.mean(random_patterns**2, axis=1, keepdims=True))
patterns = np.vstack([PATTERN, random_patterns])   # row 0: our pattern


def model_agp(pattern, w_khz):
    """-> {direction: closed-system AGP at MU_PLAN_KHZ} for one pattern and W."""
    model = fixed_n_hamiltonian(3, 1 + len(STORS), 1e-3 * w_khz * pattern, COUPLINGS_MHZ, KERR_MHZ)
    values = {}
    for label, (mode, probe) in PLAN_DIRECTIONS.items():
        differences, weights = spectral_weights(model, probe_values(model.fock_basis, [0, *STORS].index(mode), probe))
        values[label] = spectral_agp(differences, weights, MU_PLAN_KHZ / 1000, G_MHZ)
    return values


# agp_model[direction]: (pattern, W)
agp_model = {label: np.zeros((len(patterns), len(W_PLAN_KHZ))) for label in PLAN_DIRECTIONS}
for p, pattern in enumerate(patterns):
    for w, w_khz in enumerate(W_PLAN_KHZ):
        for label, value in model_agp(pattern, w_khz).items():
            agp_model[label][p, w] = value

storage_labels = ["S4", "S5", "S6", "S7"]
storage_mean = np.mean([agp_model[label] for label in storage_labels], axis=0)   # (pattern, W)
fig, axes = plt.subplots(1, len(PLAN_DIRECTIONS) + 1, figsize=(16, 3.6), sharey=True,
                         constrained_layout=True)
for ax, (label, values) in zip(axes, list(agp_model.items()) + [("mean of S4..S7", storage_mean)]):
    low, mid, high = np.percentile(values[1:], [16, 50, 84], axis=0)
    ax.fill_between(W_PLAN_KHZ, low, high, color=plt.cm.Blues(0.3), lw=0, label="random: 16-84 %")
    ax.plot(W_PLAN_KHZ, mid, color=plt.cm.Blues(0.7), lw=2, label="random: median")
    ax.plot(W_PLAN_KHZ, values[0], "k", lw=2, label="our pattern")
    ax.set(title=label, xlabel="W (kHz)", yscale="log")
    ax.grid(alpha=0.3)
axes[0].set_ylabel(f"AGP at mu = {MU_PLAN_KHZ:g} kHz (model)")
axes[0].legend(fontsize=7)
plt.show()

print("MODEL, W = 200 kHz: our pattern; random median [16-84 %]; spread = std/mean over patterns")
for label, values in list(agp_model.items()) + [("mean S4..S7", storage_mean)]:
    v = values[:, -1]
    print(f"   {label:12s} ours {v[0]:.4f}; random {np.median(v[1:]):.4f} "
          f"[{np.percentile(v[1:], 16):.4f}, {np.percentile(v[1:], 84):.4f}]; spread {np.std(v[1:]) / np.mean(v[1:]):.2f}")
single = np.std(agp_model["S7"][1:, -1]) / np.mean(agp_model["S7"][1:, -1])
four = np.std(storage_mean[1:, -1]) / np.mean(storage_mean[1:, -1])
print(f"   spread of one direction {single:.2f}; of the S4..S7 mean {four:.2f}; "
      f"4 independent patterns would give {single / 2:.2f}")

# %% Cell 7: chi_ge and chi_f from the 11-05 pulse-probe spectra (measurement script cell 8,
# JOB-20261105-00206..00209), against the probe offset from the config's f_ge or f_ef. Each
# spectrum uses the quadrature with the larger contrast (e and f differ in Q, not in I); one
# Lorentzian per spectrum (fitting.fitting.fitlor) for the line position.
import h5py

import fitting.fitting as fitter

SPECTRA = {"ge, M1 |0>": "JOB-20261105-00206", "ge, M1 |1>": "JOB-20261105-00207",
           "ef, qubit e, M1 |0>": "JOB-20261105-00208", "ef, qubit e, M1 |1>": "JOB-20261105-00209"}
lines = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for label, job_id in SPECTRA.items():
    path = DATA_1105 / f"{job_id}_PulseProbeSpectroscopyExperiment.h5"
    with h5py.File(path, "r") as handle:
        freq = np.asarray(handle["xpts"])
        quadratures = {key: np.asarray(handle[key]) for key in ("avgi", "avgq")}
        qubit_cfg = json.loads(handle.attrs["config"])["device"]["qubit"]
    smooth = {key: np.convolve(value[1:-1], np.ones(9) / 9, mode="valid")
              for key, value in quadratures.items()}
    quadrature = max(smooth, key=lambda key: np.ptp(smooth[key]))
    signal = quadratures[quadrature]
    center = qubit_cfg["f_ge" if label.startswith("ge") else "f_ef"][0]
    popt, pcov = fitter.fitlor(freq[1:-1], signal[1:-1])
    lines[label] = (popt[2], np.sqrt(pcov[2][2]), abs(popt[3]))
    ax = axes[0 if label.startswith("ge") else 1]
    ax.plot(freq[1:-1] - center, signal[1:-1], ".-", lw=0.8, ms=3, label=f"{label} ({quadrature})")
    ax.plot(freq[1:-1] - center, fitter.lorfunc(freq[1:-1], *popt), "k", lw=0.8)
    print(f"{label} ({quadrature}): line at {popt[2]:.4f} MHz (config + {popt[2] - center:+.4f}), "
          f"+- {lines[label][1]:.4f}, HWHM {lines[label][2]:.4f} MHz; {path}")
for ax, name in zip(axes, ("f_ge", "f_ef")):
    ax.set(xlabel=f"probe - config {name} (MHz)", ylabel="I or Q (ADC units)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.show()
chi_ge = lines["ge, M1 |1>"][0] - lines["ge, M1 |0>"][0]
chi_ef = lines["ef, qubit e, M1 |1>"][0] - lines["ef, qubit e, M1 |0>"][0]
print(f"chi_ge = {chi_ge:+.4f} MHz; ef shift per photon (config chi_ef) = {chi_ef:+.4f} MHz; "
      f"chi_f = chi_ge + chi_ef = {chi_ge + chi_ef:+.4f} MHz")
print("config: chi_ge -0.3025, chi_ef +1.2312; from the f_n-g_n+1 table: chi_f about -0.571")


# %% Cell 8: M1 reset check (measurement script cells 9-11). P(e) of the last readout, thresholded at
# the config's readout threshold (e above it, as in MM_base.active_reset); the error is binomial.
# "left" = P(e) - floor = M1 photon left after the reset (mapped to e by f0-g1 pi, e0-f0 pi).
RESET_LABELS = ["floor: no photon, M1 reset", "g,1 no M1 reset", "g,1 M1 reset", "e,1 no M1 reset",
                "e,1 M1 reset", "f,1 no M1 reset", "f,1 M1 reset"]


def reset_check(first_job, tag, show_histograms=False):
    """-> {label: (P(e), error)} for 7 consecutive jobs from first_job; prints a table."""
    results, finals = {}, {}
    for k, label in enumerate(RESET_LABELS):
        job_id = f"JOB-20261105-{first_job + k:05d}"
        path = DATA_1105 / f"{job_id}_QsimExperiment.h5"
        with h5py.File(path, "r") as handle:
            idata = np.asarray(handle["idata"])
            cfg = json.loads(handle.attrs["config"])
        read_num = int(cfg["read_num"])
        final = idata.reshape(-1, read_num)[:, -1]
        p = float(np.mean(final > cfg["device"]["readout"]["threshold"][0]))
        results[label] = (p, np.sqrt(p * (1 - p) / final.size))
        finals[label] = final
        print(f"{tag} | {label:28s} {job_id}: P(e) = {p:.4f} +- {results[label][1]:.4f}; {path}")
    floor = results[RESET_LABELS[0]][0]
    for state in ("g", "e", "f"):
        kept, reset = results[f"{state},1 no M1 reset"][0], results[f"{state},1 M1 reset"][0]
        print(f"{tag} | qubit {state}: photon left {reset - floor:+.4f} of {kept - floor:.4f} "
              f"without the M1 reset ({100 * (reset - floor) / (kept - floor):.1f} %)")
    if show_histograms:
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        for label in (RESET_LABELS[0], "g,1 no M1 reset"):
            ax.hist(finals[label], bins=100, histtype="step", label=label)
        ax.axvline(cfg["device"]["readout"]["threshold"][0], color="k", lw=0.8, label="threshold")
        ax.set(xlabel="I, last readout", ylabel="shots", title=f"{tag}: last-readout I")
        ax.legend(fontsize=8)
        plt.show()
    return results


baseline_reset = reset_check(210, "baseline 6-pulse reset", show_histograms=True)

# %% Cell 9: the same check with the new 3-pulse reset and the measured chi (JOB-20261105-00217..00223).
new_reset = reset_check(217, "new 3-pulse reset")

# %% Cell 10: the trimmed reset (offsets 0, chi_ge, 2 chi_ge with the old chi_ge; JOB-20261105-00225..00231).
trimmed_reset = reset_check(225, "trimmed 3-pulse reset")

# %% Cell 11: sign-flipped trial reset (offsets 0, -chi_ge, -chi_f = +0.3175, +0.593 MHz;
# JOB-20261105-00233..00239).
flipped_reset = reset_check(233, "sign-flipped 3-pulse reset")

# %% Cell 12: sign-flipped pulses in the order f, e, g (JOB-20261105-00241..00247).
reordered_reset = reset_check(241, "sign-flipped f,e,g reset")

# %% Cell 13: MODEL, for the plan of 11-05..08. (a) Where the low-W peak of the AGP (mu = 10 kHz) is,
# for our pattern, M1 and S5 directions, on a fine W grid. (b) The random pattern of cell 6 whose
# AGP(W) is closest to the median in all five directions: score = mean over directions and W of
# (log10 AGP - log10 median)^2.
W_FINE_KHZ = np.arange(0., 80.1, 2.5)
fine = {label: np.array([model_agp(PATTERN, w)[label] for w in W_FINE_KHZ]) for label in ("M1 (Kerr)", "S5")}
for label, values in fine.items():
    peak = W_FINE_KHZ[np.argmax(values)]
    half = W_FINE_KHZ[values >= 0.5 * (values.max() + values[0])]
    print(f"our pattern, {label}: AGP(W=0) {values[0]:.4f}, peak {values.max():.4f} at W = {peak:g} kHz; "
          f"above half-way from W=0 to peak: W = {half.min():g}..{half.max():g} kHz")

log_median = {label: np.log10(np.median(values[1:], axis=0)) for label, values in agp_model.items()}
score = np.mean([(np.log10(values[1:]) - log_median[label])**2 for label, values in agp_model.items()],
                axis=(0, 2))
best = int(np.argmin(score)) + 1   # row in patterns (row 0 is ours)
ours_score = np.mean([(np.log10(values[0]) - log_median[label])**2 for label, values in agp_model.items()])
TYPICAL_PATTERN = patterns[best]
print(f"typical pattern: row {best}, score {score[best - 1]:.4f} (ours {ours_score:.4f}; "
      f"random median {np.median(score):.4f}); S4..S7 = {np.round(TYPICAL_PATTERN, 4).tolist()}; "
      f"detunings at W = 200 kHz (kHz): {np.round(200 * TYPICAL_PATTERN, 1).tolist()}")

fig, axes = plt.subplots(1, len(PLAN_DIRECTIONS), figsize=(16, 3.6), sharey=True, constrained_layout=True)
for ax, (label, values) in zip(axes, agp_model.items()):
    low, mid, high = np.percentile(values[1:], [16, 50, 84], axis=0)
    ax.fill_between(W_PLAN_KHZ, low, high, color=plt.cm.Blues(0.3), lw=0, label="random: 16-84 %")
    ax.plot(W_PLAN_KHZ, mid, color=plt.cm.Blues(0.7), lw=2, label="random: median")
    ax.plot(W_PLAN_KHZ, values[0], "k", lw=1.5, label="our pattern")
    ax.plot(W_PLAN_KHZ, values[best], color="C3", lw=2, label="typical pattern")
    ax.set(title=label, xlabel="W (kHz)", yscale="log")
    ax.grid(alpha=0.3)
axes[0].set_ylabel(f"AGP at mu = {MU_PLAN_KHZ:g} kHz (model)")
axes[0].legend(fontsize=7)
plt.show()


# %%

s5scan =  CentralBosonAGPDisorderScan.from_h5file(r"C:\experiments\261027AGPlowKerr\assembled_data\CentralBosonAGPDisorderScan_261105_S5_direction5.h5")

