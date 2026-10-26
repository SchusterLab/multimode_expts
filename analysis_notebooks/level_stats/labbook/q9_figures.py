"""Figures for the report from the Q2 and Q4 caches (no new computation).

q9_observers.png: P(r<0.25) true / naive / counting per hardware case and Kerr (Q2).
q9_sep10_test.png: the Sep 10 frozen test, r histograms of the counted fit and the model (Q4).
Run: pixi run python analysis_notebooks/level_stats/labbook/q9_figures.py
"""
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CACHE, FIGURES, gap_ratios, goe_r_density, poisson_r_density

CASE_LABELS = {"sep10": "Sep 10 hardware\n(g 29 kHz, 200 us)", "now_92us": "now, 92 us\n(g 20 kHz)",
               "now_184us": "now, 184 us\n(g 20 kHz)", "highK_29": "high Kerr\n(g 29 kHz, 128 us, decay x2)"}
KERRS = (-0.13, -0.9, -1.8)

q2 = np.load(CACHE / "q2_resolution.npz")
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True, layout="constrained")
for ax, (case, label) in zip(axes, CASE_LABELS.items()):
    x = np.arange(len(KERRS))
    for j, (field, color) in enumerate((("true", "#555555"), ("naive", "#eb6834"), ("counting", "#2a78d6"))):
        vals = [np.mean(q2[f"{case}|per_photon_0.5kHz|{k}|1.7|{field}"] < 0.25) for k in KERRS]
        ax.bar(x + (j - 1) * 0.27, vals, 0.27, color=color, label={"true": "model (perfect resolution)",
               "naive": "naive: a merged pair is one level", "counting": "counting: integer weights"}[field])
    ax.axhline(0.40, color="k", ls=":", lw=1)
    ax.set_xticks(x, [f"K/g {k}" for k in KERRS])
    ax.set_title(label, fontsize=10)
axes[0].set_ylabel("P(r < 0.25), W/g = 1.7, N = 3")
axes[0].legend(fontsize=7.5, loc="upper right")
fig.suptitle("What the best unbiased estimator keeps (Cramér-Rao, merge below 4 errors); dotted: Poisson")
fig.savefig(FIGURES / "q9_observers.png", dpi=130)
plt.close(fig)

q4 = pickle.load(open(CACHE / "q4_sep10_counting.pkl", "rb"))
bins = np.linspace(0, 1, 11)
centers = 0.5 * (bins[1:] + bins[:-1])
r = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
fit_r = np.concatenate([gap_ratios(levels) for levels in q4["results"]["T1T3"]["counted"]])
model_r = np.concatenate([gap_ratios(levels) for levels in q4["exact"]])
obs_r = np.concatenate([gap_ratios(levels) for levels in q4["observed"]])
ax.hist(model_r, bins, density=True, color="#d6d6d6", label=f"model, same 9 patterns (P(r<.25) {np.mean(model_r < .25):.2f})")
ax.step(centers, np.histogram(obs_r, bins, density=True)[0], where="mid", color="#555555", lw=1.5,
        label=f"model through the counting observer ({np.mean(obs_r < .25):.2f})")
ax.step(centers, np.histogram(fit_r, bins, density=True)[0], where="mid", color="#2a78d6", lw=2.5,
        label=f"data: fit + counting ({np.mean(fit_r < .25):.2f} +/- {q4['results']['T1T3']['err']:.2f})")
ax.plot(r, poisson_r_density(r), "k:", lw=1, label="Poisson")
ax.plot(r, goe_r_density(r), "k-.", lw=1, label="GOE")
ax.set(xlabel="gap ratio r", ylabel="density", title="Sep 10 ensemble (K/g -0.13, W/g 1.71), 9 realizations")
ax.legend(fontsize=7.5)
fig.savefig(FIGURES / "q9_sep10_test.png", dpi=130)
print("written")

# q9_fitter_check.png: Q10, the real fitter + counting against the exact values and the CR observer.
options = {"idle_276us_250reps": "idle (g 10 kHz)\n276 us, 250 reps, K/g -1.8",
           "idle_184us_250reps": "idle (g 10 kHz)\n184 us, 250 reps, K/g -1.8",
           "now_92us_500reps": "now (g 20 kHz)\n92 us, 500 reps, K/g -0.9",
           "now_184us_250reps": "now (g 20 kHz)\n184 us, 250 reps, K/g -0.9"}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True, layout="constrained")
for ax, (name, label) in zip(axes, options.items()):
    res = pickle.load(open(CACHE / f"q10_fitter_check_{name}.pkl", "rb"))
    kerrs = sorted({r["kerr"] for r in res}, reverse=True)
    for j, (field, color, text) in enumerate((("exact", "#555555", "exact (model)"),
                                              ("observer", "#eb6834", "CR counting model (4-error cut)"),
                                              ("fitter", "#2a78d6", "real fitter + counting"))):
        vals = [np.mean([r[field] for r in res if r["kerr"] == k]) for k in kerrs]
        ax.bar(np.arange(2) + (j - 1) * 0.27, vals, 0.27, color=color, label=text)
    ax.set_xticks(range(2), ["linear (K = 0)", f"K/g {kerrs[1]:g}"])
    ax.axhline(0.40, color="k", ls=":", lw=1)
    ax.set_title(label, fontsize=9.5)
axes[0].set_ylabel("P(r < 0.25), W/g = 2.4, 16 shapes")
axes[0].legend(fontsize=7.5, loc="upper right")
fig.suptitle("Synthetic complete-basis traces: the frozen fitter with counting is close to the exact values")
fig.savefig(FIGURES / "q9_fitter_check.png", dpi=130)
plt.close(fig)
print("fitter check figure written")
