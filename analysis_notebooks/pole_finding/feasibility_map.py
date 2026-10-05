# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feasibility map: where in the (K/g, delta/g) plane can our spectroscopy see the level statistics?
#
# guan's question (2026-10-05): is it worth measuring deep in the K > 0, delta > 0 bulk (high Kerr,
# high disorder, many realizations, long averaging)? First pass, theory and bounds only, at the
# Sep 10 hardware: g 29.2 kHz, T2 166 us, 468 samples at 0.428 us, 35 rows (complete basis).
#
# Per point of the plane:
# 1. the model's gap-ratio distribution from many disorder draws (what there is to see);
# 2. what our resolution keeps: per draw, the Cramér-Rao gap errors (real amplitudes, row offsets
#    with a prior of 0.8 kHz, the residual after a per-photon correction), adjacent levels merged
#    while their gap is under 4 errors, then the gap ratios (a proxy for the best fitter: T1T3
#    on the Sep 10 data is near this bound);
# 3. (dropped 2026-10-05: "realizations to tell <r> from Poisson" compared the biased proxy with
#    Poisson, so the merging bias made it look easy. The right count compares two points' kept
#    distributions; not done yet.)
#
# Two noise levels: 0.16 per sample (realizations 1-8 of Sep 10) and 0.06 (realization 0, the
# remeasurement with more averaging). Writes the page to the lab vault. About 15 min.

# %%
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.mbr_disorder import disorder_direction
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.poles.design import Design, gap_errors
from fitting.qsim.poles.statistics import bulk_gap_ratios
from fitting.qsim.poles.views import INK, MODEL_COLOR, goe_density, poisson_density

VAULT = Path("G:/Shared drives/SLab/Multimode/Lab/guan/qsim_analysis/pole_finding")
FIGURES = VAULT / "figures"
PREFIX = "2026-10-05_feasibility"
G_MHz = 1 / (4 * 40 * 92 / 430.08)                 # 29.217 kHz, the Sep 10 coupling
DT_US, SAMPLES, DECAY = 2 * 92 / 430.08, 468, 0.006
TIME_US = np.arange(SAMPLES) * DT_US
BIN_MHz = 1 / (SAMPLES * DT_US)
KERR_OVER_G = (0.0, -0.13, -0.5, -1.0, -2.0, -4.0)
DISORDER_OVER_G = (0.5, 1.0, 2.0, 3.4, 6.0, 10.0)
NOISE = {"0.16 (standard)": 0.16, "0.06 (more averaging)": 0.06}
THEORY_DRAWS, BOUND_DRAWS = 400, 8
POISSON_R, GOE_R = 2 * np.log(2) - 1, 0.5359
SEP10 = (-0.1288, 3.4226)


def model(kerr_over_g, disorder_over_g, seed):
    detunings = -disorder_over_g * G_MHz * disorder_direction(seed, 4)
    return fixed_n_hamiltonian(3, 5, detunings, [G_MHz] * 4, kerr_over_g * G_MHz)


def merge_unresolved(levels, errors, weights, min_snr=4.):
    """-> levels after merging, smallest gap / error first, every adjacent pair under ``min_snr``
    errors into its weighted mean (the errors of a merged gap: the larger of its neighbours')."""
    levels, errors, weights = list(levels), list(errors), list(weights)
    while len(levels) > 2:
        snr = np.diff(levels) / np.asarray(errors)
        k = int(np.argmin(snr))
        if snr[k] >= min_snr:
            break
        total = weights[k] + weights[k + 1]
        levels[k:k + 2] = [(weights[k] * levels[k] + weights[k + 1] * levels[k + 1]) / total]
        weights[k:k + 2] = [total]
        neighbours = [e for e in (errors[k - 1] if k > 0 else None, errors[k + 1] if k + 1 < len(errors) else None) if e is not None]
        del errors[k]
        if k > 0 and neighbours:
            errors[k - 1] = max(neighbours)
    return np.asarray(levels)


# %%
design = Design(amplitudes="real", offsets="free", offset_prior_MHz=0.8e-3, decay_per_us=DECAY)
results = {}
start = time.time()
for K in KERR_OVER_G:
    for W in DISORDER_OVER_G:
        theory = [model(K, W, 20_000 + s).energies_MHz for s in range(THEORY_DRAWS)]
        cell = dict(theory=np.concatenate([bulk_gap_ratios(levels) for levels in theory]),
                    sff=np.mean([np.abs(np.exp(-2j * np.pi * np.outer(TIME_US, levels)).sum(axis=1)) ** 2
                                 for levels in theory], axis=0) / 35 ** 2)
        for label, noise in NOISE.items():
            merged, small, small_kept = [], 0, 0
            for s in range(BOUND_DRAWS):
                m = model(K, W, 20_000 + s)
                errors = gap_errors(TIME_US, m.energies_MHz, m.basis_eigenstate_weights, np.asarray(m.fock_basis, float),
                                    np.full(35, noise), BIN_MHz, design)
                gaps = np.diff(m.energies_MHz)
                is_small = gaps < 0.5 * gaps.mean()
                small += int(is_small.sum())
                small_kept += int(np.sum(is_small & (gaps >= 4 * errors)))
                merged.append(bulk_gap_ratios(merge_unresolved(m.energies_MHz, errors, np.ones(35))))
            cell[label] = dict(ratios=np.concatenate(merged), small=small, small_kept=small_kept,
                               levels_kept=np.mean([len(x) + 2 for x in merged]))
        results[K, W] = cell
    print(f"K/g {K}: {time.time() - start:.0f} s", flush=True)


# %% [markdown]
# ## Figures

# %%
def grid(value):
    return np.array([[value(results[K, W]) for W in DISORDER_OVER_G] for K in KERR_OVER_G])


panels = [("model <r> (what there is to see)", grid(lambda c: np.mean(c["theory"])), "r")]
for label in NOISE:
    panels.append((f"<r> our resolution keeps, noise {label}", grid(lambda c, l=label: np.mean(c[l]["ratios"])), "r"))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
for ax, (title, values, _) in zip(axes, panels):
    image = ax.imshow(values, origin="lower", cmap="Blues", vmin=POISSON_R, vmax=GOE_R, aspect="auto")
    for i in range(len(KERR_OVER_G)):
        for j in range(len(DISORDER_OVER_G)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > 0.47 else INK)
    ax.set_xticks(range(len(DISORDER_OVER_G)), [f"{w:g}" for w in DISORDER_OVER_G])
    ax.set_yticks(range(len(KERR_OVER_G)), [f"{k:g}" for k in KERR_OVER_G])
    ax.set(xlabel="disorder delta / g", ylabel="Kerr K / g", title=title)
    ax.plot(np.interp(SEP10[1], DISORDER_OVER_G, range(6)), np.interp(-SEP10[0], [-k for k in KERR_OVER_G], range(6)),
            marker="s", ms=22, mfc="none", mec="#eb6834", mew=2)
fig.colorbar(image, ax=axes, label=f"<r>  (Poisson {POISSON_R:.3f}, GOE {GOE_R:.3f})", shrink=0.85)
fig.suptitle("Mean gap ratio over the plane at the Sep 10 hardware (orange square: the Sep 10 point)", color=INK)
fig.savefig(FIGURES / f"{PREFIX}_r_maps.png", dpi=110)
plt.close(fig)

SHOWN = [(-0.13, 3.4), (-1.0, 3.4), (-2.0, 6.0), (-4.0, 6.0), (-1.0, 1.0), (0.0, 3.4)]
edges, r = np.linspace(0, 1, 11), np.linspace(0, 1, 201)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True, layout="constrained")
for ax, point in zip(axes.ravel(), SHOWN):
    cell = results[point]
    ax.hist(cell["theory"], bins=edges, density=True, color="#d6d6d6", edgecolor="white", linewidth=2, label="model")
    for label, color in zip(NOISE, ("#eb6834", "#2a78d6")):
        density, _ = np.histogram(cell[label]["ratios"], bins=edges, density=True)
        ax.step(edges, np.r_[density, density[-1]], where="post", color=color, lw=2, label=f"resolution, noise {label.split()[0]}")
    ax.plot(r, poisson_density(r), color=INK, lw=1, ls="--", label="Poisson")
    ax.plot(r, goe_density(r), color=INK, lw=1, ls=":", label="GOE")
    ax.set_title(f"K/g {point[0]:g}, delta/g {point[1]:g}: <r> {np.mean(cell['theory']):.3f}", fontsize=10, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0, 0].legend(fontsize=7.5, frameon=False)
fig.supxlabel("gap ratio r", color=INK)
fig.supylabel("probability density", color=INK)
fig.suptitle("The model's r distribution, and what our resolution keeps of it", color=INK)
fig.savefig(FIGURES / f"{PREFIX}_r_histograms.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
for point, color in zip(SHOWN, ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4")):
    ax.semilogy(TIME_US, results[point]["sff"] * np.exp(-2 * DECAY * TIME_US), color=color, lw=1.4,
                label=f"K/g {point[0]:g}, delta/g {point[1]:g}")
ax.axhline(1 / 35, color=INK, lw=0.8, ls=":", label="1/D")
ax.set(xlabel="t (us)", ylabel="<|Tr U(t)|^2> / D^2, with T2 166 us", ylim=(1e-4, 1.2))
ax.grid(alpha=0.25)
ax.legend(fontsize=8, frameon=False, ncols=2)
ax.set_title(f"The model's form factor ({THEORY_DRAWS} draws) with our decay", color=INK, fontsize=11)
fig.savefig(FIGURES / f"{PREFIX}_form_factors.png", dpi=110)
plt.close(fig)

np.savez(VAULT / "feasibility_map_results.npz", K=KERR_OVER_G, W=DISORDER_OVER_G,
         theory_r=grid(lambda c: np.mean(c["theory"])),
         **{f"kept_r_{v}": grid(lambda c, l=l: np.mean(c[l]["ratios"])) for l, v in zip(NOISE, ("std", "avg"))})
print("figures written", time.time() - start)
for i, K in enumerate(KERR_OVER_G):
    print(K, [f"{np.mean(results[K, W]['theory']):.3f}/{np.mean(results[K, W]['0.16 (standard)']['ratios']):.3f}/"
              f"{np.mean(results[K, W]['0.06 (more averaging)']['ratios']):.3f}" for W in DISORDER_OVER_G])
