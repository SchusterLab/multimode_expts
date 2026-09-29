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
# # Row offsets, and fitters C and D
#
# docs/qsim/pole_finding.md sections 4.2, 4.3 and 6; the findings are in
# `docs/log/2026-09-28_pole-finding-phase1.md`. The row-to-row frequency offset sigma on the
# August sets is 0.5-1 kHz (`sigma.py`). Here: what it does to A, B, C and D on synthetic data
# at the August point, and the four fitters on the real August sets. About 15 min (C is slow).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.job_paths import data_root
from experiments.qsim.pole_data import load_spectra
from fitting.qsim.poles import joint_pencil, joint_refined, per_row_clustered, per_row_reconciled
from fitting.qsim.poles.bench_plots import FITTER_COLORS, plot_resolution
from fitting.qsim.poles.bench_summaries import resolution_curve, scores_table, small_gap_table
from fitting.qsim.poles.benchmarks import run_nonideal_bench
from fitting.qsim.poles.matching import display_match, level_tolerances, match_poles
from fitting.qsim.poles.real_benchmarks import MODEL_ERROR_MHz, multiplet_weights
from fitting.qsim.poles.registry import data_set
from fitting.qsim.poles.statistics import small_gap_ratio_fraction
from fitting.qsim.poles.synthetic import Hardware, Nonideal, sample_phase_diagram

pd.set_option("display.width", 220)
FITTERS = {"A": (per_row_reconciled.fit, per_row_reconciled.ANALYSIS_SETTINGS),
           "B": (joint_pencil.fit, joint_pencil.JointPencilSettings()),
           "C": (joint_refined.fit, joint_refined.JointRefinedSettings()),
           "D": (per_row_clustered.fit, per_row_clustered.PerRowClusteredSettings())}

# %% [markdown]
# ## Synthetic: the August point with row offsets
#
# g 8.615 kHz, K/g -1.22, delta/g 5.80, 300 samples at 1.4509 us, 10 rows, SNR 100, decay
# 0.01 per us; one offset per row, Gaussian of width sigma: 0, the floor (0.5 kHz) and the
# upper bound (1.06 kHz). 5 disorder draws.

# %%
august = Hardware(coupling_MHz=8.615e-3, dt_us=1.4509, samples=300, partial_rows=10)
points = sample_phase_diagram([-1.22], [5.80], draws=5, hardware=august, seed=100)
sigmas_MHz = [0., 0.5e-3, 1.06e-3]
synthetic = run_nonideal_bench(points, [Nonideal(snr=100, decay_per_us=0.01, offset_sigma_MHz=s) for s in sigmas_MHz],
                               seeds=1, hardware=august, fitters=FITTERS)
table = scores_table(synthetic)
gaps = small_gap_table(synthetic)

# %%
means = table.groupby(["offset_sigma_MHz", "fitter"])[["resolved", "false_poles", "seconds"]].mean().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for fitter, rows in means.groupby("fitter"):
    kw = dict(color=FITTER_COLORS[fitter], marker="o", lw=2, label=fitter)
    axes[0].plot(1e3 * rows.offset_sigma_MHz, rows.resolved, **kw)
    axes[1].plot(1e3 * rows.offset_sigma_MHz, rows.false_poles, **kw)
    g = gaps[gaps.fitter == fitter].sort_values("offset_sigma_MHz")
    axes[2].plot(1e3 * g.offset_sigma_MHz, g.found, **kw)
axes[2].plot(1e3 * np.array(sigmas_MHz), gaps.groupby("offset_sigma_MHz").true.first(), "k--", lw=1, label="true")
for ax, label in zip(axes, ["levels resolved (of 35)", "false poles", "P(r < 0.25)"]):
    ax.set(xlabel="row offset sigma (kHz)", ylabel=label)
    ax.axvspan(0.5, 1.06, color="0.9", zorder=0)
axes[0].legend(fontsize=8)
axes[2].legend(fontsize=8)
fig.suptitle("Synthetic August point: the grey band is the measured sigma (floor to upper bound)", fontsize=10)
fig.tight_layout()
means.pivot(index="offset_sigma_MHz", columns="fitter").round(2)

# %%
plot_resolution(resolution_curve(synthetic, august.bin_MHz), unit="bins");

# %% [markdown]
# ## Real: august_N3 (complete basis, delta = 0, 10 levels)
#
# Top of each panel: the found poles; bottom: the model levels; matched pairs joined
# (within max(level tolerance, 0.3 kHz)). The bars: the found weight per model multiplet
# against its multiplicity.

# %%
n3 = load_spectra(data_set("august_N3"), data_root())[0]
disorder = load_spectra(data_set("august_disorder"), data_root())
fits = {(s.label, name): fit(s.A, s.time_us, settings) for s in [n3, *disorder] for name, (fit, settings) in FITTERS.items()}


def show_matches(spectrum, title):
    fig, axes = plt.subplots(len(FITTERS), 1, figsize=(11, 1.9 * len(FITTERS)), sharex=True)
    tolerance = np.maximum(level_tolerances(spectrum.levels_MHz, spectrum.bin_MHz), MODEL_ERROR_MHz)
    for ax, name in zip(axes, FITTERS):
        f = fits[spectrum.label, name]
        match = match_poles(f.frequencies_MHz, spectrum.levels_MHz, tolerance, 1 / spectrum.dt_us)
        display_match(match, f.frequencies_MHz, spectrum.levels_MHz, spectrum.bin_MHz, ax=ax,
                      title=f"{name}: {len(f.frequencies_MHz)} poles, {len(match.level_index)} of "
                            f"{len(spectrum.levels_MHz)} levels matched")
        ax.title.set_fontsize(9)
        if ax is not axes[0]:
            ax.get_legend().remove()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()


show_matches(n3, "august_N3")

# %%
fig, ax = plt.subplots(figsize=(10, 3.2))
x = np.arange(len(n3.levels_MHz))
ax.bar(x - 0.4, n3.multiplicities, width=0.16, color="0.7", label="model multiplicity")
for k, name in enumerate(FITTERS):
    sums, far = multiplet_weights(fits[n3.label, name], n3)
    error = np.mean(np.abs(sums - n3.multiplicities))
    ax.bar(x - 0.24 + 0.16 * k, sums, width=0.16, color=FITTER_COLORS[name],
           label=f"{name} (mean error {error:.2f}, {far} far poles)")
ax.set(xticks=x, xticklabels=[f"{1e3 * level:.1f}" for level in n3.levels_MHz], xlabel="model level (kHz)",
       ylabel="summed pole weight", title="august_N3: weight per multiplet")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout()

# %% [markdown]
# ## Real: august_disorder (4 realizations, 10 rows each)

# %%
show_matches(disorder[0], disorder[0].label)

# %%
rows = []
for s in disorder:
    tolerance = np.maximum(level_tolerances(s.levels_MHz, s.bin_MHz), MODEL_ERROR_MHz)
    for name in FITTERS:
        f = fits[s.label, name]
        rows.append(dict(fitter=name, poles=len(f.frequencies_MHz),
                         matched=len(match_poles(f.frequencies_MHz, s.levels_MHz, tolerance, 1 / s.dt_us).level_index)))
real = pd.DataFrame(rows).groupby("fitter").mean()
real["P(r<0.25)"] = [small_gap_ratio_fraction([fits[s.label, name].frequencies_MHz for s in disorder]) for name in real.index]
model_I = small_gap_ratio_fraction([s.levels_MHz for s in disorder])
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(real.index, real["P(r<0.25)"], color=[FITTER_COLORS[name] for name in real.index])
ax.axhline(model_I, color="k", ls="--", lw=1, label=f"model {model_I:.2f}")
ax.set(ylabel="P(r < 0.25), pooled", title="august_disorder")
ax.legend(fontsize=8)
fig.tight_layout()
real.round(2)
