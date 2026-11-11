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
# # AGP summary, 2026-11-10: all disorder scans with the current code
#
# The first scans (10-17..24: S7, Kerr) and the 11-05..07 scans (S5, and S7 and Kerr at small W).
# This notebook only loads, analyzes and lays out figures; the analysis is in
# `experiments/qsim/agp.py` (`load_scan`, `analyze`, `agp_table`). Each scan loads from its one
# HDF5 (`load_scan`); a scan saved before 2026-11-10 reads all its job files until
# `tools/rewrite_scan_h5.py` saves it again.
#
# Estimator `agp_connected` (C_1 minus a fitted A e^{-t/tau} from `BACKGROUND_US`). A point is
# "usable" when shot + calibration noise and 0.13 x the decaying-DC term (lab book Q5) are under
# 30 % of the value. Theory: the closed system, infinite time, on the recorded pattern, couplings
# and Kerr; and the same at K = 0 (free bosons). Figures go to `figures_20261110/`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.agp import agp_table, load_scan

OUT = Path("figures_20261110")
OUT.mkdir(exist_ok=True)
BACKGROUND_US = 30.
SHOWN_MU = [10., 20.]
SCANS = {  # label: registry key
    "S7 fine (10-17)": "s7_fine",
    "S7 coarse (10-17/23)": "s7_coarse",
    "S7 high W (10-29)": "s7_high_w",
    "S7 small W (11-07)": "261106_Storage_directions7",
    "Kerr (10-24)": "kerr",
    "Kerr W=0 (10-24)": "kerr_w0",
    "Kerr small W (11-07)": "261106_Storage_directions0",
    "S5 (11-05)": "261105_S5_direction5",
    "S5 small W (11-06)": "261106_Storage_directions5",
}
PROBES = {"S7": "S7", "M1": "Kerr", "S5": "S5"}  # probe mode: panel title

# %% [markdown]
# ## Load, analyze, tabulate

# %%
scans = {label: load_scan(key) for label, key in SCANS.items()}
for scan in scans.values():
    scan.analyze(background_start_us=BACKGROUND_US)
table = agp_table(scans, SHOWN_MU)
table.round(4)

# %% [markdown]
# ## The module's stages, per scan

# %%
for label, scan in scans.items():
    for name, figure in [("traces", scan.display_traces()),
                         ("spectrum", scan.display_spectrum(mu=SHOWN_MU, background_start_us=BACKGROUND_US)),
                         ("agp_mu", scan.display_agp_mu())]:
        figure.savefig(OUT / f"{SCANS[label]}_{name}.png", dpi=150)
        plt.close(figure)

# %% [markdown]
# ## AGP against W, per probe: measured, the closed system at the recorded K, and at K = 0

# %%
MARKERS = "osD^v<>"
fig, axes = plt.subplots(len(SHOWN_MU), len(PROBES), figsize=(13, 7), sharex="col", squeeze=False,
                         constrained_layout=True)
for column, (probe, title) in enumerate(PROBES.items()):
    members = list(dict.fromkeys(table[table.probe == probe].scan))
    reference = max((scans[label] for label in members), key=lambda s: np.max(s.data["disorder_khz"]))
    dense = np.linspace(0, table[table.probe == probe].W.max(), 201)
    with_kerr = reference.theory_agp(dense, mu=SHOWN_MU)
    free = reference.theory_agp(dense, mu=SHOWN_MU, kerr_mhz=0.)
    kerr_khz = 1e3 * reference.children[0].recorded_kerr_mhz()
    for row, cutoff in enumerate(SHOWN_MU):
        ax = axes[row, column]
        ax.plot(dense, with_kerr[:, row], color="#eb6834", lw=1.6, label=f"closed, K = {kerr_khz:.1f} kHz")
        ax.plot(dense, free[:, row], color="#2a78d6", lw=1.4, ls="--", label="closed, K = 0")
        for marker, label in zip(MARKERS, members):
            points = table[(table.scan == label) & (table.mu == cutoff)]
            for usable in (True, False):
                chosen = points[points.usable == usable]
                if len(chosen):
                    ax.errorbar(chosen.W, chosen.value.abs(), chosen.total, fmt=marker, ms=4, capsize=2,
                                color="k", mfc="k" if usable else "none", lw=0.8,
                                label=label if usable or not points.usable.any() else None)
        ax.set(yscale="log", ylim=(1e-4, 1), title=f"{title} probe, μ = {cutoff:g} kHz")
        if row == len(SHOWN_MU) - 1:
            ax.set_xlabel("W (kHz)")
        if column == 0:
            ax.set_ylabel(r"$g^2\|A_\mu\|^2$ (filled: usable)")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.savefig(OUT / "agp_vs_W_all.png", dpi=150)

# %% [markdown]
# ## Is there signal? Oscillation of the background-subtracted C_1 over its noise

# %%
fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
for marker, label in zip("os^Dv<>xP", scans):
    points = table[(table.scan == label) & (table.mu == SHOWN_MU[0])]
    ax.plot(points.W, points.signal, marker + "-", ms=4, lw=0.8, label=label)
ax.axhline(1, color="0.5", lw=0.8, ls=":")
ax.set(yscale="log", xlabel="W (kHz)", ylabel="std(C₁ − background) / shot error, t > 5 µs")
ax.legend(fontsize=6, frameon=False)
fig.savefig(OUT / "signal_vs_W.png", dpi=150)

# %% [markdown]
# ## Measured over the closed system, with Kerr and at K = 0 (usable points)

# %%
fig, axes = plt.subplots(1, len(PROBES), figsize=(13, 3.6), sharey=True, constrained_layout=True)
for ax, (probe, title) in zip(axes, PROBES.items()):
    members = list(dict.fromkeys(table[table.probe == probe].scan))
    for marker, label in zip(MARKERS, members):
        points = table[(table.scan == label) & (table.mu == 20.) & table.usable]
        if not len(points):
            continue
        ax.plot(points.W, points.value / points.theory, marker, color="#eb6834", ms=5, label=f"{label}: / K = −27 kHz")
        ax.plot(points.W, points.value / points.free, marker, color="#2a78d6", mfc="none", ms=5, label=f"{label}: / K = 0")
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set(title=f"{title} probe, μ = 20 kHz", xlabel="W (kHz)", ylim=(0.4, 2.2))
    ax.legend(fontsize=6, frameon=False)
axes[0].set_ylabel("measured / closed-system AGP")
fig.savefig(OUT / "ratio_vs_W.png", dpi=150)

# %% [markdown]
# ## Spectra for the slides: measured (background subtracted) and the closed system, same window

# %%
PICKS = [("S5 (11-05)", 20.), ("S5 (11-05)", 100.), ("S5 (11-05)", 200.),
         ("Kerr small W (11-07)", 59.), ("S7 small W (11-07)", 16.), ("S7 high W (10-29)", 200.)]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, constrained_layout=True)
for ax, (label, W) in zip(axes.ravel(), PICKS):
    data = scans[label].data
    i = int(np.argmin(np.abs(np.asarray(data["disorder_khz"]) - W)))
    f = np.asarray(data["frequency_khz"])
    shown = f <= 150
    measured, theory, error = data["spectrum"][i][shown], data["spectrum_theory"][i][shown], data["spectrum_err"][i][shown]
    top = 1.1 * max(theory.max(), measured.max())
    for cutoff, alpha in [(10., 0.22), (20., 0.12)]:
        ax.fill_between(f[shown], 0, top * 4 * cutoff**2 * f[shown]**2 / (f[shown]**2 + cutoff**2)**2,
                        color="0.45", alpha=alpha, lw=0, label=f"AGP filter, μ = {cutoff:g} kHz (scaled)")
    ax.fill_between(f[shown], measured - error, measured + error, color="#2a78d6", alpha=0.25, lw=0)
    ax.plot(f[shown], theory, color="#eb6834", lw=1.6, label="closed system, same window")
    ax.plot(f[shown], measured, color="#2a78d6", lw=1.2, label="measured, background subtracted")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set(title=f"{label.split()[0]} probe, W = {W:g} kHz ({label.split('(')[1][:-1]})", xlim=(0, 150))
for ax in axes[1]:
    ax.set_xlabel("frequency (kHz)")
for ax in axes[:, 0]:
    ax.set_ylabel("S(f) (1/kHz)")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.savefig(OUT / "spectra_slide.png", dpi=150)
