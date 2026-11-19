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
# # Near the integrable lines of the central boson: does the preserving direction grow a low-frequency tail?
#
# The qualitative claim of Kim and Polkovnikov (arXiv:2308.09745, reproduced on their c-XXZ model in
# `kim_polkovnikov_cxxz.py`): just off an integrable line, the spectral function of the integrability-preserving
# perturbation grows a low-frequency tail, about $(\text{breaking})^2/\omega^2$ above a frequency that scales with the
# breaking strength, and it rises above the spectral function of the breaking perturbation. No cutoff $\mu$ is used
# here.
#
# Model: 3 photons, 1 hub and 4 storage modes, $g = 20$ kHz (`theory/qsim/spectral_ensemble.py`). The two
# perturbations are the derivatives of $H$ in the $(K, W)$ plane, both in MHz units, so their spectral functions
# can be compared directly:
#
# - $\partial H/\partial K = n_0(n_0-1)/2$ (hub Kerr),
# - $\partial H/\partial W = V = \sum_s \epsilon_s n_s$ (the disorder direction, with the unit-RMS pattern that sets $H$).
#
# On the line $K = 0$ (free bosons) $V$ preserves and Kerr breaks. On the line $W = 0$ (dark modes decoupled) Kerr
# preserves and $V$ breaks. Spectra: weight per MHz on log bins, mean over disorder patterns.
#
# Caution: the sector has $D = 35$ states and a mean level spacing of about 10 kHz, so everything below about
# 10 kHz is below the level spacing of one pattern. The low-frequency weight here comes from pairs of levels that
# are close in some patterns. In the paper the tail lies above the Heisenberg scale.

# %%
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.spectral_ensemble import StarSetup, ensemble_spectra

SETUP = StarSetup(photon_number=3, storage_modes=4, coupling_mhz=0.02)
PROBES = {"Kerr, dH/dK": (0, "kerr"), "disorder V, dH/dW": (None, "disorder")}
EDGES = np.geomspace(1e-6, 1, 70)
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
SAMPLES = 3000
LOW_BAND = (1e-4, 1e-3)   # MHz: 0.1 to 1 kHz, far below the level spacing


def scan(points, seed=1):
    return [ensemble_spectra(SETUP, k, w, SAMPLES, EDGES, probes=PROBES, seed=seed) for k, w in points]


def band_mean(density, band=LOW_BAND):
    inside = (CENTERS > band[0]) & (CENTERS < band[1])
    return density[inside].mean()



# %% [markdown]
# ## T1. The free line $K = 0$ at $W = 35$ kHz: the preserving $V$ against the breaking Kerr
#
# Expect: at $K = 0$, $V$ has no weight below the single-particle band (about 10 kHz). For small $|K|$ a low-frequency
# part appears; if it follows the paper, it is a plateau that ends at a frequency proportional to $|K|$, with a
# $K^2/\omega^2$ fall after it, so the curves collapse against $\omega/|K|$ (right panel). The breaking Kerr probe
# should change little.

# %%
T1_K = [0.0, -0.001, -0.003, -0.01, -0.03]
t1 = scan([(k, 0.035) for k in T1_K])
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for res, k in zip(t1, T1_K):
    for ax, name in zip(axes[:2], PROBES):
        d = res[name]["density"]
        ax.loglog(CENTERS, np.where(d > 0, d, np.nan), label=f"K = {k * 1e3:g} kHz")
    if k != 0:
        d = res["disorder V, dH/dW"]["density"]
        axes[2].loglog(CENTERS / abs(k), np.where(d > 0, d, np.nan), label=f"K = {k * 1e3:g} kHz")
x = np.geomspace(1, 30, 5)
axes[2].loglog(x, 1 / x**2, "k:", label=r"$(\omega/|K|)^{-2}$")
for ax, name in zip(axes[:2], PROBES):
    ax.set_title(f"K = 0 line, W = 35 kHz\n{name}", fontsize=8)
    ax.axvspan(*LOW_BAND, color="gray", alpha=0.1)
    ax.set_xlabel("frequency (MHz)")
axes[2].set_title("disorder V against frequency / |K|", fontsize=8)
axes[2].set_xlabel(r"$\omega/|K|$")
axes[0].set_ylabel("weight / MHz")
for ax in axes:
    ax.set_ylim(1e-6, 1e3)
    ax.legend(fontsize=7)
fig.tight_layout()
for res, k in zip(t1, T1_K):
    print(f"K = {k * 1e3:5g} kHz: mean weight/MHz in 0.1-1 kHz: V {band_mean(res['disorder V, dH/dW']['density']):.3g}, "
          f"Kerr {band_mean(res['Kerr, dH/dK']['density']):.3g}")

# %% [markdown]
# ## T2. The dark-mode line $W = 0$ at $K = -27$ kHz: the preserving Kerr against the breaking $V$
#
# Expect: the low-frequency weight of the Kerr probe grows from zero as $W^2$ for small $W$ (right panel, against the
# $W^2$ guide). The breaking $V$ has its weight near $\omega \approx W$ at small $W$ (see `spectral_function.py`, Q6).

# %%
T2_W = [0.0003, 0.001, 0.003, 0.01, 0.03]
t2 = scan([(-0.027, w) for w in T2_W])
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for res, w in zip(t2, T2_W):
    for ax, name in zip(axes[:2], PROBES):
        d = res[name]["density"]
        ax.loglog(CENTERS, np.where(d > 0, d, np.nan), label=f"W = {w * 1e3:g} kHz")
for name, marker in zip(PROBES, ("o-", "s--")):
    axes[2].loglog(np.array(T2_W) * 1e3, [band_mean(r[name]["density"]) for r in t2], marker, label=name)
ws = np.array(T2_W[:3]) * 1e3
axes[2].loglog(ws, band_mean(t2[0]["Kerr, dH/dK"]["density"]) * (ws / ws[0])**2, "k:", label=r"$\propto W^2$")
for ax, name in zip(axes[:2], PROBES):
    ax.set_title(f"W = 0 line, K = -27 kHz\n{name}", fontsize=8)
    ax.axvspan(*LOW_BAND, color="gray", alpha=0.1)
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylim(1e-6, 1e3)
    ax.legend(fontsize=7)
axes[0].set_ylabel("weight / MHz")
axes[2].set_xlabel("W (kHz)")
axes[2].set_ylabel("mean weight / MHz in 0.1-1 kHz")
axes[2].legend(fontsize=7)
fig.tight_layout()
for res, w in zip(t2, T2_W):
    print(f"W = {w * 1e3:5g} kHz: mean weight/MHz in 0.1-1 kHz: Kerr {band_mean(res['Kerr, dH/dK']['density']):.3g}, "
          f"V {band_mean(res['disorder V, dH/dW']['density']):.3g}")
