"""The closed-system spectral function averaged over disorder patterns, and what a finite window does to it.

Fig D (``disorder_average.png``): the line weights of fig A, averaged over random unit-RMS patterns
(i.i.d. normal, then scaled to RMS 1, as the experiment's pattern is), in log-spaced frequency bins,
per W, for K = 0 and K = -27 kHz. Tick marks: the mean level spacing of the sector at that W.
The mean AGP over patterns is the AGP of this mean spectrum, since the AGP is linear in the weights.

Fig E (``disorder_average_window.png``): the island regime (W = 60, K = -27 kHz) with the measured
pattern, as sticks, as the same lines seen through a Hann window of T = 244 us (1/T = 4.1 kHz, the
S7 coarse scan), and the pattern average through the same window. Run from this folder.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.agp import probe_values, spectral_weights, split_dc
from theory.qsim.central_boson import fixed_n_hamiltonian

OUT = Path("figures")
g = 0.02036; K = -0.0271
MEASURED = np.array([-0.71, 0.19, -1.03, 1.55])  # S7 scans' pattern
PROBES = {"probe $n_{S7}$": (4, "number"), "probe Kerr $n_M(n_M-1)/2$": (0, "kerr")}
WS = [5., 15., 30., 60., 90., 150.]
REALIZATIONS = 400
EDGES = np.logspace(-1, np.log10(600), 46)  # kHz
T_US = 244.
rng = np.random.default_rng(0)


def patterns(count):
    p = rng.normal(size=(count, 4))
    return p / np.sqrt(np.mean(p**2, axis=1, keepdims=True))


def lines(W, kerr, pattern):
    """-> {probe: (|f| kHz, weight / C(0))} of the connected lines (DC removed), and the mean spacing (kHz)."""
    m = fixed_n_hamiltonian(3, 5, 1e-3 * W * pattern, [g] * 4, kerr)
    out = {}
    for name, args in PROBES.items():
        d, w = spectral_weights(m, probe_values(m.fock_basis, *args))
        _, dc = split_dc(d, w, 1e-9)
        out[name] = (1e3 * np.abs(d[~dc]), w[~dc] / w.sum())
    return out, 1e3 * np.mean(np.diff(m.energies_MHz))


def averaged(W, kerr, count=REALIZATIONS):
    """-> {probe: mean density per kHz in EDGES bins}, mean level spacing (kHz)."""
    total = {name: np.zeros(len(EDGES) - 1) for name in PROBES}
    spacing = 0.
    for pattern in patterns(count):
        out, s = lines(W, kerr, pattern)
        spacing += s / count
        for name, (f, w) in out.items():
            total[name] += np.histogram(f, EDGES, weights=w)[0] / count
    return {name: v / np.diff(EDGES) for name, v in total.items()}, spacing


def hann(f_khz, f_lines, w_lines):
    """-> S(f) per kHz of the lines seen over [0, T] with the half-Hann window, the transform of display_spectrum."""
    t = np.linspace(0, T_US, 2001)
    c = np.cos(2 * np.pi * 1e-3 * np.outer(t, f_lines)) @ w_lines
    trap = np.full(len(t), t[1]); trap[[0, -1]] /= 2
    kernel = 4 * trap * np.cos(np.pi * t / (2 * T_US))**2 * np.cos(2 * np.pi * 1e-3 * np.outer(f_khz, t))
    return kernel @ c / 1e3


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
centers = np.sqrt(EDGES[1:] * EDGES[:-1])
colors = plt.cm.viridis(np.linspace(0, 0.9, len(WS)))
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row")
for j, (kerr, title) in enumerate([(0., "linear, $K = 0$"), (K, "$K = -27$ kHz")]):
    for W, col in zip(WS, colors):
        density, spacing = averaged(W, kerr)
        for i, name in enumerate(PROBES):
            a = ax[i, j]
            a.loglog(centers, density[name], color=col, lw=1.6, label=f"W = {W:g} kHz")
            a.plot(spacing, 1.5e-6, "|", color=col, ms=12, mew=2)
    for i, name in enumerate(PROBES):
        a = ax[i, j]
        a.axvspan(5, 20, color="0.92", zorder=0)
        a.set_ylim(1e-6, 1e-1); a.set_xlim(EDGES[0], EDGES[-1])
        if i == 0: a.set_title(title)
        if j == 0: a.set_ylabel(name + "\nmean S(f) / C(0)  (1/kHz)")
        if i == 1: a.set_xlabel("f (kHz)")
ax[0, 1].legend(frameon=False, fontsize=8)
fig.suptitle(f"Closed system, N = 3, D = 35, mean over {REALIZATIONS} unit-RMS patterns, DC removed. "
             "Ticks: mean level spacing. Grey: μ = 5–20 kHz.", fontsize=9)
fig.tight_layout(); OUT.mkdir(exist_ok=True); fig.savefig(OUT / "disorder_average.png", dpi=140)

# Fig E
f = np.linspace(0, 200, 801)
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
one, _ = lines(60., K, MEASURED)
pool = [lines(60., K, p)[0] for p in patterns(REALIZATIONS)]
for a, name in zip(ax, PROBES):
    fl, wl = one[name]
    u, inv = np.unique(np.round(fl, 3), return_inverse=True)
    wu = np.bincount(inv, weights=wl)
    a.vlines(u, 0, wu * T_US / 1e3, color="0.7", lw=0.8, label="measured pattern: lines, height = weight × T")
    a.plot(f, hann(f, fl, wl), color="#2a78d6", lw=1.6, label="measured pattern, Hann window T = 244 µs")
    mean = np.mean([hann(f, *p[name]) for p in pool[:100]], axis=0)
    a.plot(f, mean, color="k", lw=1.6, label="mean over 100 patterns, same window")
    a.set_ylabel(name + "\nS(f) / C(0)  (1/kHz)")
ax[0].set_title("Island regime, W = 60 kHz, K = -27 kHz, closed system")
ax[0].legend(frameon=False, fontsize=8); ax[1].set_xlabel("f (kHz)"); ax[1].set_xlim(0, 200)
fig.tight_layout(); fig.savefig(OUT / "disorder_average_window.png", dpi=140)
