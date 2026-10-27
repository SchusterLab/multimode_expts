"""Which ergodicity signatures a time-trace AGP could reach as the photon number N grows.

Island set point (K = -27.1 kHz, g = 20.36 kHz, W = 60 kHz, the scans' pattern), closed system.
Per N: mean level spacing Delta; the typical frequency of the probe's non-DC weight (how fast the
probe relaxes); and the connected memory (DC - Vbar^2) / (C(0) - Vbar^2), which ETH sends to 0.
Compare with the measured decay rate of C_1 and the resolvable mu.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.agp import probe_values, spectral_weights, split_dc
from theory.qsim.central_boson import fixed_n_hamiltonian

OUT = Path(__file__).parent / "figures"
g, K, W = 0.02036, -0.0271, 60.
p = np.array([-0.71, 0.19, -1.03, 1.55])
NS = np.arange(2, 11)
C = ["#2a78d6", "#eb6834", "#1baf7a"]
rows = []
for N in NS:
    m = fixed_n_hamiltonian(int(N), 5, 1e-3 * W * p, [g] * 4, K)
    e = np.sort(m.energies_MHz)
    out = dict(N=N, D=len(e), spacing=1e3 * np.mean(np.diff(e)))
    for name, (mode, kind) in {"S7": (4, "number"), "Kerr": (0, "kerr")}.items():
        v = probe_values(m.fock_basis, mode, kind)
        d, w = spectral_weights(m, v)
        dc, mask = split_dc(d, w, 1e-9)
        vbar2 = v.mean()**2
        off = w[~mask]
        # Split the connected DC: the part that follows energy (survives ETH) and the scatter around it.
        diag = np.einsum("fi,f,fi->i", m.states, v, m.states)[np.argsort(m.energies_MHz)]
        smooth = np.convolve(diag, np.ones(max(5, len(v) // 10)) / max(5, len(v) // 10), mode="same")
        edge = max(5, len(v) // 10) // 2
        smooth[:edge], smooth[-edge:] = diag[:edge].mean(), diag[-edge:].mean()
        variance = w.sum() - vbar2
        out[f"memory_{name}"] = (dc - vbar2) / variance
        out[f"energy_{name}"] = np.mean((smooth - v.mean())**2) / variance
        out[f"scatter_{name}"] = np.mean((diag - smooth)**2) / variance
        bulk = slice(len(v) // 4, 3 * len(v) // 4)
        out[f"bulk_{name}"] = np.mean((diag[bulk] - smooth[bulk])**2) / variance
        out[f"freq_{name}"] = 1e3 * np.sqrt(np.sum(off * d[~mask]**2) / off.sum())
    rows.append(out)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})

r = {k: np.array([row[k] for row in rows]) for k in rows[0]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
a = ax[0]
a.semilogy(NS, r["spacing"], "o-", color="k", lw=2, label="mean level spacing Δ (needs T ≳ 1/Δ)")
a.semilogy(NS, r["freq_S7"], "o-", color=C[0], lw=2, label="typical frequency of the S7 probe")
a.semilogy(NS, r["freq_Kerr"], "o-", color=C[1], lw=2, label="typical frequency of the Kerr probe")
a.axhspan(5, 10, color="0.88", zorder=0)
a.text(10, 7, "resolvable μ today", ha="right", va="center", fontsize=8)
a.axhline(1e3 / (2 * np.pi * 300), color="0.4", ls=":", lw=1)
a.text(10, 0.6, "decay of C₁ today, 1/(2π·300 µs)", ha="right", va="bottom", fontsize=8, color="0.3")
a.set(xlabel="photon number N", ylabel="kHz", title="time scales against N", ylim=(0.05, 300))
a.legend(frameon=False, fontsize=7.5, loc="lower left")
a = ax[1]
for name, c in [("S7", C[0]), ("Kerr", C[1])]:
    a.plot(NS, r[f"memory_{name}"], "o-", color=c, lw=2, label=f"{name}: all of the DC")
    a.plot(NS, r[f"energy_{name}"], "s--", color=c, lw=1.2, label=f"{name}: part that follows energy (stays under ETH)")
    a.plot(NS, r[f"scatter_{name}"], "^:", color=c, lw=1.2, label=f"{name}: scatter around it (ETH → 0)")
a.set(xlabel="photon number N", ylabel="fraction of the connected C(0)", ylim=(0, 1),
      title="what the DC of C₁ is made of")
a.legend(frameon=False, fontsize=7)
fig.suptitle("Island set point (K = −27 kHz, W = 60 kHz), closed system, one disorder pattern", fontsize=9)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "photon_number.png", dpi=140)
