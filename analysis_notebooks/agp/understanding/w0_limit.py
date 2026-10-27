"""The W -> 0 limit of the AGP: degenerate dark-mode weight that a small W moves to omega ~ c W.

At W = 0 the three dark storage modes are degenerate. The part of the probe inside that
manifold is at omega = 0 (DC; the AGP filter gives it zero). A small W splits the manifold
linearly, omega_k = c_k W, so that weight w_k gives g^2 w_k (c_k W)^2 / ((c_k W)^2 + mu^2)^2:
zero at W = 0, a bump of height g^2 w_k / (4 mu^2) at W = mu / c_k.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.agp import probe_values, spectral_agp, spectral_weights
from theory.qsim.central_boson import fixed_n_hamiltonian

OUT = Path(__file__).parent / "figures"
g, K = 0.02036, -0.0271
p = np.array([-0.71, 0.19, -1.03, 1.55])
PROBES = {"probe $n_{S7}$": (4, "number"), "probe Kerr": (0, "kerr")}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def weights(W_khz, name):
    m = fixed_n_hamiltonian(3, 5, 1e-3 * W_khz * p, [g] * 4, K)
    return spectral_weights(m, probe_values(m.fock_basis, *PROBES[name]))


# Single-particle dark manifold: the disorder projected on the 3 modes orthogonal to the bright one.
bright = np.ones(4) / 2
dark = np.linalg.svd(np.eye(4) - np.outer(bright, bright))[0][:, :3]
shifts = np.linalg.eigvalsh(dark.T @ np.diag(-p) @ dark)
print("dark single-particle shifts per unit W:", shifts.round(3))

# Fragile lines: at a tiny W, the lines with omega << g are the split dark manifold. slope c = omega / W.
W_tiny = 1e-3  # kHz
for name in PROBES:
    d, w = weights(W_tiny, name)
    sel = (d > 1e-12) & (np.abs(d) < 1e-4) & (w > 1e-12)
    c = np.round(1e3 * d[sel] / W_tiny, 2)
    u, inv = np.unique(c, return_inverse=True)
    wk = np.bincount(inv, weights=2 * w[sel])
    print(f"{name}: fragile weight {wk.sum():.4f}; slopes c (kHz per kHz of W) and weights:",
          list(zip(u, wk.round(4))))
    if name == "probe $n_{S7}$":
        slopes, fragile = u, wk

W = np.logspace(-2, 2.3, 400)
MU = [1, 2, 5, 10]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for i, name in enumerate(PROBES):
    a = ax[i]
    for mu, col in zip(MU, C):
        y = [spectral_agp(*weights(x, name), mu * 1e-3, g) for x in W]
        y0 = spectral_agp(*weights(0., name), mu * 1e-3, g)
        a.loglog(W, y, color=col, lw=2, label=f"μ = {mu} kHz")
        a.axhline(y0, color=col, lw=0.8, ls=":")
        if i == 0:
            om = np.abs(slopes)[None, :] * W[:, None]
            pred = y0 + g**2 * 1e6 * (fragile[None, :] * om**2 / (om**2 + mu**2)**2).sum(1)
            a.loglog(W[W < 3 * mu], pred[W < 3 * mu], color="k", lw=0.9, ls="--")
    a.set(title=name, xlabel="disorder strength W (kHz), log scale")
ax[0].set_ylabel(r"$g^2\|A_\mu\|^2$")
ax[0].plot([], [], "k--", lw=0.9, label="degenerate perturbation theory")
ax[0].plot([], [], color="0.4", ls=":", lw=0.8, label="value at W = 0 exactly")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Near W = 0 (K = −27 kHz): the n_S7 AGP has a bump at W ≈ μ / c whose height grows as 1/μ²; the Kerr probe has none", fontsize=9)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "w0_limit.png", dpi=140)
