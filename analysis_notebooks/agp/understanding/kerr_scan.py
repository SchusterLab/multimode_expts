"""What a scan of the Kerr set point K would show, at fixed disorder W (closed system, N = 3).

Units of g. For W/g = 1.5, 3, 5: the AGP along K (Kerr probe: the scan's own direction) and along
S7, at mu/g = 0.25, 0.5, 1; next to the level-spacing ratio <r> (one pattern, and the mean over
random patterns) and the mean Fock participation ratio of the eigenstates.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.agp import probe_values, spectral_agp, spectral_weights
from theory.qsim.central_boson import fixed_n_hamiltonian

OUT = Path(__file__).parent / "figures"
p = np.array([-0.71, 0.19, -1.03, 1.55])
KG = np.linspace(0, 3, 121)          # |K| / g, device sign K < 0
WG = [1.5, 3., 5.]
MUG = [0.25, 0.5, 1.]
C = ["#2a78d6", "#eb6834", "#1baf7a"]
rng = np.random.default_rng(1)
PATTERNS = rng.standard_normal((200, 4))
PATTERNS /= np.sqrt((PATTERNS**2).mean(1, keepdims=True))


def r_ratio(e):
    s = np.diff(np.sort(e))
    s = s[s > 1e-9 * np.abs(e).max()]
    return np.mean(np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1]))


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True)
for col, wg in enumerate(WG):
    agp = {"Kerr": [], "S7": []}
    r1, rmean, pr = [], [], []
    for kg in KG:
        m = fixed_n_hamiltonian(3, 5, wg * p, [1.] * 4, -kg)
        for name, (mode, kind) in {"Kerr": (0, "kerr"), "S7": (4, "number")}.items():
            agp[name].append(spectral_agp(*spectral_weights(m, probe_values(m.fock_basis, mode, kind)), np.array(MUG), 1.))
        r1.append(r_ratio(m.energies_MHz))
        pr.append(np.mean(1 / np.sum(np.abs(m.states)**4, 0)) / len(m.energies_MHz))
        rmean.append(np.mean([r_ratio(fixed_n_hamiltonian(3, 5, wg * q, [1.] * 4, -kg).energies_MHz) for q in PATTERNS[:60]]))
    for row, name in enumerate(["Kerr", "S7"]):
        a = ax[row, col]
        for j, (mu, c) in enumerate(zip(MUG, C)):
            a.semilogy(KG, np.array(agp[name])[:, j], color=c, lw=1.8, label=f"μ = {mu:g} g")
        a.set_title(f"W = {wg:g} g: AGP along {'K (Kerr probe)' if name == 'Kerr' else 'δ7 (S7 probe)'}")
    a = ax[2, col]
    a.plot(KG, rmean, color="k", lw=2, label="⟨r⟩, mean of 60 patterns")
    a.plot(KG, r1, color="0.6", lw=1, label="⟨r⟩, scans' pattern")
    a.plot(KG, pr, color=C[1], lw=1.5, ls="--", label="participation ratio / D")
    a.axhline(0.386, color="0.5", ls=":", lw=1); a.axhline(0.531, color="0.5", ls=":", lw=1)
    a.text(3, 0.39, "Poisson", ha="right", fontsize=7, color="0.4"); a.text(3, 0.535, "GOE", ha="right", fontsize=7, color="0.4")
    a.set(xlabel="|K| / g", ylim=(0, 0.65))
    a.axvline(27.1 / 20.36, color="0.3", lw=0.8, ls="-.")
ax[0, 0].legend(frameon=False, fontsize=7.5); ax[2, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
for r in range(2): ax[r, 0].set_ylabel(r"$g^2\|A_\mu\|^2$")
fig.suptitle("Kerr set-point scan at fixed W (closed system, N = 3). Dash-dot: today's |K|/g = 1.33", fontsize=9)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "kerr_scan.png", dpi=140)
