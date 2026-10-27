"""Eigenstate expectation values <E_i|V|E_i> against E_i (the ETH picture), N = 3 and N = 10.

ETH: a thin smooth curve. Memory of the start state: a broad cloud. Island set point, one pattern.
The infinite-temperature C_1 (all Fock starts) averages over the whole cloud.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.agp import probe_values
from theory.qsim.central_boson import fixed_n_hamiltonian

OUT = Path(__file__).parent / "figures"
g, K, W = 0.02036, -0.0271, 60.
p = np.array([-0.71, 0.19, -1.03, 1.55])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
fig.subplots_adjust(hspace=0.45, wspace=0.3)
for row, (name, mode, kind) in enumerate([("S7 probe $n_{S7}$", 4, "number"), ("Kerr probe", 0, "kerr")]):
    for col, (N, kerr) in enumerate([(3, K), (10, K), (10, K * 3 / 10)]):
        m = fixed_n_hamiltonian(N, 5, 1e-3 * W * p, [g] * 4, kerr)
        v = probe_values(m.fock_basis, mode, kind)
        diag = np.einsum("fi,f,fi->i", m.states, v, m.states)
        pr = 1 / np.sum(m.states**4, 0) / len(v)
        a = ax[row, col]
        sc = a.scatter(1e3 * m.energies_MHz / N, diag / N if kind == "number" else diag / (N * (N - 1) / 2),
                       c=pr, s=6, cmap="viridis", vmin=0, vmax=0.3)
        lab = f"N = {N}, K = {1e3 * kerr:.1f} kHz" + (" (K·N fixed)" if col == 2 else "")
        name_short = "S7" if kind == "number" else "Kerr"
        a.set(title=f"{name_short}: {lab}", xlabel="E_i / N (kHz)",
              ylabel=r"$\langle V\rangle_i$ / max" if col == 0 else None, ylim=(-0.03, 1.03))
fig.colorbar(sc, ax=ax, label="participation ratio / D", shrink=0.6)
fig.suptitle("ETH picture: eigenstate expectation values against energy (W = 60 kHz, g = 20.4 kHz)", fontsize=9)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "eth_scatter.png", dpi=140, bbox_inches="tight")
