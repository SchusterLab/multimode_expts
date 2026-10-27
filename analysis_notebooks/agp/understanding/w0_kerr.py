"""Why the Kerr-probe AGP is low and its memory high at W = 0.

At W = 0, H = (M1 + bright mode B dimer) + 3 dark modes at zero energy. V = n_M (n_M - 1)/2 acts
on M1 only, so it lives in the dimer with N' = N - d photons (d = photons in the dark modes).
Part (a): the dimer decomposition at N = 3. Part (b): memory (DC / C(0)) and AGP against W.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.agp import probe_values, spectral_agp, spectral_weights, split_dc
from theory.qsim.central_boson import fixed_n_hamiltonian

OUT = Path(__file__).parent / "figures"
g, K = 0.02036, -0.0271
p = np.array([-0.71, 0.19, -1.03, 1.55])
C = ["#2a78d6", "#eb6834", "#1baf7a"]

# (a) The dimer M1-B: coupling 2g (four equal g, B = sum a_s / 2), N' photons.
print("dimer sectors at W = 0, N = 3 (d = dark photons, dark degeneracy C(d+2, 2)):")
total_c0 = total_dc = 0.
for d in range(4):
    n = 3 - d
    deg = (d + 1) * (d + 2) // 2
    basis = np.arange(n + 1)                     # n_M = 0..n
    h = np.diag(0.5 * K * basis * (basis - 1))
    for k in range(n):                           # a_M^dag a_B between n_M = k and k + 1
        h[k + 1, k] = h[k, k + 1] = 2 * g * np.sqrt((k + 1) * (n - k))
    e, u = np.linalg.eigh(h)
    v = basis * (basis - 1) / 2
    vm = u.T @ np.diag(v) @ u
    c0, dc = deg * np.sum(v**2), deg * np.sum(np.diag(vm)**2)
    total_c0 += c0; total_dc += dc
    lines = sorted({round(1e3 * abs(a - b), 1) for a in e for b in e if a > b})
    print(f"  d={d}: dimer N'={n}, {n+1} levels x {deg} dark states; sum V^2 = {c0:.1f}, DC part = {dc:.2f};"
          f" lines (kHz) {lines}")
print(f"  DC / C(0) = {total_dc / total_c0:.3f}  (sector mean over D = 35)")

# (b) Against W, with the full model.
Ws = np.arange(0, 200.1, 1.)
memory, agp, nlines = [], [], []
for W in Ws:
    m = fixed_n_hamiltonian(3, 5, 1e-3 * W * p, [g] * 4, K)
    d, w = spectral_weights(m, probe_values(m.fock_basis, 0, "kerr"))
    dc, mask = split_dc(d, w, 1e-9)
    memory.append(dc / w.sum())
    agp.append(spectral_agp(d, w, 0.01, g))
    sel = (d > 1e-9) & (w > 1e-3 * w.sum())
    nlines.append(np.sum(w[sel & (d < 0.03)]) * 2 / w.sum())
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
ax[0].plot(Ws, memory, color=C[0], lw=2)
ax[0].set(title="memory: DC / C(0) of the Kerr probe", ylim=(0, 0.6), xlabel="W (kHz)")
ax[1].plot(Ws, nlines, color=C[2], lw=2)
ax[1].set(title="weight in lines below 30 kHz / C(0)", xlabel="W (kHz)")
ax[2].semilogy(Ws, agp, color=C[1], lw=2)
ax[2].set(title=r"$g^2\|A_\mu\|^2$, μ = 10 kHz", xlabel="W (kHz)")
for a in ax:
    a.axvspan(0, 2, color="0.88", zorder=0)
fig.suptitle("Kerr probe, K = −27 kHz, N = 3: W = 0 is the M1–bright dimer only; disorder lets the dark modes in", fontsize=9)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "w0_kerr.png", dpi=140)
