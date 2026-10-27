import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from theory.qsim.agp import probe_values, spectral_weights, spectral_agp, split_dc
from theory.qsim.central_boson import fixed_n_hamiltonian
from pathlib import Path
OUT = str(Path(__file__).parent / "figures") + "/"
g = 0.02036; K = -0.0271; p = np.array([-0.71, 0.19, -1.03, 1.55])
resid = 1e-3*np.array([0.4, -2.6, -2.6, -1.6])  # Q3b fitted W=0 residual detunings (10-23)
PROBES = {"probe $n_{S7}$ (detuning direction)": (4, "number"), "probe Kerr $n_M(n_M-1)/2$": (0, "kerr")}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
def sw(W, kerr, extra=0.):
    m = fixed_n_hamiltonian(3, 5, 1e-3*W*p + extra, [g]*4, kerr)
    return m, {n: spectral_weights(m, probe_values(m.fock_basis, *a)) for n, a in PROBES.items()}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
regimes = [("linear: $K=0$, $W=60$ kHz", 60, 0.), ("superintegrable: $W=0$, $K=-27$ kHz", 0, K), ("island: $W=60$, $K=-27$ kHz", 60, K)]
# Fig A: stick spectra
fig, ax = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True, sharey=True)
for j, (lab, W, kerr) in enumerate(regimes):
    _, out = sw(W, kerr)
    for i, (name, (d, w)) in enumerate(out.items()):
        a = ax[i, j]; c0 = w.sum(); dc, _ = split_dc(d, w, 1e-9)
        sel = (d > 1e-9) & (w > 1e-9*c0)
        a.vlines(1e3*d[sel], 1e-6, 2*w[sel]/c0, color=C[j], lw=1.2)
        a.axvspan(0, 10, color="0.9", zorder=0)
        a.set_yscale("log"); a.set_ylim(1e-5, 1); a.set_xlim(0, 260)
        a.text(0.97, 0.95, f"DC = {dc/c0:.2f} of $C(0)$\n{len(np.unique(np.round(1e6*d[sel])))} lines", transform=a.transAxes, ha="right", va="top")
        if i == 0: a.set_title(lab)
        if j == 0: a.set_ylabel(name + "\nweight / $C(0)$")
        if i == 1: a.set_xlabel(r"line frequency $\omega_{ij}/2\pi$ (kHz)")
fig.suptitle("Where the probe's weight sits (closed system, N = 3, D = 35). Grey: |ω| < 10 kHz, the part a μ = 10 kHz AGP weighs most.", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig_a_spectra.png", dpi=140)
# Fig B: AGP vs mu
mu = np.logspace(np.log10(0.3e-3), np.log10(150e-3), 200)
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
cases = [(regimes[0][0], 60, 0., 0., C[0], "-"), (regimes[1][0], 0, K, 0., C[1], "-"),
         ("$W=0$ + 2 kHz residual detunings", 0, K, resid, C[1], "--"), (regimes[2][0], 60, K, 0., C[2], "-")]
for i, name in enumerate(PROBES):
    a = ax[i]
    for lab, W, kerr, ex, col, ls in cases:
        _, out = sw(W, kerr, ex); d, w = out[name]
        a.loglog(1e3*mu, spectral_agp(d, w, mu, g), color=col, ls=ls, lw=2, label=lab)
    # sum rule mu^-4 for island
    _, out = sw(60, K); d, w = out[name]; M2 = np.sum(w*d**2)
    a.loglog(1e3*mu, g**2*M2/mu**4, color="0.5", ls=":", lw=1.2, label=r"sum rule $g^2M_2/\mu^4$ (large $\mu$)")
    for s, y0 in [(-1, 3e-2), (-2, 3e-1)]:
        x = np.array([1, 4.]); a.loglog(x, y0*(x/1)**s, color="k", lw=1); a.text(4.2, y0*4.**s, rf"$\mu^{{{s}}}$")
    a.set_ylim(1e-4, 1e3); a.set_title(name); a.set_xlabel(r"cutoff $\mu/2\pi$ (kHz)")
    a.axvspan(5, 20, color="0.92", zorder=0)
ax[0].set_ylabel(r"$g^2\|A_\mu\|^2$"); fig.legend(*ax[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower center", ncol=3)
fig.suptitle("AGP against μ, closed system. Grey band: μ = 5–20 kHz, about what the data resolve.", fontsize=9)
fig.tight_layout(rect=(0,0.13,1,1)); fig.savefig(OUT + "fig_b_mu.png", dpi=140)
# Fig C: AGP vs W
Ws = np.arange(0, 200.1, 1.)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for i, name in enumerate(PROBES):
    a = ax[i]
    for kerr, col, lab in [(0., C[0], "$K=0$ (linear)"), (K, C[2], "$K=-27$ kHz")]:
        for m_, ls in [(5e-3, ":"), (10e-3, "-"), (20e-3, "--")]:
            y = [spectral_agp(*sw(W, kerr)[1][name], m_, g) for W in Ws]
            a.semilogy(Ws, y, color=col, ls=ls, lw=1.6, label=f"{lab}, μ = {m_*1e3:.0f} kHz")
    a.set_title(name); a.set_xlabel("disorder strength W (kHz)")
ax[0].set_ylabel(r"$g^2\|A_\mu\|^2$"); ax[1].legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(OUT + "fig_c_W.png", dpi=140)
