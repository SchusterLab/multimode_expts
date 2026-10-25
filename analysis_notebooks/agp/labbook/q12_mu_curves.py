"""Q12: the measured AGP against mu, per W, next to the closed-system curve.

Measured: exp fit (Q5), shot error band; grey where mu < mu_min of Q7 (not trustworthy).
Theory: closed system, infinite time; dotted: the sum-rule limit g^2 M_2 / mu^4 (Q2), which does
not depend on W. Where data and theory both bend away from that limit, the data see low-frequency
spectral weight.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from load_scans import VAULT, load_scans
from theory.qsim.agp import spectral_agp

MU = np.geomspace(2, 60, 30)
MU_MIN = np.load("cache/q7_mu_min.npy", allow_pickle=True).item()

scans = load_scans()
fig, axes = plt.subplots(3, 9, figsize=(24, 8.5), constrained_layout=True, sharey="row")
for r, (name, scan) in enumerate(scans.items()):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    times = scan.data["time_us"]
    for c in range(9):
        ax = axes[r, c]
        if c >= len(scan.children):
            ax.axis("off")
            continue
        part = scan.children[c]
        g = float(part.data["coupling_mhz"])
        diff, w = part.theory_spectrum()
        data, error = scan.data["C1_corrected"][c], scan.data["C1_err"][c]
        fit = fit_exponential_background(times, data, error, 30.)
        trace = data - fit["amplitude"] * np.exp(-fit["rate_per_us"] * times)
        kernels = np.array([(2 * np.pi * g)**2 * agp_kernel_weights(times, m / 1e3) for m in MU])
        value, noise = kernels @ trace, np.sqrt(kernels**2 @ error**2)
        theory = spectral_agp(diff, w, MU / 1e3, g)
        m2 = np.sum(w * diff**2)  # cyclic MHz^2
        mu_min = MU_MIN.get((name, part.disorder_khz), {}).get("now", np.nan)
        good = MU >= mu_min if np.isfinite(mu_min) else np.zeros(len(MU), bool)
        ax.plot(MU, theory, "k-", lw=1.5, label="closed theory")
        ax.plot(MU, g**2 * m2 / (MU / 1e3)**4, "k:", lw=1, label="g^2 M_2 / mu^4")
        ax.fill_between(MU, value - noise, value + noise, color="C0", alpha=.3, lw=0)
        ax.plot(MU[good], value[good], "C0o", ms=3, label="measured, exp fit")
        ax.plot(MU[~good], np.abs(value[~good]), "o", color="0.6", ms=3, label="|measured|, mu < mu_min")
        if np.isfinite(mu_min):
            ax.axvline(mu_min, color="C3", lw=.8)
        ax.set(xscale="log", yscale="log", ylim=(1e-4, 3), title=f"{name}, W = {part.disorder_khz:g}")
        if r == 2:
            ax.set_xlabel("mu (kHz)")
    axes[r, 0].set_ylabel(r"$g^2\|A_\mu\|^2$")
axes[0, 0].legend(fontsize=6)
fig.suptitle("Q12 [measured vs model]: AGP against the cutoff mu (red line: mu_min from Q7)")
fig.savefig(VAULT / "q12_mu_curves.png", dpi=100)
