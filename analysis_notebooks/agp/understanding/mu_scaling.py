"""Why the AGP slope against mu reads the low-frequency shape of S(omega).

g^2 ||A_mu||^2 = g^2 int_0^inf S(w) F_mu(w) dw,  F_mu(w) = w^2 / (w^2 + mu^2)^2.
Toy spectra with unit weight and a bandwidth cutoff Lambda; frequencies in kHz.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

OUT = Path(__file__).parent / "figures"
LAM, GAMMA, W0 = 100., 2., 0.01  # bandwidth, Lorentzian half-width, low cut of 1/w (kHz)
C = ["#2a78d6", "#eb6834", "#1baf7a"]


def normalized(f):
    z = quad(f, 0, 1e5, limit=500, points=[W0, GAMMA, LAM])[0]
    return lambda w: f(w) / z


SPECTRA = {
    "flat near 0 (ETH-like)": normalized(lambda w: np.exp(-w / LAM)),
    r"$\propto 1/\omega$ (broad spread of slow rates)": normalized(lambda w: (w > W0) * np.exp(-w / LAM) / np.maximum(w, W0)),
    rf"Lorentzian, $\Gamma$ = {GAMMA:g} kHz (one slow rate)": normalized(lambda w: GAMMA / (w**2 + GAMMA**2) * np.exp(-w / LAM)),
}
mu = np.logspace(-1, 3, 120)
w = np.logspace(-2, 3, 600)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for m, ls in [(2, ":"), (10, "-"), (50, "--")]:
    ax[0].semilogx(w, m**2 * w**2 / (w**2 + m**2)**2, color="k", ls=ls, lw=1.4, label=f"μ = {m} kHz")
ax[0].set(title=r"filter $\mu^2 F_\mu(\omega)$: a band at $\omega\approx\mu$, width $\sim\mu$", xlabel="ω (kHz)", ylim=(0, 0.3))
ax[0].legend(frameon=False)
for (name, S), col in zip(SPECTRA.items(), C):
    ax[1].loglog(w, S(w), color=col, lw=2, label=name)
    agp = [quad(lambda x: S(x) * x**2 / (x**2 + m**2)**2, 0, 1e5, limit=500, points=[W0, GAMMA, m, LAM])[0] for m in mu]
    ax[2].loglog(mu, agp, color=col, lw=2)
ax[1].set(title=r"toy spectra $S(\omega)$, unit weight", xlabel="ω (kHz)", ylim=(1e-6, 1e2))
ax[1].legend(frameon=False, fontsize=7.5, loc="lower left")
for s, x0, y0 in [(-1, 0.3, 3e-1), (-2, 2, 2e-2), (-3, 10, 1e-4), (-4, 300, 2e-8)]:
    x = np.array([x0, 3 * x0])
    ax[2].loglog(x, y0 * (x / x0)**s, color="k", lw=1)
    ax[2].text(3.3 * x0, y0 * 3.**s, rf"$\mu^{{{s}}}$")
ax[2].axvline(LAM, color="0.6", lw=1, ls=":"); ax[2].text(LAM * 1.1, 1e-1, "Λ", color="0.4")
ax[2].set(title=r"$\int S\,F_\mu\,d\omega$ against μ", xlabel="μ (kHz)")
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "mu_scaling.png", dpi=140)
