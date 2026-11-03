"""Measured spectral function S_V(omega) against theory, for a few W.

S(f) = (2/pi) int_0^T C(t) w(t) cos(2 pi f t) dt with a half-Hann window w = cos^2(pi t / 2T).
Measured: C_1 minus its fitted decaying background A e^{-t/tau} (lab book Q5, fit from 30 us).
Theory: closed-system lines (DC removed), each damped at the fitted envelope rate 1/tau, same grid
and window. So both curves have the same resolution; the comparison is line by line.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.agp import fit_exponential_background
from measured import measured_scans
from theory.qsim.agp import probe_values, spectral_weights, split_dc
from theory.qsim.central_boson import fixed_n_hamiltonian

OUT = Path("figures")
S = measured_scans()
PICK = [("S7 fine", 30.), ("S7 fine", 60.), ("Kerr", 20.), ("Kerr", 100.), ("Kerr", 180.), ("Kerr W=0", 0.)]
PROBE = {"S7 fine": (4, "number"), "Kerr": (0, "kerr"), "Kerr W=0": (0, "kerr")}
f = np.linspace(0, 250, 1001)  # kHz


def transform(t, c):
    w = np.cos(np.pi * t / (2 * t[-1]))**2
    dt = np.gradient(t)
    return (2 / np.pi) * (np.cos(2 * np.pi * 1e-3 * f[:, None] * t[None, :]) @ (c * w * dt))


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
for a, (name, W) in zip(ax.ravel(), PICK):
    s = S[name]
    i = int(np.argmin(np.abs(s["W"] - W)))
    t, data, err = s["times"], s["data"][i], s["err"][i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_exponential_background(t, data, err, 30.)
    measured = transform(t, data - fit["amplitude"] * np.exp(-fit["rate_per_us"] * t))
    pattern = S["Kerr"]["pattern"] if name == "Kerr W=0" else s["pattern"]
    m = fixed_n_hamiltonian(3, 5, 1e-3 * W * pattern, s["couplings"], s["kerr"])
    d, w = spectral_weights(m, probe_values(m.fock_basis, *PROBE[name]))
    _, mask = split_dc(d, w, 1e-9)
    osc = (np.cos(2 * np.pi * t[:, None] * d[~mask][None, :]) @ w[~mask]) * np.exp(-fit["rate_per_us"] * t)
    theory = transform(t, osc)
    a.axhline(0, color="0.8", lw=0.8)
    # The AGP filter omega^2 / (omega^2 + mu^2)^2, scaled to the panel: where the AGP looks.
    top = 1.05 * max(theory.max(), measured.max())
    for mu, alpha in [(10., 0.22), (20., 0.12)]:
        shape = 4 * mu**2 * f**2 / (f**2 + mu**2)**2
        a.fill_between(f, 0, top * shape, color="0.45", alpha=alpha, lw=0,
                       label=f"AGP filter, μ = {mu:g} kHz (scaled)")
    a.plot(f, theory, color="#eb6834", lw=1.8, label="closed theory, same window")
    a.plot(f, measured, color="#2a78d6", lw=1.4, label="measured")
    a.set_title(f"{'S7' if name.startswith('S7') else 'Kerr'} probe, W = {W:g} kHz (T = {t[-1]:.0f} µs)")
    a.set_xlim(0, 250)
for a in ax[1]:
    a.set_xlabel("frequency (kHz)")
for a in ax[:, 0]:
    a.set_ylabel(r"$S_V(f)$ (µs)")
ax[0, 0].legend(frameon=False, fontsize=8.5)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "spectra_measured.png", dpi=150)
