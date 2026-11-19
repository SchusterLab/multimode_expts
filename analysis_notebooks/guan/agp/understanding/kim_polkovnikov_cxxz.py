# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reproducing Kim and Polkovnikov (arXiv:2308.09745) on the c-XXZ model
#
# Goal: check that our spectral-function and AGP tools give the paper's results before we use them on
# the central boson. Model, sectors and conventions as in the paper (`theory/qsim/spin_chains.py`):
# $\Delta = 1.2$, $h = 1.6$, open chain of $L$ spins, sectors $m = 0, \pm 2$ (even $L$), only the central 50%
# of the eigenstates of each sector as $|n\rangle$, cutoff $\mu = \alpha L / D_s$ with $D_s = \binom{L}{\lfloor L/2 \rfloor}$,
# Gaussian broadening $\Gamma = \beta\,\omega_H$ with $\omega_H$ the typical (median) level spacing of the central states.
# Directions: $h$ preserves integrability on the line $g = 1$, $g$ breaks it.
#
# The paper uses $L = 14$ to $17$; this notebook uses smaller $L$ (set below) so that it runs in about a minute.
#
# Claims to reproduce:
#
# - **R1** (Fig. 3a): at $g = 1$, $\Phi_h(\omega \to 0) \to 0$, while $\Phi_g$ saturates to a constant.
# - **R2** (Fig. 17): near $g = 1$, $\Phi_h(\omega) \sim (g-1)^2/\omega^2$ at low frequency, so curves for several
#   $g - 1$ collapse against $\omega/(g-1)$ onto $(\omega/(g-1))^{-2}$.
# - **R3** (Eq. 18): $\chi_h \sim (g-1)^2/\mu^3$ and $\chi_g \sim 1/\mu$, so they cross at $|g-1|_c \sim \mu$, and the
#   minimal direction of the QGT switches from $h$ (below) to $g$ (above).

# %%
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.agp import broadened_spectral_function, qgt_directions
from theory.qsim.spin_chains import (cxxz_operators, cxxz_sectors, cxxz_spectrum, largest_sector_dimension,
                                     pooled_sticks, qgt, typical_spacing)

LENGTH = 12
DELTA, H0 = 1.2, 1.6
OPERATORS = {ups: cxxz_operators(LENGTH, ups, DELTA) for ups in cxxz_sectors(LENGTH)}
D_S = largest_sector_dimension(LENGTH)
print(f"L = {LENGTH}: sectors (up spins) {cxxz_sectors(LENGTH)}, D_s = {D_S}, mu = 2 L / D_s = {2 * LENGTH / D_S:.4f}")


def averaged_phi(g, h_values, direction, omega, beta):
    """-> (Phi_n(omega), omega_H): sticks of all h realizations pooled (weights / realizations), broadened."""
    sticks, spacings = [], []
    for h in h_values:
        sectors = cxxz_spectrum(LENGTH, h, g, DELTA, operators=OPERATORS)
        sticks.append(pooled_sticks(sectors, direction))
        spacings.append(typical_spacing(sectors)[0])
    omega_h = float(np.mean(spacings))
    f = np.concatenate([s[0] for s in sticks])
    w = np.concatenate([s[1] for s in sticks]) / len(h_values)
    return broadened_spectral_function(f, w, omega, beta * omega_h), omega_h


# %% [markdown]
# ## R1. On the integrable line $g = 1$: $\Phi_h \to 0$ at low frequency, $\Phi_g$ saturates
#
# 11 realizations of $h$ within 5% of 1.6 and $\Gamma = 0.5\,\omega_H$, as in Fig. 3a. The dotted line marks $\omega_H$.

# %%
omega = np.geomspace(1e-4, 20, 300)
h_realizations = H0 * (1 + np.linspace(-0.05, 0.05, 11))
r1 = {name: averaged_phi(1.0, h_realizations, n, omega, 0.5) for name, n in (("h (preserving)", (1, 0)),
                                                                             ("g (breaking)", (0, 1)))}
fig, ax = plt.subplots(figsize=(6, 4))
for name, (phi, omega_h) in r1.items():
    ax.loglog(omega, phi, label=name)
ax.axvline(r1["h (preserving)"][1], color="gray", ls=":", lw=0.8)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\Phi_n(\omega)$")
ax.set_title(f"c-XXZ, L = {LENGTH}, g = 1, h = 1.6 (11 realizations), Gamma = 0.5 omega_H", fontsize=9)
ax.legend()
fig.tight_layout()
low = omega < 3 * r1["h (preserving)"][1]
for name, (phi, omega_h) in r1.items():
    print(f"{name}: omega_H = {omega_h:.2e}; Phi at the lowest omega {phi[0]:.2e}, max below 3 omega_H {phi[low].max():.2e}, "
          f"max overall {phi.max():.2e}")

# %% [markdown]
# ## R2. Near $g = 1$: $\Phi_h \sim (g-1)^2/\omega^2$, a collapse against $\omega/(g-1)$
#
# 3 realizations of $h$ within 1% and $\Gamma = 0.1\,\omega_H$, as in Fig. 17. Right: the same against $\omega/(g-1)$,
# with the $(\omega/(g-1))^{-2}$ line (dotted).

# %%
G_MINUS_1 = [0.02, 0.05, 0.1, 0.2]
h3 = H0 * (1 + np.array([-0.01, 0.0, 0.01]))
r2 = {dg: averaged_phi(1 + dg, h3, (1, 0), omega, 0.1) for dg in G_MINUS_1}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for dg, (phi, omega_h) in r2.items():
    ax1.loglog(omega, phi, label=f"g - 1 = {dg}")
    ax2.loglog(omega / dg, phi, label=f"g - 1 = {dg}")
x = np.geomspace(0.1, 100, 10)
ax2.loglog(x, 1 / x**2, "k:", label=r"$(\omega/(g-1))^{-2}$")
ax1.set_xlabel(r"$\omega$")
ax2.set_xlabel(r"$\omega/(g-1)$")
ax1.set_ylabel(r"$\Phi_h(\omega)$")
ax1.legend(fontsize=7)
ax2.legend(fontsize=7)
fig.tight_layout()
for dg, (phi, omega_h) in r2.items():
    band = (omega / dg > 1) & (omega / dg < 10) & (omega > 3 * omega_h) & (omega < 0.3)
    slope = np.polyfit(np.log(omega[band]), np.log(phi[band]), 1)[0] if band.sum() > 5 else np.nan
    print(f"g - 1 = {dg}: slope of log Phi_h over 1 < omega/(g-1) < 10 (and 3 omega_H < omega < 0.3): {slope:.2f}")

# %% [markdown]
# ## R3. AGP norms against $g - 1$: $\chi_h \sim (g-1)^2/\mu^3$, $\chi_g \sim 1/\mu$, and the switch of the minimal direction
#
# Left: $\chi_h$ and $\chi_g$ against $g - 1$ for several $\mu = \alpha L/D_s$. Middle: $\chi_h \mu^3/(g-1)^2$ and $\chi_g \mu$,
# which should be flat in the perturbative range. Right: the angle of the minimal direction ($0$ = $h$, $\pi/2$ = $g$)
# against $(g-1)/\mu$; the switch should sit at $(g-1)/\mu$ of order 1 for all $\mu$.
# On the line $g = 1$, $\chi_h$ is not zero: it is a constant in $\mu$ from the high-frequency part of $\Phi_h$. So the middle
# panel uses the excess $\chi_h(g) - \chi_h(1)$.

# %%
ALPHAS = [2, 4, 8, 16]
DG_SCAN = np.geomspace(3e-3, 0.5, 14)
spectra = {dg: [cxxz_spectrum(LENGTH, h, 1 + dg, DELTA, operators=OPERATORS) for h in h3] for dg in DG_SCAN}
mus = [a * LENGTH / D_S for a in ALPHAS]
tensors = {mu: np.array([np.mean([qgt(s, mu) for s in spectra[dg]], axis=0) for dg in DG_SCAN]) for mu in mus}
on_line = [cxxz_spectrum(LENGTH, h, 1.0, DELTA, operators=OPERATORS) for h in h3]
baseline = {mu: np.mean([qgt(s, mu) for s in on_line], axis=0) for mu in mus}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
for mu, color in zip(mus, ("C0", "C1", "C2", "C3")):
    t = tensors[mu]
    ax1.loglog(DG_SCAN, t[:, 0, 0], "-o", color=color, ms=3, label=f"chi_h, mu = {mu:.3f}")
    ax1.loglog(DG_SCAN, t[:, 1, 1], "--", color=color, label=f"chi_g, mu = {mu:.3f}")
    ax2.loglog(DG_SCAN, (t[:, 0, 0] - baseline[mu][0, 0]) * mu**3 / DG_SCAN**2, "-o", color=color, ms=3)
    ax2.loglog(DG_SCAN, t[:, 1, 1] * mu, "--", color=color)
    angles = [qgt_directions(m)["min_angle"] for m in t]
    angles = np.minimum(angles, np.pi - np.array(angles))   # fold to [0, pi/2]: distance from the h axis
    ax3.semilogx(DG_SCAN / mu, angles, "-o", color=color, ms=3, label=f"mu = {mu:.3f}")
ax1.set_xlabel("g - 1")
ax1.set_ylabel("AGP norm")
ax1.legend(fontsize=6)
ax2.set_xlabel("g - 1")
ax2.set_ylabel(r"$[\chi_h - \chi_h(g=1)] \mu^3/(g-1)^2$ (solid), $\chi_g \mu$ (dashed)")
ax3.axhline(np.pi / 4, color="gray", lw=0.8)
ax3.set_xlabel(r"$(g-1)/\mu$")
ax3.set_ylabel("angle of n_min from the h axis (rad)")
ax3.legend(fontsize=7)
fig.tight_layout()

for mu in mus:
    t = tensors[mu]
    cross = DG_SCAN[np.argmax(t[:, 0, 0] > t[:, 1, 1])] if np.any(t[:, 0, 0] > t[:, 1, 1]) else np.nan
    excess = t[:, 0, 0] - baseline[mu][0, 0]
    small = DG_SCAN < mu   # the perturbative range
    slope = np.polyfit(np.log(DG_SCAN[small]), np.log(excess[small]), 1)[0] if small.sum() > 2 else np.nan
    print(f"mu = {mu:.4f}: chi_h(g=1) = {baseline[mu][0, 0]:.3f}, chi_g(g=1) = {baseline[mu][1, 1]:.3f}; "
          f"slope of chi_h - chi_h(g=1) in g-1 for g-1 < mu: {slope:.2f}; "
          f"first g-1 with chi_h > chi_g: {cross:.3g} ((g-1)/mu = {cross / mu:.2f})")

chi_g_line = np.array([baseline[mu][1, 1] for mu in mus])
excess_line = np.array([tensors[mu][3, 0, 0] - baseline[mu][0, 0] for mu in mus])
print("slope of chi_g(g=1) in mu: %.2f (paper: -1)" % np.polyfit(np.log(mus), np.log(chi_g_line), 1)[0])
print("slope of chi_h - chi_h(g=1) in mu at g-1 = %.3g: %.2f (paper: -3)"
      % (DG_SCAN[3], np.polyfit(np.log(mus), np.log(excess_line), 1)[0]))
