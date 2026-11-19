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
# # Spectral function of a small central-boson model
#
# All parameters are given by hand, in cyclic MHz: 1 hub (M1) and 4 storage modes, 3 photons (sector
# dimension $D = 35$), $g = 20$ kHz, $K = -27$ kHz unless stated. Run from this folder. One cell group per question.
#
# **Setup.** The spectral function of a probe $O$ is the list of sticks $(\Delta_{ij}, w_{ij})$ with
# $\Delta_{ij} = E_i - E_j$ and $w_{ij} = |\langle i|O|j\rangle|^2 / D$. Then
# $C(t) = \mathrm{Tr}[O(t)\,O]/D = w_\mathrm{DC} + 2\sum_{\Delta_{ij}>0} w_{ij}\cos(2\pi\Delta_{ij}t)$, where $w_\mathrm{DC}$
# is the weight at $\Delta = 0$ and includes the trivial constant $\langle O\rangle^2$. The disorder pattern $\epsilon$
# (4 storage detunings) is iid normal, mean removed, scaled to RMS $W$: uniform on the sphere of fixed RMS in the
# zero-mean plane (C7).
#
# **Probes.** Hub Kerr $n_0(n_0-1)/2$, hub number $n_0$, storage number $n_s$ (one mode; the 4 modes are equivalent
# after averaging, Q2), and the disorder direction $V = \sum_s \epsilon_s n_s$. In a fixed-$N$ sector
# $n_0 = N - \sum_s n_s$, so the hub number is the sum of all storage numbers, not one of them.
#
# **Which operators preserve integrability** (for discussion; my reading of the integrable lines).
# At $W = 0$ the dark modes decouple and $K$ runs along the line: the preserving perturbation is hub Kerr
# ($\partial H/\partial K$), and the breaking one is $V$ ($\partial H/\partial W$). At $K = 0$, $H$ is free, and any
# quadratic operator preserves it, so both $n_0$ and $n_s$ do. The disorder ensemble moves along $V$, not along
# $n_0$, because the mean of the pattern is removed.
#
# **What is checked** (C1 to C7): the Hamiltonian, the weights, $C(t)$, the sum rule, the free limit, the
# pattern, the binning against independent calculations. **What the figures show:** Q3 and Q4 the spectra at
# the corners of the $(K, W)$ plane with numbered pointers; Q5 how the total weight scales toward the
# integrable lines (hub Kerr falls as $1/K^2$ at large $|K|$; number probes keep a constant weight; at large $W$ the
# single-mode and hub probes fall about as $1/W$ and $V$ as $1/W^2$); Q6 that the single-mode storage probe is not a
# good proxy for $V$ at low frequency.
#
# **Open.** The origin of the Kerr-made low-frequency shoulder and the dip (pointers 2, 3, 4, 6); the
# explanation of the $1/W$ law (pointer 11); the sign conventions are checked against the code only, not the
# device; `tests/test_mbr_analysis_golden.py` errors in this checkout (a fixture signature problem); and how
# to measure $V$ in the experiment, since a single-mode readout gives only the diagonal terms of
# $C_V(t) = \sum_{s,s'} \epsilon_s \epsilon_{s'} C_{ss'}(t)$.

# %%
import matplotlib.pyplot as plt
import numpy as np

from theory.qsim.agp import probe_values, spectral_weights
from theory.qsim.central_boson import fixed_n_hamiltonian

PHOTONS = 3
STORAGE_MODES = 4
COUPLING_MHZ = 0.02
KERR_MHZ = -0.027
DISORDER_RMS_MHZ = 0.06   # W: RMS of the detunings over the storage modes
SAMPLES = 5000
rng = np.random.default_rng(0)


def disorder_pattern(rng, count, rms_mhz):
    """-> ``count`` detunings: iid normal, mean removed, scaled to RMS ``rms_mhz``.

    This is uniform on the sphere of fixed RMS in the zero-mean plane.
    """
    pattern = rng.normal(size=count)
    pattern -= pattern.mean()
    return pattern * rms_mhz / np.sqrt(np.mean(pattern**2))


def build_model(detunings, kerr_mhz=KERR_MHZ):
    return fixed_n_hamiltonian(
        photon_number=PHOTONS,
        mode_count=STORAGE_MODES + 1,
        detunings=detunings,
        couplings_MHz=[COUPLING_MHZ] * STORAGE_MODES,
        physical_kerr_MHz=kerr_mhz,
    )


def stick_spectrum(model, values):
    """-> (frequencies, weights, dc) for a probe given by its diagonal ``values`` in the Fock basis."""
    differences, weights = spectral_weights(model, values)   # both (D, D)
    f, w = differences.ravel(), weights.ravel()
    keep = f > 1e-9
    return f[keep], w[keep], w[np.abs(f) <= 1e-9].sum()


def spectrum(model, mode_index, probe="number"):
    """-> (frequencies, weights, dc): positive-frequency sticks and the zero-frequency weight.

    ``mode_index`` 0 is the hub M1, 1 to ``STORAGE_MODES`` are the storage modes.
    ``probe`` is ``number`` or ``kerr``. The spectrum is mirrored, so ``dc`` is
    the whole zero-frequency weight and the sticks carry half of the rest.
    """
    return stick_spectrum(model, probe_values(model.fock_basis, mode_index=mode_index, probe=probe))


def sticks(model, mode_index):
    """-> (frequencies, weights) of the positive-frequency sticks of the probe n of one mode."""
    return spectrum(model, mode_index)[:2]


# %% [markdown]
# ## Q1. What does one realization look like, and the ensemble average of the hub probe?

# %%
f_one, w_one = sticks(build_model(disorder_pattern(rng, STORAGE_MODES, DISORDER_RMS_MHZ)), 0)

pooled = [sticks(build_model(disorder_pattern(rng, STORAGE_MODES, DISORDER_RMS_MHZ)), 0)
          for _ in range(SAMPLES)]
f_all = np.concatenate([f for f, _ in pooled])
w_all = np.concatenate([w for _, w in pooled])
edges = np.geomspace(f_all.min(), f_all.max(), 60)
# Weight per realization per unit frequency.
density = np.histogram(f_all, edges, weights=w_all)[0] / SAMPLES / np.diff(edges)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
ax1.vlines(f_one, 0, w_one)
ax1.set_xlabel("E_i - E_j (MHz)")
ax1.set_ylabel("weight")
ax1.set_title("one realization")
ax2.stairs(density, edges)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("E_i - E_j (MHz)")
ax2.set_ylabel("weight / MHz")
ax2.set_title(f"mean of {SAMPLES} realizations")
fig.tight_layout()

# %% [markdown]
# ## Q2. After the ensemble average, are all storage modes equivalent?
#
# The couplings are equal and the pattern distribution is symmetric under exchange of storage
# modes, so the averaged spectral function of $n_s$ should be the same for every storage mode $s$.
# The same realizations are used for all probes. A spread between modes is compared with the
# standard error of the mean over realizations.

# %%
bins = np.geomspace(1e-4, 1.0, 40)
rng = np.random.default_rng(1)
per_realization = np.zeros((SAMPLES, STORAGE_MODES, len(bins) - 1))   # weight per MHz
for r in range(SAMPLES):
    model = build_model(disorder_pattern(rng, STORAGE_MODES, DISORDER_RMS_MHZ))
    for s in range(STORAGE_MODES):
        f, w = sticks(model, s + 1)
        per_realization[r, s] = np.histogram(f, bins, weights=w)[0] / np.diff(bins)

mean = per_realization.mean(axis=0)                                  # (modes, bins)
sem = per_realization.std(axis=0, ddof=1) / np.sqrt(SAMPLES)
# Per bin, the spread of the 4 means in units of their standard error (4 curves, same data).
spread = (mean.max(axis=0) - mean.min(axis=0)) / sem.mean(axis=0)
centers = np.sqrt(bins[:-1] * bins[1:])

print("worst spread / SEM over bins with signal:", np.nanmax(spread[mean.mean(axis=0) > 0]).round(2))
print("total weight per mode:", (mean * np.diff(bins)).sum(axis=1).round(5))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
for s in range(STORAGE_MODES):
    ax1.errorbar(centers, mean[s], sem[s], label=f"storage {s + 1}", alpha=0.7)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel("weight / MHz")
ax1.legend()
ax2.plot(centers, spread, "o-")
ax2.set_xlabel("E_i - E_j (MHz)")
ax2.set_ylabel("(max - min) / SEM")
fig.tight_layout()


# %% [markdown]
# ## Q3. One point in the (Kerr, disorder) plane, four probes: how to show it
#
# Probes: hub Kerr $n_0(n_0-1)/2$, hub number $n_0$, storage number $n_s$ (mean of the 4 equivalent storage modes),
# and the disorder direction $V = \sum_s \epsilon_s n_s$ with the unit-RMS pattern $\epsilon$ that also sets $H$
# (the perturbation that $W$ controls, see Q6).
# Top row: ensemble-mean weight per MHz (the spectral function) on log-log axes, band is one standard error.
# Bottom row: cumulative weight of the positive-frequency sticks, which needs no bin choice.
# The title gives the zero-frequency (DC) weight and the positive-frequency total.

# %%
PROBES = {
    "hub Kerr": (0, "kerr"),
    "hub number": (0, "number"),
    "storage number": (None, "number"),   # None: mean over the storage modes
    "disorder V": (None, "disorder"),     # sum_s e_s n_s, with the unit-RMS pattern e that sets H
}


def probe_sticks(model, unit, mode, kind):
    """-> (frequencies, weights, dc) of one probe of ``PROBES``; ``unit`` is the unit-RMS pattern of ``model``.

    ``mode`` None averages the single-mode probe over the storage modes. ``disorder`` is V = sum_s e_s n_s.
    """
    if kind == "disorder":
        return stick_spectrum(model, np.asarray(model.fock_basis, dtype=float)[:, 1:] @ unit)
    modes = range(1, STORAGE_MODES + 1) if mode is None else [mode]
    parts = [spectrum(model, m, kind) for m in modes]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]) / len(modes),
            sum(p[2] for p in parts) / len(modes))
FREQ_EDGES = np.geomspace(1e-5, 0.5, 60)


def ensemble_spectra(kerr_mhz, rms_mhz, samples, seed=0, edges=FREQ_EDGES):
    """-> {probe name: dict(f, w, density, sem, dc)} for one (Kerr, disorder) point.

    ``f``, ``w``: all sticks of all realizations pooled (storage probe: all modes pooled, weights / modes),
    so the weights sum to ``samples`` times the mean positive-frequency weight. ``density``, ``sem``: mean
    and standard error over realizations of the binned weight per MHz. ``dc``: mean zero-frequency weight.
    """
    rng = np.random.default_rng(seed)
    binned = {name: np.zeros((samples, len(edges) - 1)) for name in PROBES}
    pooled = {name: ([], []) for name in PROBES}
    dc = {name: np.zeros(samples) for name in PROBES}
    for r in range(samples):
        unit = disorder_pattern(rng, STORAGE_MODES, 1.0)
        model = build_model(rms_mhz * unit, kerr_mhz=kerr_mhz)
        for name, (mode, kind) in PROBES.items():
            f, w, dc[name][r] = probe_sticks(model, unit, mode, kind)
            pooled[name][0].append(f)
            pooled[name][1].append(w)
            binned[name][r] = np.histogram(f, edges, weights=w)[0] / np.diff(edges)
    return {name: dict(f=np.concatenate(pooled[name][0]), w=np.concatenate(pooled[name][1]),
                       density=binned[name].mean(0), sem=binned[name].std(0, ddof=1) / np.sqrt(samples),
                       dc=dc[name].mean())
            for name in PROBES}


def plot_point(spectra, samples, title):
    centers = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])
    fig, axes = plt.subplots(2, len(spectra), figsize=(4 * len(spectra), 5.5), sharex=True)
    for col, (name, s) in enumerate(spectra.items()):
        top, bottom = axes[:, col]
        density, sem = s["density"], s["sem"]
        top.stairs(np.maximum(density, 1e-300), FREQ_EDGES)
        top.fill_between(centers, np.maximum(density - sem, 1e-300), density + sem, alpha=0.3)
        top.set_xscale("log")
        top.set_yscale("log")
        top.set_ylim(density.max() * 1e-5, density.max() * 3)   # cut the empty-bin floor
        top.set_title(f"{name}\nDC {s['dc']:.3f}, f > 0 total {s['w'].sum() / samples:.3f}", fontsize=9)
        order = np.argsort(s["f"])
        bottom.plot(s["f"][order], np.cumsum(s["w"][order]) / samples)
        bottom.set_xlabel("E_i - E_j (MHz)")
    axes[0, 0].set_ylabel("weight / MHz")
    axes[1, 0].set_ylabel("cumulative weight")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


POINT_KERR_MHZ, POINT_RMS_MHZ = -0.012, 0.035
fig = plot_point(ensemble_spectra(POINT_KERR_MHZ, POINT_RMS_MHZ, SAMPLES), SAMPLES,
                 f"K = {POINT_KERR_MHZ * 1e3:.0f} kHz, W = {POINT_RMS_MHZ * 1e3:.0f} kHz, "
                 f"g = {COUPLING_MHZ * 1e3:.0f} kHz, {PHOTONS} photons, {SAMPLES} patterns")


# %% [markdown]
# ## Checks of the calculation
#
# Each cell below checks one step with an independent calculation and asserts the result.
# Checks C1 to C3 use one random pattern at the Q3 point. Q2 above is the check that all
# storage modes are equivalent after averaging.

# %%
from functools import reduce

from scipy.linalg import expm

CHECK_RNG = np.random.default_rng(11)
check_detunings = disorder_pattern(CHECK_RNG, STORAGE_MODES, POINT_RMS_MHZ)
check_model = build_model(check_detunings, kerr_mhz=POINT_KERR_MHZ)

# Independent build: truncated boson operators on all modes (full space), then the sector of PHOTONS.
CUT = PHOTONS + 1
_lower = np.diag(np.sqrt(np.arange(1, CUT)), 1)
_modes = STORAGE_MODES + 1


def full_space_operator(single, k):
    return reduce(np.kron, [single if j == k else np.eye(CUT) for j in range(_modes)])


annihilate = [full_space_operator(_lower, k) for k in range(_modes)]
number = [a.T @ a for a in annihilate]
sector = np.where(np.isclose(sum(number).diagonal(), PHOTONS))[0]


def independent_hamiltonian(detunings, kerr_mhz):
    """-> H (MHz) of the sector, from the operator form H = sum -d_s n_s + K/2 n0 (n0 - 1) + g (a0^+ a_s + h.c.)."""
    dim = CUT**_modes
    H = kerr_mhz / 2 * number[0] @ (number[0] - np.eye(dim))
    for s, d in enumerate(detunings, start=1):
        H = H - d * number[s] + COUPLING_MHZ * (annihilate[0].T @ annihilate[s] + annihilate[s].T @ annihilate[0])
    return H[np.ix_(sector, sector)]


# %% [markdown]
# ### C1. The Hamiltonian: eigenvalues of `fixed_n_hamiltonian` against the operator-form build

# %%
H_independent = independent_hamiltonian(check_detunings, POINT_KERR_MHZ)
error = np.abs(np.sort(check_model.energies_MHz) - np.linalg.eigvalsh(H_independent)).max()
print(f"sector dimension {len(sector)}; max |E_code - E_independent| = {error:.1e} MHz")
assert len(sector) == len(check_model.energies_MHz) and error < 1e-12

# %% [markdown]
# ### C2. The weights: $C(t) = \mathrm{Tr}[O(t)\,O]/D$ from a matrix exponential of the independent $H$, against the stick sum

# %%
from theory.qsim.agp import two_time_correlation

check_times = np.linspace(0, 300, 7)   # us
operators = {   # name: (operator in the full space, mode index, probe kind)
    "hub Kerr": (number[0] @ (number[0] - np.eye(CUT**_modes)) / 2, 0, "kerr"),
    "hub number": (number[0], 0, "number"),
    "storage 2 number": (number[2], 2, "number"),
}
for name, (full, mode, kind) in operators.items():
    O = full[np.ix_(sector, sector)]
    direct = []
    for t in check_times:
        U = expm(-2j * np.pi * H_independent * t)
        direct.append(np.trace(U.conj().T @ O @ U @ O).real / len(sector))
    differences, weights = spectral_weights(check_model, probe_values(check_model.fock_basis, mode, kind))
    error = np.abs(np.array(direct) - two_time_correlation(differences, weights, check_times)).max()
    print(f"{name}: max |C_direct - C_sticks| = {error:.1e}   (C(0) = {direct[0]:.4f})")
    assert error < 1e-10

# %% [markdown]
# ### C3. The sum rule: $w_\mathrm{DC} + 2\sum_{\Delta>0} w = \overline{O^2}$, the mean over the Fock states
#
# The factor 2 is the mirrored negative-frequency half. The right side is counted from the Fock
# basis alone, with no diagonalization.

# %%
for name, (mode, kind) in {"hub Kerr": (0, "kerr"), "hub number": (0, "number"), "storage 2 number": (2, "number")}.items():
    f, w, dc = spectrum(check_model, mode, kind)
    expected = np.mean(probe_values(check_model.fock_basis, mode, kind) ** 2)
    print(f"{name}: DC {dc:.4f} + 2 x {w.sum():.4f} = {dc + 2 * w.sum():.6f}   mean O^2 = {expected:.6f}")
    assert abs(dc + 2 * w.sum() - expected) < 1e-12

# %% [markdown]
# ### C4. The free limit: at $K = 0$ the number-probe sticks lie on the single-particle differences
#
# The photon-number-1 sector is the single-particle problem. Its 5 levels give 10 differences.

# %%
worst = 0
for _ in range(100):
    detunings = disorder_pattern(CHECK_RNG, STORAGE_MODES, POINT_RMS_MHZ)
    single = fixed_n_hamiltonian(1, STORAGE_MODES + 1, detunings, [COUPLING_MHZ] * STORAGE_MODES, 0.0).energies_MHz
    lines = np.abs(single[:, None] - single[None, :]).ravel()
    lines = lines[lines > 1e-9]
    free_model = build_model(detunings, kerr_mhz=0.0)
    for mode in (0, 1):
        f, w, _ = spectrum(free_model, mode, "number")
        f = f[w > 1e-10]   # sticks with weight
        worst = max(worst, np.abs(f[:, None] - lines[None, :]).min(axis=1).max())
print(f"worst distance of a stick from a single-particle difference: {worst:.1e} MHz")
assert worst < 1e-9

# %% [markdown]
# ### C5. The disorder pattern: zero mean, RMS equal to $W$, and a symmetric ensemble

# %%
patterns = np.array([disorder_pattern(CHECK_RNG, STORAGE_MODES, POINT_RMS_MHZ) for _ in range(20000)])
print("max |mean| of a pattern:", np.abs(patterns.mean(axis=1)).max())
print("max |RMS - W| / W:", np.abs(np.sqrt((patterns**2).mean(axis=1)) / POINT_RMS_MHZ - 1).max())
print("mean of each mode (0 expected):", patterns.mean(axis=0) / POINT_RMS_MHZ)
print("correlation of mode pairs, off-diagonal (-1/3 expected):",
      np.round(np.corrcoef(patterns.T)[np.triu_indices(STORAGE_MODES, 1)], 3))
assert np.abs(patterns.mean(axis=1)).max() < 1e-12
assert np.abs(np.sqrt((patterns**2).mean(axis=1)) / POINT_RMS_MHZ - 1).max() < 1e-12

# %% [markdown]
# ### C6. The binning: the plotted density integrates to the pooled weight inside the bin range
#
# The weight outside the range (below 10 Hz or above 0.5 MHz) is what the plot does not show.

# %%
check_samples = 200
spectra = ensemble_spectra(POINT_KERR_MHZ, POINT_RMS_MHZ, check_samples, seed=3)
inside = (lambda f: (f >= FREQ_EDGES[0]) & (f < FREQ_EDGES[-1]))
for name, s in spectra.items():
    shown = (s["density"] * np.diff(FREQ_EDGES)).sum()
    pooled_inside = s["w"][inside(s["f"])].sum() / check_samples
    outside = s["w"][~inside(s["f"])].sum() / check_samples
    print(f"{name}: plotted {shown:.6f}, pooled inside range {pooled_inside:.6f}, outside range {outside:.2e} "
          f"of {s['w'].sum() / check_samples:.4f}")
    assert abs(shown - pooled_inside) < 1e-12


# %% [markdown]
# ### C7. The pattern ensemble: Gaussian draw against the uniform-cube draw
#
# Both are mean-removed and scaled to RMS 1, so a pattern is a point on a sphere (radius 2) in the 3-dimensional
# zero-mean plane. Uniform on the sphere means that the component along any axis, as a fraction of the radius, is
# uniform on $[-1, 1]$. The table gives the KS distance from that, along 4 axes, and the largest $|\epsilon_s|/W$.

# %%
from scipy import stats

basis = np.linalg.qr(np.eye(STORAGE_MODES) - 1 / STORAGE_MODES)[0][:, :STORAGE_MODES - 1].T   # (3, 4)
axes_3d = [np.array(a, dtype=float) / np.linalg.norm(a) for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1])]


def normalize(p):
    p = p - p.mean(axis=1, keepdims=True)
    return p / np.sqrt((p**2).mean(axis=1, keepdims=True))


rng_c7 = np.random.default_rng(1)
draws = {"gaussian (used)": normalize(rng_c7.normal(size=(400000, STORAGE_MODES))),
         "uniform cube (old)": normalize(rng_c7.uniform(-1, 1, (400000, STORAGE_MODES)))}
for name, p in draws.items():
    unit = p @ basis.T / np.sqrt(STORAGE_MODES)   # unit vector in the plane
    ks = [stats.kstest(unit @ a, stats.uniform(-1, 2).cdf).statistic for a in axes_3d]
    print(f"{name}: KS along 4 axes {np.round(ks, 4)}, mean max|e|/W {np.abs(p).max(axis=1).mean():.3f}, "
          f"99.9% {np.quantile(np.abs(p).max(axis=1), 0.999):.3f} (bound {np.sqrt(STORAGE_MODES - 1):.3f})")
    if name.startswith("gaussian"):
        assert max(ks) < 0.006   # noise level for 400000 draws

# %% [markdown]
# ## Q4. Corners of the (Kerr, disorder) plane: do the curves look sensible?
#
# Rows: the generic point, the free line ($K = 0$), small disorder, exactly zero disorder (dark modes
# decoupled), large disorder ($W \gg g$, detuned) and large Kerr ($|K| \gg g$, self-trapped hub).
# Vertical gray lines mark the scales $g$ (dash-dot), $W$ (dashed) and $|K|$ (dotted).
# The numbers in circles point at the features discussed in the cell after the figure.
# The $W = 0$ row is one exact spectrum with no averaging: lines of equal frequency are summed, and the
# y-axis is the weight per line, not per MHz.

# %%
CORNERS = {   # name: (K in MHz, W in MHz)
    "generic": (-0.027, 0.035),
    "free, K = 0": (0.0, 0.035),
    "small W": (-0.027, 0.001),
    "W = 0 (exact)": (-0.027, 0.0),
    "large W": (-0.027, 0.3),
    "large K": (-0.3, 0.035),
}
POINTERS = {   # row: [(frequency in MHz, number)], drawn in all three panels of the row
    "generic": [(0.114, "1"), (1e-3, "2"), (7e-3, "3")],
    "free, K = 0": [(0.01, "4"), (1e-3, "5"), (2e-3, "6")],
    "small W": [(1e-3, "7"), (0.05, "8")],
    "W = 0 (exact)": [(0.04, "9"), (0.08, "10")],
    "large W": [(1e-3, "11"), (0.03, "12"), (0.3, "13")],
    "large K": [(0.3, "14"), (0.015, "15")],
}
CORNER_EDGES = np.geomspace(1e-5, 5, 80)
CORNER_SAMPLES = 5000


def exact_lines(kerr_mhz):
    """-> {probe name: dict(f, w, dc)} at W = 0, one realization, equal frequencies summed."""
    model = build_model(np.zeros(STORAGE_MODES), kerr_mhz=kerr_mhz)
    out = {}
    for name, (mode, kind) in PROBES.items():
        if kind == "disorder":
            continue   # below
        modes = range(1, STORAGE_MODES + 1) if mode is None else [mode]
        parts = [spectrum(model, m, kind) for m in modes]
        f = np.concatenate([p[0] for p in parts])
        w = np.concatenate([p[1] for p in parts]) / len(modes)
        lines, inverse = np.unique(np.round(f, 9), return_inverse=True)
        out[name] = dict(f=lines, w=np.bincount(inverse, weights=w), dc=sum(p[2] for p in parts) / len(modes))
    # Disorder direction at W = 0: H does not depend on the pattern, so the pattern covariance
    # (4/3)(I - J/4) gives the line weights (4/3) sum_s w_s - w_hub / 3 exactly. DC is not given (n/a).
    hub = spectrum(model, 0, "number")
    stor = [spectrum(model, m, "number") for m in range(1, STORAGE_MODES + 1)]
    f = np.concatenate([p[0] for p in stor] + [hub[0]])
    w = np.concatenate([4 / 3 * p[1] for p in stor] + [-hub[1] / 3])
    lines, inverse = np.unique(np.round(f, 9), return_inverse=True)
    combined = np.bincount(inverse, weights=w)
    assert combined.min() > -1e-9   # a negative line weight would mean the covariance formula failed
    out["disorder V"] = dict(f=lines, w=np.maximum(combined, 0), dc=np.nan)
    return out


def mark(ax, x, label):
    """A numbered pointer at the top of the panel, with a short tick down to the curve."""
    ax.annotate(label, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 9), textcoords="offset points",
                ha="center", va="center", fontsize=7, bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="C3"),
                arrowprops=dict(arrowstyle="-", color="C3", lw=0.8))


def plot_corners(corners, pointers, samples, edges):
    centers = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(len(corners), len(PROBES), figsize=(4 * len(PROBES), 2.5 * len(corners)),
                             sharex=True, squeeze=False)
    results = {}
    for row, (label, (kerr, rms)) in enumerate(corners.items()):
        exact = rms == 0
        results[label] = spectra = exact_lines(kerr) if exact else ensemble_spectra(kerr, rms, samples, seed=row, edges=edges)
        for col, (name, s) in enumerate(spectra.items()):
            ax = axes[row, col]
            if exact:
                ax.vlines(s["f"], 1e-300, s["w"])
                top = s["w"].max()
                dc_text = "n/a" if np.isnan(s["dc"]) else f"{s['dc']:.3f}"
                subtitle = f"{name}: DC {dc_text}, {len(s['f'])} lines"
            else:
                ax.stairs(np.maximum(s["density"], 1e-300), edges)
                ax.fill_between(centers, np.maximum(s["density"] - s["sem"], 1e-300), s["density"] + s["sem"], alpha=0.3)
                top = max(s["density"].max(), 1e-12)
                subtitle = f"{name}: DC {s['dc']:.3f}, f>0 {s['w'].sum() / samples:.4f}"
            ax.axvline(COUPLING_MHZ, color="gray", ls="-.", lw=0.8)
            if rms > 0:
                ax.axvline(rms, color="gray", ls="--", lw=0.8)
            if kerr != 0:
                ax.axvline(abs(kerr), color="gray", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(top * 1e-6, top * 3)
            ax.set_title(subtitle, fontsize=8, pad=14 if pointers.get(label) else 4)
            for x, number in pointers.get(label, []):
                mark(ax, x, number)
        unit = "weight per line" if exact else "weight / MHz"
        axes[row, 0].set_ylabel(f"{label}\nK={kerr * 1e3:.0f}, W={rms * 1e3:.0f} kHz\n{unit}", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("E_i - E_j (MHz)")
    fig.tight_layout()
    return fig, results


fig, corner_results = plot_corners(CORNERS, POINTERS, CORNER_SAMPLES, CORNER_EDGES)


# %% [markdown]
# ### What the numbers in Q4 point at
#
# Status: **checked** = shown by a calculation in this notebook, **scan** = from Q5, **open** = a guess, not tested.
#
# 1. Generic point, all probes: the high-frequency cutoff at about 0.11 MHz. It is the largest single-particle level difference (mean 0.114 MHz at $W = 35$ kHz). **checked** (C4 shows the $K = 0$ sticks are on those differences; row 2 shows this part does not move with $K$).
# 2. Generic, number probes: the shoulder from about $10^{-5}$ to $3\times10^{-3}$ MHz. It is absent for the number probes at $K = 0$ (row 2), so Kerr makes it. **open**: why it has this shape.
# 3. Generic, hub number: the dip between the shoulder and the single-particle band. **open**.
# 4. $K = 0$, hub number: almost no weight below about 10 kHz. **open**: I think the hub weight sits in the bright single-particle levels only.
# 5. $K = 0$, hub Kerr: a plateau already at $K = 0$. Hub Kerr is a two-body operator, so its frequencies are sums and differences of two single-particle energies. **open**, consistent with the operator form.
# 6. $K = 0$, storage number: weight down to about 1 kHz, from pairs of single-particle levels that are close. **open**.
# 7. Small $W$, storage number: a cluster of weight at $0.4\,W$ to $2\,W$ (5 to 95 percent), median $1.0\,W$, see the cell after Q5. The hub probes have almost none of it. This is a property of the single-mode probe $n_s$, not of the perturbation $V = \sum_s \epsilon_s n_s$: Q6 shows that $V$ has almost no weight there (about $10^{-2}$ against $10^{2}$ per MHz for the proxy), because the eigenstates of the degenerate manifold diagonalize $V$ at first order. **checked** (the cluster) and **checked** in Q6 (it is absent for $V$).
# 8. Small $W$: the main lump is the single-particle band, as at $W = 0$. **checked** by comparison with row 4.
# 9. $W = 0$, storage number: the strongest line is at 40 kHz $= 2g$, the bright-to-dark single-particle difference $g\sqrt{4} = 2g$. **checked** (strongest lines printed in the cell after Q5).
# 10. $W = 0$: the line at 80 kHz $= 4g$ is the difference of the two bright levels, $\pm g\sqrt{4}$. Hub number has only these lines (75 to 95 kHz after the Kerr splitting), because the dark modes have no hub weight. **checked**.
# 11. Large $W$: the plateau from $3\times10^{-4}$ to $10^{-2}$ MHz in the single-mode and hub probes. **open**: I think it comes from rare patterns with a storage detuning within about $g$ of the hub (probability about $g/W$). Q5 shows these probes fall as $1/W$, which fits this, while the disorder direction $V$ falls as $1/W^2$: a resonant mode has $\epsilon_s \approx 0$, so it carries almost no weight in $V$. Not tested.
# 12. Large $W$: the bump at 20 to 50 kHz, the scale of $g$ and $K$. **open**.
# 13. Large $W$: the smooth tail, which ends at about $W$. **open**.
# 14. Large $K$, hub Kerr: the line at $|K|$, and a total that falls as $1/K^2$. **scan**.
# 15. Large $K$, storage number: the broad bump at the $g$ scale, with a constant total as $|K|$ grows, so the hub keeps hopping while $n_0 \le 1$. **scan**, the explanation is **open**.

# %% [markdown]
# ## Q5. How the total weight scales toward the integrable lines
#
# The positive-frequency total of each probe, mean and standard error over patterns, as $W$ grows ($K = -27$ kHz)
# and as $|K|$ grows ($W = 35$ kHz). Dashed lines are guides: $1/W$ and $1/W^2$ (left), $1/K^2$ (right).
# At large $W$ the average is dominated by rare patterns, so the standard error is larger there.

# %%
SCAN_SAMPLES = 2000
W_SCAN = [0.1, 0.2, 0.5, 1.0, 2.0]               # MHz
K_SCAN = [-0.1, -0.2, -0.5, -1.0, -2.0, -5.0]    # MHz


def positive_totals(kerr_mhz, rms_mhz, samples, seed=0):
    """-> {probe: (mean, standard error)} of the positive-frequency weight total over patterns."""
    rng = np.random.default_rng(seed)
    totals = {name: np.zeros(samples) for name in PROBES}
    for r in range(samples):
        unit = disorder_pattern(rng, STORAGE_MODES, 1.0)
        model = build_model(rms_mhz * unit, kerr_mhz=kerr_mhz)
        for name, (mode, kind) in PROBES.items():
            totals[name][r] = probe_sticks(model, unit, mode, kind)[1].sum()
    return {name: (t.mean(), t.std(ddof=1) / np.sqrt(samples)) for name, t in totals.items()}


def scan(points):
    return [positive_totals(kerr, rms, SCAN_SAMPLES, seed=i) for i, (kerr, rms) in enumerate(points)]


def log_slope(x, y, err, last=3):
    """-> slope of log y against log x over the last ``last`` points, and its error (weighted fit)."""
    x, y, err = np.log(x[-last:]), np.log(y[-last:]), err[-last:] / y[-last:]
    fit, cov = np.polyfit(x, y, 1, w=1 / err, cov="unscaled")
    return fit[0], np.sqrt(cov[0, 0])


w_results = scan([(-0.027, w) for w in W_SCAN])
k_results = scan([(k, 0.035) for k in K_SCAN])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, xs, results, label in ((ax1, np.array(W_SCAN), w_results, "W (MHz)"),
                               (ax2, np.abs(K_SCAN), k_results, "|K| (MHz)")):
    for name in PROBES:
        y = np.array([r[name][0] for r in results])
        e = np.array([r[name][1] for r in results])
        ax.errorbar(xs, y, e, marker="o", label=name)
        slope, error = log_slope(xs, y, e)
        print(f"{label.split()[0]} scan, {name}: slope of the last 3 points {slope:.2f} +- {error:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("positive-frequency weight")
anchor = np.array([r["storage number"][0] for r in w_results])[-1]
ax1.plot(W_SCAN, anchor * (np.array(W_SCAN) / W_SCAN[-1]) ** -1.0, "k--", lw=0.8, label="1/W")
ax1.plot(W_SCAN, anchor * (np.array(W_SCAN) / W_SCAN[-1]) ** -2.0, "k:", lw=0.8, label="1/W^2")
anchor = np.array([r["hub Kerr"][0] for r in k_results])[-1]
ax2.plot(np.abs(K_SCAN), anchor * (np.abs(K_SCAN) / np.abs(K_SCAN[-1])) ** -2.0, "k--", lw=0.8, label="1/K^2")
ax1.legend(fontsize=7)
ax2.legend(fontsize=7)
fig.tight_layout()

# %% [markdown]
# ### Facts quoted in the Q4 pointers (small-$W$ cluster and $W = 0$ lines)

# %%
cluster = []
rng = np.random.default_rng(0)
for _ in range(1500):
    f, w, _ = spectrum(build_model(disorder_pattern(rng, STORAGE_MODES, 0.001)), 1, "number")
    keep = f < 0.008   # the low cluster; the single-particle band starts above 0.01 MHz
    cluster.append((f[keep] / 0.001, w[keep]))
ratio = np.concatenate([c[0] for c in cluster])
weight = np.concatenate([c[1] for c in cluster])
order = np.argsort(ratio)
cdf = np.cumsum(weight[order]) / weight.sum()
quantiles = [ratio[order][np.searchsorted(cdf, q)] for q in (0.05, 0.5, 0.95)]
print("W = 1 kHz, storage number: cluster weight %.4f, weight quantiles of f/W (5, 50, 95 percent):" % (weight.sum() / 1500),
      np.round(quantiles, 2))

for name, s in corner_results["W = 0 (exact)"].items():
    strong = s["w"] > 0.05 * s["w"].max()
    print(f"W = 0, {name}: lines above 5% of the strongest (kHz):", np.round(s["f"][strong] * 1e3, 1))
print("g = %.0f kHz: 2 g = %.0f kHz, 4 g = %.0f kHz" % (COUPLING_MHZ * 1e3, 2e3 * COUPLING_MHZ, 4e3 * COUPLING_MHZ))


# %% [markdown]
# ## Q6. What does the single-mode storage probe measure?
#
# The perturbation that $W$ controls is $V = \sum_s \epsilon_s n_s$, with the unit-RMS pattern $\epsilon$ that also sets $H$.
# Exact: the spectrum of $V$ for each pattern, averaged over patterns ("exact").
# Proxy: if the pattern and $H$ were independent, the pattern covariance $\tfrac{4}{3}(I - J/4)$ would give
# $\tfrac{4}{3}\left(4 S_\mathrm{storage} - S_\mathrm{hub}/4\right)$, with $S_\mathrm{storage}$ the single-mode spectrum and
# $S_\mathrm{hub}$ the hub-number spectrum, because $\sum_s n_s = N - n_0$ ("proxy").
# Left: both curves (weight / MHz). Right: (exact − proxy) / proxy, with a one-standard-error band
# from the paired difference; it is cut at $\pm 1$. The proxy is valid for the sticks at $\Delta > 0$ only: the
# constant $N$ in $\sum_s n_s = N - n_0$ matters for DC.

# %%
Q6_SAMPLES = 3000
Q6_CORNERS = {name: kw for name, kw in CORNERS.items() if kw[1] > 0}


def disorder_probe_spectra(kerr_mhz, rms_mhz, samples, edges, seed=0):
    """-> dict(exact, proxy, difference) of (mean, sem) binned weight / MHz, and the mean totals.

    The same patterns give both, so the paired difference has its own standard error.
    """
    rng = np.random.default_rng(seed)
    bins = len(edges) - 1
    exact, proxy = np.zeros((samples, bins)), np.zeros((samples, bins))
    dc = np.zeros(samples)
    for r in range(samples):
        unit = disorder_pattern(rng, STORAGE_MODES, 1.0)
        model = build_model(rms_mhz * unit, kerr_mhz=kerr_mhz)
        values = np.asarray(model.fock_basis, dtype=float)[:, 1:] @ unit
        f, w, dc[r] = stick_spectrum(model, values)
        assert abs(dc[r] + 2 * w.sum() - np.mean(values**2)) < 1e-9   # the sum rule of C3
        exact[r] = np.histogram(f, edges, weights=w)[0] / np.diff(edges)
        hub = spectrum(model, 0, "number")
        stor = [spectrum(model, m, "number") for m in range(1, STORAGE_MODES + 1)]
        # Weighted histograms of the same sticks, combined with the covariance weights.
        h_hub = np.histogram(hub[0], edges, weights=hub[1])[0] / np.diff(edges)
        h_stor = np.mean([np.histogram(f, edges, weights=w)[0] for f, w, _ in stor], axis=0) / np.diff(edges)
        proxy[r] = 4 / 3 * (4 * h_stor - h_hub / 4)
    diff = exact - proxy
    sem = lambda x: x.std(0, ddof=1) / np.sqrt(samples)
    return dict(exact=(exact.mean(0), sem(exact)), proxy=(proxy.mean(0), sem(proxy)),
                difference=(diff.mean(0), sem(diff)), dc=dc.mean(0), edges=edges)


def plot_disorder_probe(corners, samples, edges):
    centers = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(len(corners), 2, figsize=(9, 2.4 * len(corners)), sharex=True, squeeze=False)
    results = {}
    for row, (label, (kerr, rms)) in enumerate(corners.items()):
        results[label] = res = disorder_probe_spectra(kerr, rms, samples, edges, seed=row)
        ax, ax_ratio = axes[row]
        for name, color in (("exact", "C0"), ("proxy", "C1")):
            mean, err = res[name]
            ax.stairs(np.maximum(mean, 1e-300), edges, color=color, label=name)
            ax.fill_between(centers, np.maximum(mean - err, 1e-300), mean + err, color=color, alpha=0.25)
        top = max(res["exact"][0].max(), res["proxy"][0].max())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(top * 1e-6, top * 3)
        ax.set_ylabel(f"{label}\nK={kerr * 1e3:.0f}, W={rms * 1e3:.0f} kHz\nweight / MHz", fontsize=8)
        mean_d, err_d = res["difference"]
        ratio = np.divide(mean_d, res["proxy"][0], out=np.full_like(mean_d, np.nan), where=res["proxy"][0] > 0)
        band = np.divide(err_d, res["proxy"][0], out=np.full_like(mean_d, np.nan), where=res["proxy"][0] > 0)
        ax_ratio.plot(centers, ratio, ".-")
        ax_ratio.fill_between(centers, ratio - band, ratio + band, alpha=0.3)
        ax_ratio.axhline(0, color="gray", lw=0.8)
        ax_ratio.set_ylim(-1, 1)
        total = [(res[n][0] * np.diff(edges)).sum() for n in ("exact", "proxy")]
        ax.set_title(f"f>0 total: exact {total[0]:.3f}, proxy {total[1]:.3f}; DC (exact) {res['dc']:.3f}",
                     fontsize=7)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title("(exact - proxy) / proxy", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("E_i - E_j (MHz)")
    fig.tight_layout()
    return fig, results


fig, q6_results = plot_disorder_probe(Q6_CORNERS, Q6_SAMPLES, CORNER_EDGES)
