"""Q6: what does the AGP at the resolvable mu tell? Closed-system model on a dense W grid.

(a) spectral function S(omega) (weights binned) at a few W with the AGP filters for mu = 5, 10, 20 kHz
(b) g^2 ||A_mu||^2 against W (1 kHz steps) for mu = 2, 5, 10, 20 kHz, both probes, with the measured
    best estimate (envelope + tail) at mu = 10, 20 kHz: shot errors and the Q5 systematic
    (0.13 x the decaying-DC term); filled where the total error is below 30 % of the value
(c) localization measures of the same model: mean Fock participation ratio of the eigenstates / D,
    and the DC fraction of C_1 (sum_i V_ii^2 / sum V^2: the eigenstate memory of the probe)
(d) the AGP shapes against W, normalized, next to the participation ratio
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from load_scans import VAULT, load_scans
from theory.qsim.agp import probe_values, spectral_agp, spectral_weights, split_dc, window_agp
from theory.qsim.central_boson import fixed_n_hamiltonian

DENSE = np.arange(0, 200.1, 1.)
MU = np.array([2., 5., 10., 20.])
SYSTEMATIC = 0.13  # Q5: fraction of the decaying-DC term left after the background subtraction

scans = load_scans()
for scan in scans.values():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=list(MU))
ref = scans["S7 fine"].children[1]
pattern = scans["S7 fine"].disorder_pattern()
couplings, kerr = ref.recorded_couplings_mhz(), ref.recorded_kerr_mhz()
g = float(np.mean(couplings))

theory = {"S7": [], "Kerr": []}
locality = dict(pr=[], dc_s7=[], dc_kerr=[], spacing=[])
for W in DENSE:
    model = fixed_n_hamiltonian(3, 5, 1e-3 * W * pattern, couplings, kerr)
    for name, (mode, probe) in {"S7": (4, "number"), "Kerr": (0, "kerr")}.items():
        values = probe_values(model.fock_basis, mode, probe)
        diff, w = spectral_weights(model, values)
        theory[name].append(spectral_agp(diff, w, MU / 1e3, g))
        dc, _ = split_dc(diff, w)
        locality["dc_s7" if name == "S7" else "dc_kerr"].append(dc / w.sum())
    pr = 1 / np.sum(np.abs(model.states)**4, axis=0)
    locality["pr"].append(pr.mean() / len(pr))
    locality["spacing"].append(1e3 * np.mean(np.diff(np.sort(model.energies_MHz))))
theory = {k: np.array(v) for k, v in theory.items()}
locality = {k: np.array(v) for k, v in locality.items()}
pick = [0, 30, 60, 90, 120, 150]
print("mean level spacing (kHz) at W =", pick, locality["spacing"][pick].round(1))
print("participation ratio / D:", locality["pr"][pick].round(3))
print("DC fraction S7:", locality["dc_s7"][pick].round(2), " Kerr:", locality["dc_kerr"][pick].round(2))

# The measured best estimate with its error budget.
measured = {}
for name, scan in scans.items():
    times = scan.data["time_us"]
    rows = []
    for row, part in enumerate(scan.children):
        data, error = scan.data["C1_corrected"][row], scan.data["C1_err"][row]
        tau = fit_exponential_background(times, data, error, 30.)["lifetime_us"]
        compensated = data * np.exp(times / tau)
        level = np.mean(compensated[times >= 0.9 * times[-1]])
        scale = (2 * np.pi * g)**2
        diff, w = part.theory_spectrum()
        dc, _ = split_dc(diff, w)
        for mu in MU:
            a = 2 * np.pi * mu / 1e3
            weights = agp_kernel_weights(times, mu / 1e3, tau)
            value = scale * weights @ data - scale * level * times[-1] * np.exp(-a * times[-1]) / (2 * a)
            noise = scale * np.sqrt(weights**2 @ error**2)
            dc_term = window_agp(np.zeros(1), np.array([dc]), mu / 1e3, g, times[-1], 1 / tau)
            rows.append((part.disorder_khz, mu, value, noise, SYSTEMATIC * dc_term))
    measured[name] = np.array(rows)

fig = plt.figure(figsize=(16, 10), constrained_layout=True)
grid = fig.add_gridspec(2, 3)
ax = fig.add_subplot(grid[0, 0])
bins = np.linspace(0, 400, 201)
for W, color in zip([0, 40, 90, 150], ["C0", "C1", "C2", "C3"]):
    model = fixed_n_hamiltonian(3, 5, 1e-3 * W * pattern, couplings, kerr)
    diff, w = spectral_weights(model, probe_values(model.fock_basis, 4))
    off = np.abs(diff.ravel()) * 1e3 > 1e-6
    hist, _ = np.histogram(np.abs(diff.ravel()[off]) * 1e3, bins, weights=w.ravel()[off])
    ax.step(bins[:-1], np.maximum(hist, 1e-6), where="post", color=color, lw=1, label=f"W = {W} kHz")
ax2 = ax.twinx()
for mu, ls in zip([5, 10, 20], [":", "--", "-"]):
    f = bins**2 / (bins**2 + mu**2)**2
    ax2.plot(bins, f / f.max(), "k", ls=ls, lw=.8, label=f"filter mu = {mu} kHz")
ax.set(xlabel="|omega| / 2 pi (kHz)", ylabel="S7 spectral weight per 2 kHz bin", yscale="log",
       ylim=(1e-4, 1), title="(a) S7 spectral function, AGP filters")
ax.legend(fontsize=7, loc="upper right")
ax2.legend(fontsize=7, loc="center right")
ax2.set_ylabel("filter (normalized)")
for c, name in enumerate(["S7", "Kerr"]):
    ax = fig.add_subplot(grid[0, 1 + c])
    for k, mu in enumerate(MU):
        ax.plot(DENSE, theory[name][:, k], color=f"C{k}", lw=1, label=f"closed theory, mu = {mu:g} kHz")
    sets = [("S7 fine", "o"), ("S7 coarse", "s")] if name == "S7" else [("Kerr", "o")]
    for scan_name, marker in sets:
        rows = measured[scan_name]
        for k in (2, 3):
            sel = rows[:, 1] == MU[k]
            W, value, total = rows[sel, 0], rows[sel, 2], np.hypot(rows[sel, 3], rows[sel, 4])
            good = total < 0.3 * np.abs(value)
            ax.errorbar(W[good], value[good], total[good], fmt=marker, color=f"C{k}", ms=5, capsize=2)
            ax.errorbar(W[~good], np.abs(value[~good]), total[~good], fmt=marker, mfc="none",
                        color=f"C{k}", ms=5, capsize=2, alpha=.6)
    ax.set(yscale="log", ylim=(1e-4, 3), xlabel="W (kHz)", ylabel=r"$g^2\|A_\mu\|^2$",
           title=f"(b) {name}: theory (lines), measured (filled: error < 30 %)")
    ax.legend(fontsize=7)
ax = fig.add_subplot(grid[1, 0])
ax.plot(DENSE, locality["pr"], "k", label="mean Fock participation ratio / D")
ax.plot(DENSE, locality["dc_s7"], "C0", label="DC fraction of C_1, S7 (model)")
ax.plot(DENSE, locality["dc_kerr"], "C1", label="DC fraction of C_1, Kerr (model)")
for scan_name, marker in [("S7 fine", "o"), ("S7 coarse", "s"), ("Kerr", "^")]:
    data = scans[scan_name].data
    amps = [fit_exponential_background(data["time_us"], data["C1_corrected"][r], data["C1_err"][r], 30.)["amplitude"]
            / data["C1_corrected"][r][0] for r in range(len(data["disorder_khz"]))]
    ax.plot(data["disorder_khz"], amps, marker, color="C1" if scan_name == "Kerr" else "C0", mfc="none",
            label=f"measured A / C_1(0), {scan_name}")
ax.set(xlabel="W (kHz)", title="(c) localization measures", ylim=(0, 1.05))
ax.legend(fontsize=7)
for c, name in enumerate(["S7", "Kerr"]):
    ax = fig.add_subplot(grid[1, 1 + c])
    for k, mu in enumerate(MU):
        ax.plot(DENSE, theory[name][:, k] / theory[name][:, k].max(), color=f"C{k}", lw=1, label=f"mu = {mu:g} kHz")
    ax.plot(DENSE, locality["pr"] / locality["pr"].max(), "k--", lw=1, label="participation ratio (scaled)")
    ax.set(xlabel="W (kHz)", ylabel="normalized to its maximum", title=f"(d) {name}: AGP shape vs localization")
    ax.legend(fontsize=7)
fig.suptitle("Q6 [model; measured points in (b), (c)]: what the AGP at each mu sees")
fig.savefig(VAULT / "q6_physics.png", dpi=110)

for name, rows in measured.items():
    print(name)
    for W in np.unique(rows[:, 0]):
        sel = rows[:, 0] == W
        print(f"  W={W:5.0f} " + "  ".join(f"mu={mu:g}: {v:.2e} +/- {n:.1e} shot +/- {s:.1e} syst"
                                          for mu, v, n, s in rows[sel][2:, 1:]))
