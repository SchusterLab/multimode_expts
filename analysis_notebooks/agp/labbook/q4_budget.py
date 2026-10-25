"""Q4: what is the plain kernel integral made of? Budget per (scan, W, mu).

target      closed system, infinite time (spectral_agp)
window      closed system, same window, exact (window_agp)
open model  loss + M1 dephasing 0.0106 (+ the fitted static detunings at W = 0), on the grid
DC decay    the closed-system DC weight A times e^{-t/tau}, tau the envelope fitted to the data
measured    the plain kernel integral of the measured C_1; noise = its shot standard error
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from load_scans import VAULT, load_scans
from open_models import part_model
from theory.qsim.agp import spectral_agp, split_dc, window_agp

MU_KHZ = np.array([2., 5., 10., 20., 40.])
# q3b_w0.py fits (kHz, S4..S7): the static residual detunings of the W = 0 parts.
W0_DELTA_KHZ = {"S7 fine": [7.16, 2.13, 2.13, 2.56], "S7 coarse": [0.36, -2.58, -2.58, -1.61]}
LOSS_DEPH = "loss + M1 deph 0.0106"

scans = load_scans()
budget = {}
for name, scan in scans.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=list(MU_KHZ))
    times = scan.data["time_us"]
    window_us = times[-1]
    weights = np.array([agp_kernel_weights(times, m / 1e3) for m in MU_KHZ])
    rows = []
    for row, part in enumerate(scan.children):
        g = float(part.data["coupling_mhz"])
        scale = (2 * np.pi * g)**2
        diff, w = part.theory_spectrum()
        dc, _ = split_dc(diff, w)
        data, error = scan.data["C1_corrected"][row], scan.data["C1_err"][row]
        tau = fit_exponential_background(times, data, error, 30.)["lifetime_us"]
        entry = dict(W=part.disorder_khz, dc=dc, tau=tau,
                     target=spectral_agp(diff, w, MU_KHZ / 1e3, g),
                     window=window_agp(diff, w, MU_KHZ / 1e3, g, window_us),
                     measured=scale * weights @ data,
                     noise=scale * np.sqrt(weights**2 @ error**2),
                     dc_decay=window_agp(np.zeros(1), np.array([dc]), MU_KHZ / 1e3, g, window_us, 1 / tau),
                     dc_static=window_agp(np.zeros(1), np.array([dc]), MU_KHZ / 1e3, g, window_us))
        if part.disorder_khz == 0 and name in W0_DELTA_KHZ:
            delta = 1e-3 * np.asarray(W0_DELTA_KHZ[name])
            d2, w2 = part.theory_spectrum(detunings=delta)
            entry["target_delta"] = spectral_agp(d2, w2, MU_KHZ / 1e3, g)
            entry["dc_delta"] = split_dc(d2, w2)[0]
            from theory.qsim.agp import probe_values  # noqa: local, one-off
            from open_models import LOSS, MODEL
            from theory.qsim.open_system import sector_correlation
            from theory.qsim.central_boson import fixed_n_hamiltonian
            sector = fixed_n_hamiltonian(3, 5, delta, part.recorded_couplings_mhz(), part.recorded_kerr_mhz()).fock_basis
            c_open = sector_correlation(MODEL, sector, lambda b: probe_values(b, part.mode_index), times,
                                        part.recorded_couplings_mhz(), delta, part.recorded_kerr_mhz(),
                                        LOSS, np.r_[0.0106, np.zeros(4)])[0]
        else:
            c_open, _ = part_model(part, LOSS_DEPH)
        entry["open"] = scale * weights @ c_open
        rows.append(entry)
    budget[name] = rows

np.save(VAULT.parent / "labbook_figures" / "q4_budget.npy", budget, allow_pickle=True) if False else None
for name, rows in budget.items():
    print(f"== {name}  (mu = {MU_KHZ.tolist()} kHz)")
    for e in rows:
        print(f" W={e['W']:5.0f} DC={e['dc']:.2f} tau={e['tau']:4.0f}us")
        for key in ("target", "window", "open", "measured", "dc_decay", "noise") + (("target_delta",) if "target_delta" in e else ()):
            print(f"    {key:12s}" + " ".join(f"{v:9.2e}" for v in e[key]))
        print("    meas/target " + " ".join(f"{v:9.2f}" for v in e["measured"] / e["target"]))
        print("    meas/open   " + " ".join(f"{v:9.2f}" for v in e["measured"] / e["open"]))
        print("    noise/targ  " + " ".join(f"{v:9.2f}" for v in e["noise"] / e["target"]))
        print("    dcdec/targ  " + " ".join(f"{v:9.2f}" for v in e["dc_decay"] / e["target"]))
        if "dc_delta" in e:
            print(f"    W=0 with fitted static delta: DC {e['dc']:.2f} -> {e['dc_delta']:.2f}; target ratio "
                  + " ".join(f"{v:7.2f}" for v in e["target_delta"] / e["target"]))

fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
for r, (name, rows) in enumerate(budget.items()):
    W = np.array([e["W"] for e in rows])
    for c, k in enumerate([1, 2, 3, 4]):
        ax = axes[r, c]
        get = lambda key: np.array([e[key][k] for e in rows])
        ax.plot(W, get("target"), "k-", lw=2, label="target: closed, infinite time")
        ax.plot(W, get("window"), "k:", label="closed, same window")
        ax.plot(W, get("open"), "C1-s", ms=4, label="open model, same window")
        ax.errorbar(W, get("measured"), 2 * get("noise"), fmt="C0o", ms=5, capsize=2, label="measured plain (2 sigma shot)")
        ax.plot(W, get("dc_decay"), "C2--", label="decaying DC alone")
        if r < 2:
            ax.plot([0], [rows[0]["target_delta"][k]], "k*", ms=12, label="target with W=0 residual delta")
        ax.set(yscale="log", title=f"{name}, mu = {MU_KHZ[k]:g} kHz", xlabel="W (kHz)")
    axes[r, 0].set_ylabel(r"$g^2\|A_\mu\|^2$")
axes[0, 0].legend(fontsize=7)
fig.suptitle("Q4 [model + measured]: budget of the plain kernel integral")
fig.savefig(VAULT / "q4_budget.png", dpi=110)
