"""Q3: what makes C_1 decay? Measured C_1 against the open-system model, no free parameters.

Per part: chi2/dof of each damping model (C1_err only), and the envelope 1/rate of A e^{-rate t}
fitted for t >= 30 us, to the data and to each model trace in the same way.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.agp import fit_exponential_background
from load_scans import VAULT, load_scans
from open_models import DAMPING, part_model

START_US = 30.
scans = load_scans()
summary = {}
for name, scan in scans.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    times = scan.data["time_us"]
    rows = []
    for row, part in enumerate(scan.children):
        data, error = scan.data["C1_corrected"][row], scan.data["C1_err"][row]
        fit = fit_exponential_background(times, data, error, START_US)
        entry = dict(W=part.disorder_khz, data_life=fit["lifetime_us"],
                     data_life_err=np.sqrt(fit["covariance"][1, 1]) / fit["rate_per_us"]**2)
        for hamiltonian in ("recorded", "dressed"):
            for damping in DAMPING:
                c, _ = part_model(part, damping, hamiltonian)
                entry[hamiltonian, damping, "chi2"] = np.mean(((c - data) / error)**2)
                if damping != "closed":
                    entry[hamiltonian, damping, "life"] = fit_exponential_background(
                        times, c, error, START_US)["lifetime_us"]
        rows.append(entry)
    summary[name] = rows
    print(name)
    for e in rows:
        print(f"  W={e['W']:5.0f} data 1/rate {e['data_life']:5.0f}+/-{e['data_life_err']:3.0f} | " +
              " | ".join(f"{d[:14]:14s} chi2 {e['recorded', d, 'chi2']:6.1f}/{e['dressed', d, 'chi2']:6.1f}"
                         + (f" life {e['recorded', d, 'life']:4.0f}/{e['dressed', d, 'life']:4.0f}" if d != "closed" else "")
                         for d in DAMPING))

colors = dict(zip(list(DAMPING)[1:], ["C1", "C2", "C3"]))
fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
for column, (name, rows) in enumerate(summary.items()):
    W = np.array([e["W"] for e in rows])
    ax = axes[0, column]
    ax.errorbar(W, [e["data_life"] for e in rows], [e["data_life_err"] for e in rows], fmt="ko",
                capsize=3, label="measured")
    for damping, color in colors.items():
        ax.plot(W, [e["recorded", damping, "life"] for e in rows], "-", color=color, label=f"{damping} (recorded H)")
        ax.plot(W, [e["dressed", damping, "life"] for e in rows], "--", color=color, label="dressed H")
    ax.set(title=f"{name}: envelope 1/rate", xlabel="W (kHz)", ylabel="1/rate (us)")
    ax = axes[1, column]
    for damping in DAMPING:
        ax.plot(W, [e["recorded", damping, "chi2"] for e in rows], "o-", color=colors.get(damping, "k"), label=damping)
        ax.plot(W, [e["dressed", damping, "chi2"] for e in rows], "s--", color=colors.get(damping, "k"))
    ax.set(yscale="log", xlabel="W (kHz)", ylabel="chi2 / point (C1_err only)",
           title=f"{name}: chi2 (solid recorded H, dashed dressed)")
axes[0, 0].legend(fontsize=6)
axes[1, 0].legend(fontsize=6)
fig.suptitle("Q3 [model vs measured]: the decay of C_1, no free parameters (envelope: A exp(-rate t) fitted for t >= 30 us)")
fig.savefig(VAULT / "q3_envelope_chi2.png", dpi=120)
