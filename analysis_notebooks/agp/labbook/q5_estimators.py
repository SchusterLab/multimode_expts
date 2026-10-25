"""Q5: which estimator gets the closed-system AGP back? Tested on the Q3 open model (not a toy).

Data = open-model C_1 on the measured grid + Gaussian noise at the measured C1_err (DRAWS draws).
Target = closed-system infinite-time AGP of the same Hamiltonian. Estimators:
  plain        kernel integral over the window
  exp fit      subtract A e^{-t/tau} fitted for t >= 30 us, then integrate
  envelope     multiply by e^{t/tau} (tau fitted as above), integrate (= analyze(t1_us=tau))
  envelope+tail  as envelope, plus the analytic tail beyond T of the late-time level (mean of the last 10 %)
Output: median(estimate/target) and the 16-84 % spread, per scan, W, mu. Then the same estimators
on the measured traces, against the closed-system target (recorded H; W = 0 with the residual delta).
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from load_scans import VAULT, load_scans
from open_models import realistic
from theory.qsim.agp import spectral_agp

MU_KHZ = np.array([2., 5., 10., 20., 40.])
DRAWS = 60
START_US = 30.
METHODS = ["plain", "exp fit", "envelope", "envelope+tail"]
RNG = np.random.default_rng(7)


def estimate(times, trace, error, g_mhz):
    """-> {method: array over MU_KHZ} of g^2 ||A_mu||^2 estimates for one trace."""
    scale = (2 * np.pi * g_mhz)**2
    fit = fit_exponential_background(times, trace, error, START_US)
    tau = fit["lifetime_us"]
    background = fit["amplitude"] * np.exp(-times / tau)
    compensated = trace * np.exp(times / tau)
    level = np.mean(compensated[times >= 0.9 * times[-1]])
    out = {m: [] for m in METHODS}
    for mu in MU_KHZ:
        plain = agp_kernel_weights(times, mu / 1e3)
        out["plain"].append(scale * plain @ trace)
        out["exp fit"].append(scale * plain @ (trace - background))
        envelope = scale * agp_kernel_weights(times, mu / 1e3, tau) @ trace
        out["envelope"].append(envelope)
        a, window = 2 * np.pi * mu / 1e3, times[-1]
        out["envelope+tail"].append(envelope - scale * level * window * np.exp(-a * window) / (2 * a))
    return {m: np.asarray(v) for m, v in out.items()}


scans = load_scans()
results = {}
for name, scan in scans.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    times = scan.data["time_us"]
    for row, part in enumerate(scan.children):
        g = float(part.data["coupling_mhz"])
        clean, (diff, w) = realistic(name, part)
        target = spectral_agp(diff, w, MU_KHZ / 1e3, g)
        error = scan.data["C1_err"][row]
        draws = {m: [] for m in METHODS}
        for _ in range(DRAWS):
            for m, v in estimate(times, clean + RNG.normal(0, error), error, g).items():
                draws[m].append(v / target)
        measured = estimate(times, scan.data["C1_corrected"][row], error, g)
        results[name, part.disorder_khz] = dict(
            target=target,
            model={m: np.percentile(np.array(v), [16, 50, 84], axis=0) for m, v in draws.items()},
            measured={m: v / target for m, v in measured.items()})

for name in scans:
    print(f"== {name}: median estimate/target on the open model [16-84 %], then measured/target")
    for (n, W), r in results.items():
        if n != name:
            continue
        print(f"  W={W:5.0f} target " + " ".join(f"{v:8.2e}" for v in r["target"]))
        for m in METHODS:
            lo, mid, hi = r["model"][m]
            print(f"    {m:14s} model " + " ".join(f"{b:6.2f}[{a:5.2f},{c:5.2f}]" for a, b, c in zip(lo, mid, hi)))
            print(f"    {'':14s} meas  " + " ".join(f"{v:6.2f}{'':14s}" for v in r["measured"][m]))

colors = dict(zip(METHODS, ["C0", "C1", "C2", "C3"]))
fig, axes = plt.subplots(3, len(MU_KHZ), figsize=(18, 10), constrained_layout=True, sharey=True)
for r_, name in enumerate(scans):
    keys = [k for k in results if k[0] == name]
    W = np.array([k[1] for k in keys])
    for c_, mu in enumerate(MU_KHZ):
        ax = axes[r_, c_]
        for i, m in enumerate(METHODS):
            band = np.array([results[k]["model"][m][:, c_] for k in keys])
            ax.fill_between(W, band[:, 0], band[:, 2], color=colors[m], alpha=.2, lw=0)
            ax.plot(W, band[:, 1], "-", color=colors[m], lw=1, label=f"{m} (model)")
            ax.plot(W + 1.5 * (i - 1.5), [results[k]["measured"][m][c_] for k in keys], "o", ms=4,
                    color=colors[m], label=f"{m} (measured)")
        ax.axhline(1, color="k", lw=.8)
        ax.set(yscale="symlog", ylim=(-1, 100), title=f"{name}, mu = {mu:g} kHz", xlabel="W (kHz)")
    axes[r_, 0].set_ylabel("estimate / closed-system target")
axes[0, 0].legend(fontsize=6, ncol=2)
fig.suptitle("Q5: estimators on the open model (bands: 16-84 % over noise draws) and on the data (dots)")
fig.savefig(VAULT / "q5_estimators.png", dpi=110)
np.save("cache/q5_results.npy", results, allow_pickle=True)
