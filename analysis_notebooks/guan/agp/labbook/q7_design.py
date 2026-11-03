"""Q7: design rules. (a) the smallest usable mu per W; (b) does a longer window help? (measured traces).

(a) On the Q3 open model + noise at the measured C1_err: the envelope + tail estimate over DRAWS
    draws. mu_min = the smallest mu with |median/target - 1| + half the 16-84 % spread < 30 %, for it
    and every larger mu. Also with all damping rates / 2 and / 4, and with 16x the shots.
(b) The envelope + tail estimate of the measured traces cut at shorter windows (Kerr: 245 us of
    data; S7 coarse: 244 us), against the closed-system target.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.agp import agp_kernel_weights, fit_exponential_background
from load_scans import VAULT, load_scans
from open_models import realistic
from theory.qsim.agp import spectral_agp

TOLERANCE = 0.3


def envelope_tail(times, data, error, mu_khz, g):
    """-> (estimate, shot error) of g^2 ||A_mu||^2: e^{t/tau} compensation plus the analytic tail."""
    tau = fit_exponential_background(times, data, error, 30.)["lifetime_us"]
    level = np.mean((data * np.exp(times / tau))[times >= 0.9 * times[-1]])
    scale, a = (2 * np.pi * g)**2, 2 * np.pi * mu_khz / 1e3
    weights = agp_kernel_weights(times, mu_khz / 1e3, tau)
    value = scale * (weights @ data - level * times[-1] * np.exp(-a * times[-1]) / (2 * a))
    return value, scale * np.sqrt(weights**2 @ error**2), tau


scans = load_scans()
MU_FINE = np.array([2., 3., 4., 5., 6., 8., 10., 12., 15., 20., 25., 30., 40., 50.])
DRAWS = 30
RNG = np.random.default_rng(3)
VARIANTS = {"now": ("loss + M1 deph 0.0106", 1.), "rates / 2": ("rates / 2", 1.),
            "rates / 4": ("rates / 4", 1.), "16x shots": ("loss + M1 deph 0.0106", .25)}
table = {}
for name, scan in scans.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=[10.])
    times = scan.data["time_us"]
    for row, part in enumerate(scan.children):
        g = float(part.data["coupling_mhz"])
        error = scan.data["C1_err"][row]
        mins = {}
        for label, (damping, noise_factor) in VARIANTS.items():
            clean, (diff, w) = realistic(name, part, damping)
            target = spectral_agp(diff, w, MU_FINE / 1e3, g)
            draws = np.array([[envelope_tail(times, clean + RNG.normal(0, noise_factor * error),
                                             noise_factor * error, mu, g)[0] for mu in MU_FINE]
                              for _ in range(DRAWS)]) / target
            lo, mid, hi = np.percentile(draws, [16, 50, 84], axis=0)
            ok = np.abs(mid - 1) + (hi - lo) / 2 < TOLERANCE
            bad = np.where(~ok)[0]
            mins[label] = MU_FINE[0] if not len(bad) else (MU_FINE[bad.max() + 1] if bad.max() + 1 < len(MU_FINE) else np.nan)
        table[name, part.disorder_khz] = mins

print("mu_min (kHz), open model + noise, envelope + tail: |bias| + half 16-84 spread < 30 % for this and all larger mu")
print(f"{'':18s}" + "".join(f"{label:>12s}" for label in VARIANTS))
for (name, W), mins in table.items():
    print(f"{name:10s} W={W:4.0f} " + "".join(f"{mins[label]:12.0f}" for label in VARIANTS))
np.save("cache/q7_mu_min.npy", table, allow_pickle=True)

# (b) window cuts on the measured traces
cuts = {"Kerr": [60., 90., 120., 180., 245.], "S7 coarse": [60., 90., 120., 180., 244.]}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, (name, windows) in zip(axes, cuts.items()):
    scan = scans[name]
    times = scan.data["time_us"]
    print(f"(b) {name}: envelope + tail / closed target, mu = 10 kHz, window cut at {windows} us")
    for row, part in enumerate(scan.children):
        g = float(part.data["coupling_mhz"])
        diff, w = part.theory_spectrum()
        target = spectral_agp(diff, w, 10. / 1e3, g)
        ratios, errors = [], []
        for window in windows:
            keep = times <= window + 1e-9
            value, noise, _ = envelope_tail(times[keep], scan.data["C1_corrected"][row][keep],
                                            scan.data["C1_err"][row][keep], 10., g)
            ratios.append(value / target)
            errors.append(noise / target)
        print(f"   W={part.disorder_khz:4.0f}: " + " ".join(f"{r:5.2f}+/-{e:4.2f}" for r, e in zip(ratios, errors)))
        ax.errorbar(windows, ratios, errors, fmt="o-", ms=3, capsize=2, label=f"W = {part.disorder_khz:g}")
    ax.axhline(1, color="k", lw=.8)
    ax.set(xlabel="window cut (us)", ylabel="estimate / closed target", title=f"(b) {name}, mu = 10 kHz",
           ylim=(-1, 3))
    ax.legend(fontsize=6, ncol=2)
fig.suptitle("Q7 [measured]: the envelope + tail estimate against the window length")
fig.savefig(VAULT / "q7_window.png", dpi=110)
