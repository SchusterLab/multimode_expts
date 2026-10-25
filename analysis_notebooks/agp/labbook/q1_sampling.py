"""Q1: is the time grid fine enough? Aliasing of the AGP kernel integral (closed-system model only).

For each scan and W: the exact window integral (no grid), the same kernel on the measured grid
(piecewise-linear samples, what analyze() does), and the infinite-time AGP. Also the spectral
weight above the grid's Nyquist frequency.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.agp import agp_kernel_weights
from load_scans import VAULT, load_scans
from theory.qsim.agp import spectral_agp, split_dc, two_time_correlation, window_agp

MU_KHZ = [5., 10., 20., 40.]

scans = load_scans()
rows = []
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for ax, (name, scan) in zip(axes, scans.items()):
    times = np.asarray(scan.children[0].data.get("time_us", None) if scan.children[0].data else [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan.analyze(mu=MU_KHZ)
    times = scan.data["time_us"]
    dt, window = times[1] - times[0], times[-1]
    nyquist_khz = 1e3 / (2 * dt)
    for part in scan.children:
        diff, w = part.theory_spectrum()
        g = float(part.data["coupling_mhz"])
        dc, mask = split_dc(diff, w)
        above = w[np.abs(diff) * 1e3 > nyquist_khz].sum() / (w.sum() - dc)
        exact = window_agp(diff, w, np.array(MU_KHZ) / 1e3, g, window)
        theory = two_time_correlation(diff, w, times)
        grid = np.array([(2 * np.pi * g)**2 * agp_kernel_weights(times, m / 1e3) @ theory for m in MU_KHZ])
        infinite = spectral_agp(diff, w, np.array(MU_KHZ) / 1e3, g)
        rows.append((name, part.disorder_khz, nyquist_khz, above, exact, grid, infinite,
                     np.abs(diff[~mask]).max() * 1e3))
        ax.plot(part.disorder_khz, grid[1] / exact[1] - 1, "o", color="C0")
        ax.plot(part.disorder_khz, grid[0] / exact[0] - 1, "s", color="C1")
    ax.set(title=f"{name}: dt = {dt:.2f} us, Nyquist {nyquist_khz:.0f} kHz", xlabel="W (kHz)",
           ylabel="grid / exact - 1")
    ax.axhline(0, color="k", lw=.8)
axes[0].legend(["mu = 10 kHz", "mu = 5 kHz"])
fig.suptitle("Q1 [model]: sampling error of the kernel integral (closed system, same window)")
VAULT.mkdir(exist_ok=True)
fig.savefig(VAULT / "q1_sampling.png", dpi=120)

print("scan, W, Nyquist kHz, weight above Nyquist (of non-DC), max |freq| kHz")
print("   per mu", MU_KHZ, ": grid/exact - 1 ; exact window / infinite")
for name, strength, nyq, above, exact, grid, infinite, fmax in rows:
    print(f"{name:10s} W={strength:5.0f} nyq={nyq:6.0f} above={above:.2e} fmax={fmax:6.0f}")
    print("     grid/exact-1", np.array2string(grid / exact - 1, precision=4),
          " window/inf", np.array2string(exact / infinite, precision=3))
