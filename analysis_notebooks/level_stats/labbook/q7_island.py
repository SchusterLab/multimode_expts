"""Q7: where along the disorder axis the Kerr changes the small-gap fraction (perfect resolution, N = 3).

Paired: the same disorder shapes at every (K, W). Shows P(r < 0.25) against W/g for several K/g, the
Kerr contrast P(K) - P(K=0), and its per-shape spread (what sets the realization count).
Writes figure q7_island.png and cache/q7_island.npz.
Run: pixi run python analysis_notebooks/level_stats/labbook/q7_island.py   (about 3 min)
"""
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CACHE, FIGURES, disorder_shape, gap_ratios, spectrum

DISORDERS = np.array([0.02, 0.05, 0.1, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 2.4, 3.0, 4.0, 5.0, 6.5, 8.0])
KERRS = (0.0, -0.9, -1.8, -4.0)
DRAWS = 1000
R0 = 0.25


def cell(args):
    kerr, w = args
    rng = np.random.default_rng(99)
    return [np.mean(gap_ratios(spectrum(3, kerr, w, disorder_shape(rng))[0]) < R0) for _ in range(DRAWS)]


if __name__ == "__main__":
    jobs = [(k, w) for k in KERRS for w in DISORDERS]
    with ProcessPoolExecutor(30) as pool:
        p = np.array(list(pool.map(cell, jobs))).reshape(len(KERRS), len(DISORDERS), DRAWS)
    np.savez(CACHE / "q7_island.npz", p=p, kerrs=KERRS, disorders=DISORDERS)
    mean = p.mean(axis=2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for i, k in enumerate(KERRS):
        axes[0].semilogx(DISORDERS, mean[i], "o-", ms=3, label=f"K/g = {k:g}")
    axes[0].axhline(2 * (1 - 1 / 1.25), color="k", ls=":", lw=1, label="Poisson")
    axes[0].axhline(0.158, color="k", ls="-.", lw=1, label="GOE (surmise)")
    axes[0].set(xlabel="disorder W/g (RMS onsite)", ylabel="P(r < 0.25), bulk, 1000 shapes",
                title="N = 3, perfect resolution")
    axes[0].legend(fontsize=8)
    for i, k in enumerate(KERRS[1:], start=1):
        d = p[i] - p[0]
        m_needed = (3 * d.std(axis=1) / np.abs(d.mean(axis=1))) ** 2
        axes[1].semilogx(DISORDERS, d.mean(axis=1), "o-", ms=3, label=f"K/g = {k:g}")
        for w, dm, mm in zip(DISORDERS, d.mean(axis=1), m_needed):
            if w in (1.7, 2.4):
                axes[1].annotate(f"{mm:.0f}", (w, dm), textcoords="offset points", xytext=(4, -10), fontsize=7)
        print(f"K/g {k}: contrast " + " ".join(f"{w:g}:{dm:+.3f}" for w, dm in zip(DISORDERS, d.mean(axis=1))))
        print(f"   paired shapes for 3 sigma: " + " ".join(f"{w:g}:{mm:.0f}" for w, mm in zip(DISORDERS, m_needed)))
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xlabel="disorder W/g", ylabel="P(r<0.25; K) - P(r<0.25; K=0), same shapes",
                title="Kerr contrast against the linear system (numbers: shapes for 3 sigma)")
    axes[1].legend(fontsize=8)
    fig.savefig(FIGURES / "q7_island.png", dpi=130)
