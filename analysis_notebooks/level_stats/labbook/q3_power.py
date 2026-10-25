"""Q3: how many disorder realizations a frozen test needs to tell two points apart.

Perfect resolution (the model's levels): a lower bound on what any measurement needs.
Test statistic, fixed before any data: the pooled P(r < 0.25) of the bulk ratios (one
number per point, realizations are the independent units). Two forms:

- two-point: both points measured, M realizations each, z = difference / its sd;
- one-point: one point measured, compared with the model's prediction at the other.

Writes cache/q3_power.npz and figure q3_power.png.
Run: pixi run python analysis_notebooks/level_stats/labbook/q3_power.py   (about 2 min)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CACHE, FIGURES, disorder_shape, gap_ratios, spectrum

R0 = 0.25
DRAWS = 3000
REFERENCE = (-0.13, 1.7)  # the Sep 10 point (K/g, W/g): low Kerr
TARGETS = [(-0.9, 1.7), (-1.8, 1.7), (-0.9, 3.0), (-1.8, 3.0), (-2.5, 2.4)]
M_GRID = np.arange(3, 61)


def per_realization(n, kerr, disorder, seed):
    """-> (#{r < R0}, #r) per realization."""
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(DRAWS):
        r = gap_ratios(spectrum(n, kerr, disorder, disorder_shape(rng))[0])
        counts.append((np.sum(r < R0), len(r)))
    return np.array(counts, float)


def z_curves(a, b):
    """Expected z of the pooled-fraction difference with M realizations at each point
    (two-point) and with M at b against the exact model value at a (one-point)."""
    pa, pb = a[:, 0].sum() / a[:, 1].sum(), b[:, 0].sum() / b[:, 1].sum()
    # variance of a ratio estimator per realization (realizations are the units)
    va = np.var(a[:, 0] - pa * a[:, 1]) / np.mean(a[:, 1]) ** 2
    vb = np.var(b[:, 0] - pb * b[:, 1]) / np.mean(b[:, 1]) ** 2
    two = abs(pb - pa) / np.sqrt((va + vb) / M_GRID)
    one = abs(pb - pa) / np.sqrt(vb / M_GRID)
    return pa, pb, two, one


if __name__ == "__main__":
    out, fig = {}, plt.figure(figsize=(12, 4.5), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    for n, ax in zip((3, 4), axes):
        ref = per_realization(n, *REFERENCE, seed=7)
        for target in TARGETS:
            tgt = per_realization(n, *target, seed=7)
            pa, pb, two, one = z_curves(ref, tgt)
            m3 = M_GRID[np.argmax(two >= 3)] if np.any(two >= 3) else np.inf
            m3_one = M_GRID[np.argmax(one >= 3)] if np.any(one >= 3) else np.inf
            print(f"N={n} ref {REFERENCE} P={pa:.3f}  target {target} P={pb:.3f}: "
                  f"z at M=9: two-point {two[M_GRID == 9][0]:.2f}, one-point {one[M_GRID == 9][0]:.2f}; "
                  f"M for z=3: two-point {m3}, one-point {m3_one}")
            out[f"N{n}_{target}"] = np.array([pa, pb, m3, m3_one])
            ax.plot(M_GRID, two, label=f"K/g {target[0]}, W/g {target[1]}  ({pa:.2f} vs {pb:.2f})")
        ax.axhline(3, color="k", lw=0.8, ls=":")
        ax.set(xlabel="realizations M per point", title=f"N = {n}: z of the P(r<0.25) difference vs the Sep 10 point",
               ylim=(0, 8))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("expected z (two-point test, perfect resolution)")
    fig.savefig(FIGURES / "q3_power.png", dpi=130)
    np.savez(CACHE / "q3_power.npz", **{k.replace(" ", ""): v for k, v in out.items()})
