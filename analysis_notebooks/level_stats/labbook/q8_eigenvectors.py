"""Q8: eigenvector statistics as a chaos test that needs less resolution (perfect resolution first, N = 3).

A complete diagonal basis measures w_{b,lambda} = |<b|lambda>|^2 (the pole amplitudes per row).
Chaotic (GOE-like) eigenvectors: D w is Porter-Thomas (chi^2 with 1 dof) in the bulk, the same for every b.
Per realization, over the bulk eigenvectors (10 % cut each edge), statistics:
- ipr: mean over lambda of D * sum_b w^2 (GOE: 3, more for structured states);
- small: fraction of D w < 0.05 (Porter-Thomas: 0.177);
- logw: mean of log(D w) (Porter-Thomas: -1.27 = -(gamma + ln 2)).
Paired shapes, K/g = 0 vs K/g at W/g; realizations for 3 sigma.
Run: pixi run python analysis_notebooks/level_stats/labbook/q8_eigenvectors.py
"""
import numpy as np
from scipy.stats import chi2

from common import disorder_shape, spectrum

DRAWS = 1000
D = 35
PT_SMALL = chi2.cdf(0.05, 1)


def stats(levels, weights):
    order = np.argsort(levels)
    cut = int(np.ceil(0.1 * D))
    w = weights[:, order][:, cut:-cut] * D
    return np.array([np.mean(np.sum(w ** 2, axis=0) / D), np.mean(w < 0.05), np.mean(np.log(np.maximum(w, 1e-12)))])


if __name__ == "__main__":
    names = ("ipr", "small", "logw")
    print(f"Porter-Thomas: ipr 3, small {PT_SMALL:.3f}, logw -1.27")
    for w_over_g in (1.7, 2.4, 0.7):
        rng = np.random.default_rng(5)
        shapes = [disorder_shape(rng) for _ in range(DRAWS)]
        base = np.array([stats(*spectrum(3, 0.0, w_over_g, u)) for u in shapes])
        for k in (-0.9, -1.8, -4.0):
            s = np.array([stats(*spectrum(3, k, w_over_g, u)) for u in shapes])
            d = s - base
            m3 = (3 * d.std(axis=0) / np.abs(d.mean(axis=0))) ** 2
            print(f"W/g {w_over_g}, K/g {k}: " + "; ".join(
                f"{n} {base[:, i].mean():.3f} -> {s[:, i].mean():.3f} (paired shapes for 3 sigma {m3[i]:.1f})"
                for i, n in enumerate(names)))
