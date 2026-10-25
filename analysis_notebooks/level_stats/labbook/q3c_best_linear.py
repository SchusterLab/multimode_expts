"""Q3c: the best linear statistic of the per-realization r histogram (Fisher discriminant),
trained on one set of shapes and tested on another (no in-sample gain). Perfect resolution.
Run: pixi run python analysis_notebooks/level_stats/labbook/q3c_best_linear.py
"""
import numpy as np

from common import disorder_shape, gap_ratios, spectrum

BINS = np.linspace(0, 1, 6)
DRAWS = 2000


def hists(n, kerr, disorder, seed):
    rng = np.random.default_rng(seed)
    return np.array([np.histogram(gap_ratios(spectrum(n, kerr, disorder, disorder_shape(rng))[0]), BINS)[0]
                     for _ in range(DRAWS)], float)


if __name__ == "__main__":
    n = 3
    ref_train, ref_test = hists(n, -0.13, 1.7, 1), hists(n, -0.13, 1.7, 2)
    for target in [(-0.9, 1.7), (-1.8, 1.7), (-2.5, 1.7)]:
        tr, te = hists(n, *target, 1), hists(n, *target, 2)
        frac = lambda h: h / h.sum(axis=1, keepdims=True)
        A, B = frac(ref_train), frac(tr)
        cov = 0.5 * (np.cov(A.T) + np.cov(B.T))
        w = np.linalg.lstsq(cov[:-1, :-1], (B - A).mean(0)[:-1], rcond=None)[0]   # one bin is redundant
        sa, sb = frac(ref_test)[:, :-1] @ w, frac(te)[:, :-1] @ w
        d = (sb.mean() - sa.mean()) / np.sqrt(0.5 * (sa.var() + sb.var()))
        first = lambda h: frac(h)[:, 0] + 0.25 * 0  # P(r<0.2) for comparison
        fa, fb = frac(ref_test)[:, 0], frac(te)[:, 0]
        d0 = (fb.mean() - fa.mean()) / np.sqrt(0.5 * (fa.var() + fb.var()))
        print(f"{target}: separation per realization, best linear {abs(d):.3f} (M for z=3 two-point: "
              f"{int(np.ceil(2 * (3 / d) ** 2))}); P(r<0.2) alone {abs(d0):.3f} ({int(np.ceil(2 * (3 / d0) ** 2))}); "
              f"weights {np.round(w / np.abs(w).max(), 2)}")
