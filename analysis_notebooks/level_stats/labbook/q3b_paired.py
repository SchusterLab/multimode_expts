"""Q3b: the same test with paired disorder patterns (one shape measured at both Kerr values).

Per shape i: d_i = P_B,i(r<0.25) - P_A,i(r<0.25); z = mean(d) / (sd(d) / sqrt(M)).
Run: pixi run python analysis_notebooks/level_stats/labbook/q3b_paired.py
"""
import numpy as np

from common import disorder_shape, gap_ratios, spectrum

R0, DRAWS = 0.25, 2000
REFERENCE = (-0.13, 1.7)
TARGETS = [(-0.9, 1.7), (-1.8, 1.7), (-2.5, 1.7), (-1.8, 3.0)]

if __name__ == "__main__":
    for n in (3, 4):
        rng = np.random.default_rng(11)
        shapes = [disorder_shape(rng) for _ in range(DRAWS)]
        ref = np.array([np.mean(gap_ratios(spectrum(n, *REFERENCE, u)[0]) < R0) for u in shapes])
        for target in TARGETS:
            tgt = np.array([np.mean(gap_ratios(spectrum(n, *target, u)[0]) < R0) for u in shapes])
            d = tgt - ref
            m3 = int(np.ceil((3 * d.std() / abs(d.mean())) ** 2))
            print(f"N={n} {REFERENCE} -> {target}: mean diff {d.mean():+.3f}, sd per shape {d.std():.3f} "
                  f"(unpaired sd {np.sqrt(ref.var() + tgt.var()):.3f}); paired M for z=3: {m3}; "
                  f"z at M=6: {abs(d.mean()) / (d.std() / np.sqrt(6)):.2f}, M=9: {abs(d.mean()) / (d.std() / 3):.2f}")
