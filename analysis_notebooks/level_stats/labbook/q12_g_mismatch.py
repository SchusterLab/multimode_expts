"""Q12: the paired test when the N = 3 coupling is 4 % larger than the N = 1 coupling (10-24 Kerr log).
Null case: the N = 3 system is linear with g' = 1.04 g, the reference is the sums of the N = 1 levels (g).
-> the bias and the extra spread of d per shape. Also with Kerr, for the contrast.
Run: pixi run python analysis_notebooks/level_stats/labbook/q12_g_mismatch.py
"""
import numpy as np

from common import disorder_shape, gap_ratios, spectrum
from q5_cost import linear_levels

W, DRAWS = 2.4, 2000
frac = lambda levels: np.mean(gap_ratios(np.sort(levels)) < 0.25)
rng = np.random.default_rng(3)
null, contrast, ideal = [], [], []
for _ in range(DRAWS):
    u = disorder_shape(rng)
    ref = frac(linear_levels(u, W, 1.0))
    null.append(frac(spectrum(3, 0.0, W / 1.04, u, g=1.04)[0]) - ref)      # same onsite energies, g x 1.04
    contrast.append(frac(spectrum(3, -1.8 / 1.04, W / 1.04, u, g=1.04)[0]) - ref)
    ideal.append(frac(spectrum(3, -1.8, W, u)[0]) - ref)
null, contrast, ideal = map(np.array, (null, contrast, ideal))
for name, d in (("null (linear, g x 1.04)", null), ("K/g -1.8, g x 1.04", contrast), ("K/g -1.8, no mismatch", ideal)):
    print(f"{name}: mean d {d.mean():+.4f} +/- {d.std() / np.sqrt(DRAWS):.4f}, sd per shape {d.std():.3f}, "
          f"shapes for 3 sigma {(3 * d.std() / abs(d.mean())) ** 2 if abs(d.mean()) > 0.02 else float('nan'):.1f}")
