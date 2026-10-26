"""Q11b: the five sync-34 rows with one common storage offset eps and one common g (row offsets free).
Run: pixi run python analysis_notebooks/level_stats/labbook/q11b_common.py
"""
import json

import numpy as np
from scipy.optimize import least_squares

import q11_onebody as q

first = json.load(open(q.WORK + "idle_check.json"))["runs"]["34"]["trace_files"][0]
others = json.load(open(q.WORK + "idle_onebody.json"))["runs"]["34"]["trace_files"]
loaded = [q.load(f) for f in [first] + others]
t = loaded[0][0]
a = np.array([x[1] for x in loaded])
noise = np.array([np.std(np.diff(row)) / np.sqrt(2) for row in a])
expand = lambda p: np.r_[np.full(4, p[0]), np.full(4, p[1]), p[2:6], p[6]]
best = None
for eps0 in np.linspace(-0.06, 0.06, 13):
    for g0 in (0.004, 0.007, 0.010):
        f = least_squares(lambda p: q.residuals(expand(p), t, a, noise), np.r_[eps0, g0, np.zeros(4), 0.004])
        if best is None or f.cost < best.cost:
            best = f
err = np.sqrt(np.diag(np.linalg.pinv(best.jac.T @ best.jac)))
print(f"common model: eps {1e3 * best.x[0]:+.2f} +/- {1e3 * err[0]:.2f} kHz, g {1e3 * best.x[1]:.2f} +/- "
      f"{1e3 * err[1]:.2f} kHz (expected 10.18), chi2/point {2 * best.cost / best.fun.size:.2f}, "
      f"row offsets {np.round(1e3 * np.r_[best.x[2:6], -best.x[2:6].sum()], 1)} kHz")
E, V = q.onebody(np.full(4, best.x[0]), np.full(4, best.x[1]))
print("M1 row weights", np.round(V[0] ** 2, 3), "levels", np.round(1e3 * E, 1))
