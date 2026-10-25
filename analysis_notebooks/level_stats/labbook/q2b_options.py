"""Q2b: the counting observer for hardware options that reach K/g <= -1.8 (same method as Q2).
Run: pixi run python analysis_notebooks/level_stats/labbook/q2b_options.py
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import q2_resolution as q2

q2.CASES.clear()
q2.CASES.update({
    # present Kerr (-19 kHz), g halved (pi/80 swaps or an idle per cycle): K/g = -1.8, 368 us window
    "g10_368us": (10.2e-3, 1.228, 300, 0.006, 0.10),
    "g10_368us_noise0.14": (10.2e-3, 1.228, 300, 0.006, 0.14),
    # present g, a high-Kerr set point K = -37 kHz, M1 T2 assumed 60 us (level decay 0.009)
    "g20_highK_184us": (20.4e-3, 0.614, 300, 0.009, 0.10),
    # present everything, 276 us window
    "now_276us": (20.4e-3, 0.614, 450, 0.006, 0.10),
})
q2.OFFSETS.clear()
q2.OFFSETS.update({"per_photon_0.5kHz": 0.5e-3})
q2.POINTS[:] = [(-0.13, 1.7), (-0.9, 1.7), (-1.8, 1.7)]

if __name__ == "__main__":
    jobs = [(c, o, p, 50_000 + s) for c, o, p, s in itertools.product(q2.CASES, q2.OFFSETS, q2.POINTS, range(24))]
    with ProcessPoolExecutor(30) as pool:
        res = list(pool.map(q2.one, jobs, chunksize=2))
    for case in q2.CASES:
        print(case)
        for k, w in q2.POINTS:
            rows = [r for r in res if r["key"] == (case, "per_photon_0.5kHz", k, w)]
            tr, na, co = (np.concatenate([r[f] for r in rows]) for f in ("true", "naive", "counting"))
            print(f"  K/g {k:+.2f}: P(r<.25) true {np.mean(tr < .25):.2f} naive {np.mean(na < .25):.2f} counting {np.mean(co < .25):.2f}")
