"""Q2c: does the counting observer survive larger row offsets? (Q2 method; per-photon pattern free,
random rest per row with prior 0.5 / 2 / 5 kHz.)
Run: pixi run python analysis_notebooks/level_stats/labbook/q2c_offsets.py
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import q2_resolution as q2

q2.CASES.clear()
q2.CASES.update({"sep10": (29.2e-3, 0.428, 468, 0.006, 0.16), "now_184us": (20.4e-3, 0.614, 300, 0.006, 0.10)})
q2.OFFSETS.clear()
q2.OFFSETS.update({"0.5kHz": 0.5e-3, "2kHz": 2e-3, "5kHz": 5e-3})
q2.POINTS[:] = [(-0.13, 1.7), (-1.8, 1.7)]

if __name__ == "__main__":
    jobs = [(c, o, p, 50_000 + s) for c, o, p, s in itertools.product(q2.CASES, q2.OFFSETS, q2.POINTS, range(24))]
    with ProcessPoolExecutor(30) as pool:
        res = list(pool.map(q2.one, jobs, chunksize=2))
    for case, off in itertools.product(q2.CASES, q2.OFFSETS):
        line = []
        for k, w in q2.POINTS:
            rows = [r for r in res if r["key"] == (case, off, k, w)]
            tr, co = (np.concatenate([r[f] for r in rows]) for f in ("true", "counting"))
            line.append(f"K/g {k}: true {np.mean(tr < .25):.2f} counting {np.mean(co < .25):.2f}")
        print(f"{case}, offset rest {off}: " + " | ".join(line))
