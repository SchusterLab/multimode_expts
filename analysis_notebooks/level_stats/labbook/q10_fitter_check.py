"""Q10: does the real fitter + counting give what the counting observer (Q2, Q5b) predicts?

Synthetic complete-basis traces at the cheapest option of Q5b (idle: g 10.2 kHz, dt 1.228 us,
225 samples = 276 us, decay 0.006 /us, noise 0.14 per complex sample = 250 reps), W/g = 2.4, the same
shapes at K = 0 (the linear reference) and K/g = -1.8. Fitter: the frozen one of Q4 (sparse non-negative
fit), started from the joint pencil for band and decay scale, with the true row offsets (0): on synthetic
data the Hamiltonian fit recovers the offsets to 0.02-0.05 kHz, so this stands in for its offsets.
Counting as in Q4. Writes cache/q10_fitter_check.pkl.
Run: pixi run python analysis_notebooks/level_stats/labbook/q10_fitter_check.py [option]   (about 15 min)
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, disorder_shape, gap_ratios  # noqa: E402
from q2_resolution import merge  # noqa: E402
from fitting.qsim.poles import joint_pencil, sparse_fit  # noqa: E402
from fitting.qsim.poles.design import Design, gap_errors  # noqa: E402
from theory.qsim.central_boson import fixed_n_hamiltonian  # noqa: E402

# Options: name -> (g MHz, dt us, samples, decay 1/us, noise, Kerr/g). Default: the cheapest Q5 option.
OPTIONS = {
    "idle_276us_250reps": (10.2e-3, 1.228, 225, 0.006, 0.14, -1.8),
    "idle_184us_250reps": (10.2e-3, 1.228, 150, 0.006, 0.14, -1.8),
    "now_92us_500reps": (20.4e-3, 0.614, 150, 0.006, 0.10, -0.9),
    "now_184us_250reps": (20.4e-3, 0.614, 300, 0.006, 0.14, -0.9),
}
OPTION = sys.argv[1] if len(sys.argv) > 1 else "idle_276us_250reps"
G, DT, SAMPLES, DECAY, NOISE, KERR = OPTIONS[OPTION]
W_OVER_G = 2.4
KERRS = (0.0, KERR)
SHAPES = 16


def one(args):
    kerr, seed = args
    shape = disorder_shape(np.random.default_rng(seed))
    m = fixed_n_hamiltonian(3, 5, -W_OVER_G * G * shape, [G] * 4, kerr * G)
    t = np.arange(SAMPLES) * DT
    Z = np.exp(np.outer(-DECAY - 2j * np.pi * m.energies_MHz, t))
    rng = np.random.default_rng(seed + 1_000_000 + int(100 * abs(kerr)))
    A = m.basis_eigenstate_weights @ Z
    A += NOISE / np.sqrt(2) * (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape))
    start = joint_pencil.fit(A, t)
    fit = sparse_fit.fit(A, t, sparse_fit.SparseFitSettings(), start=start, offsets_MHz=np.zeros(len(A)))
    w = np.real(fit.amplitudes.sum(axis=0))
    k = np.rint(w).astype(int)
    counted = np.repeat(fit.frequencies_MHz[k > 0], k[k > 0])
    design = Design(amplitudes="real_sums", offsets="per_photon", offset_prior_MHz=0.5e-3, decay_per_us=DECAY)
    errors = gap_errors(t, m.energies_MHz, m.basis_eigenstate_weights, np.asarray(m.fock_basis, float),
                        np.full(35, NOISE), 1 / (SAMPLES * DT), design)
    pos, wt = merge(m.energies_MHz, errors, np.ones(35))
    frac = lambda levels: float(np.mean(gap_ratios(np.sort(levels)) < 0.25))
    return dict(kerr=kerr, seed=seed, exact=frac(m.energies_MHz), observer=frac(np.repeat(pos, wt.astype(int))),
                fitter=frac(counted), total=int(k[k > 0].sum()), poles=len(fit.frequencies_MHz))


if __name__ == "__main__":
    jobs = [(k, 70_000 + s) for k in KERRS for s in range(SHAPES)]
    with ProcessPoolExecutor(16) as pool:
        res = list(pool.map(one, jobs))
    pickle.dump(res, open(CACHE / f"q10_fitter_check_{OPTION}.pkl", "wb"))
    print("option", OPTION)
    for k in KERRS:
        rows = [r for r in res if r["kerr"] == k]
        print(f"K/g {k}: exact {np.mean([r['exact'] for r in rows]):.3f}, observer {np.mean([r['observer'] for r in rows]):.3f}, "
              f"fitter+counting {np.mean([r['fitter'] for r in rows]):.3f}; counted totals {[r['total'] for r in rows]}")
    a = np.array([r["fitter"] for r in res if r["kerr"] == 0.0])
    b = np.array([r["fitter"] for r in res if r["kerr"] == KERR])
    d = b - a
    print(f"paired fitter difference {d.mean():+.3f} +/- {d.std() / np.sqrt(len(d)):.3f} ({SHAPES} shapes), "
          f"z = {abs(d.mean()) / (d.std() / np.sqrt(len(d))):.1f}")
