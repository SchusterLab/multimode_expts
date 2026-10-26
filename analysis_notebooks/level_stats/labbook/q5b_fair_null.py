"""Q5b: Q5 with the fair null: the linear system (K = 0, same shape, same g) measured by the SAME
counting procedure (Cramér-Rao merge at the option's grid, decay and noise). Also 250 reps (noise 0.14).
Run: pixi run python analysis_notebooks/level_stats/labbook/q5b_fair_null.py
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import q5_cost as q5
from common import CACHE, disorder_shape, gap_ratios
from q2_resolution import merge
from fitting.qsim.poles.design import Design, gap_errors
from theory.qsim.central_boson import fixed_n_hamiltonian

q5.OPTIONS.clear()
q5.OPTIONS.update({
    "K/g -0.9: g 20.4, 184 us": (20.4e-3, 0.614, 300, 0.006, 0.10, -0.9),
    "K/g -1.8: g 10.2 (idle), 368 us": (10.2e-3, 1.228, 300, 0.006, 0.10, -1.8),
    "K/g -1.8: g 10.2 (idle), 368 us, 250 reps": (10.2e-3, 1.228, 300, 0.006, 0.14, -1.8),
    "K/g -1.8: g 10.2 (idle), 276 us, 250 reps": (10.2e-3, 1.228, 225, 0.006, 0.14, -1.8),
    "K/g -1.8: g 20.4 (new set point), 184 us": (20.4e-3, 0.614, 300, 0.009, 0.10, -1.8),
})
DISORDERS = (2.0, 2.4)


def observed(kerr, shape, disorder, g, dt, samples, decay, noise):
    m = fixed_n_hamiltonian(3, 5, -disorder * g * shape, [g] * 4, kerr * g)
    design = Design(amplitudes="real_sums", offsets="per_photon", offset_prior_MHz=0.5e-3, decay_per_us=decay)
    errors = gap_errors(np.arange(samples) * dt, m.energies_MHz, m.basis_eigenstate_weights,
                        np.asarray(m.fock_basis, float), np.full(35, noise), 1 / (samples * dt), design)
    pos, w = merge(m.energies_MHz, errors, np.ones(35))
    return np.mean(gap_ratios(np.repeat(pos, w.astype(int))) < 0.25)


def one(args):
    name, disorder, seed = args
    g, dt, samples, decay, noise, kerr = q5.OPTIONS[name]
    shape = disorder_shape(np.random.default_rng(seed))
    return name, disorder, observed(0.0, shape, disorder, g, dt, samples, decay, noise), \
        observed(kerr, shape, disorder, g, dt, samples, decay, noise)


if __name__ == "__main__":
    jobs = list(itertools.product(q5.OPTIONS, DISORDERS, range(70_000, 70_160)))
    with ProcessPoolExecutor(30) as pool:
        res = list(pool.map(one, jobs, chunksize=2))
    out = {}
    print("option | W/g | linear, counted | K, counted | shapes for 3 sigma | device hours (500 reps) ")
    for name, disorder in itertools.product(q5.OPTIONS, DISORDERS):
        a, b = np.array([r[2:] for r in res if r[0] == name and r[1] == disorder]).T
        d = b - a
        m3 = (3 * d.std() / abs(d.mean())) ** 2
        samples = q5.OPTIONS[name][2]
        trace_min = q5.TRACE_MIN_PER_150 * samples / 150 * (0.5 if "250 reps" in name else 1)
        hours = np.ceil(m3) * 40 * (q5.STARK_MIN + trace_min) / 60
        out[f"{name}|{disorder}"] = np.array([a.mean(), b.mean(), m3, hours])
        print(f"{name} | {disorder} | {a.mean():.3f} | {b.mean():.3f} | {m3:.1f} | {hours:.0f}")
    np.savez(CACHE / "q5b_fair_null.npz", **out)
