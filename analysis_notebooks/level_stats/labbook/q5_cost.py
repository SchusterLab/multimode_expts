"""Q5: realizations and device hours for a 3-sigma detection, per hardware option.

The test (frozen): per disorder shape, d = P_N3(r < 0.25) measured (counting observer of Q2 at the
option's grid, decay and noise) minus P(r < 0.25) of the linear reference of the SAME shape (all
sums of the 5 single-particle levels, which an N = 1 measurement gives). z = mean(d) / (sd(d)/sqrt(M)).
The counting bias raises P_N3 (towards Poisson), so it can hide a change but not make one.

Device time per N = 3 occupation: Stark calibration 5.5 min + trace 11 min per 150 samples (500 reps),
from the 2026-10-24 jobs; the N = 1 reference adds 5 occupations of the same cost.
Writes cache/q5_cost.npz. Run (about 15 min on 32 cores):
pixi run python analysis_notebooks/level_stats/labbook/q5_cost.py
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, disorder_shape, gap_ratios  # noqa: E402
from q2_resolution import merge  # noqa: E402
from fitting.qsim.poles.design import Design, gap_errors  # noqa: E402
from theory.qsim.central_boson import fixed_n_hamiltonian  # noqa: E402

# name: (g MHz, dt us, samples, level decay 1/us, noise, kerr/g)
OPTIONS = {
    "K/g -0.9: g 20.4, 92 us": (20.4e-3, 0.614, 150, 0.006, 0.10, -0.9),
    "K/g -0.9: g 20.4, 184 us": (20.4e-3, 0.614, 300, 0.006, 0.10, -0.9),
    "K/g -1.8: g 10.2 (idle), 184 us": (10.2e-3, 1.228, 150, 0.006, 0.10, -1.8),
    "K/g -1.8: g 10.2 (idle), 368 us": (10.2e-3, 1.228, 300, 0.006, 0.10, -1.8),
    "K/g -1.8: g 20.4 (new set point), 184 us": (20.4e-3, 0.614, 300, 0.009, 0.10, -1.8),
}
DISORDERS = (1.7, 2.4)
DRAWS = 160
R0 = 0.25
STARK_MIN, TRACE_MIN_PER_150 = 5.5, 11.0


def linear_levels(shape, disorder, g):
    """-> the N = 3 levels of the linear system: all sums of three single-particle levels."""
    one = fixed_n_hamiltonian(1, 5, -disorder * g * shape, [g] * 4, 0.0).energies_MHz
    return np.sort([one[i] + one[j] + one[k] for i in range(5) for j in range(i, 5) for k in range(j, 5)])


def one(args):
    name, disorder, seed = args
    g, dt, samples, decay, noise, kerr = OPTIONS[name]
    shape = disorder_shape(np.random.default_rng(seed))
    m = fixed_n_hamiltonian(3, 5, -disorder * g * shape, [g] * 4, kerr * g)
    design = Design(amplitudes="real_sums", offsets="per_photon", offset_prior_MHz=0.5e-3, decay_per_us=decay)
    errors = gap_errors(np.arange(samples) * dt, m.energies_MHz, m.basis_eigenstate_weights,
                        np.asarray(m.fock_basis, float), np.full(35, noise), 1 / (samples * dt), design)
    pos, w = merge(m.energies_MHz, errors, np.ones(35))
    frac = lambda levels: np.mean(gap_ratios(levels) < R0)
    return name, disorder, frac(linear_levels(shape, disorder, g)), frac(m.energies_MHz), frac(np.repeat(pos, w.astype(int)))


if __name__ == "__main__":
    jobs = list(itertools.product(OPTIONS, DISORDERS, range(70_000, 70_000 + DRAWS)))
    with ProcessPoolExecutor(30) as pool:
        res = list(pool.map(one, jobs, chunksize=2))
    out = {}
    print("option | W/g | P linear ref | P true | P observed | M for 3 sigma (true, observed) | device hours (observed)")
    for name, disorder in itertools.product(OPTIONS, DISORDERS):
        rows = np.array([r[2:] for r in res if r[0] == name and r[1] == disorder])
        ref, true, obs = rows.T
        m_true = (3 * np.std(true - ref) / abs(np.mean(true - ref))) ** 2
        m_obs = (3 * np.std(obs - ref) / abs(np.mean(obs - ref))) ** 2 if np.mean(obs - ref) < 0 else np.inf
        samples = OPTIONS[name][2]
        hours = np.ceil(m_obs) * 40 * (STARK_MIN + TRACE_MIN_PER_150 * samples / 150) / 60
        out[f"{name}|{disorder}"] = np.array([ref.mean(), true.mean(), obs.mean(), m_true, m_obs, hours])
        print(f"{name} | {disorder} | {ref.mean():.3f} | {true.mean():.3f} | {obs.mean():.3f} | "
              f"{m_true:.0f}, {m_obs:.0f} | {hours:.0f}")
    np.savez(CACHE / "q5_cost.npz", **out)
