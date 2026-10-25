"""Q2: what the best unbiased estimator can keep of the small gaps, per hardware case.

Per model draw: the Cramér-Rao error of every adjacent gap (``fitting.qsim.poles.design``:
real amplitudes with column sums = complete diagonal basis, per-photon row offsets plus a
random rest), adjacent levels merged while their gap is under 4 errors. Two observers:

- naive: a merged pair is one level (what a fitter without the weights does);
- counting: a merged cluster of weight m is m coincident levels (complete basis, weights known).

Writes cache/q2_resolution.npz. Run (about 10 min on 32 cores):
pixi run python analysis_notebooks/level_stats/labbook/q2_resolution.py
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, disorder_shape, gap_ratios  # noqa: E402
from fitting.qsim.poles.design import Design, gap_errors  # noqa: E402
from theory.qsim.central_boson import fixed_n_hamiltonian  # noqa: E402

# Hardware cases: (g MHz, dt us, samples, level decay 1/us, noise per complex sample)
CASES = {
    "sep10": (29.2e-3, 0.428, 468, 0.006, 0.16),          # the Sep 10 ensemble as measured
    "now_92us": (20.4e-3, 0.614, 150, 0.006, 0.10),       # 0.25 mA now, the present trace length
    "now_184us": (20.4e-3, 0.614, 300, 0.006, 0.10),      # 0.25 mA now, twice the window
    "highK_29": (29.2e-3, 0.428, 300, 0.012, 0.16),       # K ~ -52 kHz, M1 T2 ~ 35 us
}
OFFSETS = {"per_photon_0.5kHz": 0.5e-3, "known": 1e-6}
POINTS = [(-0.13, 1.7), (-0.9, 1.7), (-1.8, 1.7), (-0.13, 3.0), (-0.9, 3.0), (-1.8, 3.0)]
DRAWS = 24
N = 3


def merge(levels, errors, weights, min_snr=4.0):
    """Merge, smallest gap/error first, every adjacent pair under ``min_snr`` errors.
    -> (positions, integer weights)."""
    levels, errors, weights = list(levels), list(errors), list(weights)
    while len(levels) > 2:
        snr = np.diff(levels) / np.asarray(errors)
        k = int(np.argmin(snr))
        if snr[k] >= min_snr:
            break
        total = weights[k] + weights[k + 1]
        levels[k:k + 2] = [(weights[k] * levels[k] + weights[k + 1] * levels[k + 1]) / total]
        weights[k:k + 2] = [total]
        near = [errors[i] for i in (k - 1, k + 1) if 0 <= i < len(errors)]
        del errors[k]
        if k > 0 and near:
            errors[k - 1] = max(near)
    return np.asarray(levels), np.asarray(weights)


def one(args):
    case, offsets, (kerr, disorder), seed = args
    g, dt, samples, decay, noise = CASES[case]
    t = np.arange(samples) * dt
    shape = disorder_shape(np.random.default_rng(seed))
    m = fixed_n_hamiltonian(N, 5, -disorder * g * shape, [g] * 4, kerr * g)
    design = Design(amplitudes="real_sums", offsets="per_photon", offset_prior_MHz=OFFSETS[offsets],
                    decay_per_us=decay)
    D = len(m.energies_MHz)
    errors = gap_errors(t, m.energies_MHz, m.basis_eigenstate_weights, np.asarray(m.fock_basis, float),
                        np.full(D, noise), 1 / (samples * dt), design)
    positions, weights = merge(m.energies_MHz, errors, np.ones(D))
    counted = np.repeat(positions, weights.astype(int))
    gaps = np.diff(m.energies_MHz)
    return dict(key=(case, offsets, kerr, disorder), true=gap_ratios(m.energies_MHz),
                naive=gap_ratios(positions), counting=gap_ratios(counted),
                gap_over_error=gaps / errors, gap_over_mean=gaps / gaps.mean())


if __name__ == "__main__":
    jobs = [(c, o, p, 50_000 + s) for c, o, p, s in
            itertools.product(CASES, OFFSETS, POINTS, range(DRAWS))]
    with ProcessPoolExecutor(30) as pool:
        results = list(pool.map(one, jobs, chunksize=2))
    out = {}
    for key in {r["key"] for r in results}:
        rows = [r for r in results if r["key"] == key]
        name = "|".join(str(x) for x in key)
        for field in ("true", "naive", "counting", "gap_over_error", "gap_over_mean"):
            out[f"{name}|{field}"] = np.concatenate([r[field] for r in rows])
    np.savez(CACHE / "q2_resolution.npz", **out)
    for case, offsets in itertools.product(CASES, OFFSETS):
        print(f"\n{case}, offsets {offsets}:  P(r<0.25) true / naive / counting   <r> true / naive / counting   "
              "small gaps (<0.5 mean) resolved")
        for kerr, disorder in POINTS:
            name = f"{case}|{offsets}|{kerr}|{disorder}"
            tr, na, co = (out[f"{name}|{f}"] for f in ("true", "naive", "counting"))
            go, gm = out[f"{name}|gap_over_error"], out[f"{name}|gap_over_mean"]
            small = gm < 0.5
            print(f"  K/g {kerr:+.2f} W/g {disorder}:  {np.mean(tr < .25):.2f} / {np.mean(na < .25):.2f} / "
                  f"{np.mean(co < .25):.2f}    {tr.mean():.3f} / {na.mean():.3f} / {co.mean():.3f}    "
                  f"{np.mean(go[small] >= 4):.2f}")
