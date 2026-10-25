"""Q4: the frozen counting test on the existing Sep 10 ensemble (low Kerr, K/g -0.13, W/g 1.71).

Frozen before looking (2026-10-25 evening):
- fitter: the cached sparse non-negative fit started from the Hamiltonian fit's row offsets
  (survey of 2026-10-02, file suffix ``T1T3``), unchanged;
- counting: m = round(sum over the 35 rows of a pole's amplitude); m = 0 poles dropped; m >= 2 gives
  m coincident levels;
- statistic: pooled bulk P(r < 0.25), 10 % of the levels cut at each edge;
- comparison: the recorded model of each realization, (a) exact, (b) through the Cramér-Rao
  counting observer at this data's grid and noise (as Q2).
Also shown, not part of the test: the same fit without counting (naive), and the other fitters.

Run: pixi run python analysis_notebooks/level_stats/labbook/q4_sep10_counting.py   (about 3 min)
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, gap_ratios  # noqa: E402
from q2_resolution import merge  # noqa: E402
from experiments.job_paths import data_root  # noqa: E402
from experiments.qsim.pole_data import load_spectra  # noqa: E402
from fitting.qsim.poles.design import Design, gap_errors  # noqa: E402
from fitting.qsim.poles.noise import row_noise  # noqa: E402
from fitting.qsim.poles.pole_fit import normalize_to_initial_return  # noqa: E402
from fitting.qsim.poles.registry import data_set  # noqa: E402
from theory.qsim.central_boson import fixed_n_hamiltonian  # noqa: E402

SURVEY = data_root() / "260818_qsim_spectroscopy" / "derived_data" / "pole_finding" / "sep10_survey"
FITTERS = ("T1T3", "T1T3_merge200", "C", "B", "A")
R0 = 0.25


def counted_levels(fit):
    """-> the levels with integer multiplicities from the pole weights, and the weights."""
    w = np.real(np.sum(fit.amplitudes, axis=0))
    m = np.rint(w).astype(int)
    keep = m > 0
    return np.repeat(fit.frequencies_MHz[keep], m[keep]), w


def stat(level_sets):
    r = np.concatenate([gap_ratios(levels) for levels in level_sets])
    return np.mean(r < R0), r


def bootstrap(level_sets, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ratios = [gap_ratios(levels) for levels in level_sets]
    values = []
    for _ in range(n):
        pick = rng.integers(0, len(ratios), len(ratios))
        r = np.concatenate([ratios[i] for i in pick])
        values.append(np.mean(r < R0))
    return np.std(values)


if __name__ == "__main__":
    spectra = load_spectra(data_set("sep10_full_K3p6_g29p2"), data_root())
    results = {}
    for name in FITTERS:
        counted, naive, totals = [], [], []
        for s in spectra:
            path = SURVEY / f"fit_{s.label.replace('/', '_')}_{name}.pkl"
            if not path.exists():
                break
            with open(path, "rb") as f:
                fit, _ = pickle.load(f)
            levels, w = counted_levels(fit)
            counted.append(levels)
            naive.append(np.sort(fit.frequencies_MHz[w > 0.5]))
            totals.append(len(levels))
        if len(counted) < len(spectra):
            print(f"{name}: only {len(counted)} realizations cached, skipped in the table")
            if name != "T1T3_merge200":
                continue
        p_c, r_c = stat(counted)
        p_n, _ = stat(naive)
        results[name] = dict(counted=counted, p_counted=p_c, p_naive=p_n, err=bootstrap(counted), totals=totals)
        print(f"{name:14s} counted P(r<.25) {p_c:.3f} +/- {results[name]['err']:.3f} (<r> {r_c.mean():.3f}); "
              f"naive {p_n:.3f}; counted levels per realization {totals}")

    # The model, exact and through the counting observer at this grid and noise.
    exact, observed = [], []
    design = Design(amplitudes="real_sums", offsets="per_photon", offset_prior_MHz=0.5e-3, decay_per_us=0.006)
    for s in spectra:
        p = s.model_parameters
        m = fixed_n_hamiltonian(p["photon_number"], p["mode_count"], p["detunings_MHz"], p["couplings_MHz"], p["kerr_MHz"])
        exact.append(m.energies_MHz)
        noise = row_noise(normalize_to_initial_return(s.A), s.dt_us, s.levels_MHz, 2)
        noise = np.broadcast_to(np.asarray(noise, float), (len(m.energies_MHz),))
        errors = gap_errors(s.time_us, m.energies_MHz, m.basis_eigenstate_weights, np.asarray(m.fock_basis, float),
                            noise, s.bin_MHz, design)
        pos, wts = merge(m.energies_MHz, errors, np.ones(len(m.energies_MHz)))
        observed.append(np.repeat(pos, wts.astype(int)))
    p_e, _ = stat(exact)
    p_o, _ = stat(observed)
    print(f"model, same 9 patterns: exact {p_e:.3f} +/- {bootstrap(exact):.3f}; through the counting observer "
          f"{p_o:.3f} +/- {bootstrap(observed):.3f}; Kerr {[round(1e3 * s.model_parameters['kerr_MHz'], 2) for s in spectra[:1]]} kHz")
    with open(CACHE / "q4_sep10_counting.pkl", "wb") as f:
        pickle.dump(dict(results=results, exact=exact, observed=observed,
                         labels=[s.label for s in spectra]), f)
