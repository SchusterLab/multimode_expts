# 2026-09-29: pole finding T1, a fit of the Hamiltonian

Plan: `docs/qsim/pole_finding_explore.md`, T1. Branch `qsim-analysis`.

**Caveat (guan), for every result below: the levels of a model fit carry the model's
statistics.** Their gaps and P(r < 0.25) are a check of the model or a start for a free fit,
never the answer. Model-derived results are marked (model).

## What was done

- `fitting/qsim/poles/hamiltonian_fit.py`. The model is `fixed_n_hamiltonian`, which is linear
  in its 9 parameters (4 detunings, 4 couplings, Kerr): H(p) = sum_k p_k D_k. Row b is
  `s_b exp(-(gamma + 2 pi i delta_b) t) sum_lambda V_{b,lambda}^2 exp(-2 pi i E_lambda t)`: one
  decay, one offset per row (prior 0.5 kHz, as C and F), and a complex scale s_b per row that is
  projected out (A_b(0) is noisy and is 0.4-0.7 on the real data). Variant `amplitudes="free"`
  (T1free): the model's levels, the amplitudes >= 0 free per row (NNLS, as F). chi^2 is in
  F's units (noise per row out of band). The Jacobian is analytic (Daleckii-Krein for the
  eigenvectors, smooth through degeneracies; checked against finite differences). Many starts:
  the given parameters, then 11 random ones around them (1 kHz, 5 %, 1 kHz). Interface
  `fit(A, time_us, settings, row_groups=None, *, occupations, start) -> PoleFit`;
  `fit_hamiltonian` gives the parameters, their errors (Fisher), chi^2 and all the starts.
  The PoleFit merges levels closer than 1e-3 bin (as the synthetic truth).
- Additive change to shared code: `RealSpectrum.model_parameters` (optional, default None),
  set by `experiments/qsim/pole_data.spectrum` (the recorded detunings, couplings, Kerr, photon
  number, mode count). Nothing else changed.
- `analysis_notebooks/pole_finding/hamiltonian_fit.py` (fits, caches, tables; `T1_STAGES`
  picks the stages when run as a script); `tests/test_hamiltonian_fit.py` (6 tests, 10 s).
- Caches next to the fixed set: `fits_T1.h5` (80 cases, with the parameters in an attribute),
  `fits_T1free.h5` (the 8 subset cases), `fits_T1F.h5` (F started from T1, the 2 hardest cases).

## Recovery on the fixed set

Start: the true parameters off by the size of the real model error (detunings and Kerr 1 kHz,
couplings 5 %; 1-2 kHz in the levels). T1 on all 80 cases, 5-15 s per fit. Mean over 5 draws:

| point | rows | z rms | max z | detuning error (kHz) | coupling error (kHz) | Kerr error (kHz) | reduced chi^2 |
|---|---|---|---|---|---|---|---|
| August | 10 | 0.8-1.1 | 1.6-2.0 | 0.03-0.08 | 0.04-0.08 | 0.08-0.17 | 1.00 |
| August | 35 | 0.8-0.9 | 1.3-1.7 | 0.02-0.05 | 0.02-0.03 | 0.03-0.06 | 1.00 |
| September | 10 | 1.0-1.2 | 1.7-2.0 | 0.11-0.16 | 0.08-0.11 | 0.09-0.15 | 1.00 |
| September | 35 | 0.9-1.0 | 1.4-1.8 | 0.04-0.06 | 0.04-0.05 | 0.04-0.10 | 1.00 |

z = (fit - truth) / Fisher error: the errors are right (rms about 1). T2 is found to 1 %, the
row offsets to 0.02-0.05 kHz (the model fixes the levels, so the offsets are well fixed). 7-12
of the 12 starts reach the best chi^2: near the recorded values the problem has one minimum.
T1free recovers the August parameters as well (z rms 0.8-1.4), but on September 10 rows its
detuning and coupling errors are 1-17 kHz (free amplitudes leave directions of the parameters
unfixed; its levels are still right).

## The gap score on the fixed set (model)

T1's levels are the model's, on data made by the same model: this score is the best case for a
model fit, not a fitter result. All 80 cases: T1 finds 33-34 of 34 gaps, all the resolvable
ones, 0-0.4 false poles, median |gap error| / bound 0.1-0.2 (the bound is for free levels; 9
parameters fix 35 levels much better). Its P(r < 0.25) is the truth's (0.18-0.20 against
0.184; 0.352 against 0.352): the caveat in numbers.

The subset (draw 0, 10 rows), resolvable gaps found (real bound):

| point | T2 (us) | offset (kHz) | resolvable | C | F (from C) | T1 (model) | T1free (model) | F from T1 |
|---|---|---|---|---|---|---|---|---|
| August | 100 | 0.5 | 22 | 8 | 8 | 22 | 22 | **19** |
| August | 100 | 1 | 20 | 6 | 8 | 20 | 20 | **18** |
| August | 200 | 0.5 | 27 | 16 | 24 | 27 | 27 | - |
| August | 200 | 1 | 25 | 14 | 16 | 25 | 25 | - |
| September | 100 | 0.5 | 30 | 21 | 24 | 30 | 30 | - |
| September | 100 | 1 | 30 | 21 | 24 | 30 | 30 | - |
| September | 200 | 0.5 | 30 | 24 | 28 | 30 | 30 | - |
| September | 200 | 1 | 30 | 26 | 28 | 30 | 30 | - |

F from T1 on the hardest case (August, T2 100 us, 10 rows): false poles 0 and 2, median |gap
error| / bound 0.35 and 0.40, 570-620 s per fit (F refines all 35 T1 levels: 230 s for the first
refinement alone). **Started from T1, F finds 18-19 of 20-22 resolvable gaps, where from C it
finds 8.** The levels it keeps are its own (free, real, >= 0 amplitudes), so this is not a model
result; but its P(r < 0.25) is still low (0.045-0.048 against 0.28 of draw 0): F drops or
merges most small gaps, which are under the bound here. Caution: the synthetic data are exactly
the model, and the start is the truth plus 1-2 kHz; on real data the model is worse (below).

## Real data: the model check

From the recorded parameters (kHz; errors 0.02-0.07 kHz, from Fisher, not scaled by chi^2):

| spectrum | d1 | d2 | d3 | d4 | g1 | g2 | g3 | g4 | K | T2 (us) | red. chi^2 | in-band excess mean / max | level shift max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| august_N3 recorded | 0 | 0 | 0 | 0 | 8.62 | 8.62 | 8.62 | 8.62 | -10.5 | | | | |
| august_N3 | -1.41 | -0.92 | -1.13 | -1.34 | 8.56 | 8.47 | 8.63 | 8.77 | -9.16 | 246 | 2.16 | 3.96 / 6.76 | 4.1 |
| august_disorder/0 recorded | 13.50 | 34.21 | -21.64 | -26.06 | 8.62 x 4 | | | | -10.5 | | | | |
| august_disorder/0 | 13.28 | 34.25 | -21.92 | -27.23 | 8.34 | 8.28 | 9.08 | 8.15 | -8.72 | 241 | 1.43 | 2.14 / 3.71 | 2.2 |
| august_disorder/1 recorded | 28.97 | -39.36 | -0.18 | 10.57 | 8.62 x 4 | | | | -10.5 | | | | |
| august_disorder/1 | 28.28 | -39.10 | -0.56 | 9.71 | 8.57 | 9.66 | 8.69 | 8.72 | -8.78 | 208 | 1.31 | 1.75 / 2.44 | 2.1 |
| august_disorder/2 recorded | 3.88 | 28.24 | -40.26 | 8.14 | 8.62 x 4 | | | | -10.5 | | | | |
| august_disorder/2 | 3.41 | 28.42 | -40.21 | 7.23 | 8.32 | 8.38 | 9.62 | 8.84 | -8.55 | 250 | 1.51 | 2.34 / 3.54 | 2.2 |
| august_disorder/3 recorded | -11.29 | 41.47 | -25.02 | -5.17 | 8.62 x 4 | | | | -10.5 | | | | |
| august_disorder/3 | -11.91 | 41.15 | -25.42 | -6.42 | 8.94 | 8.19 | 9.00 | 8.52 | -9.26 | 256 | 1.89 | 3.78 / 6.09 | 2.7 |

In-band excess: `diagnosis.residual_excess` (in-band residual power over the out-of-band noise,
1 if noise only; the band around the recorded model's levels). Level shift: fitted minus
recorded model levels. 9-12 of 12 starts reach the best chi^2.

T1free (model levels, free amplitudes) leaves less: excess 1.22 (N3), 1.36-2.22 (disorder),
reduced chi^2 1.15-1.45; its parameters are close to T1's on the disorder sets (within about
1 kHz), but on august_N3 they are not fixed (g2 0.1 +- 124 kHz: with zero detunings the star
is degenerate and free amplitudes hide the directions). For comparison (log 2026-09-28): the
in-band excess of C on august_disorder/0 is 0.6-1.7, of the oracle 0.5-1.1.

Found:

- **The recorded Kerr is too large on every set:** -8.5 to -9.3 kHz fitted against -10.5
  recorded (15-20 %; 20-30 errors). This is the largest single model error.
- **The detunings are off by 0.2-1.2 kHz**; on august_N3 all four by about -1.2 kHz together
  (a shift of the storage frame against M1). **The couplings differ by link:** 8.2-9.7 kHz
  against 8.62 (up to 12 %).
- The fitted model moves the levels by up to 2-4 kHz: larger than the 1-2 kHz of the
  diagnosis log, which compared the oracle with the recorded model only where a level was held.
- **The model does not reach the noise:** T1's in-band excess is 1.75-4 (C: 0.6-1.7). With the
  model's amplitudes the rows cannot be fitted to the noise; with free amplitudes (T1free) it is
  closer, 1.2-2.2. So the model misses something (amplitudes more than levels): a check for
  T4 on the September data, where the basis is complete. Real-data T2 (from one decay): 208-256
  us.

## Next

- Use T1 as the start of F (and of T3) on real data: F from T1 on august_disorder, compared with
  F from C. On synthetic data it is the one thing that recovers the hard case.
- F from T1 is slow (10 min per fit; F refines all 35 levels at once). T5's analytic Jacobian
  would help most here. The rest of the subset for T1F waits (`T1_STAGES=F`, resumes).
- Why F from T1 still loses most small gaps (P 0.05 against 0.28): look at the dropped poles.
- The model check on the September data (T4): Kerr, couplings per link, and whether the
  residual excess of the model amplitudes stays.
