# 2026-09-29: pole finding T3, a convex sparse fit

Plan: `docs/qsim/pole_finding_explore.md`, T3. Branch `qsim-analysis`. Scored on T0's fixed set
(`docs/log/2026-09-29_pole-finding-t0.md`).

## What was done

- `fitting/qsim/poles/sparse_fit.py` (fitter T3). All rows on one grid of 0.05 bin, one decay
  (the start's median), amplitudes real and >= 0 per row and grid point, the row offsets fixed
  (C's, from its cached fit). Minimize 1/2 chi^2 + lambda x sum over grid points of the l2 norm
  over rows (a non-negative group lasso). Then: runs of grid spikes to poles, then merge of close
  poles and drop of weak ones by chi^2 (`merge_chi2` 25, as F's `drop_chi2`).
- Solver: cvxpy 1.6.7 (Clarabel; SCS if Clarabel fails, 1 case of 80) on a working set of grid
  points; the optimality test on the whole grid by FFT adds the worst points until none is over
  lambda (exact at the end, within 2%). The whole grid in one cvxpy problem took 6 min for one
  10-row case (2000 points, one decay); the working set takes 5-15 s (10 rows), 70-115 s
  (35 rows). A hand-written FISTA came first (dropped: after 20000 steps the support was still
  moving).
- `analysis_notebooks/pole_finding/sparse_fit.py` (scores, lambda curve, batch cell);
  `tests/test_sparse_fit.py`.
- Fit caches in the set folder: `fits_T3.h5` (80 cases), `fits_T3oracle.h5` (true offsets; 10
  rows all draws, 35 rows draw 0), `fits_T3F.h5` (F's `pursuit.fit` started from T3; draw 0, 10
  rows), `lambda_curve_T3.jsonl`. No shared module changed.

## Why the group norm, not l1

With c >= 0 the l1 norm is linear and, for a diagonal row, almost fixed by the data (sum_lambda
c = a_b(0) = 1): it shrinks every row's total, but does not count grid points. The group norm
(l2 over rows per grid point) shares the support across rows, as the levels are shared. It
also gives lambda a meaning: a grid point enters iff sum_b max(Re <v, r_b> / s_b, 0)^2 > lambda^2,
which is F's candidate gain. So lambda^2 is in F's chi^2 units.

## Found

**One decay only.** With 3 decays (0.7, 1, 1.4 x the median) one level splits into spikes of
different decays: August T2 100 us, 10 rows, true offsets: 13 false poles; one decay: 3.

**The lambda curve** (draw 0, 10 rows, the 8 conditions summed; 214 resolvable gaps):

| lambda^2 | merge | C offsets: found / false | true offsets: found / false | time (s, 8 cases, C offsets) |
|---|---|---|---|---|
| 4 | on | 172 / 26 | 203 / 2 | 487 |
| 9 | on | 170 / 28 | 203 / 2 | 149 |
| 25 | on | 170 / 30 | 202 / 0 | 75 |
| 100 | on | 163 / 21 | 191 / 2 | 28 |
| 400 | on | 96 / 16 | 114 / 9 | 8 |
| 25 | off | 121 / 111 | 166 / 56 | 47 |
| 100 | off | 145 / 47 | 164 / 36 | 21 |
| 400 | off | 93 / 25 | 108 / 17 | 6 |

Without the merge the lasso splits levels (many false poles at small lambda). With it the curve
is flat for lambda^2 <= 25. I used 25 (the default); 100 is a fair choice too (fewer false
poles, 3 times faster, 7 gaps fewer).

**The subset of T0 (draw 0, 10 rows): found of resolvable (real bound) / false poles.**

| point | T2 (us) | offset (kHz) | C | F | T3 | T3F | T3 true offsets |
|---|---|---|---|---|---|---|---|
| August | 100 | 0.5 | 8 / 4 | 8 / 4 | 13 / 4 | 11 / 3 | 18 / 0 |
| August | 100 | 1 | 6 / 5 | 8 / 4 | 9 / 8 | 10 / 4 | 15 / 0 |
| August | 200 | 0.5 | 16 / 2 | 24 / 0 | 23 / 3 | 24 / 1 | 25 / 0 |
| August | 200 | 1 | 14 / 4 | 16 / 3 | 14 / 7 | 16 / 3 | 24 / 0 |
| September | 100 | 0.5 | 21 / 1 | 24 / 1 | 27 / 2 | 27 / 2 | 30 / 0 |
| September | 100 | 1 | 21 / 1 | 24 / 1 | 28 / 2 | 28 / 2 | 30 / 0 |
| September | 200 | 0.5 | 24 / 1 | 28 / 1 | 28 / 2 | 28 / 2 | 30 / 0 |
| September | 200 | 1 | 26 / 1 | 28 / 1 | 28 / 2 | 28 / 2 | 30 / 0 |
| mean | | | 17.0 / 2.4 | 20.0 / 1.9 | 21.3 / 3.8 | 21.5 / 2.4 | 25.3 / 0 |

Median |gap error| / bound: C 1.7, F 0.8, T3 1.1, T3F 0.6, T3 true offsets 0.4. Time per fit:
C 114 s, F 94 s (its part), T3 9 s, T3F 69 s (its part).

**All 5 draws, C against T3 (found of resolvable / false poles), and P(r < 0.25) pooled**
(true 0.184 August, 0.352 September):

| point | rows | T2 (us) | C | T3 | T3 true offsets | P: C | P: T3 |
|---|---|---|---|---|---|---|---|
| August | 10 | 100 | 6.1 / 2.4 | 16.3 / 3.3 | 23.0 / 0.3 | 0.02-0.03 | 0.03-0.05 |
| August | 10 | 200 | 20.9 / 1.6 | 26.0 / 2.0 | 29.3 / 0.1 | 0.09 | 0.12 |
| August | 35 | 100 | 17.2 / 0.8 | 21.7 / 6.2 | (draw 0 only) | 0.01 | 0.15-0.19 |
| August | 35 | 200 | 27.4 / 0.9 | 27.9 / 4.0 | (draw 0 only) | 0.07-0.10 | 0.19-0.22 |
| September | 10 | 100 | 24.9 / 0.5 | 28.9 / 1.0 | 29.9 / 0.1 | 0.16-0.18 | 0.31-0.32 |
| September | 10 | 200 | 27.1 / 0.9 | 29.8 / 2.1 | 31.1 / 0.4 | 0.23-0.25 | 0.32 |
| September | 35 | 100 | 28.1 / 0.4 | 28.8 / 3.3 | (draw 0 only) | 0.26 | 0.38-0.43 |
| September | 35 | 200 | 30.0 / 0.3 | 28.5 / 4.4 | (draw 0 only) | 0.32 | 0.40-0.46 |

(Ranges and means over the two offset widths.)

- **The hardest case (August, T2 100 us, 10 rows) is much better**: T3 finds 14-18 of 26
  resolvable gaps (5 draws), C 6. On draw 0: T3 13 and 9, F 8 and 8. The convex fit does not
  merge what C merged.
- **With the true offsets T3 is near the bound everywhere at 10 rows** (23-31 of 26-31, almost no
  false poles). So what is left is the offsets, not the sparse fit.
- **The offsets from C are the limit, and alternating does not fix them.** C's offsets on the hard
  case are up to 1.8 kHz off. The sparse objective is lower at C's offsets than at the true ones
  (3741 against 3801; chi^2 3424 against 3464): the data with the sparse model prefer the wrong
  offsets, so no alternation of this objective can find the true ones. Tried: offsets per row by
  a scan with the grid poles fixed (3 rounds: 13 found, 9 false, against 12 / 11), and F's refine
  of poles and offsets (1 round, 310 s: 10 found, 14 false). Both off by default.
- **F from T3 (T3F) against F from C**: better or equal in all 8 cases, better in 4 (mean 21.5 against 20.0
  found), false poles 2.4 against 1.9, gap errors smaller (0.6 against 0.8 of the bound). On the
  hard case +2-3 gaps, but F's loop drops some of T3's (13 to 11 on draw 0).
- **35 rows: T3 has more false poles** (2-7 per case, C 0.2-1.6) and finds about as many as C at
  T2 200 us. The false poles push P(r < 0.25) over the truth on September (0.38-0.46 against
  0.352). At 35 rows the merge threshold (chi^2 25) is probably too low: chi^2 grows with the
  row count. Not tuned.
- P(r < 0.25) is less biased than C's everywhere, most at August 35 rows (0.15-0.22 against 0.184;
  C 0.01-0.10), but still low at August 10 rows (0.03-0.12).
- F's `refine` + `prune` from T3's 35 poles is slow (over 9 min, one case); so T3 leaves it to F's
  own loop (T3F).

## Next

- The offsets: a fit that finds them without C, or a joint fit of offsets and grid amplitudes
  that is not the sparse objective (e.g. the rows' own lines: each row's spectrum against the
  shared grid poles, weighed by the Cramér-Rao offset error). T1's Hamiltonian fit might give
  better offsets as a start: worth a try (T3 takes any start PoleFit, `start=`).
- The merge threshold per row count (35 rows), then T3F on the full set.
- Real data (august_disorder) not done yet.
