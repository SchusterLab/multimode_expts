# 2026-09-29 (evening): T3 started from T1, and the review of T1 and T3

T1 (`docs/log/2026-09-29_pole-finding-t1.md`) and T3 (`docs/log/2026-09-29_pole-finding-t3.md`)
ran as two parallel agents in this worktree; the main session reviewed and committed them. guan
allowed new dependencies (a library over hand-written numerics): cvxpy 1.6.7 is in the
environment (1.8 fails on import with our numpy < 2).

## What was done

Both reports named the same next step: T3 is limited by C's row offsets (up to 1.8 kHz wrong in
the hardest case; the sparse objective itself prefers them), and T1 finds the offsets to
0.02-0.05 kHz on synthetic data. So T3 with T1's fit as the start (`sparse_fit.fit(...,
start=<T1 fit>)`), on the subset (draw 0, 10 rows). Notebook
`analysis_notebooks/pole_finding/combined_starts.py`; cache `fits_T1T3.h5`.

## Result

Resolvable gaps found (real bound) / false poles, and P(r < 0.25) found (per case, 34 levels:
noisy). T1 alone is model-derived (best case, not a fitter result); T3oracle has the true
offsets.

| case | resolvable | C | F | T3 | T3F | T1F | T3oracle | T1T3 | P true |
|---|---|---|---|---|---|---|---|---|---|
| August T2 100, 0.5 kHz | 22 | 8/4 0.00 | 8/4 0.00 | 13/4 0.12 | 11/3 0.06 | 19/0 0.05 | 18/0 0.05 | 19/0 0.10 | 0.28 |
| August T2 100, 1 kHz | 20 | 6/5 0.00 | 8/4 0.00 | 9/8 0.12 | 10/4 0.00 | 18/2 0.05 | 15/0 0.05 | 17/0 0.10 | 0.28 |
| August T2 200, 0.5 kHz | 27 | 16/2 0.32 | 24/0 0.09 | 23/3 0.39 | 24/1 0.23 | - | 25/0 0.10 | 26/0 0.14 | 0.28 |
| August T2 200, 1 kHz | 25 | 14/4 0.26 | 16/3 0.14 | 14/7 0.32 | 16/3 0.23 | - | 24/0 0.10 | 24/0 0.10 | 0.28 |
| September T2 100, 0.5 kHz | 30 | 21/1 0.25 | 24/1 0.36 | 27/2 0.48 | 27/2 0.48 | - | 30/0 0.62 | 30/0 0.62 | 0.64 |
| September T2 100, 1 kHz | 30 | 21/1 0.25 | 24/1 0.36 | 28/2 0.52 | 28/2 0.57 | - | 30/0 0.62 | 30/0 0.62 | 0.64 |
| September T2 200, 0.5 kHz | 30 | 24/1 0.50 | 28/1 0.59 | 28/2 0.58 | 28/2 0.58 | - | 30/0 0.62 | 30/0 0.62 | 0.64 |
| September T2 200, 1 kHz | 30 | 26/1 0.48 | 28/1 0.55 | 28/2 0.62 | 28/2 0.58 | - | 30/0 0.62 | 30/0 0.62 | 0.64 |

Time per fit: T1 5-10 s, then T3 3-5 s (C 20-350 s, F from T1 about 600 s).

Found:

- **T1T3 is as good as T3 with the true offsets, in every case, with no false poles**: the
  offsets were the whole limit of T3. On September it finds all resolvable gaps and P(r < 0.25)
  within 0.02 of the truth. And it is the fastest route (about 10 s, no C).
- **On August P(r < 0.25) stays low (0.10-0.14 against 0.28), and this is mostly the data, not
  the fitter:** of the 12 small gaps of draw 0, only 0-1 (T2 100 us) and 3-5 (T2 200 us) are
  resolvable at all (real bound). No fitter can give the true P there; the target on such data
  is "P of what is resolvable", and the missing small gaps must be counted (or corrected for) in
  the statistics, not fitted.
- F after T3 (T3F) or after T1 (T1F) adds nothing over T1T3 and is much slower.

Caveat: T1's offsets are right here because the synthetic data come from the model T1 fits. On
real data the model is off (Kerr 15-20 %, residual 1.8-4x the noise; T1 log), and T1's offsets
may absorb some of that. Real data are the test.

## Next

- T1T3 on the real sets (august_disorder, august_N3), with the in-band residual and the
  diagnosis (spec 5.5) as the check; then on all 80 fixed-set cases (cheap: about 10 s each; 35
  rows more).
- T3's merge threshold per row count (false poles at 35 rows, T3 log).
- The statistics side: P(r < 0.25) from partly resolved spectra (count the unresolvable small
  gaps from the bound, or compare with the model's P restricted to what is resolvable).
- T5 (speed) matters less now: the fast route needs neither C nor F.
