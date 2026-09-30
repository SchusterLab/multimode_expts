# Pole finding: exploration plan (tasks for parallel sessions)

Status: plan, 2026-09-29 (guan and Claude). Tasks for the next sessions; each can run in its own
worktree from `guan`. The method spec stays `docs/qsim/pole_finding.md`; what was found and why
is in `docs/log/2026-09-28_pole-finding-diagnostics.md`. When a task ends, its session adds a
log entry, and the spec takes over what is kept.

## Where we are

- The goal is the small gaps of the spectrum (P(r < 0.25)), on the August disorder sets (10 of
  35 occupations) and on future sets.
- The per-level diagnosis (spec 5.5) and the Cramér-Rao design calculator (spec 5.6) say: on
  august_disorder/0 (T2 about 200 us) 24 of 35 levels are resolvable with the fitters' free
  complex amplitudes, 33 with real amplitudes (`<b|P|b>`, true of diagonal rows; the August data
  are consistent with it). T2 dominates everything else; the window (about 2 T2) is right.
- Fitters: C (pencil, then joint refinement with row offsets) is the best so far; F (pursuit:
  poles added and dropped by chi^2, real non-negative amplitudes) finds 1-2 more levels than C
  on synthetic August data, 2-7 min per fit.
- Row alignment by the rank of the stacked Hankel matrix (guan's "visual continuity" idea,
  tried 2026-09-29): its minimum is at the true offsets and wide, but shallow; C's offsets are as
  good or better, and C matches the pencil given the true offsets. **Even with the true offsets
  the pencil resolves 22-33 of 35 (35 rows) and 11-18 (10 rows): the offsets are not the limit;
  the pencil and what it assumes are.**
- Real complete-basis disorder data exist and are not converted: September 10-14, N=3, 9
  realizations x 35 occupations, K about 3.6 kHz, g 29.2 kHz (jonginn,
  `docs/job_list_and_nb_labeling/Job_list.md`).

## Rules for every task

- One new module per task under `fitting/qsim/poles/`, one notebook under
  `analysis_notebooks/pole_finding/`, tests in its own file; do not edit another task's module.
  Shared modules (`pole_fit`, `diagnosis`, `design`, `synthetic`) only by T0, or by a small
  additive change named in the log.
- Every fitter keeps the interface `fit(A, time_us, settings, row_groups=None) -> PoleFit`, so
  that `display_pole_fit`, `diagnose_levels` and the T0 score take it unchanged.
- Score on T0's fixed synthetic set first, real data second. Report against the Cramér-Rao
  bound of the same design (spec 5.6), so that "how far from possible" is always visible.
- Commit as you go; a log entry per session.

## T0 (first; blocks the comparisons of the others): a gap score and a fixed benchmark set

The score of benchmarks 2-4 asks for absolute positions within 0.3-0.57 kHz; with free
offsets a level is only fixed to about 0.4 kHz absolute, while the statistics need only the
gaps (log, 2026-09-29, fitter F). So:

- `fitting/qsim/poles/gap_score.py`: match in order (tolerance half the nearest gap); per
  adjacent gap its found value, error, and error over its Cramér-Rao error; small gaps (under
  half the mean) resolved; false poles; P(r < 0.25) found against true.
- A fixed synthetic set, saved once (HDF5 with seeds): the August point and one point of the
  September regime (K/g about 0.12), T2 100 and 200 us, offsets 0.5 and 1 kHz, 10 rows and 35
  rows, 5 draws each; C's fits cached (C is slow).
- Done when: C and F are scored on it, and the table is in the log.

## T1: a fit of the Hamiltonian (guan: worthwhile; may not be the final choice)

- `hamiltonian_fit.py`: fit detunings (per mode), couplings (per link), Kerr, one decay and the
  row offsets to all rows at once; the levels are the model's eigenvalues and the amplitudes its
  `|<b|lambda>|^2` (or: the model's levels with free non-negative amplitudes). Many starts
  (about 15 parameters; the recorded values plus spread).
- Uses: (a) the model check (how far off are the recorded detunings, couplings, Kerr; the
  1-2 kHz shifts of the diagnosis); (b) a start for a free fit (F, T3).
- The logic caveat (guan): levels from a model fit carry the model's statistics; for P(r <
  0.25) only as a start or a check, not as the answer. Say which in every result.
- Done when: on synthetic data with a perturbed model it recovers the parameters, and on
  august_N3 and august_disorder it gives the parameters and the in-band residual.

## T2: the full weight structure of a complete basis

For a complete diagonal basis `W_{b,lambda} = |<b|lambda>|^2` is real, >= 0, every row sums to
1 and every level's column to its multiplicity (doubly stochastic up to the multiplicities).
The design calculator has only the column sums, with local bounds.

- First the bound: a global (not local) Fisher matrix for 35 rows x 35 levels with both sums
  (`design.py`, additive: a `row_sums` option), at the August point and the September regime.
  Does the full structure buy levels over "real" alone?
- Only if it does: a fitter option (constrained non-negative least squares, or a Sinkhorn step
  on the amplitudes) in F or T3.
- Done when: the bound table (10 / 35 rows; real / real + sums / real + both sums) is in the
  log, and the go / no-go for a fitter option.

## T3: a convex sparse fit (global for fixed offsets)

- `sparse_fit.py`: all rows on one fine frequency grid (0.05-0.1 bin), a few decay values,
  amplitudes real and >= 0 per row and grid point; minimize chi^2 + lambda x (number of grid
  points used), convex as a non-negative group lasso. The offsets from C at first; later
  alternate (offsets <-> sparse fit). Grid spikes clustered, then F's `refine` for the final
  poles.
- Check first whether a convex solver is in the pixi environment (cvxpy); else FISTA / active
  set by hand.
- Done when: scored on T0 against C and F, and lambda chosen by false poles against levels.

## T4: convert the September complete-basis disorder data (on pippin)

- 9 realizations x 35 occupations (630 spectroscopy jobs) and their calibrations, from
  `Job_list.md`, with `tools/migrate_mbr_jobs.py`; registry entries
  (`analysis_notebooks/pole_finding/registry.yaml`) with the analysis frame.
- Then the diagnosis (spec 5.5) and the design calculator on them: how many levels are
  resolvable with 35 rows at this Kerr and T2. The data answer "is the complete basis worth
  measuring every time" (guan: if yes, measure it every time).
- Owner: guan (with jonginn for the job lists). Only T4 edits the registry.

## T5: speed

- An analytic Jacobian (variable projection, Golub-Pereyra) for the refinement shared by C,
  F and the oracle; C takes 30-120 s and F 2-7 min per fit, which limits T0 and the benchmarks.
- Done when: the same fits to 1e-6 bin, and the time per fit in the log.

## Not now

- Off-diagonal data (orthogonality, Hamiltonian tomography): measuring time and no Stark
  calibration for off-diagonal entries (guan, 2026-09-29).
- More work on row alignment by rank: C already does as well (above). Kept only as an offset
  step inside T3 if T3 needs one.

## Order

T0 first (about half a day). Then T1, T2, T3, T5 in parallel, and T4 whenever pippin time
allows. After them: choose the fitter (and the design: 10 or 35 rows) on the T0 score and on the
September data, freeze it, and switch `MBRDisorderEnsembleExperiment` (through `main`).
