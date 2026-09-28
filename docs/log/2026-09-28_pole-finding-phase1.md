# 2026-09-28: pole finding, phase 1

Session record; not edited after this date. The current state is `docs/STATUS.md`; the
design is `docs/qsim/pole_finding.md`. On pippin, `guan` worktree. Follows
`2026-09-28_pole-finding-design.md` (same day, on the MacBook).

## Done

Phase 1 of the spec, all in `fitting/qsim/poles/` (13 modules, each under 160 lines):
`PoleFit`; the rank rules (`rank.py`: threshold and MDL); fitters A (`per_row_reconciled`,
the frozen current code with the 7-1 campaign settings), B (`joint_pencil`) and E
(`fft_peaks`); the model generator and phase-diagram sampler (`synthetic.py`); matching
(`matching.py`); the small-gap-ratio statistic (`statistics.py`); benchmarks 1 and 2
(`benchmarks.py`), their tables (`bench_summaries.py`), plots (`bench_plots.py`) and HDF5
output (`bench_io.py`); the data set registry format (`registry.py`,
`analysis_notebooks/pole_finding/registry.yaml`, 4 entries).

Report: `analysis_notebooks/pole_finding/report.py`, headless with `pixi run pole-report`
(`tools/pole_report.py`). The small run (A, B, E; about 17 min, fitter A is 95% of it) is at
`C:\experiments\260818_qsim_spectroscopy\derived_data\pole_finding\260928_154200_small_ABE\`.

Tests: `tests/test_poles.py`, `tests/test_pole_bench_ideal.py` (benchmark 1),
`tests/test_pole_finding_spec.py` (every function the spec names exists): 22 tests, about 8 s.

## Decisions

- guan: the notation is K (Kerr), g (coupling), delta (disorder strength); a disorder draw
  is a zero-mean unit vector as in `mbr_disorder.disorder_direction`. The synthetic cases
  sample the (K/g, delta/g) plane.
- guan: the score is P(r < r0) of the gap ratio (no unfolding), not P(s < s0).
- guan: benchmark 2 keeps two questions apart: how good a method is (long windows, where
  the answer can be found), and what the measured grid allows (100 samples, T2 decay). In
  the experiment the traces can run long, but past about T2 = 100 us they are flat.
- Claude, written in the spec: benchmark 1 holds B to 1e-6 bin only where the Vandermonde
  conditioning of the true levels on the grid is at least 1e-2, and runs on 400 samples; A
  and E are scored but not held to it. Joint pencil defaults: MDL, floor 1e-12, L = 2N/3.
- Claude, written in the spec: each level is matched within min(0.25 bin, 1/4 of its
  nearest separation). With a fixed 0.25 bin (near the mean spacing), 35 random poles
  "resolved" 10-20% of the levels closer than a bin, and fitter A 60% of those closer than
  0.25 bin; with the cap, random poles resolve 1-8%.

## Found

- **The measured window cannot resolve all 35 levels, for any fitter.** At the 7-1 point
  (K/g = -2.9, delta/g = 3.3; 100 samples of 0.818 us, bin 12.2 kHz) the 35 levels lie in
  about 25 bins, mean spacing below one bin. The singular values of the clean pencil past
  about the 25th are at machine precision. Joint pencil (MDL) rank on clean / SNR 1000 /
  SNR 30 data: 28 / 19 / 17 at 100 samples, 35 / 28 / 26 at 200, 35 / 35 / 34 at 400.
- Benchmark 1 (400 samples, 40 cases): B is exact (below 1e-6 bin) at every case with
  conditioning at least 1e-2.
- Benchmark 2a (SNR 100, 20 points x 2 seeds; means over the plane; "levels" 29.8 on
  average, as delta = 0 has 7-10):

  | samples | resolved A / B / E | false poles A / B / E | lambda_eff (bins) A / B | I(r0) bias A / B / E |
  |---|---|---|---|---|
  | 100 | 11.5 / 6.0 / 0.6 | 14.5 / 6.0 / 4.9 | 0.5 / 0.75 | +0.05 / -0.05 / -0.12 |
  | 200 | 19.2 / 12.8 / 3.5 | 8.2 / 5.5 / 5.4 | 0.125 / 0.375 | +0.10 / -0.04 / -0.18 |
  | 400 | 24.8 / 21.0 / 7.4 | 4.8 / 2.8 / 6.4 | 0 / 0.125 | +0.09 / -0.02 / -0.16 |

  (no decay; the T2 decay of 0.01 per us changes these little at SNR 100, even at 400
  samples, where the signal has fallen to 4% but is still above the noise.)
- So: A (per row, 35 poles forced) resolves close levels better than B, but half its poles
  are false on the measured grid, and its I(r0) bias goes *positive* (more Poisson) and does
  not shrink with a longer window. B's bias is negative (merging, more repulsion) and goes
  to 0 as the window grows (std 0.09 at 400). E is far behind. The per-row advantage
  suggests fitter D (per-row pencil, rank rule, clustering) is worth writing; B at the
  measured grid is limited by its rank rule stopping at about 18.
- Benchmark 2b (100 samples, decay 0.01 per us): offsets of sigma = 0.25 bin (3 kHz)
  raise B's false poles per case from 7 to 12 and lower its resolved levels from 5 to 3; A's
  resolved levels fall from 11 to 5.

## Next

- Tune B on benchmark 2 (rank rule, pencil length); write fitter D.
- Estimate the real SNR per data set, to place the 7-1 data on these curves.
- Phase 2: benchmarks 3 and 4 on the registry; the three sigma estimates.

## Later the same day: speed

The report took 17 min, 95% of it fitter A. A profile of one A fit (0.5-0.7 s, 35 rows,
100 samples) shows no AttrDict cost: the time is the 3675 eigenproblems of its rank sweeps
(one per rank per row, part of the algorithm) and about 220k scalar Python calls in its pole
tracking (`_track_poles`: `wrap`, `distance`). A is frozen, so it was not changed. Instead
(`benchmarks.score_cases`): the fits run in 8 worker processes with one BLAS thread each
(on these small matrices BLAS threads only add overhead: A at 400 samples 1.6 s on one
thread, 3.2 s on the default). A 72-fit test: 30 s serial, 4.3 s parallel; the poles agree to
1e-12 bin, the scores exactly. 8 workers, not 32, because this PC's CPU is suspected in its
crashes.

Then the parallel report failed: a worker died in benchmark 2b. With `faulthandler` in the
workers: `Windows fatal exception: code 0x80000003` (a breakpoint trap), always at
`matrix_pencil.py:326` in `_track_poles` (a numpy scalar read), at a different task each run
(tasks 168, 240, 1563, 642 of 1800), 5 to 47 s into 8 workers, in 5 of 5 parallel runs.
Serial runs never crashed (several thousand A fits, about 50 CPU-min). An 8-process stress
test of similar numpy loops without our code ran 24 s without a crash. No Windows
Application Error event. The load dependence fits the suspected CPU fault (see the
workstation notes: 5 BSODs, CPU hardware leading); the fixed line fits a software cause as
well; not settled. Load tests stopped, because a BSOD on pippin also stops the job worker
and server. `benchmarks.WORKERS` is 1 (serial) until this is understood; the parallel
code path and its test stay.
