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

## Later the same day: first look at the real data

guan: before larger benchmarks, get a coarse sense of each fitter on real data.
`analysis_notebooks/pole_finding/first_look.py` (16 s for 25 spectra): A (the 7-1 ensemble
analysis settings), B and E on july_N3, august_N3, august_disorder (4 realizations) and the 7-1
set (19), each in the frame of its analysis notebook. The model's levels come from each
spectrum's detunings, couplings and Kerr; a match is within max(level tolerance, 0.3 kHz).

- **7-1 (100 samples, bin 12.2 kHz):** no fitter matches the model much better than random
  poles of the same count (A 7.0 vs 6.0, B 5.2 vs 4.4); P(r < 0.25) is A 0.01, B 0.10 vs the
  model's 0.24: the small gaps are lost. Synthetic data at the same point with 9 partial rows
  give the same picture (I(r0) A 0.00, B 0.12 vs 0.21), at SNR 15 as at 100: the window, not
  the fitter or the noise, is the limit. B's residual gives a noise near 0.08 per sample
  (SNR about 12-18). The campaign's mean r agreed with theory (0.498); with the small-gap tail
  gone that says nothing about repulsion.
- **august_N3 (complete, 150 samples, bin 2.3 kHz, delta = 0):** B finds every model level but
  one (5.7 kHz, weight 1) with weights near the multiplicities (1.07, 3.12, 0.89, 3.37, 3.02,
  1.08; total 35.1). The 6- and 10-fold multiplets come out split by 1-2 kHz (about a bin):
  the device is apparently not as symmetric as the model (zero detunings, equal couplings).
  A finds 29 poles; 13 have |w| < 0.3.
- **august_disorder (300 samples, 435 us):** A and B find 31-35 poles; P(r < 0.25) A 0.16,
  B 0.15 vs the model's 0.18. The matched poles are off by about 0.25 kHz, the size of the
  known 0.3 kHz gap between the rebuilt and the recorded theory: at this bin, the model's own
  error sets the match.
- **july_N3 (40 samples at 8.2 us):** aliased (Nyquist 61 kHz, levels over -113..+81 kHz);
  aliased multiplets land on each other.
- **A against B:** on the August sets close in residual (0.22-0.30), model match and
  P(r < 0.25). A finds 85-93% of B's poles; B finds 45-77% of A's; A's extra poles are
  mostly weak (the cap of 35 pads). B is 50-100 times faster and needs no cap.

So: the window decides, as the synthetic benchmarks said; B is a serious candidate. On
real data, the weight checks (benchmark 3) are the firmer test; a match to the model
(benchmark 4) needs a model-error floor of about 0.3 kHz.

Two follow-up checks (same notebook, same 16 s):

- **Weights per multiplet** (each pole to its nearest model level): on august_N3 the mean
  |sum - multiplicity| is B 0.28, A 0.42; B's worst is the missed 5.7 kHz level (1.00), where A
  gives 0.17. A leaves 14 poles farther than half a bin from any level (total weight 1.24), B
  one (-0.10). july_N3 (aliased) is poor for both (1.4-1.8).
- **Poles only one fitter finds** (none from the other within a quarter bin): A's are near a
  model level about as often as random poles (august_N3 12% vs 9%, august_disorder 20% vs 16%,
  7-1 31% vs 29%) and weak (median |w| 0.05-0.25): padding, not weak levels. B's are near a
  level about twice as often as random on the August sets (august_disorder 30% vs 16%) and
  heavier; on 7-1 neither is better than random.

So on the real data B is at least as good as A and cleaner; B is the lead fitter.

## Later the same day: B tuned (task 1 of 3)

guan: wrap up the small remaining tasks today (tune B, sigma estimates, benchmarks 3 and 4);
freezing B and switching `MBRDisorderEnsembleExperiment` waits for the new registry entries,
because it changes results other users rely on and has to go through `main`.

`analysis_notebooks/pole_finding/tune_b.py` (66 s): both rank rules x L = 0.5, 0.67, 0.8 N, on
synthetic data at the August point (g 8.615 kHz, K/g -1.22, delta/g 5.80, 300 samples at
1.4509 us, 10 rows, decay 0.01 per us, 20 draws x 2 seeds), then on the real August sets.

- MDL beats the threshold rule at every SNR (at SNR 100: 24.8 of 35 resolved, 3.3 false poles
  with MDL and L = 2N/3; threshold, N/2: 16.6); L matters little (N/2 slightly worse at high
  SNR). On the real sets MDL with L = 2N/3 has the smallest multiplet error on august_N3 (0.28;
  threshold 0.8-1.1) and the P(r < 0.25) closest to the model's on august_disorder (0.15 vs
  0.18). **The defaults stay: MDL, L = 2N/3.**
- B's rank on the real August data (about 32) matches the synthetic rank near SNR 100, not
  near the SNR 13 that B's residual suggests (0.075 per sample): most of that residual is
  probably not white noise but structure the sum-of-exponentials model does not capture.
- At SNR 100 B still loses small gaps on synthetic August data (P(r < 0.25) 0.12 vs 0.22; at
  300, 0.16): the real agreement (0.15 vs 0.18) may be partly luck, with merged pairs and
  false poles cancelling.

## Later the same day: the offset sigma (task 2 of 3)

`fitting/qsim/poles/offsets.py` (`calibration_sigma`, `model_offset_sigma`; tests recover
injected offsets to 0.05 kHz) and `analysis_notebooks/pole_finding/sigma.py` (36 s):

| | floor (Stark slope errors) | upper bound (per-row offset vs model) |
|---|---|---|
| August | median 0.50 kHz, max 0.79 (0.22-0.34 bin) | N3 0.59 kHz (0.26 bin); disorder 1.06 kHz (0.46 bin) |
| 7-1 | median 0.89 kHz, max 2.1 (0.07-0.18 bin) | 2.2 kHz; 71 of 184 rows match no offset in +-5 kHz (large model misfit) |

Floor and upper bound are close on the August sets, so sigma is about 0.5-1 kHz there:
0.2-0.5 of the August bin.

What it costs B (synthetic, August conditions, SNR 100, 20 draws x 2 seeds):

| sigma | resolved (of 35) | false poles | P(r < 0.25) found / true |
|---|---|---|---|
| 0 | 24.8 | 3.3 | 0.12 / 0.22 |
| 0.5 kHz | 16.7 | 8.8 | 0.15 / 0.22 |
| 1.06 kHz | 7.3 | 20.0 | 0.19 / 0.22 |

The last row is the pattern of the real august_disorder fits (about 32 poles, about 11
matched, about 20 false, P(r < 0.25) 0.15 vs 0.18). **So the real agreement of P(r < 0.25) is
most likely made by the row offsets**: their false poles add small gaps and cancel the
merging. It is not evidence that the statistic is measured right. Consequences: fitter C (one
offset per row group, with the calibration prior; spec 4.2) is needed; and the 1-2 kHz
splitting of august_N3's degenerate multiplets may be row offsets rather than an asymmetric
device (a hypothesis for C to test).

## Later the same day: benchmarks 3 and 4 (task 3 of 3)

- The registry entries now carry their analysis frame (`registry.Analysis`), and
  `experiments/qsim/pole_data.py` (`load_spectra`) builds each spectrum and its model from the
  entry alone (in `experiments/`, as it needs the experiment classes).
- `fitting/qsim/poles/real_benchmarks.py`: `run_self_consistency_bench` (residual, d,
  P(r < 0.25) and after barycenter re-merging at 1.25 and 1.5 lambda_eff, weights per
  multiplet or heavy poles) and `run_model_bench` (match within max(level tolerance, 0.3 kHz)
  vs random, errors, P(r < 0.25) vs the model). Notebook
  `analysis_notebooks/pole_finding/real_benchmarks.py` (32 s).
- Fitter A has two presets now: `CAMPAIGN_SETTINGS` (the class default, 0.1-bin dedup), and
  `ANALYSIS_SETTINGS` (the 0.5-bin tolerances the 7-1 notebook ran, from which the saved 7-1
  results come). With the first, A's weights on the August sets blow up (multiplet error
  1.6e4, weight errors to 1e8: near-duplicate poles in the least squares); the real
  benchmarks use the second.
- B's lambda_eff at each set's conditions (benchmark 2, SNR 100, sigma at the floor): 0.88 bin
  (7-1), 1.0 bin (August). With it, d = 0.30-0.38 on both disorder sets (flag: 0.25), and
  re-merging at 1.25 lambda_eff drops P(r < 0.25) from 0.15 to 0.01 on august_disorder:
  **benchmark 3 flags both disorder sets as at their resolution limit, with no model.**
- august_N3: multiplet error A 0.42 (14 poles more than half a bin from any level), B 0.28 (1).

## State at the end of the day

Tasks 1-3 are done. B (MDL, L = 2N/3) is the lead fitter; A adds only padding poles. The real
limit is the data: the 7-1 window, and on the August sets the row offsets (sigma 0.5-1 kHz,
0.2-0.5 bin), which turn merged pairs into false poles and make P(r < 0.25) look right for the
wrong reason. Next: fitter C (spec 4.2: B plus one offset per row group with the calibration
prior), tested on synthetic August data with sigma 0.5-1 kHz and then on august_N3 (do its
split multiplets close?). Freezing a fitter and switching `MBRDisorderEnsembleExperiment` wait
for C and for the new registry entries.

## Correction, end of the day: A against B under row offsets

guan asked whether B really loses most levels under realistic offsets. Same synthetic August
conditions (SNR 100, 20 draws x 2 seeds), A (`ANALYSIS_SETTINGS`) against B:

| sigma | resolved A / B | matched A / B | false poles A / B | P(r < 0.25) A / B (true 0.22) |
|---|---|---|---|---|
| 0 | 25.9 / 24.8 | 29.3 / 29.0 | 5.1 / 3.3 | 0.14 / 0.12 |
| 0.5 kHz (floor) | 22.9 / 16.7 | 27.8 / 24.3 | 6.7 / 8.8 | 0.17 / 0.15 |
| 1.06 kHz (upper bound) | 8.5 / 7.3 | 17.2 / 15.7 | 17.9 / 20.0 | 0.22 / 0.19 |

At the floor A holds up clearly better: a row's offset shifts only that row's poles, and A's
clustering across rows absorbs it, while B's shared-pole model splits a smeared level. At the
upper bound both collapse alike (the data smear dominates). **This corrects "A adds only
padding" and "B is the lead fitter" above**: without offsets B equals A at 50-100x the speed;
with realistic offsets B needs its offset correction (fitter C), and a per-row fitter without
A's defects (fitter D) is a real candidate again. Next session: C and D against A under
sigma = 0.5-1 kHz.

## Later the same day: fitters C and D, sketched

guan: sketch C and D now. `fitting/qsim/poles/joint_refined.py` (C: B, then nonlinear least
squares over E, gamma and one offset per row group with a Gaussian prior of width
sigma_cal = 0.5 kHz, amplitudes by variable projection; two pencil-then-refine rounds, the
second on offset-corrected rows) and `per_row_clustered.py` (D: pencil and MDL rank per row,
row poles clustered within 0.5 bin, one per row per cluster).

Synthetic August conditions (SNR 100, 3 draws only), levels resolved of 35 / false poles:

| sigma | A | B | C | D |
|---|---|---|---|---|
| 0 | 23.7 / 5.7 | 25.7 / 2.3 | 26.3 / 2.7 | 20.3 / 12.0 |
| 0.5 kHz | 20.7 / 9.0 | 12.0 / 12.0 | 26.3 / 3.7 | 14.7 / 16.3 |
| 1.06 kHz | 8.7 / 19.7 | 6.3 / 21.3 | 25.7 / 3.7 | 6.0 / 28.7 |

C removes the offset loss (for offsets of the form it assumes: one constant per row). D as
sketched is worse than A (false poles). Known limit of C (strict xfail test): with few rows
the pencil splits each level into one pole per offset row, and the split absorbs the offsets.

Real August sets (21 min, almost all C):

- august_N3: matched 9 of 10 (A 8, B 6), multiplet error 0.16 (A 0.42, B 0.28), no pole far
  from a level; C finds the weight-1 level at 5.7 kHz (0.81) that A and B miss. The 6-fold
  multiplets stay split by about 1.2 kHz and the 10-fold by about 1 kHz after the offsets are
  fitted: **the splitting is partly real** (an asymmetric device), not only row offsets.
- august_disorder: matched 12 (A 12, B 11; random 4.7); P(r < 0.25) **0.09** (B 0.15, model
  0.18). With the offsets fitted, the false poles that inflated B's value are gone; what is
  left is the merging loss (C on synthetic data: about 0.13 vs 0.21). So B's 0.15 was
  inflated, and about 0.09 is the honest value on this data set.
- C's residual (0.58-0.69) is not comparable: `fit_residual` models the rows without C's
  offsets. To fix: return the offsets with the fit.

**C is now the lead candidate.** To do: an analytic Jacobian (C takes 7-33 s per fit on
synthetic data, minutes on august_N3); return the offsets (residual, benchmark 3); the
few-rows limit; then C on the full benchmark 2 and the new registry entries. D needs
rework or can be dropped.

Plots of the C and D comparison: `analysis_notebooks/pole_finding/offsets_cd.py` (14 min),
HTML at `C:\experiments\260818_qsim_spectroscopy\derived_data\pole_finding\260928_193122_offsets_cd\`.
With 5 draws the synthetic result holds (resolved of 35 at sigma 0 / 0.5 / 1.06 kHz: A 25.8 /
20.6 / 7.8, B 25.6 / 14.4 / 6.2, C 27.0 / 26.8 / 26.2, D 23.0 / 15.8 / 5.8). Even C's
P(r < 0.25) stays at about 0.10-0.11 vs the true 0.18 at every sigma: the offsets fixed, the
window-and-noise merging loss remains. A's P(r < 0.25) crosses the true value inside the
measured sigma band only because its false poles grow.
