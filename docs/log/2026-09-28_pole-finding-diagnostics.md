# 2026-09-28: pole finding, a figure for any fitter

Session record; not edited after this date. The current state is `docs/STATUS.md`; the
design is `docs/qsim/pole_finding.md`. On pippin, `guan` worktree. Follows
`2026-09-28_pole-finding-phase1.md` (same day).

## Decisions

guan and Claude, at the start: none of A-D does well enough on the real data, and the end
scores (levels found) do not say why. Before more fitters, find the cause of each miss:

1. A figure that takes the result of any fitter (this session).
2. One table per data set, one row per model level, with the checks in order: is the level
   in the measured rows (model weight); can any method resolve it (Cramér-Rao bound for its
   neighbours, weight, SNR, window, decay); do the data hold it (refinement started at the
   model levels, with offsets); is the model a little off (fit K, g, detunings to the found
   poles); does fitter X find it. The first "no" is the cause. The same on synthetic data at
   each set's conditions, one non-ideality at a time.
3. Then choose where the work goes: fitter, calibration, model or acquisition.

Two ideas stay small checks inside step 2, not new fitters:

- **Summed trace + one pencil:** a weaker case of B. The same noise gain, but the sum loses
  the row-to-row difference of amplitudes that lets B split close levels. Row offsets make
  each level a non-exponential decay in the sum (about exp(-2 pi^2 sigma^2 t^2); at 1 kHz and
  435 us, 0.025 left), which the pencil pads with false poles. C's offsets cannot be applied
  after the sum; correcting the rows first and then summing is only a check of C's offsets.
- **ESPRIT:** B's `shift_invariance_poles` is least-squares ESPRIT, and stacking the rows is
  multi-channel ESPRIT. The total-least-squares form is at most an option of B.

## Done

- `PoleFit` has the offset of each row (`row_offsets_MHz`, None but for C) and
  `returns(time_us)`, the fitted rows with the offsets. C fills the offsets.
  `real_benchmarks.fit_residual` uses `returns`, so C's residual is now right (it modelled
  the rows without the offsets before). `RealSpectrum` has the rows' occupations.
- `fitting/qsim/poles/pole_plots.py`, `display_pole_fit`: per-row FFT of the data and of the
  residual, found poles (at each row's own offset, size |c_b|) and model levels (where the row
  has weight above 0.05); the row sums with found weights up and model weights down; the
  residual per row. Zoomed on the model's span by default. On a spectrum:
  `MBRSpectrumExperiment.display_poles(fit)`.
- `analysis_notebooks/pole_finding/diagnose.py`: A, B and C on august_N3, august_disorder/0
  and 7-1/0 (about 5 min, almost all C).
- Tests (`tests/test_poles.py`): C returns injected offsets (8 rows, SNR 100, to 0.02 bin);
  `returns` applies the offsets; the figure draws for B and C.

## Found

- **C fails at high SNR** (new strict xfail `test_joint_refined_returns_the_row_offsets_at_high_snr`):
  8 rows, 4 levels, offsets 0.1 bin. At SNR 100 C finds the 4 levels and the offsets to
  0.007 bin; at SNR 1000 B's rank is 12 (each level smeared into 3 poles), and the refinement
  keeps the split, which absorbs the offsets. So C's limit is not only "few rows": the better
  the data, the more of the smear B keeps as poles.
- **C's offsets on august_N3 follow the first mode's occupation:** rows with a photon in it
  +0.63 +- 0.40 kHz, the others -0.47 +- 0.44 kHz. A random calibration error would not do
  that. A hypothesis for step 2: a model error (for example a first-mode detuning) that C
  takes up as offsets. Spread of C's offsets: august_N3 0.69 kHz, august_disorder/0 1.05,
  7-1/0 1.16 (the prior is 0.5 kHz).
- First look at the figures (not yet analyzed): on the August sets the residual (0.2-0.3 of
  the signal) is spread over the whole band, with some rows much worse (august_N3
  (0,0,0,3,0)). On 7-1/0 the residual has peaks on the strongest data lines of several rows:
  there the line shape, not a missing pole, is not fitted.

## Next

Step 2: the per-level table (spec to be written in `pole_finding.md`), first on august_N3.
