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

## Later the same day: step 2, why each level is missed

guan: commit step 1 (`1a23362`), then step 2. `fitting/qsim/poles/diagnosis.py` and
`resolution.py` (spec 5.5), `display_causes`, notebook `analysis_notebooks/pole_finding/level_causes.py`.
One row per model level: in the rows (weight >= 0.1); resolvable (both adjacent gaps at least
4 Cramér-Rao errors, with free row offsets, prior 0.5 kHz); held (C's refinement started at
the model: weight, place, drop-one chi^2 >= 25); found. The noise per row is measured out of
band.

Three fixes on the way, all in the spec:

- The full Fisher matrix of the 7-1 levels is beyond double precision (bounds of 1e14 kHz);
  each bound now uses only the levels within 3 bins, and a local matrix above condition 1e12
  counts as no bound.
- One oracle pole per unresolvable cluster (at its weighted mean) pulled the neighbouring
  oracle poles away by 10-50 kHz; the oracle now starts at every level and merges only as far
  as its pole matrix needs (condition 1e8; on these sets that merges only on 7-1).
- C's poles on august_disorder sit about 1 kHz above A's and B's for the levels at 70-90 kHz,
  with the mean offset near 0: a level held by a few rows moves with those rows' offsets. So
  all fitters are compared where the rows see a pole (`seen_frequencies`: E plus the
  |c|-weighted mean offset of its rows). Before this, C "missed" levels it had found.

| | levels | not resolvable | resolvable: found A / B / C | resolvable: search A / B / C |
|---|---|---|---|---|
| august_N3 | 10 | 0 | 9 / 7 / 10 | 1 / 3 / 0 |
| august_disorder/0 | 35 | 18 (6 clusters of 2-5) | 11 / 14 / 13 | 6 / 3 / 4 |
| 7-1/0 | 34 (+1 not in rows) | 32 (in clusters) | the 2 edge levels | 0-1 |

(Fitters also match some levels inside clusters, by luck of placement; those are "found" in
the table of the notebook.) No resolvable level failed "held" on the August sets after the
fixes.

So:

- **august_N3:** the fitter was the limit. All 10 levels are resolvable (gap at least 22 of
  its errors) and held; C finds all 10.
- **august_disorder/0:** the data are the limit. About half the levels (18 of 35) are in
  clusters no method can split at this window, SNR and offset prior; the fitters lose only 3-6
  more. A better fitter can gain at most about 4 levels here.
- **7-1/0:** 32 of 34 levels are one chain of unresolvable gaps: the set cannot give level
  statistics, with any fitter.
- **The model is off by more than the assumed 0.3 kHz:** oracle minus model on
  august_disorder/0 is +0.8 to +1.3 kHz for the five top levels (58-90 kHz; Cramér-Rao
  errors about 0.5 kHz) and up to 2 kHz elsewhere; on august_N3 up to 0.9 kHz. The oracle's
  in-band residual on august_N3 is 0.5-1.9 times the noise, mean about 1.2 (the split multiplets, which one
  pole per model level does not take).
- **The August residual is mostly noise:** C's and the oracle's in-band residual is about the
  out-of-band noise (excess on august_disorder/0: C 0.6-1.7, oracle 0.5-1.1). The large relative residual
  (0.2-0.3) is white noise, SNR about 13 per sample, not structure. B and C on 7-1/0 leave 2-6
  times the noise in band (the unresolved clusters).

Open: the thresholds (4 errors, chi^2 25) are choices; the counts move with them. The CR
bound assumes the model's amplitudes; the offset prior sets how much the offsets cost.

Next (step 3): choose where the work goes. From this table: for the disorder sets, the data
(window, SNR, offsets), not the fitter; a model check (refit detunings, couplings, Kerr to the
held levels) because the model errors reach 1-2 kHz.

## Correction, same evening: the decay, and what the eye sees

guan: a bare `display()` of august_disorder/0 (`260924_195535_MBRSpectrumExperiment`, the
same spectrum) shows data and theory alike; half the peaks are not missing. Are the eyes
wrong?

No: the table above was too pessimistic. **The Cramér-Rao bounds used an assumed decay of 0.01
per us; the data's (the oracle's median) is 0.005 per us (T2 about 200 us).** The bounds now
use the oracle's decay (`diagnose_levels` runs the oracle first). august_disorder/0, 35 levels:

| | resolvable (gaps >= 4 CR errors) | at >= 2 errors |
|---|---|---|
| decay 0.01 (before), offsets free | 17 | 24 |
| decay 0.005, offsets free (prior 0.5 kHz) | **24** | 33 |
| decay 0.005, offsets known | 31 | 33 |

Of the 24 resolvable levels, A finds 15, B 18, C 19 (search misses 8 / 5 / 4); one is not held.
The 11 not resolvable are 4 pairs and 1 triple, gaps 0.6-2.3 kHz (0.3-1 bin), CR error of the
gap 0.4-0.6 kHz. 7-1/0 is unchanged (its decay is 0.014 per us; 32 of 34 in one chain).
august_N3 unchanged.

Why the eye and the display agree with the model: per row, the theory's finite-time FFT shows
nearly every level the row holds (53 peaks for 52 level-row entries with weight > 0.05; 4-7 per
row), and the close pairs live mostly in *different* rows (row overlap 0.1-0.45 for 5 of the 6
unresolvable pairs). So each row shows its own peaks at the right place, and the display, which
compares with the theory at the same finite-time resolution, looks right. What the data cannot
give is the gap of such a pair to better than about 0.5 kHz: the level statistics need exactly
those small gaps. The offsets matter because a pair held by different rows can trade its gap
against those rows' offsets (24 resolvable with free offsets, 31 with known ones).

So the revised reading for august_disorder/0: the fitters, not the data, lose most levels
(C 5 of 24, B 6, A 9), and a better offset calibration would add about 7 resolvable levels.
This replaces "the data are the limit; a better fitter gains at most about 4" above.

## Later the same evening: step 3, the design calculator

guan: commit step 2 (`5730b26`); close levels mostly sit in different rows, and the offsets may
depend on the occupation (a Stark calibration worse for some occupations, or a systematic
error); resolve as many levels as possible, as T2 = 200 us is on the high side and falls with
Kerr; does a complete basis with whole-number weights help? Agreed: first a Cramér-Rao design
calculator, no new fitter (spec 5.6, `fitting/qsim/poles/design.py`,
`analysis_notebooks/pole_finding/design.py`, 1 min). Also: `RealSpectrum` carries the model's
weights for every occupation; the condition test is now on the unit-free matrix
(`resolution.scaled_condition`; the unit-dependent test before could flag or pass by units).

First, two checks on the fits:

- Offsets per photon: a linear fit in the photon numbers takes 35-40% of C's and the oracle's
  offset variance on august_N3 (+0.43-0.48 kHz per photon in the first mode, both fits); the
  rest scatters by 0.5-0.65 kHz, the size of the calibration floor. 10 rows (august_disorder)
  are too few to test it.
- T2 at the august_disorder/0 model, offsets free (prior 0.5 kHz): median gap error 0.35 /
  0.82 / 1.73 / 4.2 kHz at T2 200 / 100 / 67 / 50 us; resolvable 24 / 17 / - / 2 of 35.

The design grid, august_disorder/0 model, noise 0.075 per sample, levels resolvable of 35 and
(small gaps resolved of 8); "time x3.5" is 3.5 times the measuring time of now:

| design (offsets free, prior 0.5 kHz) | T2 100 us | T2 200 us |
|---|---|---|
| 10 measured, x1, complex (now) | 19 (0) | 24 (2) |
| 10 measured, x1, real | 25 (3) | 33 (7) |
| 10 measured, x3.5, complex | 20 (0) | 33 (7) |
| 10 measured, x3.5, real | 31 (6) | 33 (7) |
| 35 complete, x1, complex | 20 (0) | 29 (5) |
| 35 complete, x1, real | 27 (4) | 33 (7) |
| 35 complete, x1, real + sums | 29 (5) | 33 (7) |
| 35 complete, x3.5, real + sums | 31 (6) | 35 (8) |

With offsets known to 0.1 kHz: now 24 (2) / 29 (5); real 29 (5) / 33 (7).

Found:

- **Real amplitudes are the largest gain, and they cost no measuring time:** 24 -> 33 of 35
  (small gaps 2 -> 7 of 8) at T2 200 us, 19 -> 25 at 100 us, on the data we have. Every fitter
  now fits free complex amplitudes; for a diagonal row c = <b|P|b> is real (and not negative)
  once the phase frame is right. With real amplitudes the offset calibration hardly matters at
  T2 200 us (33 at every prior). This holds only if the data are in that frame: the offsets
  are a linear phase roll per row, a constant phase is removed by the normalization, anything
  else (a phase drift not linear in t) breaks it. To check on the data.
- **A complete basis at the same total time adds little** (complex 24 -> 29 at T2 200, 19 -> 20
  at 100; real 25 -> 27 at 100), and the whole-number sums little more on top of real
  amplitudes (27 -> 29 at T2 100, same at 200). Their main use stays the weight check of
  benchmark 3 (counting merges).
- **A per-photon offset pattern does not help unless it is calibrated elsewhere:** with its
  shifts free it is a weaker assumption than free offsets of the total spread (10 rows, T2 100:
  11 vs 17).
- **Offset calibration** matters with complex amplitudes (24 -> 29 at 0.25 kHz), hardly with real.
- **T2** dominates everything else (above).

Next: fitter C with real amplitudes (an option of the variable projection); on synthetic data,
does it reach the bound; on the real August sets, does the in-band residual stay at the noise
(if it rises, the phase frame is not right and realness cannot be used).
