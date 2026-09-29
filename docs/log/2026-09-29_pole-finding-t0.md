# 2026-09-29: pole finding T0, the gap score and the fixed synthetic set

Plan: `docs/qsim/pole_finding_explore.md`, T0. Branch `qsim-analysis` (the plan said `guan`;
corrected, guan).

## What was done

- `fitting/qsim/poles/gap_score.py`: the score by gaps. (1) The one common shift the data
  cannot fix (E -> E + s, every row offset -> delta_b - s) is removed: on synthetic data from
  the offsets, s = mean(found offsets) - mean(true offsets) (`gauge_shift`); else searched
  (`common_shift`). (2) Found poles are matched to the levels in order (monotone, one-to-one,
  within half the level's nearest gap; dynamic programming). (3) A gap is found if its two
  levels are matched to adjacent poles; its error is scored against its Cramér-Rao error. (4)
  Small gaps (under half the mean) found, false poles, P(r < 0.25) found against true.
- `fitting/qsim/poles/fixed_set.py`: 80 cases, saved once with their gap bounds (complex and
  real amplitudes; `design.gap_errors`, offsets free with the true prior): the August point and
  the September point x T2 100 and 200 us x row offsets 0.5 and 1 kHz x 10 and 35 rows x 5
  draws; noise 0.075 per sample in every row (not equal total time). Fit caches per fitter;
  each fit carries a digest of the rows it fitted, so a rebuilt case is never scored or resumed
  with an old fit. Files: `C:\experiments\260818_qsim_spectroscopy\derived_data\pole_finding\fixed_set\`
  (`set.h5`, `fits_C.h5`, `fits_F.h5`; `set_v0_august_grid.h5` and
  `fits_C_backup_before_september_grid.h5` are the first build, kept until F is done).
- `tools/pole_fixed_set.py` (`build`, `fit C|F [--draws] [--rows]`, resumes);
  `analysis_notebooks/pole_finding/gap_score.py` (the tables); `tests/test_gap_score.py`.
- Fitter F takes a cached C fit as its start (`pursuit.fit(..., start=)`); its time in the
  cache is then F's own part only.

## The September point: timing and disorder

The September files carry no timing (no `derived_params`, no `config_versions`, no sidecar
entry). A subagent recovered it (guan pointed at the provenance tools), by three routes that
agree: the job database export (`tools/export_job_provenance.py --range ... -o <temp>`, then
`resolve_timing(job_id, cfg, data, record=rec)`), the pickled programs (an Unpickler that finds
the classes moved to `experiments/qsim/deprecated/`), and `cfg.expt.floquet_cycle_us`, which 588
of the 630 spectroscopy jobs carry (not the first 42, Sept 10). All 700 jobs (70 calibration,
630 spectroscopy) give Floquet config CFG-FL-20260909-00043: cycle 0.2139137 us (92 tProc ticks
at 430.08 MHz), pi fractions 40 for all modes, g 29.2174 kHz; samples every 2 cycles, dt
0.427827 us, 468 samples, a 200 us window, 5.0 kHz bins, 1/dt 2337 kHz. Found on the way: the
pickled dataset's file name (CFG-FL-20260722-00001.csv) is stale, its values are those of
CFG-FL-20260909-00043; the cycle from the CSV lengths by hand (0.196 us) is 9% short (clock
rounding); the table's realization labels are off by one (row 0 "manual", rows 1-8 seeded 0-7).
For T4: the export route works for the whole campaign.

The disorder: the config's "disorder strength 50 kHz" is not the model's delta. The recorded
onsite vector has norm 0.100 MHz (zero mean), and the model's delta is that norm: delta / g
3.4226 (K / g -0.1288). The first build had 1.71, and the August grid; it was rebuilt, the 40
August cases unchanged (checked), their C fits kept.

## C on the fixed set (mean of 5 draws; 34 gaps)

"Resolvable": the gap is at least 4 of its Cramér-Rao errors (real amplitudes); "small": under
half the mean gap. P(r < 0.25) pooled over the 5 draws; true 0.184 (August), 0.352 (September).

| point | rows | T2 (us) | poles | gaps found | resolvable | small found / resolvable | P(r < 0.25) found |
|---|---|---|---|---|---|---|---|
| August | 10 | 100 | 19 | 6-7 | 26 | 0 / 2-3 | 0.02-0.03 |
| August | 10 | 200 | 30 | 20-23 | 30 | 2-3 / 5-6 | 0.09 |
| August | 35 | 100 | 27 | 17-18 | 30 | 0-1 / 5-6 | 0.01 |
| August | 35 | 200 | 33 | 27-28 | 32 | 5 / 7 | 0.07-0.10 |
| September | 10 | 100 | 31 | 25 | 30-31 | 5 / 7 | 0.16-0.18 |
| September | 10 | 200 | 32 | 27 | 32 | 6 / 8 | 0.23-0.25 |
| September | 35 | 100 | 32 | 28 | 32 | 6-7 / 8-9 | 0.26 |
| September | 35 | 200 | 33 | 30 | 33 | 8 / 9 | 0.32 |

(Ranges: the two offset widths; they matter little.) C's time per fit: 0.5-11 min (August 35
rows the slowest).

Found:

- **C misses levels; what it finds is near the bound.** The median |gap error| / bound of the
  found gaps is 1.1-2.7. The loss is merged levels, worst where the data are hardest: August
  T2 100 us, 10 rows, C returns 19 poles for 35 levels and finds 6 of the 26 resolvable gaps.
- **P(r < 0.25) is biased low, strongly:** C merges exactly the small gaps the statistic counts
  (August 0.01-0.10 against 0.184). On September the bias is smaller (0.16-0.32 against 0.352)
  and 35 rows halve it.
- **The September point is much easier** for C than August (25-30 found of 30-33 resolvable),
  though its window is only 200 us: its levels are 3.4 times farther apart in kHz (g 29.2 vs
  8.6 kHz) at a similar ratio of disorder to g.

## F on a subset

(to follow in this entry: draw 0, 10 rows first)
