# 2026-10-02 (evening): the pole-finding survey on the Sep 10 complete-basis ensemble

guan and Claude, on pippin. guan asked to run everything we have on the new data (the stage is
still a survey of all methods): the diagnosis, the fitters, the design bounds, the T3 merge
threshold, the Matrix Pencil settings at 35 rows, and the form factor. The session ended early
(guan disconnected), so the survey runs unattended; this entry records what was started and the
first numbers.

## What was done

- `experiments/qsim/pole_data.py`: the recorded self-Kerr of a realization is optional.
  `sep10_full_K3p6_g29p2` records it for r=0 only; the loader raised a `KeyError` on the other
  eight. The model Kerr of that entry is the analysis one (`model_kerr: analysis`, the registry
  default), so nothing else changes; `model_kerr: recorded` without a recorded value now raises
  a clear error.
- `analysis_notebooks/pole_finding/survey_sep10.py`: one resumable script for the whole survey
  (one pickle per spectrum and fitter, CSV summaries, a log). Stages, in order: `ff` (the
  ensemble analysis with the default Matrix Pencil, Tr U(t), the measured and model form
  factors), `fast` (A with the analysis and the campaign settings, B, T1, T1T3, T1free on all
  10 spectra: 9 realizations and `august25_N3`), `design` (Cramér-Rao bounds: complex / real /
  real_sums amplitudes x free / per_photon offsets at 35 rows, and complex / real at 10 random
  rows; at the decay T1 finds), `merge` (T1T3 with `merge_chi2` 50, 100, 200, 400 on the first
  4 spectra; the default is 25), `diagnose` (spec 5.5 with every cached fit), `C` (all spectra,
  slow), `F` (pursuit from T1 on r=0; very slow). Output folder:
  `C:\experiments\260818_qsim_spectroscopy\derived_data\pole_finding\sep10_survey\`
  (`survey.log`, `fits_summary.csv`, `t1_model_check.csv`, `design_bounds.csv`,
  `merge_scan.csv`, `diagnosis_counts*.csv`, `diagnosis_<spectrum>.csv`, `form_factor.pkl`,
  `fit_<spectrum>_<fitter>.pkl`). Console copy: `C:\experiments\survey_sep10_console.log`.
  Running in tmux session `nb`, window `survey`. To resume or rerun a stage:
  `pixi run python analysis_notebooks/pole_finding/survey_sep10.py C,F`.
- Not started: T2's row-sum bound (an additive `row_sums` option in `design.py`), and a
  scan of the Matrix Pencil settings beyond the two settings sets A already has.

## The data set (as loaded)

9 realizations x 35 rows x 468 samples, dt 0.4278 us (200 us window, 5.0 kHz bins), 35
distinct model levels per realization (no degeneracy left at delta 50 kHz RMS), recorded
g 29.22 kHz, analysis Kerr -3.76 kHz, |A(0)| 0.23-0.90, |A(end)| about 0.05-0.14.
`august25_N3`: 35 rows x 200 samples, dt 1.637 us, 10 distinct levels, g 15.27 kHz, K -45 kHz.

## First numbers (realization 0, before the survey)

| fitter | time | poles | model levels hit (weight > 0.3) | poles far from every level | in-band residual / noise (mean, max) |
|---|---|---|---|---|---|
| A (analysis settings) | 4 s | 35 | 21 of 35 | 11 | 16.9, 106 |
| B | 0.2 s | 56 | 22 | 29 | 1.5, 4.4 |
| T1 | 17 s | 35 (model) | - | - | 1.9, 3.1 |
| T1T3 | 172 s | 52 | 22 | 18 | 1.2, 1.6 |

T1 (reduced chi^2 1.37, T2 165 us) moves the recorded parameters by up to 3 kHz (detuning 4
by -3.2 kHz, Kerr -3.76 to -3.44 kHz, couplings 28.95-29.79 against 29.22) and finds **row
offsets of 4.7 kHz rms, up to 11.7 kHz** (the prior is 0.5 kHz): about one FFT bin. This is
the residual Stark-shift error guan asked about, and on this set it is an order of magnitude
larger than on the August sets (0.5-1 kHz). Caveat: T1's offsets can absorb model error; C's
offsets (free poles) are the check, and the survey's `C` stage gives them. Until that is
known, "levels hit" against the model within half a bin undercounts (the comparison frame is
off by the offsets), so the diagnosis (`seen_frequencies`) is the better score.

## Next

- Read `survey.log` and the CSVs; put the tables in a new log entry. Questions to answer:
  how many of the 35 levels are resolvable (design, diagnosis) and found by each fitter; are
  C's offsets as large as T1's; does T1T3's false-pole count fall with `merge_chi2` at 35 rows
  without losing levels; the measured form factor against the model's; P(r < 0.25) found
  against the model's and against the resolvable subset.
- If the offsets are really 5-10 kHz: the prior of 0.5 kHz in C, F, T3 and the diagnosis is
  wrong for this set (make it a setting per data set), and the per-photon offset pattern
  (`design.py`, `offsets="per_photon"`) is worth fitting, not only bounding: 4 numbers instead
  of 35.
- Then T2's row-sum bound, and the Matrix Pencil settings scan for 35 rows.
