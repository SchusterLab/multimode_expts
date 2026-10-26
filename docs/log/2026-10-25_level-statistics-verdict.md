# 2026-10-25: level statistics verdict, can MBR spectra show level repulsion turn on? (Claude, full rein)

## Why

guan (10-25 afternoon): the pole-finding work of Sep-Oct (`docs/qsim/pole_finding_recap.md`) went from
fitter to fitter without a fixed criterion. New scope: orient, be skeptical, and give a verdict: will
any measurement we can run show a change of level repulsion from either integrable limit (W = 0
superintegrable; K = 0 linear) to the chaotic bulk? Device runs allowed until 10-26 04:00, at most
about a third of the duty cycle (seb's weekend loop), same readout block as seb.

**The vault page is the main record:**
`G:\Shared drives\SLab\Multimode\Lab\guan\LevelStatistics\LevelStatistics - overview.md`
(figures in `figures\`). This log is the short version.

## What was done (branch `theory`)

- Scripts `analysis_notebooks/level_stats/labbook/q1..q10*.py` (+ `common.py`); caches git-ignored.
  Order: model (Q1, Q7) -> Cramer-Rao bound with complete-basis weight counting (Q2) -> realization
  count (Q3) -> frozen test on the Sep 10 data (Q4) -> cost (Q5) -> real fitter on synthetic traces
  (Q10) -> device check (Q6).
- Device: `measurement_notebooks/guan/level_stats_idle_check.py`, experiment `261025_level_stats`.
  Single shot JOB-20261025-00114 (fid 0.941; phase/threshold in memory only, readout block from
  seb's CFG-HW-20261025-00020), MBR StarkCal + traces of [1,0,0,0,0] and [2,0,0,0,0] at
  `scramble_sync_cycles` 1 and 34 (00116/117, 00119/120; 00122/123 and the idle traces).
  Config: guan's 10-24 snapshot at 0.25 mA (HW CFG-HW-20261024-00002, M1 CFG-M1-20261024-00001,
  FL CFG-FL-20261023-00002, MP CFG-MP-20260121-00001).
- `job_server/jobs.db` in this worktree was an empty file (all tables 0 rows); replaced by a symbolic
  link to the main one, as in the guan worktree (old file kept as `jobs.db.empty-bak-20261025`).
- No change to library code or to `main`.

## Found

LOG_FOUND_PLACEHOLDER

## Mistakes on the way (kept so they are not repeated)

- Run 1 of the device script blocked after the single shot: the runner opened a plot window
  (no `show=False`, no Agg backend). Run 2 stopped before the traces: `MBRSpectrumExperiment`
  needs the calibration set saved first. Run 3 reused the finished jobs.
- A cost table that put the linear reference through the Cramer-Rao counting model (Q5b) made the
  contrast look larger; the real fitter (Q10) shows that model is too pessimistic. Superseded.
- GOE P(r < 0.25) is 0.158 (Atas surmise), not 0.12 as in the first figure labels (fixed).
