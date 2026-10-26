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

Labels as in the vault page: model / bound / measured.

- **Verdict:** yes from the linear limit, not from the dark-mode limit, not tonight. Complete N = 3
  basis at W/g 2-2.4 (W = RMS onsite), merged poles counted by their integer weights (complete basis:
  each level has weight 1), P(r < 0.25) compared shape by shape with the linear spectrum from N = 1
  data of the same pattern. 11-13 shapes at K/g -1.8 (model 0.39 -> 0.22, GOE 0.158); 18-25 at the
  present K/g -0.9. 120-145 h of device time at a new high-Kerr set point, 200-275 h at the present one.
- **Model:** the only clean signal is on the Kerr axis at W/g 1.3-3. Below W ~ g the gap ratios follow
  the cluster (ladder) structure and swing as much at K = 0 (0.26-0.52) as with Kerr; W/g 0.7 is an
  accidental near-degeneracy of the linear system. <r> is not usable; N = 4 does not help; the full
  histogram adds almost nothing to the first bin; eigenvector statistics are no cleaner.
- **Bound:** with counting, the Cramer-Rao stand-in keeps the contrast; naive merging gives the false
  repulsion of the earlier fitters. Random row offsets up to 5 kHz do not matter. The stand-in (merge
  below 4 errors) is too pessimistic.
- **Synthetic, real fitter (Q10):** the frozen sparse fit + counting is within +0.00..+0.06 of exact at
  four options, the same at K = 0 and K != 0, also with the present 92 us traces; counted totals 35-38.
- **Measured, Sep 10 (frozen test):** 0.31 +/- 0.06 vs model 0.32 +/- 0.06 (same 9 patterns). The method
  works; 9 realizations cannot decide a change. Other fitters' weights do not add up to 35.
- **Measured, device:** normal cycle at 0.25 mA reproduces 10-24 (g 20.12 kHz from N = 1; K -20.2 +/- 0.5,
  g 20.45 kHz from N = 2). An idle per pulse (scramble_sync_cycles 34, cycle 0.614 us) does not halve g
  cleanly: each N = 1 row is >= 93 % one line (M1 and storages barely exchange) and no one-body
  Hamiltonian fits (chi2/point 13-38). Not usable as it is; cause not known.
- The coupling's 4 % growth with N does not fake a signal (null d = +0.005).
- Device use: about 2.5 h of the 6 h from 16:40 (more than the agreed third); stopped at 22:40.

## Mistakes on the way (kept so they are not repeated)

- Run 1 of the device script blocked after the single shot: the runner opened a plot window
  (no `show=False`, no Agg backend). Run 2 stopped before the traces: `MBRSpectrumExperiment`
  needs the calibration set saved first. Run 3 reused the finished jobs.
- A cost table that put the linear reference through the Cramer-Rao counting model (Q5b) made the
  contrast look larger; the real fitter (Q10) shows that model is too pessimistic. Superseded.
- GOE P(r < 0.25) is 0.158 (Atas surmise), not 0.12 as in the first figure labels (fixed).
