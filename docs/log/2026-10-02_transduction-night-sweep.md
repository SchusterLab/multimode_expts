# 2026-10-02 — transduction: night swap-length sweep and sideband tests

seb + Claude. Record with job IDs: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md` (sections
"2026-10-01 night" and "2026-10-02 morning"); scripts and tables in `ic_debug/swap_sweep/` and `ic_debug/sbenv/`.

## Done

- Night sweep (seb's plan): eta 0.35, M1-S4 and M1-S3 full-swap length, swap recalibrated per point by error
  amplification (in memory), Section-4 set per point, f0-g1 line monitor, sideband recalibration at 23:27.
  9 valid sets; 5 S3 points failed calibration (start frequency outside the scan); one S4 point locked on a wrong peak.
- Morning (seb away, green light except the coupler set point): f0-g1 with and without the env photon in S4,
  ladder sideband shift scan, 500-rep set.

## Found

- The same S4 setting gave Ic -0.015, +0.049, +0.166 over the night (rho02 0.69, 0.87, 1.03 of ideal). The best set
  reaches the ideal channel through the estimator (0.163). The rho02 loss varies over hours; inside one job it is stable.
- The env photon does not shift f0-g1 (-0.7 kHz). Ladder sideband shifts of +-30-60 kHz do not cause the linearity
  defect. The defect was absent this morning (sum rule OK in the 500-rep set, including the '-' input).
- 500-rep set: Ic +0.009 +/- 0.035, Fe 0.682 +/- 0.013 (best Fe so far). Remaining cost mostly rho02 (~0.11 of Ic).
- f0-g1 low-power line stable within +-5 kHz all night at 0.25 mA (no jumps). Sidebands drifted since 10-01 noon:
  f0-g1 -19 kHz, f1-g2 -67 kHz.
- Lead: the pre-selection kept fraction fell 0.93 -> 0.90 when the linearity defect went away (7 sets, no mechanism).

## Decisions

- seb: fix eta 0.35, sweep the swap length for S4 and S3; sideband recal every 3 h; f0-g1 trace between points.
- seb: green light on everything except the coupler set point (morning).
- Main configs not written; today's sideband values are in memory and in `ic_debug/sbenv/step1.json`.
