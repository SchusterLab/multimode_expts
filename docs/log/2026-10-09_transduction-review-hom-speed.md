# 2026-10-09 day: external review, coherence budget with decay, swap eta with S4 occupied, HOM asymmetry vs swap speed

seb + Claude. Full record: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md` (10-09 sections). Review handoff and
second opinion: `ic_debug\HANDOFF_review_2026-10-09.md`. Cases 69-74 in `ic_debug\losssplit\`. No main config changed.

## Done

- Analysis (no hardware): all 59 channel '+' jobs at eta ~0.33 pooled with one normalisation (`pool_coh033.md`); Wigner
  systematics simulated and the displacement scale fitted to the data (`sim_wigner_bias.md`, `fit_alpha_scale.md`); a
  coherence budget that includes the ordinary M1 decay (`a6_budget_033.md`; `sim_decay_seq.py`, `sim_swap_dephasing.py`).
- B1 (case 69, `hom33.md`): the swap eta with one photon in each mode vs the single-photon eta, photon-number Ramsey.
- Option 1 (cases 71-74, `homspd.md`, `figs/homspd.png`): the HOM P0 - P2 asymmetry at eta 0.333 vs the swap speed
  (0.75x / 1x / 1.5x / 2x pump gain), each speed with a converged swap calibration and a tuned eta.
- Decided by seb: a 6-pass Ic(eta) curve over the weekend (11 points 0..1 + 0.35, shuffled passes). Prepared, ON HOLD (below).

## Found

- |rho02| / ideal = 0.886 +/- 0.010 for every eta normalisation (robust). |rho13| / ideal = 0.93-0.95 +/- 0.008: the
  earlier claim "single-path coherence at the ideal" (7 jobs) was wrong as stated (reviewer right).
- The ideal channel used in all ratios has no ordinary decay. With the M1 decay (T1 340 us, MODEL) and the measured env
  vacancy, the single-path coherences are within ~2 sigma of the model; the two-path ones keep a loss: rho02 -6 +/- 1.5 %,
  rho01 -15 +/- 3.5 %.
- The Wigner tomography is not the cause: the displacement scale is right to ~0.5 % (fitted on the parities).
- The swap eta with both modes occupied = the single-photon eta within 0.006 (raw treatment 0.013); the higher channel
  "population eta" is not an eta of the swap.
- HOM P0 - P2 at eta 0.33: +0.053 +/- 0.004 (vacancy alone +0.008), reproduced 7 h later (+0.047 excess).
- vs swap speed: the excess is not pump-driven (chi2 23.8 / 4 for x ~ f); constant and 1/f both fit (k = -0.44 +/- 0.32).
  The raw P0 - P2 is ~0.06 at every speed. The env vacancy rises with the pump gain (~0 -> 3.6 %, ~2.6 sigma, tentative):
  a faster swap does not reduce the vacancy here.
- Slow swap-detuning noise and dephasing during the swap are excluded as the main cause (EA fringe width / contrast;
  wrong pattern).

## Incidents

- 11:44: an active-reset single shot returned fid nan without an exception: the plain-shot fallback did not trigger.
  Fixed in `losssplit.py` (a rejected shot bootstraps; recalibration stops without a valid readout; fid 1.0 / degenerate
  confusion matrices are rejected).
- 16:18: a waiting chained run ended on a stray Ctrl-C; the run chained on it started early and interleaved its
  calibration with the 1.5x jobs (data still valid: each job carries its own config). Fixed: Ctrl-C ignored while waiting,
  and a run also waits while any other `losssplit.py` process runs.
- From ~18:45: other projects (`260915_qsim_migration`, `261009_Calibration`) submit jobs at coupler 0.07 mA, interleaved
  with ours at 0.3 mA (neither sets avoid_yoko). The 0.75x point and the end of the 1x repeat are flagged. The weekend
  run (alternating 2x / 1x pass + Ic sweep) was put on hold at 21:05 by Claude before it submitted anything.

## Open (for seb)

- 21:55 seb: interleaving with the 0.07 mA projects is acceptable; relaunched (`losssplit.py --cases homalt@2,<icsweep_cases.txt> --ss --no-gate`).
- The cause of the two-path coherence loss and of the HOM asymmetry (likely one effect); the preparation / readout of the
  |1,1> state is a candidate again, since the asymmetry does not scale with the swap.
