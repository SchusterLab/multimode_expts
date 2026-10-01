# 2026-09-30 Transduction: Ic gap explained, notebook refactor

Who: seb, with Claude.

## Found

- **Aug-2025 "Ic matches theory" was an analysis artifact** (seb found it). qutip < 5.2
  `entropy_vn` used every nonzero eigenvalue with a complex log, so a negative eigenvalue gave a
  negative entropy term. The Aug-2025 rho_RB had eigenvalues down to -0.05 to -0.07, which gave
  Ic too high by about 0.5. Through today's pipeline (projection before the entropy) the Aug-2025
  data gives Ic = +0.01 at eta 0.35 (theory +0.21), the same as today. So nothing broke since
  Aug 2025. Record: `G:\Shared drives\SLab\Multimode\transduction\old_plots\`.
- **Ic budget at eta 0.35** (simulation with the measured errors): |2> preparation -0.10;
  extra |2> loss in the swap stage -0.05; swap-stage coherence loss -0.06 (good run) to -0.21
  (bad run); shot-noise bias of the estimator about -0.1 at 1000 reps (Fe is not biased).
  Ordinary T1/T2 of M1 and the qubit (measured 9/30) explain only a small part.
- **Rung 2 had the pi bug** that Rung 1 had before: it removed the measured phase of rho_+[2,0]
  without the ideal phase. Fixed by the shared `phase_correct` in `process_tomo.py`.

## Decisions (seb)

- Goal changes from "find the regression" to "reduce the budget errors and build a proper
  estimator".
- Keep Sections 1-3 of the notebook. Remove the debug cells. Remove the debug options of
  `WignerTomography1ModeExperiment` (`gain_offset`, `displace_detuning_MHz`,
  `displace_style='const'`): done on `main` (they were never committed).
- Other transduction notebooks (`transduction.ipynb`, `transduction_reanalysis.ipynb`) are not
  part of this refactor.

## Done

- New `experiments/transduction/sequences.py` and `process_tomo.py`; tests
  `tests/test_transduction_process_tomo.py` (24 pass; the 9/30 S3 set gives Ic +0.005,
  Fe 0.491 again).
- `transduction_sandbox.ipynb` rebuilt (60 -> 40 cells); a 4-input set is now queued as one
  batch, and each analyzed set is saved with its job IDs and the code version.

## Estimator (later the same day)

- Study (background agent, checked by Claude; `C:\experiments\260601_Transduction_sandbox\ic_debug\estimator\REPORT.md`):
  the old Ic estimator (linear inversion + projection) reads 0.12 low for a perfect channel but only
  0.03-0.06 low for a channel like ours. Real data at eta ~0.38: Ic +0.018 +/- 0.022 with the new fit,
  against +0.143 for the ideal channel through the same estimator: a hardware gap of 0.125 +/- 0.022.
- The data has a common parity offset (pp + pm)/2 of about -0.15 (checked on the S3 set); it cancels
  in the difference parity, so the fit uses the difference parity, not the separate plus/minus counts.
- seb: put the new estimator in. Done: `experiments/transduction/estimators.py`, used by
  `process_tomo.analyze_set` for Ic (Rung 1); Fe stays linear. It gives the study values again on the
  four 9/30 sets (S3 +0.013, S4 +0.047, S4 .60 -0.252, S4 .85 +0.011). 29 tests pass.
