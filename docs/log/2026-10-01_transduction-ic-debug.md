# 2026-10-01 — transduction: Ic debugging at coupler 0.25 mA

seb + Claude. Full record with job IDs: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md`
(section "2026-10-01 afternoon" and "2026-10-01 night"); scripts and tables in `ic_debug/hom/`,
`ic_debug/rho02_ab/`, `ic_debug/swap_sweep/`.

## Done

- seb moved the coupler set point to 0.25 mA and recalibrated all pulses. Fock 2: P2 0.963 (best so far).
- Section 4 set at eta 0.35: Ic -0.10 +/- 0.04, Fe 0.51. Located the loss: rho02 of '+'/'+i' at about half of the
  model-free coherent bound, rho13 at the bound; needs both the env photon and the swap.
- Ruled out (tests and simulations): Fock-2 prep, stale hpi, '+' prep, env photon before the swap, HOM
  interference (V about 0.96), 3-photon readout, cutoff/ML artifacts, pump-time dephasing (extra 2 pi swaps),
  drift inside a job (time halves), M1/M1-S4/S4 frequency noise, swap-angle jitter, static and random Kerr,
  S4-dependent readout, phase-sensitive term (+ and - same defect), delay before the swap, ladder drive power,
  input composition, post-selection.
- Supported: weaker swap keeps more rho02 (gain 6761 / 4465 / 3389: 0.57 / 0.78 / 0.80 of ideal); Ic -0.10 /
  -0.07 / +0.02. Ic budget of the gain-3389 set closes: rho02 about 0.08, linearity about 0.03, rho13 about 0.03, estimator
  noise about 0.05 at 250 reps.
- Night run started 20:25 (seb's plan): swap-length sweep for S4 and S3 at eta 0.35 with error-amplification
  recalibration per point, f0-g1 monitor, sideband recal every 3 h; calibrations in memory only.

## Decisions (seb)

- Fix eta 0.35 and sweep the swap length for S4 and S3 overnight.
- `avoid_yoko=False` everywhere: another user runs jobs at 0.07 mA on the same queue.
- Ic, not Fock-2 fidelity, decides the coupler bias.

## Found

- At 0.25 mA the swap gain matters for rho02; at 0.30 mA a strong swap gave 0.66-1.02, so the coupler bias is a
  second lever.
- `CFG-HW-20261001-00034` (not main) has the g0-e0 hpi 1.94 MHz off.
- The transduction notebook was overwritten by an old open copy at 10:48; restored from git (commit 3ebacc0).
- A rough "dry run" of mine submitted 5 intended jobs early (no harm); the night scripts now have a real `--dry`.
