# 2026-10-21: coupler-current switch test; corrections to the night 10-20/21 reading

seb ok (option 1: a deliberate 0.30 -> 0.25 -> 0.30 mA switch). Sandbox `ic_debug/losssplit` case `swtest`,
records `night1020_case_*.json`, `swtest_partial.json`, log `run_1021_swtest2.log`.

## Why

The night 10-20/21 shared the queue with guan at 0.25 mA (EA of M1-S4..S7, Floquet EA, Qsim, MBR on S4 / S7, 21:02-04:57);
the worker ramps the yoko per job (0.2 mA/s, no settle time). Hypothesis: the coupler switch before a set dephases the swap.

## Readout incident

The worker was restarted at 10:38:54 (another user); the QICK reload turned the IQ plane by ~170 deg. The active-reset
single shot then kept 3-4 % of the shots after pre-selection and still passed the check (fid 0.886) -> the first
attempt (11:10) was stopped. Fix in losssplit `single_shot`: an angle > 5 deg also counts as failed, so the plain single
shot runs first (11:21: +169.6 deg; then the active-reset one: 91 % kept, fid 0.943).

## Result (|rho02| of the '+' output at eta 0.35, one Wigner each, bootstrap error per Wigner)

- B (right after a short job at 0.25 mA): P1 0.169, S1 0.137 -> 0.153 +/- 0.010.
- A (previous job at 0.30 mA): P2 0.154, P3 0.126, P4 0.154, S2 0.136 -> 0.144 +/- 0.007.
- B - A = +0.009 +/- 0.012: no switch effect. All six: 0.147 +/- 0.006.

## Corrections (my earlier statements)

- One Wigner has +/-0.014 in |rho02| (bootstrap), not +/-0.004 (that was the spread of three 10-19 sets).
- The 10-19 Wigners used the old alpha scale; at the corrected scale (s = 0.975) their |rho02| is 0.176 / 0.174 / 0.168
  -> 0.173 +/- 0.008 (not 0.18-0.19). Against it: night 10-20/21 0.137 +/- 0.008 (3.3 sigma), this morning 0.147 +/- 0.006
  (2.6 sigma). The 10-19 vs now difference is smaller than first stated.
- One Ic set has ~+/-0.04-0.05 (per-set bootstrap). The three 0.35 sets of the night (+0.115 +/- 0.047, -0.068 +/- 0.038,
  -0.016 +/- 0.050) give chi2 9.4 for 2 dof about the pooled value (p 0.009); the two 0.5 sets chi2 3.3 / 1 (p 0.07)
  (`final/night1020_curve.md`). So the sets scatter more than shot noise: a real set-to-set variation, or noise the
  bootstrap misses (per-set readout recalibration, frame drift, parity contrast). Not decided with 3 sets. "The loss
  came and went" over-read single sets. Rule from now on (seb ok): no conclusion from a single set; report the chi2.

## Open

- 10-19 (0.173) vs now (0.14-0.15): a setting (the photon-1 rungs moved the frame by ~110 deg) or a fluctuation.
  Proposed: + Wigners at 0.35 with the 10-19 rung values vs the current ones, alternating, 4 per arm (~1 h).
