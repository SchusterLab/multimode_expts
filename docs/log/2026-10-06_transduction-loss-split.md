# 2026-10-06 — transduction: coupler 0.3 mA, Fock readout check, loss split (10-05 15:30 -> 10-06 05:30)

Who: seb (decisions), Claude (runs, analysis; overnight with seb's ok). Record:
`C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md` (10-05 end and 10-06 sections); write-up
`ic_debug/losssplit/RESULT_2026-10-06.md`; scripts in `ic_debug/setpoint_0p3/` and `ic_debug/losssplit/`.

## Done

- seb moved the coupler to **0.3 mA** and recalibrated (main CFG-HW-20261005-00148, storage CFG-M1-20261005-00023).
  First eta 0.35 set there: Ic +0.095 +/- 0.034, gap -0.068.
- Fock readout check (seb read |0> / |1> / |2> better than my checks): same configs, same pulses; with seb's notebook
  analysis there is no systematic offset (job-to-job scatter ~+-0.015 plus drift). The alpha list does not matter
  (A/B: 0.929 vs 0.927).
- Loss-split plan (`ic_debug/handoff_loss_plan.md`) run at 0.3 mA, plus extras until 05:30: HOM vs eta, S4 wait,
  two-photon wait, herald, 3-photon vs eta, four eta 0.35 sets, S4-load test, timing test.

## Found

- **Photon-number dependent loss during the partial swap.** With photons in both modes an M1 photon is missing in
  ~8 % of the shots (HOM M1 vacancy 0.086 +/- 0.015; 3-photon |1> admixture 0.123 +/- 0.022 vs 0.046 for |2> alone).
  Not the preparation (S4-load test: no change), not slow decay (T1(M1) ~340 us, T1(S4) ~680 us), qubit stays in g.
  This term alone predicts the eta shape of the 10-03 gap and ~2/3 of the mean gap at 0.3 mA (-0.105 vs -0.15).
- **Timing test (corrected the same morning):** with the partial swap directly after the env full swap, the
  env-only input read P0 0.047 +/- 0.016 lower than with a ~1 us gain-0 gap, but P1 was the same, so the swap angle
  does not change. First written up as an "input eta mismatch" between the '0' input and the others; retracted after
  seb asked what Fig 4 shows. The low P0 (P0 + P1 ~0.95) is not seen on average in the other plain env-only jobs: open.
- **Set-to-set variation follows the coherence.** Four sets in one session: gaps -0.068 / -0.080 / -0.209 / -0.249
  order exactly with mean |rho02|/ideal 0.99 / 0.92 / 0.84 / 0.79.
- The plan's gate (200 reps) cannot separate good from bad states (thresholds at the real values, noise +-0.015);
  500 reps works.

## Decisions

- seb: 0.3 mA from now on; job 342 Wigner settings for M1 checks; relax_delay 8000 us for every case with S4 photons
  (S4 T1 ~1 ms); gate with 500 reps; extras until 05:30.
- Open for seb: how to look for the swap loss
  (pump gain, delay between the swaps) and for the time-varying dephasing.

## Afternoon (same day): photon-number Ramsey, readout fix, pair loss

- New readout of photon populations: qubit Ramsey at alpha = 0 (fit of the n chi lines), with seb's active reset and
  parity pre-selection: ~3x faster than Wigner for populations (`ic_debug/ramsey_pn/`).
- The QICK server on the board restarted (~10:50, new Pyro port; seb restarted the worker). The readout IQ plane
  rotated by -48.3 deg; new single shot applied in memory only (seb).
- Found: photon pairs are lost during the swap (HOM P0 - P2 ~ +0.07 with P1 ideal; |2> -> |0> q = 0.078; HOM + full
  swap excludes pair hopping). |2> prep fails ~8 %. S4 thermal ~1.5 % (not removed by the pre-selection), M1 ~3 %
  (removed). Ic budget at eta .35: -0.160 from these three vs -0.15 +/- 0.05 measured.
- Found: `t2_ramsey.py` time axis ~7 % short (common code, not changed).
- Decisions: seb -- readout in memory only; Wigner settings of job 342; active-reset schedule with parity
  pre-selection; no M1 reset (not calibrated).

## Evening and night (same day)

- HOM preparation tested piece by piece (env photon, M1 ladder with S4 occupied, env photon after the ladder): all
  clean. The pair-loss model of the afternoon was falsified by a direct test (|2> + two full swaps: no pair loss).
  The HOM P0 > P2 (+0.03 to +0.05 after the n = 2 visibility correction) remains, cause unknown.
- Preparation failures are f-type (|1> 1.3 %, |2> 2.5 %; e/f split with a number-matched ge pi). The n = 2 Ramsey
  fringe is ~6 % weaker for an unknown reason (hpi bandwidth computed: 0.05 %, excluded).
- Per-job readout recalibration of 25 Ic sets: the set-to-set scatter does not shrink (0.067 -> 0.069) -> readout
  drift is not the cause. The readout does drift (the single shot angle moved ~16 deg between 13:14 and 21:38).
- Recalibration incl. single shot at relax 8000, then Ic at eta 0.35: +0.154 +/- 0.042, gap +0.001 (best set); the next
  two sets -0.089 / -0.090. The bare '+' / '+i' inputs (no env, no swap) had full coherence in 5 / 5 checks -> the
  coherence is lost in the channel (env + swap), and which coherence (rho02 or rho13) varies.
- The swap pump alone (detuned, no swap, 0..2x gain) does not dephase |0> + |2>.
- **Incident 23:05:** `configs/versions` in the main checkout was emptied, in the same second as the removal of the
  `multimode_expts_qsim-analysis` worktree (the worktrees reach it through a junction). Running jobs survived until
  23:13; then job snapshots could no longer be written and jobs failed. The night run was stopped (STOP file).
  Nothing was restored.

## Decisions (evening)

- seb: the reverse-ladder coherence readout was dropped (it does not map the mixed output); swap calibration at other
  lengths copied from the 10-01 sweep with seb's EA settings, reference interleaved; Ic vs eta skipped.
- Open for seb: restore configs/versions (rebuild from the job HDF5 attrs / backups, check the main pointers); readout
  in the main config.
