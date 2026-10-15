# 2026-10-03 — transduction: search for the Ic gap (10-02 afternoon to 10-03)

Who: seb (decisions, calibration), Claude (scripts, runs, analysis; autonomous while seb was away, with seb's green
light for everything but a coupler setpoint change; on 10-03 also the setpoint inside the calibrated f0-g1 range).
Full record: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md`. Scripts and results in `ic_debug/`.

## Done

- **fn-gn+1 sideband calibration (10-02).** Error amplification reads 40-70 kHz below the single-pulse resonance:
  a drive "memory" (a drive shifts later pulses by -58..-70 kHz, recovery over tens of us, ~a^1.6). New experiment
  `experiments/single_qubit/error_amplification_2d.py` (seb approved the design) for 2D frequency x gain/length maps.
  Step 4: old / new calibration at full gain give the same Ic (0.111 / 0.109 +/- 0.04); half gain worse.
- **Overnight monitor + eta sweep (10-02/03).** Ic vs eta follows the theory shape. The driven sideband drifts by
  tens of kHz per hour while the low-power line is stable -> drive power at the chip.
- **Pooled analysis (10-03).** 13 sets at eta .35: Ic 0.083 +/- 0.015 vs 0.200 ideal through the same estimator.
  Ic at eta .35 is fragile: one 3 % error (loss, missing env photon) halves it.
- **End-point tests (10-03).** No swap (identity): Ic as ideal through the estimator; Fock 2 97 %; S4 photons do
  not disturb the tomography; one-photon prep loses ~1.8 %, a full swap ~1.1 %.
- **Hypotheses tested (10-03):** missing env photon (falsified at low eta: measured gap ~0 at eta .14), cross-Kerr
  after the swap (exactly zero cost), drift between inputs (interleaved sets and a refresh before every input do
  not help), estimator compression (no). The herald after the preparation gains ~0.03-0.05.
- **Main finding (10-03):** Ic follows the P1 of the |2>-input output (r = -0.69, 18 sets). eta .35 sits on the
  3-photon interference zero (P1 = 0 at eta 1/3); the zero is "filled" by 0..0.06 in the bad sets.
- **Overnight 10-03/04:** zero scans (eta .26-.41) alternating with eta .35 sets (`ic_debug/endpoints/`).

## Decisions

- seb: avoid_yoko=False on every job; no change of the main configs (in-memory values only).
- seb: Ic first, Fe later. Setpoint change allowed inside the calibrated f0-g1 range with a full recalibration
  (not done: no evidence yet that the coupler bias is the lever).
- Claude: launch long runs through `pixi run` (WMI process create). Direct `python.exe` launches crashed with
  0xc06d007f (LAPACK DLL not on PATH) and cost ~1.5 h of hardware time.

## Found

- `process_tomo.analyze_set` thru reference uses the nominal eta (see the status file). Not changed (common code).
- The Z angle of the outputs is random from set to set: Fe needs a phase calibration per set.
