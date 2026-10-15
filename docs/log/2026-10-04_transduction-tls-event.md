# 2026-10-04 — transduction: interference zeros, pump test, device event at ~05:00

Who: seb (decisions), Claude (runs, analysis; autonomous with seb's ok). Record:
`C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md` (2026-10-04 section), scripts in `ic_debug/endpoints/`.

## Done

- Overnight (seb ok): HOM scans (|1>_M1 |1>_S4 around eta 0.5), 3-photon zero scans (|2>|1> around eta 1/3) and
  eta 0.35 sets. Both zeros are filled (min P1 0.01-0.05); one Kerr set does not fit both; fits prefer extra loss.
- Pump test: a 10 us non-resonant M1-S4 pump adds no loss. M1-S4 error amplification (24 swaps): still clean.
- M1 T1 183 +/- 32 us (9/30: 251). Qubit T1 238 us (9/30: 379), ge T2* 129 us.
- In-memory recalibration (seb: "be autonomous"): qubit ladder pulses by error amplification (small changes,
  tens of kHz); single shot found +14 deg but made the Wigner readout worse -> not applied. No main-config writes.

## Found

- **Device event at ~05:00 on 10-04:** bare f0-g1 line +40 kHz step, qubit ge +29 kHz (ef unchanged), qubit T1 down
  ~40 %: TLS-like. Ic dropped from that time (05:17 set -0.07; before +0.06..+0.12); the preparation + tomography
  of |1> / |2> is worse (0.90 / 0.88 vs 0.97 / 0.97) and did not recover with the pulse recalibration.
- `experiments/single_qubit/t2_cavity_displacement.py` (CavityDisplacementRamseyExperiment) cannot be saved by the
  worker: HDF5 save fails on a string array (`No conversion path for dtype '<U3'`). Common code: not changed.

## Decisions

- Claude: do not apply the new single-shot calibration (worse Wigner readout); keep in-memory values only.
- Open for seb: how to calibrate the M1 displacement frequency and the parity measurement; whether to wait for the
  TLS to move or re-tune; whether to fix the CavityDisplacementRamsey save bug.
