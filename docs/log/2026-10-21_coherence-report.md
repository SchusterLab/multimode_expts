# 2026-10-21: coherence report after the cooldown (qubit ge / ef, M1 0-1 / 0-2, S4 0-1)

seb ok. 0.30 mA, `CFG-HW-20261020-00034`, storage `CFG-M1-20261020-00010`, readout single shot before each part.
Sandbox `C:\experiments\260601_Transduction_sandbox\coherence\` (scripts `run_qubit_coherence.py`, `run_cavity_t2.py`,
`run_part3.py`, `run_s4_ramsey_redo.py`; results JSON and one PNG per item). Jobs JOB-20261021-00099..00125.

## Classes

- Qubit: `T1Experiment`; `RamseyCPMGExperiment` (echoes 0 / 1) for ge and ef.
- Cavities: `CavityModeRamseyExperiment` (t2_cavity_mode) for M1 and S4 0-1 and, with the new `fock=2` and
  `phase_cycle`, M1 0-2; the new `CavityModeT1Experiment` (t1_cavity_mode) for M1 |1>, |2> and S4 |1>
  (commit fcbacb3, reviewed by seb). No qsim classes (seb).

## Results

| | T1 | T2 Ramsey | T2 echo |
|---|---|---|---|
| qubit g-e | 491 +/- 14 us | 403 +/- 32 us | 473 +/- 30 us |
| qubit e-f | 184 +/- 18 us | 137 +/- 30 us per component, split 5.45 +/- 0.17 kHz (single-frequency effective 55 +/- 3 us) | 127 +/- 10 us |
| M1 0-1 | 232 +/- 5 us (map) / 201 +/- 5 us (parity) | 134 +/- 10 us | 251 +/- 14 us |
| M1 0-2 | 215 +/- 4 us (|2>, parity, rate 2/T1) | 82.5 +/- 2.6 us (Gaussian envelope) | 125 +/- 6 us |
| S4 0-1 | 1239 +/- 34 us | 1655 +/- 80 us | 1686 +/- 87 us |

Frame offsets: M1 0-1 3.737 MHz, M1 0-2 8.19 MHz, S4 -3.533 MHz.

## Findings

- M1 dephases strongly (T2* 134 vs 2 T1 ~430 us). Dephasing beyond decay, 0-2 over 0-1: ~1.45 (Ramsey), ~2 (echo);
  white frequency noise would give 4. A hint about the noise, not a result.
- |2> decays at ~2 Gamma_1 (parity vs parity: 215 vs 201 us, ratio 0.93 +/- 0.04).
- M1 |1> T1: the map and parity readouts differ by 30 us (~4 sigma); not explained (candidates: heating |1> -> |2>,
  qubit not in |g> at the readout).
- S4 lives longer (1.24 ms) and M1 shorter (0.20-0.23 ms) than the transduction MODEL assumes (0.70 / 0.34 ms); the
  MODEL predictions should be redone with these values.
- e-f: the charge-parity beat is still there but smaller (5.45 kHz vs 8-12 kHz on 10-19/20): two-frequency fit
  preferred over a single frequency with a Gaussian envelope by delta chi2 = 18 (1 parameter). The echo refocuses it.
- No revival or other structure in any trace (residuals at the noise level with the chosen envelope); a thermal-qubit
  revival near 2-3 us is below the step sizes used.
- 0-2 phase cycling was needed: the phase-independent baseline drifted by 2.6 (Ramsey) and 7.1 (echo) units as the
  decay products built up; the difference trace has no drift.

## Notes

- The worker stopped at 14:53 without a traceback during the S4 echo (job 00113 left 'running'); seb restarted it.
- The notebooks' three-step cavity Ramsey takes the first frame candidate; for S4 it took +0.930 instead of -3.535
  MHz. Take the candidate closest to the previous step (as run_part3.py does).
- `RamseyFitting` keeps a Gaussian envelope when it lowers the SSR by > 5 %; plots must use `gaussdecaysin` then.
