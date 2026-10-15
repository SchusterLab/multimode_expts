# 2026-10-07/08 night: time series at fixed settings, noise analysis (transduction)

seb + Claude. Full record: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md` (10-07 22:18 onward); scripts and
results in `ic_debug/losssplit/`: `losssplit.py` (case `tcycle`), `analyze_tseries.py` (`tseries.md`, `tseries.png`),
`coherence_noise.py` (`coherence_noise.md`), `ic_noise.py` (`ic_noise.md`).

## Done

- 20 cycles, 10-07 22:54 to 10-08 05:24 (stopped by the 05:30 limit; seb's hard deadline 06:00). Swap fixed at the mean
  10-07 calibration (gain 3754, 878.3159 MHz). Per cycle: single shot and qubit Ramsey (applied in memory), channel '+' and
  bare '+' (Wigner, 250 reps); every 2nd cycle the M1 Ramsey or the sideband refresh, and the S4 thermal population
  (full swap S4 -> M1, pre-selected photon-number Ramsey). Full recalibration every 2 h. No main config changed.
- Noise analysis of the coherences (per-job parity-count bootstrap) and of the Ic (linear Ic with a joint bootstrap).

## Found (measured)

- The channel coherence ratio has a per-job error of 0.07-0.08 at 250 reps. All series have chi2 ~ dof: time series
  |rho02| 25.7 / 19, |rho13| 27.3 / 19; bare 15.3 / 19; part C sets 1.2-10.1 / 8. So the set-to-set and cycle-to-cycle
  changes are consistent with shot noise.
- Linear Ic of the 5 part C reference sets: scatter 0.041 vs bootstrap error 0.037 (chi2 4.7 / 4). The ML Ic of the same
  sets scatters 0.073 vs its quoted error 0.040: the quoted ML error is ~1.8x too small.
- Constant channel coherence deficit: |rho02| 0.880 +/- 0.018, |rho13| 0.932 +/- 0.015 of the ideal; bare 0.990 +/- 0.006.
- S4 thermal 0.2-0.8 % (+/- 0.3 %), stable. Qubit and M1 f_ge stable within 1-2 kHz. Sidebands drift steadily ~9 kHz / h
  (-60 kHz in 6.5 h). Readout -8 deg in 6.5 h; single-shot fidelity 0.958-0.970.

## Corrections

- The "varying coherence loss in the channel" (10-06 / 10-07 reading, Claude) is not supported: it did not take the
  per-job noise into account. Two ad-hoc predictions during the night (an odd / even cycle pattern; a |rho13| recovery
  after a full recalibration) failed and were dropped.

## Open (for seb)

- Locate the constant deficit: (a) env |1> + '+', no swap; (b) '+' + partial swap with S4 empty. See the status file.
- The ML Ic error estimate (common code, `experiments/transduction/estimators.py`): ~1.8x too small.
