# Transduction (Fock-state decoder, lossy-beamsplitter channel)

**Last updated: 2026-10-12** (seb + Claude). `configs/versions` (emptied 10-06 23:05) was restored by seb on 10-07
from the backup: 18 649 / 18 666 files OK; 9 missing (`CFG-HW-20261006-00108..00116`, newer than the backup); 8 old
checksum mismatches (Jan-May 2026). Work happens on `main` (`C:\python\multimode_expts`);
the refactor came from branch `transduction_refactor` (merged 2026-09-30).

## Where things stand

- **Code:** `experiments/transduction/`
  - `channel_model.py`: ideal channel, decoder, Fe, Ic.
  - `sequences.py`: logical preparation and the channel recipe `channel_prep`.
  - `process_tomo.py`: measure / load a 4-input set, metrics, joint bootstrap, linearity check, result files
    (`<experiment>/assembled_data/*_TransductionChannelSet.h5`), `plot_sets`.
  - `estimators.py`: joint ML fit of the channel (CPTP, photon-number covariant) for Ic.
  - **Known issue (not fixed, common code):** `analyze_set(n_thru=...)` runs the ideal channel through the
    estimator at the NOMINAL eta (`cset.eta`), not the measured one. Use `ic_debug/pool/pooled.analyze_jobs`.
  - **Known issue (10-06, common code, not fixed):** the Ramsey time axis of `experiments/single_qubit/t2_ramsey.py` is
    ~7 % short (a drive shift of 0.2775 MHz moved the line by 0.257 MHz; the wait register is set in qubit-generator
    cycles, `sync` counts tProc cycles: 2.325 / 2.504 = 0.929). Ramsey frequencies (T2*, f_adjust) are ~7 % too small.
  - Tests: `tests/test_transduction_process_tomo.py`, `tests/test_transduction_estimators.py`.
- **New experiment (untracked, approved by seb 10-02):** `experiments/single_qubit/error_amplification_2d.py`
  and `tests/test_error_amplification_2d.py`. Not committed yet.
- **Device:** **coupler 0.3 mA from 10-05** (seb). Main configs `CFG-HW-20261005-00148`, storage
  `CFG-M1-20261005-00023` (M1-S4 878.318 MHz, gain 3770, 4.225 us). The driven fn-gn+1 sidebands drift ~10-20 kHz / h
  (down ~90 kHz over the night of 10-05); scripts refresh them in memory only. T1(M1) ~340 us, T1(S4) ~0.7-1 ms.
  Wigner checks: JOB-20261005-00342 settings; relax_delay 8000 us whenever S4 holds photons.
  **Readout (10-06):** the QICK server restart on the board (~10:50) rotated the IQ plane by -48.3 deg. New single shot
  (job 93, seb's active-reset schedule): phase 655.10, threshold -45.9, Ig / Ie -74.5 / -12.9, fid 0.913 -- applied
  IN MEMORY only in the ic_debug scripts; the main config still has phase 703.40 (seb to decide).
  **Thermal photons:** M1 ~3 % (removed by the parity pre-selection), S4 ~1.5 % (NOT removed; enters M1 at the env swap).
  chi(qubit-M1) = 300 kHz (277.5 kHz on the Ramsey axis), linear in n.
  **Photon-number readout without Wigner:** qubit Ramsey with active reset + parity pre-selection
  (`ic_debug/ramsey_pn/`): +-0.005-0.008 per population in ~4-11 min, no displacement, no reconstruction.
  After a frequency event (10-04 ~05:00: qubit and M1 +28 kHz) recalibrate qubit f_ge and M1 f_ge, not only the
  multiphoton table.
- **Physics state (eta 0.35, Rung 1):**
  - 0.3 mA, four sets on 10-05/06: gaps -0.068 / -0.209 / -0.080 / -0.249 (mean -0.15 +/- ~0.05); best Ic +0.095.
    0.25 mA (10-02/03): pooled Ic 0.083 +/- 0.015, gap ~-0.12.
  - **HOM (pre-selected Ramsey): P0 > P2**, P1 at the ideal value; +0.03 to +0.05 after the n = 2 visibility correction,
    +0.05 to +0.08 without it. The pair-loss model was FALSIFIED by a direct test (|2> + two full swaps: no loss); every
    piece of the HOM preparation is clean. Cause unknown.
  - **Preparation failures are f-type:** |1> 1.3 %, |2> 2.5 % (e/f split). The n = 2 Ramsey fringe is ~6 % weaker
    (cause unknown).
  - **Ic after a full recalibration incl. the single shot at relax 8000 (10-06 18:46): gap +0.001**; the next two sets
    -0.09 / -0.09. The bare inputs keep full coherence (5 / 5); the loss is in the channel. Readout drift is not the
    cause (per-job readout re-analysis of 25 sets). The pump alone does not dephase.
  - **Swap-gain scan (10-07, 9 sets, x0.5-x1.5, reference interleaved, each set with its own swap calibration):** no
    measurable dependence of the gap on the swap gain (reference mean -0.076, others -0.114, difference ~1 sigma). The
    reference sets alone scatter from -0.006 to -0.167 (std 0.076, error 0.042) with a stable swap calibration (gain
    3750-3761, freq within 1 kHz). So the variation is not the swap strength and not the swap calibration.
  - **10-08: the set-to-set variation is consistent with measurement noise.** Per-job bootstrap errors (parity counts):
    the channel coherence ratio has +-0.07-0.08 per 250-rep Wigner (the ideal channel |rho02| is .166, 3x smaller than
    the bare .5). chi2 ~ dof for the 20-cycle time series at fixed settings (10-07 22:54 - 10-08 05:24) and for the part C
    sets; the linear Ic of the 5 reference sets also scatters as its bootstrap error (chi2 4.7 / 4). The ML Ic scatters
    0.073 vs its quoted 0.040: **the ML Ic error is ~1.8x too small** (use ~0.07 per set, or repeated sets).
  - **The real effect: a constant channel coherence deficit.** Time series: |rho02| 0.880 +/- 0.018, |rho13| 0.932 +/- 0.015
    of the ideal; bare '+' 0.990 +/- 0.006. Over 10 h and all swap gains. S4 thermal 0.2-0.8 % (not the cause); qubit / M1
    stable; sidebands drift ~9 kHz / h (refresh at least every ~30 min).
  - **10-08 afternoon, the deficit located (same-session, measured eta 0.333 of this swap, not 0.35):** env photon alone
    (env |1> + '+', no swap) 0.98 +/- 0.015; swap alone ('+' + swap, S4 empty) 0.99 +/- 0.03; full channel |rho02|
    0.853 +/- 0.028 (-3.3 sigma vs the swap alone), |rho13| 0.985 +/- 0.025. With the |0>+|1> input: channel |rho01|
    0.766 +/- 0.051, |rho12| 0.981 +/- 0.017. **In both channels the coherence that needs two interfering paths (env and
    M1 photons stay / exchange) is low, the single-path one is at the ideal.** A random phase growing ~N^2 and a static
    Kerr during the swap are disfavoured. Open: reconcile with the HOM populations (10-05, read as "little
    distinguishability"). Details: `ic_debug/losssplit/locate*.md`, `samesess.md`, `phase01.md`.
  - **10-09 correction and budget (`docs/log/2026-10-09_transduction-review-hom-speed.md`):** pooled over 59 jobs at eta
    0.33, |rho02| 0.886 +/- 0.010 (robust to the eta choice), |rho13| 0.93-0.95 (NOT at the ideal; the line above is
    superseded). With the ordinary M1 decay (MODEL, missing from the ideal channel) and the env vacancy, the single-path
    coherences are explained; the two-path ones keep -6 % (rho02) and -15 % (rho01). Wigner displacement scale right to
    ~0.5 %. Swap eta with S4 occupied = single-photon eta (within 0.006). HOM excess P0 - P2 ~+0.045 at eta 0.33, not
    pump-driven, about the same at 0.75x-2x swap speed; the env vacancy rises with the pump gain (tentative).
  - (Older reading, now superseded: "the set-to-set variation follows the coherence."): the four 0.3 mA sets order exactly with mean |rho02|/ideal
    (0.99 / 0.92 / 0.84 / 0.79). Source open.
  - Wigner populations: +-0.015 per job (102 alphas); the Wigner "5 % |1> in |2>" is probably the |2> prep failures.
  - Ruled out earlier: preparation errors as the main cause, cross-Kerr after the swap, drift between the inputs of a
    set, estimator compression, swap-angle jitter (1-photon swap clean over 24 swaps), pump-induced loss.
  - Details: `ic_debug/losssplit/RESULT_2026-10-06.md` (night), `ic_debug/ramsey_pn/RESULT_2026-10-06.md` and
    `ic_debug/LOG.md` (10-06 sections), under `C:\experiments\260601_Transduction_sandbox\`.

## Next

0. **Ic(eta) curve and budget DONE (10-12; `ic_debug/final/REPORT_2026-10-12.md` v2, reviewed externally):** Ic / Fe follow
   the ideal shape; gap ~0 at full / no swap, -0.10..-0.15 at eta 0.28-0.58. M1 + S4 decay and a coherent 5 % |2> error
   explain 40-60 %; the remainder (-0.04..-0.09, -0.09 at 0.33) is a loss of the two-path coherence |rho02| (its null at
   eta 2/3 is filled). **The env vacancy is ~0** (the 10-08 "1.5-2 %" was decay counted by the A / B formula). Analysis
   rules: reference = coherent eta_c from B / A simulated with decay; frame-align sets before pooling. Next (seb): '+' at
   eta 2/3 and 0.33 with the swap frequency scanned. Details: `docs/log/2026-10-12_transduction-ic-eta-sweep.md`.
1. (10-08/09 reading, see the 10-09 correction above) The channel coherence deficit: only the two-path coherences (rho02 for |0>+|2>, rho01 for |0>+|1>) lose ~15-23 %.
   About a third of it is the env vacancy (env photon missing at the swap in ~1-2 % of the shots, ordinary prep +
   decay; opposite-sign coherences make it count 2-4x). At eta 0.33 a two-path-specific loss of 11-16 % remains
   (|rho01| 0.839 +/- 0.033 after the vacancy, 10 jobs). Excluded: distinguishability, a photon-number-sector phase,
   static Kerr, random cross-Kerr (~6 sigma), a photon-number-dependent swap angle (n2swap: linear to ~1 % for N <= 2).
   Night eta scan (10-08/09): at eta 0.2 / 0.5 / 0.8 BOTH coherence types are ~11 % low; only at 0.33 is the single-path one
   ideal. Bare and env-only controls stay at ~0.96-0.98 (readout fine). Next (seb to decide). Use the measured eta.
2. Readout: drifts ~2-3 deg / h; jumps when the QICK server restarts (10-06 -48 deg, 10-07 ~-37 deg, 10-08 ~+158 deg;
   the restart also breaks the worker's Pyro link: restart the worker). `losssplit.py` now bootstraps with a plain single
   shot when the active-reset one fails. The main config still has phase 703.40 (seb to decide).
3. HOM P0 > P2: cause open; the n = 2 Ramsey visibility (~6 % low) too.
4. Optional: rebuild the 9 missing version files from the job HDF5 attrs (seb to decide).
5. The swap EA frequency fit (product-peak Gaussian) fails at a coarse step (the peak is ~3 kHz wide): `losssplit.py`
   uses the mirror-symmetry center + guards. The same issue may affect other scripts that use the product fit.

## Docs

- `docs/log/2026-10-12_transduction-ic-eta-sweep.md`: alternating 2x / 1x HOM pass, Ic(eta) curve, analysis fixes.
- `docs/log/2026-10-09_transduction-review-hom-speed.md`: review follow-up, budget with decay, eta with S4 occupied,
  HOM asymmetry vs swap speed, shared-queue incident.
- `docs/log/2026-10-09_transduction-eta-scan-night.md`: |rho01| result, channel eta scan, n2swap, controls.
- `docs/log/2026-10-08_transduction-deficit-located.md`: the channel deficit located (pieces, same session, |0>+|1>).
- `docs/log/2026-10-08_transduction-time-series.md`: time series at fixed settings, noise analysis.
- `docs/log/2026-10-07_transduction-swap-gain-scan.md`: drift check, swap-fit fix, swap-gain scan.
- `docs/log/2026-10-06_transduction-loss-split.md`: 0.3 mA, Fock readout check, loss split.
- `docs/log/2026-10-04_transduction-tls-event.md`: interference zeros, pump test, device event.
- `docs/log/2026-10-03_transduction-gap-search.md`: 10-02 afternoon to 10-03.
- `docs/log/2026-10-02_transduction-night-sweep.md`: night sweep and morning tests.
- `docs/log/2026-10-01_transduction-ic-debug.md`: the 10-01 debugging.
- Full record: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md`.
