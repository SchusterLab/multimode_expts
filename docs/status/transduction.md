# Transduction (Fock-state decoder, lossy-beamsplitter channel)

**Last updated: 2026-09-30** (seb + Claude). Work happens on `main` (`C:\python\multimode_expts`);
the refactor came from branch `transduction_refactor` (merged 2026-09-30).

## Where things stand

- **Code:** `experiments/transduction/`
  - `channel_model.py`: ideal channel, decoder, Fe, Ic (Ic projects rho_1, rho_2 to physical
    before the entropy).
  - `sequences.py` (new): logical preparation and the channel recipe `channel_prep`.
  - `process_tomo.py` (new): measure a 4-input set (queued together), load it from the job HDF5
    files, metrics with one phase rule for data and theory (Rung 1 and Rung 2), joint bootstrap,
    linearity check, population-only fast check, results files
    (`<experiment>/assembled_data/*_TransductionChannelSet.h5`), `plot_sets`.
  - `estimators.py` (new): joint maximum-likelihood fit of the channel (CPTP, photon-number covariant,
    difference-parity likelihood) for Ic; falls back to the non-covariant fit when the chi2 check fails.
    `analyze_set` gives Fe from linear inversion and Ic from the fit (Rung 1); the old Ic stays as `Ic_linear`.
  - Tests: `tests/test_transduction_process_tomo.py`, `tests/test_transduction_estimators.py` (some use the 9/30 data on pippin).
- **Notebook:** `measurement_notebooks/QEC/transduction_sandbox.ipynb`, 40 cells: setup, encoder
  phase, decoder phase, channel checks (3a-3d), Rung 1, Rung 2, analysis from disk. The Sep-2026
  debug cells are removed; their record is
  `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md`.
- **Physics state (eta 0.35, Rung 1):** measured Ic about 0 (theory +0.21), Fe about 0.49-0.59
  (theory 0.75). The Aug-2025 "match to theory" was a qutip < 5.2 entropy artifact (negative
  eigenvalues); with today's pipeline the Aug-2025 data also gives Ic about +0.01. No regression.
- **Error budget (seb agreed, 2026-09-30):** |2> preparation (about -0.10 in Ic), estimator bias
  (new fit: about -0.01 for our channel, -0.05 for a perfect one at 250 reps; Fe not biased), swap-stage loss (-0.11 good run, up
  to -0.26 bad run; varies from run to run).

## Next

1. seb runs one real set with the new notebook (Sections 0, 3d, one Rung 1 point) and checks it.
2. Estimator: done (study in `C:\experiments\260601_Transduction_sandbox\ic_debug\estimator\REPORT.md`).
   With it, the hardware gap at eta ~0.38 is 0.125 +/- 0.022 (ideal channel through the estimator minus data).
3. Hardware: |2> preparation (f1-g2 calibration; 3% in Fock 3 for a prepared |2>), then the
   swap-stage loss (S4 T1/Ramsey, repeated swaps).

## Docs

- `docs/log/2026-09-30_transduction-refactor.md`: this session.
