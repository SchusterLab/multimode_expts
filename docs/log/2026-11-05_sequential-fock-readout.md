# Sequential Fock population readout

The user authorized implementation of the agreed standalone module and tests after
pulling the current checkout. Existing Python classes and modules must remain unchanged.
Human readability, familiar code patterns, and English repository records take priority.

## Implementation

- Added `experiments/qsim/fock_population_sequential.py` with the three planned classes:
  `SequentialMultiparityAnalysisMixin`, `FockPopulationSequentialReadoutProgram`, and
  `FockPopulationSequentialReadoutExperiment`.
- The new Program inherits preparation, evolution and `body()`. Its complete one-mode
  multiparity block reuses the existing pulse implementation, then completes the second
  measurement. Intermediate blocks reset only the qubit. Terminal measurement and relaxation
  happen once inside multiparity; the new Program's outer `measure_wrapper()` is a no-op.
- Each readout swap has a separate waveform prefix. The existing `man_stor_swap` prefix is
  already used during preparation; repeating that name reallocates the QICK envelope and
  replaces its upload metadata. The new readout uses the existing gate-list helper with a
  distinct prefix and retains the calibrated 0.2 us settling wait.
- Raw IQ remains interleaved. Mode order, absolute bit-lane pairs and herald indices are
  saved with the job. A shared herald mask preserves same-shot correspondence, including
  differing retained counts at different sweep points.
- `analyze()` stores all numerical results in `self.data`, including probabilities, raw and
  corrected moments, errors, sample counts, and the applied matrix snapshot. Normal
  `go(analyze=True, save=True)` persists those arrays.
- `calibrate_readout_steps` runs four preparations per target step through the existing
  runner and saves matrices with axes `(prepared number, measured label, readout step)`.
  Background occupations, retained counts and source files are recorded.
- `export_mode()` writes a scalar processed HDF5 with the existing Fock population field
  names, selected calibration, timing/couplings, and raw-source provenance. Existing AGP
  construction, analysis and named manifest reload can consume those projections.
- Reused `write_assembled_h5`; no new HDF5 writer or global photon-number constant remains.

## Review decisions

The user requested alignment with their commits `633a74b`, `0a51974`, `34e3df1`, and
`4064d77`. Analysis now uses local photon-number weights and explicit per-step correction
and error calculations. The implementation is restricted to the functions in the approved
plan. Existing Program, mixin, calibration, AGP, runner and persistence source files were
not edited. No git commit was made.

## Validation

- New tests cover pulse ordering/counts, exactly one terminal measurement, one-mode
  behavior, shared herald masks, per-step matrices, normal HDF5 saving/reloading,
  calibration collection, processed exports and existing AGP manifest reload.
- Mock compilation/acquisition covers all eight readout modes. No real hardware was used.
- Existing Fock population, multiparity, readout, Program ASM, acquisition and AGP
  regression tests: 229 passed.
- New sequential-readout tests: 20 passed. Together with the 229 existing regression
  tests, 249 targeted tests passed. The new tests explicitly exercise normal
  `go(save=True, analyze=True)` and confirm that analyzed `self.data` fields are HDF5
  datasets after saving and remain available on reload.
- Ruff and `git diff --check` passed. Initial test setup failures were local temporary-
  directory permissions and an incomplete pandas installation; explicit writable test
  directories and the repaired local environment resolved them.

The local installed pixi could not read the pulled lockfile. A portable pixi executable
was used with `run --as-is`; missing or partial packages were repaired only inside the
existing local `.pixi` environment. No tracked lockfile, manifest or global tool changed.

## Limits and next steps

This implements the existing two-bit/mod-four measurement; higher parity pulses are a
future change. Hardware readout fidelity and populated-input swap fidelity need device
validation. Per-step matrices are effective calibrations for their recorded sequence and
background, not a correction of loss during the target evolution.

Current AGP supports raw and corrected scalar projections. Its calibration-uncertainty
estimator additionally needs a compatible raw calibration file; the matrix-only tensor
artifact preserves corrections and provenance but does not provide that legacy raw format.
AGP therefore reports shot-only errors when that raw calibration is unavailable.
