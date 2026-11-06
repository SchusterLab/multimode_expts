# Inherited sequential AGP and a calibration Experiment

The user approved inheriting the existing AGP classes, then authorized completion
of the notebook. They required the existing code style and separate Experiments
for population acquisition, confusion calibration and AGP assembly. Existing
Python modules and the reference notebook must remain unchanged.

## Changes

- Added `experiments/qsim/agp_sequential.py` with
  `CentralBosonAGPSequentialExperiment` and
  `CentralBosonAGPSequentialDisorderScan`, inheriting the existing AGP classes.
  One direction retains the existing API and aggregate array shapes. Private
  in-memory views select processed arrays for the existing AGP calculations;
  no raw-IQ copies or exported population files are needed.
- Native sequential children, source readout counts and metadata remain intact.
  Part manifests reference the original JOB files; scan manifests reference
  the saved parts. Analysis results enter `self.data` before inherited HDF5 saving.
  Each completed W is saved immediately under an explicit name.
- Added `experiments/qsim/multiparity_confusion_matrix_sequential.py` with
  `MultiParityConfusionMatrixSequentialExperiment`. It reuses the sequential
  Program, sweeps four occupations per target step in one JOB, and saves raw IQ,
  preparation metadata, target retained counts and the `(4, 4, steps)` matrices.
  Calibration is a separate Experiment, without a calibration branch in the
  population Experiment.
- Removed the earlier `calibrate_readout_steps` function and its helper-specific
  test. Calibration now uses an ordinary `CharacterizationRunner.execute()`.
- Completed `measurement_notebooks/jonginn/agp_sequential_measurement.py` using
  the reference notebook's defaults, settings, two runners and direction loop.
  Population and AGP jobs receive the calibration HDF5 filename. The notebook
  includes population plotting, named AGP acquisition, display and manifest reload.
- Added AGP and calibration tests and retained the sequential population tests.
  No existing Python source, base class, worker, package registration or reference
  notebook was edited. No git commit was made.

## Validation and limits

- Raw and corrected storage, Kerr and non-autocorrelation results agree with
  independently constructed legacy scalar inputs. Source configuration and raw
  data remain unchanged. Direct manifest reload and completed-W recovery pass.
- Calibration tests exercise normal `go(analyze=True, save=True)`, HDF5 reload,
  target-step matrix selection, varying retained counts and actual mock acquisition.
- The new notebook runs through a short mock calibration, five-mode population walk,
  both Kerr and S7 AGP directions, two W values, plots and saved-layer reload.
  Mock IQ cannot measure calibration fidelity: a marked synthetic identity matrix
  was used only for the corrected downstream smoke check. Acquisition remained local
  with mock instruments; no real queue, hardware or vault was accessed.
- Final targeted pytest run: 66 passed across the three new sequential test
  modules and the existing AGP part, disorder-scan and population-set tests.
  Ruff passes for the changed Python files.

The calibration JOB contains real raw-buffer layout, so the inherited AGP
calibration-error reader can obtain its acquisition shot count. This preserves
the existing error convention, including its use of total acquisition shots;
actual retained target counts are also saved. No new uncertainty estimator or
readout-fidelity claim is introduced.

This implementation supersedes the earlier helper-based calibration and
export-based AGP notebook proposals. Hardware validation remains pending.
