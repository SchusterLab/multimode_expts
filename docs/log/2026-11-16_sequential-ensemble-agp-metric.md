# Heralded sequential disorder ensembles and AGP metrics

Implemented locally at jonginn's request. Code, comments and notebook prose are
English. Existing user edits were retained; nothing was committed or submitted
to the device, and no remote file was changed.

## Acquisition and calibration

- New `experiments/qsim/fock_population_sequential_ensemble.py` combines the
  existing ensemble and sequential programs by inheritance. DMEM upload,
  frequency construction, pointer increments, pulse emission and phase updates
  remain in their existing implementations.
- The ordinary heralded sequential Program/Experiment provides the notebook's
  zero-detuning walk, using the same calibration and without a disorder table.
- Added two backward-compatible hooks to `fock_population_sequential.py`:
  `decode_readouts(accepted_only=False)` preserves rejected shot slots, and an
  overridable `readout_layout()` inserts preparation readout lanes.
- New heralded sequential calibration subclasses retain existing per-step
  confusion-matrix statistics. Vacuum gets a dummy herald. This calibration
  path uses end heralds and vacuum background occupations.
- Labels have axes (time, realization, readout mode), masks (time, realization).
  The mask is common across readout modes within a shot. The full ordered
  patterns, detuning table, DDS words and original realization IDs are saved.

## Estimation and persistence

- `fitting/qsim/agp_metric.py` uses the Fock-diagonal derivatives
  `(q_M1, -n_storage, ...)`, known initial-state values and calibrated output
  label scores. Kerr uses the label function n(n-1)/2, not the mean-number
  function. Cross correlators are symmetrized before the common kernel integral.
- Realization-specific W weights act on both initial and output operators
  before accepted-shot averaging within each initial-state/time row. Patterns
  are saved separately from detunings so the W derivative survives W=0.
- `agp_metric_local` is the averaged local-control tensor; `agp_metric` is the
  realization-projected K/W tensor. Both have units kHz^-2. Multiplication by
  the recorded reference coupling in kHz squared gives `agp_metric_scaled`.
  The diagonal `agp` retains the old dimensionless convention and direction axis.
- Whole-realization bootstrap draws are shared across states, times and modes;
  calibration draws are shared across the same data. Requested and valid draw
  counts are saved, including skipped singular sampled confusion matrices.
  Shot noise and finite-ensemble variation are not independently identified.
- Missing parameter directions have NaN entries and an availability mask.
  Eigenvectors diagonalize the averaged K/W matrix, with no PSD clipping.
- `experiments/qsim/agp_sequential_ensemble.py` reuses the existing state union,
  child acquisition, save and manifest reload. The new scan saves `agp_list`
  and `agp_metric_list` as requested, alongside the canonical array keys.
- Independent theory is in `theory/qsim/agp_ensemble.py`. It diagonalizes each
  supplied realization once and saves infinite-time and analytic finite-window
  metrics. Sample-grid comparisons use the same measured kernel in the
  Experiment. Prediction arrays never enter the measured estimator.

## Notebook and verification

`measurement_notebooks/jonginn/agp_sequential_ensemble_measurement.py` preserves
the familiar calibration, ordinary walk, settings, planning, acquisition,
display, save and reload cell order. It uses one joint scan rather than
repeating acquisition per direction. A fixed-K scan saves local K/W directions;
a two-dimensional flow field still requires scans at multiple physical K values.

Verification covers real mock compilation/acquisition and HDF5 saving for the
new Program, original realization slots after herald rejection, per-step
calibration including vacuum, independent correlator/spectral checks of signed
cross terms and units, the complete occupation union, raw-IQ-free replay and
linear-scale displays. The installed QICK wheel lacks the firmware HDL used by
the pre-existing timing checker, so the new mock tests disable only that checker
inside their test fixture. This does not validate real FPGA timing.

Tests use the existing `tmp/agp_python_runner/pixi.toml` through `pixi run` and
the environment's DLL paths because the installed Pixi cannot read the root
lockfile version. No environment, dependency or hardware configuration changed.

Final checks: 64 targeted acquisition/metric/assembly and existing AGP tests
passed, plus 20 existing sequential/calibration tests. Two old mock tests in
the latter suites were deselected because their firmware timing source is
unavailable; the new path has its own mock acquisition/calibration coverage.
The notebook's actual imports, planning cells and display return types were
checked locally. Its current settings plan 35 states per W, with the original
patterns retained in every W=0 JOB override.

All functions in changed implementation modules are at most 71 lines. The
bootstrap estimator reuses label-linear sufficient statistics for central and
resampled estimates. At shape (35,200,500,5), a two-draw benchmark took 1.57 s
with 218 MiB peak traced allocation; 300 draws extrapolate to about two minutes
per W on this machine. This is a processing estimate, not a device runtime.

## Follow-up: remove diamond inheritance

Jonginn requested avoiding diamond inheritance even when a short copied block
is necessary, and documenting the source and reason in English docstrings.

- Extracted existing sequential pulse method bodies unchanged into
  `SequentialMultiparityProgramMixin`, which has no Program parent. Ordinary,
  heralded and ensemble Programs attach this behavior to one Program chain.
- The joint population Experiment now has only the heralded sequential parent.
  Its short acquisition/data-shaping methods are adapted from
  `FockPopulationEnsembleExperiment`; their docstrings disclose that this avoids
  diamond inheritance. DMEM upload and Floquet/pulse implementations remain shared.
- The heralded calibration Experiment has only its existing calibration parent.
- Joint AGP Experiment and DisorderScan each have one ensemble parent. The
  occupation-union setup is adapted from the sequential constructor with a
  source/reason docstring, and sequential plots are explicitly delegated.
- Public class names, notebook calls, HDF5 schemas and metric estimators remain
  the same. Graph regressions check that our population/assembly bases are not
  reached through duplicate inheritance paths; vendor framework inheritance is
  unchanged. No new files or hardware actions were needed for this refactor.

Refactor verification: 94 tests passed, with the same two old missing-firmware
mock tests deselected. Added ancestry and call-count checks confirm one path to
our application bases and one population-base initialization per combined
Program initialization. All six extracted sequential Program method ASTs match
the original implementations exactly. The maximum implementation function
length remains 71 lines. No fitting or theory arithmetic changed.
