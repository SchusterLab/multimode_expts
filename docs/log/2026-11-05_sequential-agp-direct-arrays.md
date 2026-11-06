# Collect sequential AGP arrays directly

The user rejected the temporary Experiment inputs and initialization bypass in
`_create_single_mode_analysis_experiments()`. This follow-up removes that path,
rather than renaming it again. Existing base modules remain unchanged.

- Added the narrow `_SequentialPopulationAnalysis` inheritance layer. It collects
  the requested readout step directly from original JOB arrays into `self.data`,
  including probabilities, moments, errors, counts, selected calibration and timing.
- `CentralBosonAGPSequentialExperiment` inherits the existing AGP calculations.
  Its population collection resolves to the sequential layer instead of the old
  scalar collector. No temporary Fock population or legacy AGP object is created.
- Original JOBs are ordered by prepared occupation. The original children/configs
  are retained, and saved manifests still point directly to the sequential files.
- Removed the old scan planner instance too: `from_pattern()` creates only the
  actual new parts, using the same existing pattern normalization.
- Updated the existing integration tests to check direct arrays and inherited
  numerical analysis. The native path is guarded against calling the legacy AGP
  factory. All 9 integration tests pass, including numerical agreement, source
  preservation and named HDF5/manifest reload.
- The short notebook mock pipeline passes again after this change: calibration,
  five-mode walk, Kerr and S7 directions, two W values, plots and saved-layer reload.
  As before, a marked synthetic matrix is used for corrected mock analysis; no
  device fidelity is inferred. Ruff and `git diff --check` pass.

This supersedes the earlier records describing temporary scalar input objects.
The optional single-mode HDF5 writer retains its clarified name
`save_single_mode_population_h5()` and is independent of AGP acquisition.
