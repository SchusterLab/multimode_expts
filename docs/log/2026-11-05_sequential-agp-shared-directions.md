# Native directions in sequential AGP

The user requested that the new sequential AGP support multiple directions
itself, using the standard acquire/analyze/save workflow. Legacy AGP remains
scalar and unchanged.

- `CentralBosonAGPSequentialExperiment` owns one union of initial occupations
  and one set of sequential JOB children per W. The union is computed from
  selected directions: N=3 needs 15 JOBs for S7, 25 for S4+S7, 19 for Kerr+S7,
  34 for all storages and 35 for all five directions.
- `analyze()` collects every readout step once and calls the existing fitting
  and theory functions for each direction. No temporary Experiment objects,
  per-direction acquisition API, or per-direction output files are used.
- Results enter `self.data` before saving: populations have
  (state, cycle, readout step, label), C1 has (direction, cycle), AGP has
  (direction, cutoff); the scan prepends W. Undefined Kerr C2 rows are NaN when
  storage C2 is present. Manifests persist directions and reference original JOBs.
- The notebook plans one scan with `direction=AGP_DIRECTIONS`, calls `acquire()`
  once, and reloads one scan manifest. Each W saves all directions together.
- Optional walk theory follows the reference notebook, using recorded timing
  and couplings. At the user's request, prepared-state evolution stays directly
  in the notebook; the extra theory helper module and its separate test file
  are removed. The notebook overlays every mode.

Validation covers selected-direction occupation counts, agreement with legacy
raw/corrected correlations, AGP/T1/spectra/theory, raw-source preservation,
step-specific correction, saved-data reload and completed-W recovery.
37 sequential tests and Ruff checks pass. The notebook's walk theory cells
also pass against a local copy of a five-mode JOB, with matching curve counts.

All implementation edits are local. Existing base, legacy AGP, readout pulses
and configs are unchanged. Pippin was used only for read-only execution-time
records; no remote files, measurements or device settings were changed.
