# Remove the unused per-mode population writer

The user requested removal of `save_single_mode_population_h5()` unless the
assembled workflow needs it. The current sequential AGP reads original JOBs
directly, so the writer is unnecessary.

- Removed the complete method, its now-unused imports and its two raw/corrected
  conversion tests. Native acquisition, whole-JOB HDF5 saving and assembled
  analysis are unchanged.
- Population JOB HDF5s retain all readout modes, raw shots and analyzed arrays.
  Calibration remains a separate JOB containing the full `(4, 4, steps)` tensor.
  AGP part and scan HDF5s save their aggregate results and reference original
  JOBs through manifests. No intermediate per-mode files are produced.
- Existing base modules and the reference notebook remain unchanged.
- The user also identified hardcoded display loops in the new calibration.
  Restored the existing calibration module's `np.ndenumerate(matrix)` style
  and derived plot ticks from matrix shape. The two-bit protocol and its
  four prepared Fock states are unchanged.
- At the user's request, further higher-parity generalization is deferred.
  A TODO by the calibration class records that pulses, decoding, prepared
  states and matrix dimensions must be extended together.

All 31 tests across the three new sequential test modules pass. Ruff and
`git diff --check` pass. No real hardware was used.

This supersedes the earlier records retaining the optional mode-export method.
