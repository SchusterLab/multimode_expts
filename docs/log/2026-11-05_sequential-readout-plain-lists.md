# Use plain lists for sequential readout configuration

The user explicitly rejected `np.tile`, `np.atleast_1d` and similar convenience
calls in the new code. Human-readable ordinary Python takes priority.

- Replaced readout scalar/list normalization with explicit branches. Scalar
  `ro_stors=0` remains supported. Calibration reuses `readout_layout` rather
  than implementing another normalization path.
- Build calibration step/number lists with explicit nested loops.
- Replaced `tile`/`repeat` in tests with ordinary lists and loops. Replaced
  shape-copy allocations with explicit shapes, and test-only index/pair helpers
  with ordinary enumeration/index loops.
- Existing base files, protocol depth, numerical equations, HDF5 field names
  and notebook measurement settings are unchanged.

Ruff passes for the three new modules, their tests and the new notebook.
All 31 sequential tests pass. A separate compatibility check confirms scalar
zero, list, tuple and NumPy-array readout inputs retain their previous behavior.

The user additionally rejected `slice(...)` objects and uninitialized
`np.empty(...)` arrays. Calibration row selection now uses named start/stop
indices and ordinary bracket indexing. Replaced uninitialized allocations
with `np.zeros` in the new source/tests, preserving shapes and dtypes.
The arrays are overwritten by the same calculations; existing NaN handling
for empty accepted ensembles is unchanged.
