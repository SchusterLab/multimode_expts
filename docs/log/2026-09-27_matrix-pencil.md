# 2026-09-27: analysis surface, first pass; Matrix Pencil cleanup

Session record; not edited after this date. The current state is `docs/STATUS.md`.

The refactor moves from cut-and-paste to line-by-line review. Two surfaces: measurement (on hold
until a run on the device) and analysis (this session). Method: pick a data set, run the
canonical load -> analyze -> display flow in the analysis notebook, then go down the call layers.

## Coarse pass: July N=3 (`analysis_notebooks/202609_qsim_migration/mbr.py`, section 1)

Off-host over the SMB mount: import 5 s, `from_manifest` 13.6 s, `analyze(spectrum_method='mpm')`
0.8 s (now 0.15 s), `display` 0.1 s.

- Loading is fixed: manifest -> relative `raw_files` -> `load_experiment` per HDF5 -> instance by
  `cls.__new__`. No database, pickle or job-ID lookup. All load time is SMB file opens
  (70 files, about 0.1-0.2 s each).
- Left over (not done): the job-ID path in `experiments/saved_jobs.py` (`load_aggregate`,
  `select_jobs`, `SavedJob`, the sidecar) is used only by tests and `tools/migrate_mbr_jobs.py`;
  `resolve_timing` keeps four fallbacks; `coherent_trace_spectrum` has its own copy of the FFT.

## Done: `fitting/qsim/matrix_pencil.py` (guan asked for it; 1100 -> 790 lines)

- Settings: `MatrixPencilSettings` (pydantic, frozen, `extra='forbid'`). Declared limits replace
  the if/raise chains; a misspelled `mpm_*` option raises. Dropped `dedup_frequency_tolerance_MHz`
  (only the removed trace function used it). The per-row calibration errors are an argument, not
  a setting.
- `analyze_matrix_pencil(reconstruction, time_us, settings, row_frequency_standard_errors_MHz)`:
  no `spectrum` argument. The FFT of the fitted returns moved to the spectrum
  (`fitting/qsim/mbr_spectrum.py`: `windowed_fft`, `local_spectrum`; `analyze_spectrum` uses the
  same function). `MBRSpectrumExperiment.analyze` attaches it as `matrix_pencil.spectra`.
- Result: nested only (`modes`, `fit`, `candidates`, `resolved`, `settings`, `sampling`,
  `row_diagnostics`). The flat duplicate keys, the always-empty lists, the three equal scores and
  the fields derivable from others are gone. Row candidates and tracks are dataclasses (most of
  the old run time was `AttrDict.__getattr__`).
- `analyze_matrix_pencil_trace` removed; the per-row step calls `_row_candidates` directly.
  `refit_occupation` -> `refit_row(result, row_trace, row)`; the spectrum-side display fields are
  in `MBRSpectrumExperiment.analyze_matrix_pencil_occupation`.
- `MBRSpectrumExperiment`: one `_spectrum_method` (was three copies; `'rowwise_matrix_pencil'`
  alias dropped); `mpm_calibration_sigma_multiplier` and `mpm_merge_frequency_tolerance_floor_kHz`
  removed (same as the settings fields `mpm_merge_frequency_tolerance_sigma` / `_floor_MHz`, same
  defaults); `_matrix_pencil_merge_options` -> `_calibration_frequency_errors_MHz`.
- Deprecated code: one class-level binding removed from `deprecated/encoding_spectroscopy.py` so
  it still imports. Its Matrix-Pencil calls (`legacy_mbr`, `mbr_spectral_validation`) now fail at
  run time; not repaired (frozen). Two `test_mbr_stage_split.py` tests of the deprecated option
  parser and two that pinned the old signature were deleted.

## Tests

- `tests/test_matrix_pencil_regression.py` (new, 3 s, no mount): two saved reconstructions
  (Aug-15 quick plot, August N=3) x four setting variants (default, disorder ensemble, 0.5 bins,
  calibration merge). Expected outputs made by the old code (commit `f57c202`); the new code
  matches to 1e-12. Mutation check: the tracking default 1.5 -> 1.49 fails 4 of 8.
- `tests/test_matrix_pencil_synthetic.py` (new, 0.4 s): known poles and weights; noise levels,
  aliasing, off-diagonal rows, a one-row pole, row refit, a pair 2 bins apart, growth, bad settings.
- Golden: the two Matrix-Pencil baselines re-blessed after checking that every field kept at the
  same path is unchanged (only the settings record changed: it holds the given values; the
  resolved ones are in `resolved`). All 15 pass with `--runxfail`.
- `tests/test_mbr_disorder_ensemble.py` (3.5 min over SMB): 17 pass with `--runxfail`. Its
  old-preview comparison ran the deprecated preview live; that code no longer runs Matrix Pencil,
  so its output was frozen once with the old `matrix_pencil.py` into
  `tests/data/disorder_preview_71_r01.npz`. Every other file passes.

## Found (open; decisions for guan)

The synthetic tests found three defects of the algorithm, pinned as `xfail(strict=True)`:

1. `requested_max_modes` caps the rank sweep as well as the number of poles kept. At the true
   pole count, the late poles cannot persist `minimum_consecutive_ranks` ranks and are dropped
   (4 poles, 1% noise, `requested_max_modes=4` -> 3 poles, 40% residual; with 30 -> all 4 within
   0.02 bin). The disorder ensemble sets it to the Hamiltonian dimension, which can equal the
   number of distinct levels.
2. Noise-free data fails the same way: the sweep stops at the numerical rank.
3. The default 1.5-bin tracking and dedup tolerances merge poles 1-1.5 bins apart (1 bin: one
   pole, 90% residual); with 0.2 bins a pair 0.5 bin apart is resolved. This bears on the
   near-degenerate levels.

Also: `clip_growth=True` biases the weight of a growing pole (1.04 against 0.75). On July N=3,
MPM finds 7 of the 8 distinct aliased levels; it misses the 80.9 kHz level (folded to -41 kHz).
