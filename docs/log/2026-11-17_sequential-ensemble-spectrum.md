# Sequential ensemble spectrum display

Jonginn reported that `reloaded_scan.display_spectrum()` failed. The new joint
scan inherited `CentralBosonAGPDisorderScan.display_spectrum()`, which expects
`C1(W, time)`. Joint scans instead save `C1(W, direction, time)`. A regression
reproduced the matrix-multiplication dimension mismatch in the inherited method.

## Change

Only `experiments/qsim/agp_sequential_ensemble.py` changes production code. The
joint scan overrides `display_spectrum()` and applies the existing
`fitting.qsim.agp.cosine_transform_weights()` along the time axis. It uses saved
raw/corrected correlators and the independently saved theory correlator, with
identical frequency grid and window. Rows are W, columns are selected local
probe directions; 0 denotes Kerr. Optional AGP filters are normalized to unit
peak. The supplied reference plot appearance and linear axes are retained.

The display works from an arrays-only assembled HDF5. It neither reloads raw
JOBs nor calls `analyze()`, bootstrap or model construction. It does not alter
the saved AGP, normalize C(0), subtract backgrounds, compensate decay or complete
unmeasured tails. No spectral uncertainty band is inferred: the saved pointwise
`C1_err` does not supply the required cross-time covariance.

## Verification

- All 10 tests in `tests/test_agp_sequential_ensemble.py` pass.
- Two new parameterized cases reload saved two-direction correlators, recover
  four known cosine frequencies and their analytic spectral peak amplitudes,
  compare the matching theory curves, select a single direction and forbid
  accidental reanalysis.
- The completed batch was read over SSH without modifying remote files; local
  all-direction and Kerr-only PNGs verify the default filter overlay path.
- `git diff --check` passes. Existing notebook edits were retained. No remote
  edits, deployment, staging or commit was performed.
