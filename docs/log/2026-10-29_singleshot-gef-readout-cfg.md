# 2026-10-29: single-shot g/e/f analysis, `readout_cfg`, one postprocessor

seb + Claude. Gap items 1 and 2 of `measurement_notebooks/guan/yoko450ua_calibration.py`
(last cell).

## Done

- **`fitting/fit_display_classes.py` `Histogram`:** the rotation is always g -> e on +I, also
  with f shots (before, f shots changed it to the e-f axis, so the angle, the ge threshold and
  the "gf" fidelity were all on the wrong axis). With f: `fids` / `thresholds` are (ge on I,
  gf on I, ef on Q); new results `assignment_matrix` (3x3, A[prepared, assigned]: I against the
  ge threshold, then Q against the ef threshold) and `f_high_q`. Plot range includes f; a third
  column with the e/f Q histograms and the 3x3 matrix. The unrotated plot no longer draws the
  rotated centers.
- **`HistogramExperiment.analyze()`** sets `self.readout_cfg`: phase, threshold, Ig, Ie and the
  confusion matrix under the active-reset key, g-e only. Phase = `expt.cfg` phase + fitted
  angle, so a second analyze gives the same phase (before: station phase + angle, which rotated
  twice). Raises `ValueError` for non-finite or coincident centers. Not saved to HDF5: all
  numbers in it come from `data` and the saved `cfg`.
- **One postprocessor:** `experiments/readout_calibration.py` is deleted. The standard
  postprocessor is two lines in the notebook (docstring of `HistogramExperiment`):
  `expt.analyze(...)`, then `station.hardware_cfg.device.readout.update(expt.readout_cfg)`.
  `experiments/qsim/readout_calibration.py` defines it for `singleshot_runner`;
  `autocal/_qubit.py` `_ss_eval` makes its change items from `readout_cfg`.
- **`device.readout.threshold_list`** is no longer written (no code read it): removed from
  autocal and `job_server/closed_loop/core.py`. Old config versions still have it.
- `tests/test_readout_calibration.py` now tests `analyze()` on synthetic g/e and g/e/f blobs.
  Checked on saved histograms (g/e, g/e/f, with and without active reset + pre-selection).

## Decisions (seb)

- No new API names: the experiment prepares `readout_cfg`, the postprocessor copies it. A
  one-line postprocessor is not factored into a module.
- g/e/f is read in the g-e frame (I: g vs e/f, Q: e vs f), not on per-pair axes or by nearest
  center: stable meaning of I and Q. On a saved g/e/f job the two rules differ by less than
  1 percentage point per matrix element.
- No new config fields for g/e/f yet; the 3x3 matrix stays in `data` / HDF5.

## Open

- Notebook consumers: done the same day. Every notebook copy of `singleshot_postproc` (guan,
  jonginn, connie, QEC, raph, 202609) is the two-line one; jonginn reloads his kernels.
- Pre-existing test failures, not from this work (same on clean HEAD):
  `test_agp_experiment.py` (2), `test_agp_disorder_scan.py` (1),
  `test_matrix_pencil_regression.py` (3).
- Gap item 3 (prepared-state-only g/e/f population readout) is next; it will use the 3x3
  assignment from `Histogram`.
