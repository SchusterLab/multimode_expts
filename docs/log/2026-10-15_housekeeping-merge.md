# 2026-10-15 Housekeeping during cooldown: guan and tests-cleanup into main

The fridge was cooling down and no job ran (no worker process; only Jupyter kernels open).
guan asked to commit, merge and make `main` clean.

## What was done

1. Committed the uncommitted edits in the `main` checkout (decision: guan), in 5 commits:
   - transduction status and the 11 session logs of Oct 1-12;
   - `job_server/worker.py`: reconnect the coupler yoko once on a dropped socket, with
     `tests/test_worker_yoko_reconnect.py`;
   - `experiments/single_qubit/error_amplification_2d.py` with its mock test;
   - jonginn's AGP work (Oct 6-12), as-is: `FockPopulationProgram` readout through the
     existing `postpulse=True` path, `agp_measurement.py`, 3 jonginn notebooks;
   - the 3 QEC notebooks edited on Oct 9 (`coupler_systematic_study_v2`,
     `multiphoton_calibration_v2`, `single_qubit_autocalibrate_v2`), as-is.
2. Merged `guan` into `main` (normal merge). Conflicts:
   - `experiments/qsim/floquet_dark_mode_readout.py`: `guan` deleted the shim (issue 5,
     `7833030`); `main` had jonginn's AGP block at its end. The shim stays deleted. The AGP block
     moved unchanged to `experiments/qsim/fock_population.py`; `agp_measurement.py` imports it
     there.
   - `measurement_notebooks/jonginn/qsim_experiments.ipynb`: retired by guan (`0349bd8`),
     edited by jonginn on Oct 10. Kept for jonginn. Its first cell still imports the deleted
     shim.
   - `docs/STATUS.md`: guan's one-branch text, plus the transduction row.
3. Merged `tests-cleanup` into `main` (guan pointed to it: it retires the tests that failed on
   the kept notebook). Conflicts: the two migration-guard tests stay deleted, although
   `0349bd8` edited them after the branch point; `test_mbr_orthogonality.py` keeps guan's
   import path `experiments.qsim.deprecated.saved_jobs`.
4. MBR analysis baselines. With the `xfail(strict=False)` markers gone, 5 baselines failed only
   on the new field `readout_refit` (added in `0349bd8`). Checked: all old fields match within
   the test's 1e-12 tolerance (largest relative change 5.6e-11). A full `MBR_GOLDEN_BLESS` run
   would replace the reference values with that noise and drop the ignored fields from the
   files. Instead each file keeps its old fields bit-for-bit and gets only `readout_refit`.
   `mbr_stark_cal_20260905.npz` is unchanged.
5. Merged `main` back into `guan`; pushed `main` and `guan`.

## Tests

Before: 1633 passed, 6 failed (3 `matrix_pencil_regression`, 3 stage-dispatch notebook checks).
After: 1366 passed, 3 failed (the known `matrix_pencil_regression` cases), 6 xfailed.
The count is lower because `tests-cleanup` deleted the guard tests.

## Not done (left on purpose)

- No branch or worktree deleted. `transduction_refactor` and `wigner-linear-fidelity` (seb) and
  `tests-cleanup` are fully in `main`. Remove a worktree only by the link-safe steps in
  `AGENTS.md` (the `configs\versions` link).
- `jonginn` (22 commits not in `main`) and `wip/ramsey-flux-excursion`: jonginn's.
- The 6 old stashes.
- The uncommitted `analysis_notebooks/202609_qsim_migration/kerr_coherence.py` in the `guan`
  worktree.

## Found

- jonginn works around `main` by cherry-picking from another computer, and on Oct 9
  (`75d2053`) reset his branch to `main`. So `guan`'s retirement of his notebooks did not reach
  him: he kept editing `qsim_experiments.ipynb`. Tell him about `fock_population.py` and the
  broken import in that notebook.
