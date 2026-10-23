# Where things stand

**Last updated: 2026-10-23** (AGP layers item added; the rest as of 2026-10-15). This file holds only what is
shared across themes: the branch rules and the cross-theme items. Each theme has its own status
file under `docs/status/`; read the one for your work. What happened, and why, is in `docs/log/`.
Read the log only if you need the history.

## Themes, branch, worktree (guan, 2026-10-06)

One branch, `guan`, worktree `C:\python\multimode_expts_guan`, for both themes. The themes keep
their own status files:

| Theme | Status file |
|---|---|
| Measurement code (`experiments/`, `measurement_notebooks/`, calibration) | `docs/status/measurement.md` |
| Analysis, numerics, fitting (`fitting/`, `analysis_notebooks/`, offline tools) | `docs/status/qsim_numerics.md` |
| Transduction (`experiments/transduction/`, `measurement_notebooks/QEC/transduction_sandbox.ipynb`) | `docs/status/transduction.md` (works on `main`, worktree `C:\python\multimode_expts`) |
| Automated calibration (`experiments/autocal/`) | `docs/status/autocal.md` (branch `autocal`, worktree `C:\python\multimode_expts_autocal`; since 2026-10-19) |

The second branch, `qsim-analysis`, was retired on 2026-10-06 (guan): the merges between the
two cost more than the separation gave.

Rules:
- Work on `guan`. For a parallel session, make a short-lived branch and worktree from `guan`,
  and merge it back within a day or so.
- Merge `guan` into `main` with a normal merge (no squash, no rebase).
- At the end of a session, overwrite the status file of your theme. Change this file only for
  cross-theme items.
- The worker runs only the code in `C:\python\multimode_expts` (`main`).
- Other people's branches (seb: `transduction_refactor`, `wigner-linear-fidelity`; jonginn:
  `jonginn`, `wip/ramsey-flux-excursion`) are theirs to keep or merge. seb's two are fully in
  `main` and can be deleted when seb agrees.

## Cross-theme items

- **Branches (2026-10-15):** `main` has all of `guan` and `tests-cleanup`, and `main` is
  merged back into `guan`. The uncommitted edits in the `main` checkout (worker yoko reconnect,
  transduction docs, jonginn's AGP work, QEC notebooks) are committed. **Users restart their
  kernels once.** The `tests-cleanup` branch and worktree are fully merged; delete the
  worktree only by the link-safe steps in `AGENTS.md`. Record:
  `docs/log/2026-10-15_housekeeping-merge.md`.
- **AGP data in three layers (2026-10-23, guan's decision, on `main`):** jonginn's
  `CentralBosonAGPExperiment` and `CentralBosonAGPDisorderScan` are now assembled classes over
  one-state `FockPopulationExperiment` jobs, with no shots above the job. His
  `convert_legacy_agp.py` is replaced by `tools/convert_fock_population_catalog.py`; his
  `agp_measurement.py` still uses the old one-job-per-state runner (works, but not the new
  classes). Record: `docs/log/2026-10-23_agp-population-layers.md`.
- **jonginn's AGP code moved:** `FockPopulationProgram`, `validate_occupations` and
  `with_storage_waits` are now in `experiments/qsim/fock_population.py` (the
  `floquet_dark_mode_readout.py` shim is deleted). `jonginn/qsim_experiments.ipynb` is kept
  for jonginn, but its first cell still imports the deleted shim; jonginn to update or retire it.
- **One data set catalog**: `configs/datasets/mbr_datasets.yaml` holds every MBR spectroscopy
  data set (job IDs, kind, folder, rough g/K labels, converted manifests); jonginn's human vault
  with plots stays. **All 25 are converted** (2026-10-06). The five old "D72" sets were mixed
  diagonal and off-diagonal, not off-diagonal; they are now `sepNN_pairs_*`. The library loads
  only converted or new job files, and the timing only from their `derived_params`
  (issue 5). Records: `docs/log/2026-10-02_dataset-catalog-and-conversion-round.md`,
  `docs/log/2026-10-06_issue5-d72-and-loaders.md`.
- **Known test failures (3), on purpose for now:**
  - `test_matrix_pencil_regression` (3): left as is while `qsim-analysis` compares fitting
    methods.
- **Jonginn notebook migration:** the newer work is absorbed or explicitly retired using
  his issue 7 decisions. The five old notebooks and all `KEPT_UNMIGRATED` exceptions are
  gone; the stage-dispatch sweep passes. Acquisition defaults are full basis + RMS with
  interleaved single-shot calibration. On `main` since 2026-10-15. Record: `docs/log/2026-10-05_jonginn-notebook-migration.md`.
- **Test suite cleanup** (2026-10-05, on `main` since 2026-10-15): the one-off migration guards
  are retired (`docs/log/2026-10-05_tests-cleanup.md`). The code still left in jonginn's
  `data_recollecting.ipynb` and `qsim_experiments_highkerr_untracked.ipynb` is tracked in
  `docs/qsim/evidence/2026-09-30_jonginn_notebook_leftovers.md`, no longer by a failing test.
- **Hardware:** the code runs on the device, but the data is not meaningful yet; the
  calibration steps need a manual check.
- **Physics audit** (details in `docs/status/qsim_numerics.md`): the numerical baselines are
  pinned and must pass. They pin the current numbers, not checked physics.

## Docs: which are current

| Doc | Status |
|---|---|
| `docs/STATUS.md` | this file: current |
| `docs/status/measurement.md`, `docs/status/qsim_numerics.md` | per-theme status, current; each lists its own design docs |
| `docs/log/` | dated session records; never edited after their day |
| `docs/archive/` | history |
