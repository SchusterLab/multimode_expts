# Where things stand

**Last updated: 2026-10-05** (Jonginn notebook retirement and local migration completion). This file holds only what is
shared across themes: the branch rules and the cross-theme items. Each theme has its own status
file under `docs/status/`; read the one for your work. What happened, and why, is in `docs/log/`
(newest: `2026-09-29_branch-sync-after-merge.md`). Read the log only if you need the history.

## Themes, branches, worktrees (guan, 2026-09-29)

| Theme | Status file | Branch | Worktree on pippin |
|---|---|---|---|
| Measurement code (`experiments/`, `measurement_notebooks/`, calibration) | `docs/status/measurement.md` | `guan` | `C:\python\multimode_expts_guan` |
| Analysis, numerics, fitting (`fitting/`, `analysis_notebooks/`, offline tools) | `docs/status/qsim_numerics.md` | `qsim-analysis` | `C:\python\multimode_expts_qsim-analysis` |

Rules:
- Put each change on the branch of its theme. A change to a module that both themes use goes on
  `guan`; then merge `guan` into `qsim-analysis`.
- Merge `guan` into `main` with a normal merge (no squash, no rebase): `qsim-analysis` holds
  `guan`'s commits, and new hashes would make conflicts. After that merge, merge `main` into
  `qsim-analysis`.
- At the end of a session, overwrite the status file of your theme. Change this file only for
  cross-theme items.
- The worker runs only the code in `C:\python\multimode_expts` (`main`).

## Cross-theme items

- **One data set catalog** (2026-10-02, on `qsim-analysis`): `configs/datasets/mbr_datasets.yaml`
  holds every MBR spectroscopy data set (job IDs, kind, folder, rough g/K labels, and the
  converted manifest). It replaces `tests/data/mbr_datasets.json` and the two markdown job lists
  (issue 6, question 1); jonginn's human vault with plots stays. 20 of 25 data sets are converted
  to the new layout on pippin (`tools/convert_mbr_catalog.py`); the 5 off-diagonal D72 sets are not
  (step 7 plan, decision 2). Record: `docs/log/2026-10-02_dataset-catalog-and-conversion-round.md`.
  The two markdown lists are also deleted on `job_id_sorting_out` (2026-10-05), and the human
  vault there points at the YAML, so the next merge of that branch brings no conflict.
- **`main` has the qsim redesign** (merge of `guan`, 2026-09-29, after the 10G device check;
  `docs/status/measurement.md`). Users restart their kernels once. `main` is merged into
  `qsim-analysis`; both theme branches are on GitHub.
- **Known test failures (3), on purpose for now:**
  - `test_matrix_pencil_regression` (3): left as is while `qsim-analysis` compares fitting
    methods.
- **Jonginn notebook migration:** the newer work is absorbed or explicitly retired using
  his issue 7 decisions. The five old notebooks and all `KEPT_UNMIGRATED` exceptions are
  gone; the stage-dispatch sweep passes. Acquisition defaults are full basis + RMS with
  interleaved single-shot calibration. This is a local change on `guan` and `qsim-analysis`,
  not yet on prod/main. Record: `docs/log/2026-10-05_jonginn-notebook-migration.md`.
- **Hardware:** the code runs on the device, but the data is not meaningful yet; the
  calibration steps need a manual check.
- **Physics audit** (details in `docs/status/qsim_numerics.md`): the numerical baselines are
  `xfail(strict=False)`.

## Docs: which are current

| Doc | Status |
|---|---|
| `docs/STATUS.md` | this file: current |
| `docs/status/measurement.md`, `docs/status/qsim_numerics.md` | per-theme status, current; each lists its own design docs |
| `docs/log/` | dated session records; never edited after their day |
| `docs/archive/` | history |
