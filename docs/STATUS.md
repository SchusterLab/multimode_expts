# Where things stand

**Last updated: 2026-10-06** (issue closing day: issues 3, 6, 7 closed; issue 5 work; branches merged into `guan`). This file holds only what is
shared across themes: the branch rules and the cross-theme items. Each theme has its own status
file under `docs/status/`; read the one for your work. What happened, and why, is in `docs/log/`
(newest: `2026-10-06_issue5-d72-and-loaders.md`). Read the log only if you need the history.

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

- **Branches (2026-10-06):** `guan` holds all of `qsim-analysis` and `job_id_sorting_out`
  (fast-forward, then a merge). Nothing went to `main`: the device was busy and the `main`
  checkout has other sessions' uncommitted edits (including `job_server/worker.py`). **Next:**
  merge `guan` into `main` when both are clear, then fast-forward `qsim-analysis` to it.
  `job_id_sorting_out` and its worktree `C:\python\multimode_expts_jonginn` are deleted
  (fully merged; guan, 2026-10-06). Record: `docs/log/2026-10-06_issue5-d72-and-loaders.md`.
- **One data set catalog**: `configs/datasets/mbr_datasets.yaml` holds every MBR spectroscopy
  data set (job IDs, kind, folder, rough g/K labels, converted manifests); jonginn's human vault
  with plots stays. **All 25 are converted** (2026-10-06). The five old "D72" sets were mixed
  diagonal and off-diagonal, not off-diagonal; they are now `sepNN_pairs_*`. The library loads
  only converted or new job files, and the timing only from their `derived_params`
  (issue 5). Records: `docs/log/2026-10-02_dataset-catalog-and-conversion-round.md`,
  `docs/log/2026-10-06_issue5-d72-and-loaders.md`.
- **`main` has the qsim redesign** (merge of `guan`, 2026-09-29, after the 10G device check;
  `docs/status/measurement.md`). Users restart their kernels once. `main` is merged into
  `qsim-analysis`; both theme branches are on GitHub.
- **Known test failures (3), on purpose for now:**
  - `test_matrix_pencil_regression` (3): left as is while `qsim-analysis` compares fitting
    methods.
- **Jonginn notebook migration:** the newer work is absorbed or explicitly retired using
  his issue 7 decisions. The five old notebooks and all `KEPT_UNMIGRATED` exceptions are
  gone; the stage-dispatch sweep passes. Acquisition defaults are full basis + RMS with
  interleaved single-shot calibration. On `guan`, not yet on prod/main. Record: `docs/log/2026-10-05_jonginn-notebook-migration.md`.
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
