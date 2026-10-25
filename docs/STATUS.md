# Where things stand

**Last updated: 2026-10-24** (`theory/` package and the jonginn AGP item; AGP layers item as of
2026-10-23; the rest as of 2026-10-15). This file holds only what is
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

- **For jonginn (from seb, 2026-10-24). Agents: tell jonginn about this item at the start of
  his next session, and help him do it.** Record: `docs/log/2026-10-24_theory-package-agp.md`.
  - Your AGP notebook in the Julia repo
    (`multimode_thermalization_sim/notebooks/jonginn/agp_storage7_mu1_to10_t1_theory_ENTIERLYCODEX.ipynb`)
    is now in the classes: `CentralBosonAGPDisorderScan.analyze(mu=[...], t1_us=..., t1_err_us=...,
    kerr_mhz=...)` gives the AGP with shot + calibration errors, the local T1 compensation and
    the closed-system theory; `display()`, `display_traces()`. It reproduces your notebook's
    numbers to 1e-12. Example: `analysis_notebooks/agp/agp_disorder_scan.py`. Please check it,
    then work from it; analysis that lives only in a notebook has no tests and no record.
  - **New rule** (`AGENTS.md`, `docs/qsim/theory_package.md`): models go in `theory/`
    (parameters -> predictions), estimators in `fitting/`, and `analyze()` calls both. Do not
    write a model or an analysis chain inline in a notebook. `fixed_n_hamiltonian` moved to
    `theory.qsim.central_boson` (your `agp_measurement.py` import is updated).
  - **Breaking:** the scan's `agp_list` / `agp_mu_list` are now `agp` (W, mu) / `agp_mu`;
    `display()` returns `(fig, axes)`.
  - Your uncommitted edits in the `main` checkout were committed as they were (c2f1e9b) before
    the merge.
  - Three AGP tests failed on `main` after the 10-23 commits (Kerr direction n_M1 >= 2,
    `agp_list`, no couplings in the fake jobs); they are fixed.
  - **Physics to check before you use small mu:** C_1 is not connected, so its constant part
    enters the finite-window integral. In the closed-system theory, the window value is 35x (fine
    scan) and 8x (coarse) the infinite-time AGP at mu = 3 kHz, W = 0; about 1 only at
    mu = 10 kHz. The default mu = g L/D is 2.9 kHz. The table is in the log. Your Kerr-direction
    scan (W = 20-200, mu = 10 kHz) now gets `C1_err` and theory too.
- **`simulation/` deleted (2026-10-24, seb).** The RAQM-era QuTiP notebooks; nothing imported
  them. Git tag `raqm-simulation-notebooks` is the last commit with them (local; push with the
  branch). guan: FYI.

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
