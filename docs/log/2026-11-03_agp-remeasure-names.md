# 2026-11-03: jonginn's AGP remeasure to-dos; assembled data sets under names

Who: guan with Claude, in one shared kernel (`agp_remeasure`, Lab server port 8765, kernelspec
`pixi-cwd`, theory worktree). Notebook: `analysis_notebooks/guan/agp/agp_remeasure_20261029.py`.
To-do list: vault note `Jonginn/000000 Separate Lab Notebook/20261001 AGP/20261029 AGP remeasure.md`.

## What was done

- **A2 (120/140 kHz):** already assembled, with 200 kHz, in `s7_high_w`. 120 kHz is
  JOB-20261029-00393..00407; the vault note had the 140 kHz list under both headings (corrected
  there). Gap: W = 160 kHz has 3 of 15 jobs. guan: no new measurement for now.
- **A4 (plots):** `display_spectrum` plots to 1.3 x the highest closed-system frequency, at
  most Nyquist, and marks closed-system weight above Nyquist. The old fixed 250 kHz cut 88 % of the
  weight of `s7_high_w` at W = 200 kHz. `s7_coarse` at W = 150 kHz: 1.4 % folds back.
- **A3 (names, guan's decision):** `AssembledExperiment.save(name)` is required; files are
  `<Class>_<name>.yaml/.h5`, never overwritten; no time-stamp fallback. A scan names its parts
  `<name>_W<W>kHz`. Old measurement notebooks and migration scripts are not changed (guan): they
  ask for a name when they run again. `process_tomo.py` (transduction) keeps its time stamp.
- **A5 (incremental):** `CentralBosonAGPDisorderScan.acquire(runner, name)` runs W by W and saves
  each W when its jobs are in; `from_saved_parts(name, directory)` after a stop. All assembly is in
  the notebook process; the worker only runs the jobs.
- **A1 (mu as an axis, guan's design):** each W keeps its AGP(mu) on one fixed axis
  (`DEFAULT_MU_KHZ`: 41 log steps 1-100 kHz, and 2, 5, 20, 50) and its S(f) on the axis of its time
  grid; g L/D is kept as `mu_gld_khz`. The scan only stacks. Settings in the HDF5 attribute `analysis`.
- **Cleanup (guan):** the two folders `agp_k44_first_scan` and `261027AGPlowKerr` were made again
  under names (6 scans, 4 walks), checked against the old files (same jobs, same C_1 or
  populations). Old files: `<experiment>/assembled_data_old_20261103`. Registries, guan's AGP
  notebooks, jonginn's vault note and `guan/KerrCheck/2026/10/2026-10-24.md` point to the names.
- `s7_ultrafine`: the old `261102_161533` labelled W = 30 kHz (detuning RMS) as 20.
- New: the 10-29 3-photon walk `walk_n3_20261029`. Not assembled: the 10-27/28 low-Kerr
  calibration walks (jonginn's OneNote AGP page; f2g3 was wrong) and 7 S6 jobs of 10-23
  (JOB-20261023-00238..00250; question to jonginn at the end of the vault note).

## Found

- A kernel started by the Lab server can lose the `qick` path: each `pixi run` in a worktree
  reinstalls the editable `qick` (deletes and writes its `.pth`). A second `pixi run` in the same
  worktree while a kernel starts removes the path for that kernel. Running kernels are not affected.
- Loading a scan reads no single shots (10 MB of 759 MB for `s7_high_w`); only the calibration's
  `idata` is opened, for its shape.

## Known

- 28 MBR tests fail: their fixtures use `tools/migrate_mbr_jobs.py`, which saves without a name.
  guan fixes this separately.
