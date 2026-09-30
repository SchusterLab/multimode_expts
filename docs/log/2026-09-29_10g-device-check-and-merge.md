# 2026-09-29: 10G device check, and the merge of guan into main

## What was done

- Worker stopped at 17:58 (queue empty; seb's last job ended 17:28). Decided by guan.
- Device check on `guan` (local mode, smoke profile, notebook config sets):
  - `multiphoton_calibration` (127 s) and `floquet_calibration` (271 s): no errors. The HDF5
    files are named after the new classes (`..._QsimExperiment.h5`).
  - `mbr` section 0 (one `MBRStarkCal` job, one `MBRTimeTrace` job, a manifest save): no errors.
  - `mbr` section 1 (new N=3 calibration): stopped by a cell timeout I set too low (900 s).
    It is 35 jobs of ~2 min (one occupation per job), so not run in full. It uses the same
    Program as section 0.
  - `mbr` section 2 (4 `MBROrthoColumn` jobs), run as a script of the notebook's setup cells and
    section 2: no errors.
- Decision (guan): for 10G, one short job per changed Program type, not the full MBR
  notebooks. The full smoke suite repeats the same Programs for a long time.
- Merge: `guan` into `main` (normal merge, `86a03a4`), made in a separate worktree and then
  fast-forwarded in the main checkout. Then `2858086`: jonginn's and connie's notebooks moved
  with `tools/rename_program_tree.py` (114 renames) and the new
  `tools/migrate_readout_flags.py` (the readout table of plan section 6). Then `8f87f42`: tests.
- `jonginn/qsim_experiments.ipynb` had uncommitted edits in the main checkout. It was moved in
  its working copy (33 renames, 19 cells), not committed. Backup of the file before:
  `%TEMP%\g10\qsim_experiments.ipynb.bak`.

- After the merge, from the main checkout: `multiphoton_calibration` (128 s),
  `floquet_calibration` (271 s) and `mbr` section 0 (164 s) locally, no errors; the values
  agree with the `guan` run (ge fidelity 94.5 %, broadband gain 24596 +/- 10, M1-S4
  878.254 MHz / 9047). Worker restarted at 18:44 in a Git Bash window, as the server runs.
  Then `mbr` section 0 through the queue: `JOB-20260929-00087` (MBRStarkCal) and `-00088`
  (MBRTimeTrace), both completed. They show user `jonginn` (the notebook's station user).

## What was found

- Calibration physics agrees with 2026-09-28, or is better: ge readout fidelity 94.4 % (91.5 %);
  the broadband frequency and gain error-amp fits converge (yesterday: fit off / failed); Floquet
  M1-S4 878.245 MHz / 9030 (878.249 / 8991).
- The section-0 Stark phase changed sign: -8.73 +/- 0.29 deg/cycle on 09-28 (code before 10B),
  +6.59 +/- 0.24 today, with the same four config versions. Not the refactor:
  - the saved `cfg` of the two jobs (00021, 00030) is the same, except the retired readout flags;
  - `MBRStarkCalProgram` compiled from the saved cfg at `57ef39b` and at `guan` gives the same
    envelopes and ASM (3041 lines);
  - today's analysis on the 09-28 manifest gives -8.73 again.
  So the device changed. The multiphoton calibration today also moved M1 by -170 kHz.
- The orthogonality data is not meaningful yet (diagonal |M_ii| 0.04-0.15, off-diagonal power up
  to 2.7x). As on 09-28: the calibration steps need the manual check (physics, not code).
- pytest on the merge result, in a new worktree: 14 failed, 23 errors. Of these, 11 failures and
  all 23 errors are from the new worktree only: `configs/versions` is not in git. The same tests
  pass on `guan`. 3 are the known matrix-pencil cases. 4 came from the kept jonginn notebooks
  (next point).
- jonginn's `qsim_experiments.ipynb` and `data_postprocess.ipynb` (retired on `guan` in `cb3f9f7`)
  were kept at the merge: jonginn changed both on Sep 15 after the extraction (disorder-ensemble
  batches, the `Sep10 K3.6 g29.2` dataset, the readout-cloud refit), and that work is not in the
  new tree. Decided by guan. They still call `analyze(stage=...)`, which raises. The tests mark
  them `KEPT_UNMIGRATED` (xfail); jonginn moves or retires them.

## Open

- jonginn: the two kept notebooks (above); the Sep 15 cells are not in the new tree.
- Push `main` and `guan` to origin (not done in this session).
- Merge `main` into `qsim-analysis` (not done: that worktree had a running job).
