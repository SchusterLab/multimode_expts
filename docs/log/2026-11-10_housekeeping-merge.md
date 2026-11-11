# 2026-11-10 Housekeeping while the fridge is idle: test failures, commits, branches, test data

Who: guan with Claude, main checkout `C:\python\multimode_expts`. No job ran (the fridge was
idle). Decisions: guan, by answers to Claude's questions.

## What was done

- The open work in the main checkout went in one commit per topic, files as they were:
  - `5076948`: `theory.qsim.central_boson.fock_number_traces` with `tests/test_central_boson.py`;
    guan's 11-05 AGP high-W scripts (`20261105_agp_highW.py`, `20261105_agp_plots.py`).
  - `359c1a1`: guan's Floquet crosstalk tests of 11-07/08 (script, config branches, log).
  - `eeea65c`: jonginn's notebooks (AGP storage directions S4 and S6, 2D disorder cells,
    `Autocalibrate.ipynb`, `agp_reanalysis`).
  - `abb8881`: guan's M1 reset trim of 11-05 (`docs/log/2026-11-05_man-reset-trim.md`). Two tests
    in `tests/test_pulse_layer_layout.py` checked the old layout (`QsimProgram` with its own
    `man_reset`) and failed; they are replaced by one test that every qsim program plays
    `MM_base.man_reset` (guan approved the text).
  - `d798769`: seb's log `2026-10-23_swap-breakdown-night.md`, found in the autocal worktree;
    `docs/status/transduction.md` now points to it. The 10-23 edit of `transduction.md` in that
    worktree is not taken: the 10-29 version replaces it.
  - The empty file `nulstart` (from a `> nul` redirect in Git Bash) is deleted.
- Branches: `theory` moved to `main` (fast-forward). Deleted: `guan` (local and origin; it held
  only a merge commit of two `main` commits), `autocal`, `tests-cleanup` and its worktree,
  `wigner-linear-fidelity`. The worktree went by the link-safe steps of `AGENTS.md`: link
  removed first, then the archive checked (22151 files before and after), then
  `git worktree remove` without `--force`.
- Not touched: the `v2` prototype in the guan worktree (guan's; its edits to `pyproject.toml`
  and `docs/STATUS.md` overlap with `main`, so `main` goes into `v2` only after guan saves that
  work); `jonginn` and `wip/ramsey-flux-excursion` (jonginn's); `transduction_refactor` (seb's,
  fully in `main`, has a worktree); `cavity-coherence` (fully in `main`, in the autocal worktree).
- Test data copied back from the NAS (guan: "if the tests need only a small amount of files"):
  344 files, 2.27 GB, to their old paths under `C:\experiments` (below).
- Memory (guan: "do both"): `fitting.qsim.agp.cosine_transform_weights` computes in place (one
  (frequency, time) array instead of three; bit-identical output for both windows), and
  `test_cosine_transform_filtered_matches_the_spectral_agp` uses 6001 x 2001 instead of
  30001 x 40001 points (the error is 3e-5 at both sizes; the tolerance is 1e-2).

## The old data on the NAS, and the files that the tests read

- The old data folders of `C:\experiments` were moved to the NAS on 2026-11-09 (disk almost
  full): they were in its nightly copy `/volume1/slab/Multimode/BF5_expts_autosync`, checked file
  by file, and then deleted on pippin. The 120 smaller folders (46 GB) are also in the Recycle
  Bin of pippin. `configs\versions` is not in `C:\experiments` and is intact.
- 47 tests failed because of this. To find what they read, a pytest plugin (not in the repo)
  recorded each job ID asked from `experiments.job_paths.resolve_job_paths` and each file opened
  under `C:\experiments`, with the test and the fixture; three rounds of copy and run until
  nothing was missing. `h5py` opens files without a Python audit event, so the children of the
  manifests came from the manifests themselves. The copy (`ssh slabnas`, `tar`, file times kept,
  no overwrite) went over Tailscale at about 1.7 MB/s.
- What the tests do with these files: the MBR regression tests convert the old job files with
  `tools/migrate_mbr_jobs.py` into the new classes in a temporary folder, run the analysis and
  compare with the baselines in `tests/data`. By fixture: `complete_basis` (August N=3, 140 jobs,
  1.29 GB, `260526_qsim_darkmode`), `stark_cal` (70 jobs, 0.46 GB), `september` (35 orthogonality
  columns, 0.24 GB), `diagonal_disorder` (70 jobs, 0.04 GB), eight small ones (0.25 GB together);
  folders `260818_qsim_spectroscopy`, `260526_qsim_darkmode`, `260814_qsim_encspec`.
- 97 % of the 2.27 GB is single shots (`idata`, `qdata`). Every fixture converts with
  `load_shots=False`; only `tests/test_mbr_stark_cal.py` reads shots, from 4 jobs. Option for
  later (guan: keep the files as they are for now): shot-free copies for the tests (about 70 MB
  and the 4 jobs whole), with the sidecar pointing to them.

## Why a test run ran out of memory

`test_cosine_transform_filtered_matches_the_spectral_agp` (from `6213cd1`, 10-26) built a
40001 x 30001 matrix, and `cosine_transform_weights` held three arrays of that size: about 29 GB
at peak on a PC with 31.7 GB of RAM. It passed, but took 51 s and pushed about 8 GB into the page
file. With the data tests now running on real data, Claude Code stopped one full run because
free memory was low (Windows logged no low-memory event). Fixed above.

## Tests (full suite in the main checkout)

- Start of the session: 1580 passed, 16 failed, 39 errors, 7 skipped, 6 xfailed.
- End of the session: 1589 passed, 7 failed, 39 errors, 6 skipped, 6 xfailed (6 min 10 s).
- Left, known:
  - 40 in the MBR files (`test_mbr_*`): `AssembledExperiment.save()` needs a name since 11-03,
    and `tools/migrate_mbr_jobs.py` saves without one.
  - 3 in `test_convert_fock_population_catalog`: the same, in that tool.
  - 3 in `test_matrix_pencil_regression` (known, on purpose).

## Found

- The nightly NAS backup is paused: `~/bin/pippin_backup.disabled` exists on slabnas since
  2026-11-09 17:57 CST, so the runs of 11-09 17:57 and 11-10 00:00 exited at once
  (`~/logs/pippin_backup/skipped.log`). Data that pippin writes from now on has no NAS copy until
  that file is deleted (guan: later).
- New jobs are not affected by the moved folders: the worker calls
  `station._initialize_output_paths()` for each job (`job_server/worker.py`,
  `_update_station_from_job_config`), and that makes the missing folders again.
- `test_mbr_base_plays_the_dark_man_reset` in `tests/test_pulse_layer_layout.py` still passes,
  but now only because both sides are `MM_base.man_reset`; its docstring describes the old
  layout.

## Open

- Delete `~/bin/pippin_backup.disabled` on slabnas when the move is finished.
- Other analysis on pippin that loads moved folders (data-set registries in
  `analysis_notebooks/datasets/`, notebooks) needs its files copied back the same way.
- guan: which branch the rules in `docs/STATUS.md` name now that `guan` is deleted.
