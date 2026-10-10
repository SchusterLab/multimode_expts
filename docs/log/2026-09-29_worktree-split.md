# 2026-09-29: work split into two themes and two worktrees

## What was done

- New branch `qsim-analysis` from `guan` at 93a112b, with its worktree at
  `C:\python\multimode_expts_qsim-analysis`. Setup: `configs\versions` is a junction to main's
  folder (as in the guan worktree); own `pixi install`; no `jobs.db` link (analysis reads HDF5
  metadata, not the job database). The display port for the `qsim-analysis` slug is 8101, in
  `~\.ipython\profile_default\startup\50-autoshow.py`. `nb` gets the slug from the folder name.
- `docs/STATUS.md` now holds only the branch rules and the cross-theme items. The status of
  each theme is in `docs/status/measurement.md` and `docs/status/qsim_numerics.md`. The content
  moved without change, except the headers. `AGENTS.md` now tells agents to read and overwrite
  the status file of their theme.

## Decisions (guan)

- Measurement work stays on `guan` and goes to `main` after validation. Analysis, numerics,
  and fitting work goes on `qsim-analysis` from now on, so that it does not wait for the
  measurement validation.
- One worktree per theme that must run in parallel. In a worktree, switch branches in place
  only with no kernel running (`%autoreload 2` reloads modules from the other branch).
- The name is `qsim-analysis`, not `analysis`, to be more specific. The first worktree
  (`multimode_expts_analysis`) was removed and made again, not moved: the pixi environment
  holds absolute paths.

## Reasons for the merge rules

`qsim-analysis` holds all of `guan`'s commits. A squash or rebase of `guan` into `main` gives
new hashes and makes conflicts in `qsim-analysis`; a normal merge does not. One status file for
both themes would conflict at every merge, so the status is split by theme.
