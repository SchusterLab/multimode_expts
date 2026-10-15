# 2026-10-05 Test suite cleanup, step 1: retire the one-off guards

Branch `tests-cleanup` (from `guan` at `fb25c40`), worktree
`C:\python\multimode_expts_tests-cleanup`. Requested by guan.

## Why

A full run took about 230 s for 1,570 cases in 73 files. Most failures meant "retire the
test", not "fix the code". About 650 cases were guards for refactors that are finished.

The survey grouped the files into four kinds:

- one-off refactor and migration guards (retired here);
- golden and end-to-end tests: ASM, acquisition, analysis baselines, mock acquire-save-reload
  (the core; kept);
- physics and numerics with known answers (kept; the long benchmark-style ones go to a separate
  run later);
- infrastructure: runners, mock mode, job paths, env (kept).

## What was done

Deleted files (decision: guan):

| File | Reason |
|---|---|
| `test_data_comparison.py`, `test_sweep_comparison.py` | manual scripts from January, no test functions |
| `test_sweep_runner_notebook.ipynb` | notebook, not a test |
| `test_deprecated_mbr.py` | both tests skipped; the file said "do not fix" |
| `test_mbr_stage_split.py` | AST compare against the pre-split commits. Its own docstring said to delete it when the classes evolve |
| `test_jonginn_notebook_migration.py`, `test_no_stage_dispatch_remains.py` | migration of jonginn's notebooks. What is left is tracked in `docs/qsim/evidence/2026-09-30_jonginn_notebook_leftovers.md`, no longer by 2 failing tests |
| `test_f0g1_source_of_truth.py` | June migration guard; its mock part skipped |
| `test_meas_namespace.py` | `test_no_live_deprecated_imports.py` covers the same boundary |

Deleted test functions that compared new code against `experiments/qsim/deprecated/`:

- `test_mbr_orthogonality.py::test_ortho_column_pulses_at_q_are_the_old_propagator_program`
  and `test_mbr_time_trace.py::test_time_trace_pulses_are_the_old_diagonal_program`. The ASM
  goldens (`ortho_column_q0`, `ortho_column_q4`, `time_trace`) pin these pulses.
- `test_mbr_stark_cal.py::test_old_program_is_a_subclass`.
- `test_mbr_disorder_ensemble.py::test_plan_matches_the_old_selection_code`, and the old-code
  line in `test_station_floquet_hardware_runs_on_a_mock_station`.

Removed `xfail(strict=False, reason="physics audit pending")` from the 6 analysis baselines in
`test_mbr_analysis_golden.py` and from the 3 compares in `test_mbr_ham_tomo.py`,
`test_mbr_orthogonality.py`, `test_mbr_disorder_ensemble.py`. A non-strict xfail cannot fail:
a changed number was reported as XFAIL. All 9 passed (XPASS) before the change. The physics
audit itself is still open (`docs/status/qsim_numerics.md`).

## What was found

- `tests/test_qsim_notebook_mock_acquisition.py`, the most end-to-end test, skips on every
  checkout without the live `jobs.db` (it needs `CFG-HW-20260904-00019`). It should use the
  pinned `tests/data/config_set/` as the ASM goldens do.
- About 35 tests (analysis baselines, MBR gates, `test_analysis_is_offline.py`) need the live
  config archive through the `configs/versions` link. Without the link they fail with
  "No archived Floquet swap config".
- Time: `test_poles.py` 60 s (one test 44 s), `test_notebook_asm_golden.py` 36 s,
  `test_hamiltonian_fit.py` 12 s. `pytest-xdist` is not installed. Note: parallel fitter-A
  workers crash on pippin (`docs/status/qsim_numerics.md`), so check this before using xdist.
- The memory rule "promote every useful script into tests/" was changed: only tests that guard
  a lasting contract go into `tests/`.

## Next

1. Merge the notebook checks (`test_notebook_imports.py`, `test_qsim_notebooks_static.py`,
   the notebook part of `test_analysis_is_offline.py`) into one file.
2. Move the long numerics benchmarks out of the default run (opt-in marker or `tools/`).
3. Make the mock acquisition test use the pinned config set.
4. Add `pytest-xdist`, and a `tests/README.md` with one line per file.
