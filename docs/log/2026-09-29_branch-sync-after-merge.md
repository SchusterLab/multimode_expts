# 2026-09-29: branch sync after the merge to main

- `guan` fast-forwarded to `main` (`6040c3c`, after the `job_id_sorting_out` merge and
  housekeeping) and pushed.
- `main` merged into `qsim-analysis` (`562f60e`, no conflicts); `qsim-analysis` pushed for the
  first time.
- The old remote `qsim-analysis` (guan's Wigner work, Feb–Apr 2026, a separate history) had
  the same name. Checked against `main`: all of it is in `main` except commit `0faea6a`, an
  unfinished lmfit migration of `fitting/fitting.py` (`fitting/models.py`,
  `tests/test_fitting.py`, `docs/lmfit_migration.md`). Renamed on GitHub to
  `archive/lmfit-draft-2026-02` (guan's decision). Most likely obsolete: the tProcV2 migration
  includes an lmfit rewrite. Look at it once when that starts, then delete it.
- Tests on the merged `qsim-analysis`: 1566 passed, 6 failed, none caused by the merge.
  - `test_mock_mode::test_mock_root_off_prod_is_repo_tmp`: the last assert used
    `tmp_path / "C:"`, which on Windows is the drive `C:` and always exists. Changed to check
    that `mock_data` is the only thing made in the working folder.
  - The 5 others are left on purpose (guan): see the known test failures in `docs/STATUS.md`.
