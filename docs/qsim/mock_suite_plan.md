# Mock suite: what it checks, and the work left (2026-09-23)

The qsim migration notebooks (`measurement_notebooks/202609_qsim_migration/`)
are both working notebooks and a test suite, run by `tools/run_qsim_suite.py`.
This file is the plan for making `--mode mock` runs pass for real reasons.

## The rule

Mock mode and real-data tests check different things.

- **Mock mode checks acquisition code:** imports and notebook plumbing, config
  assembly, qick program build and ASM compile (parameter checks), the
  acquisition loop, the HDF5 save, and reload of the saved file with the
  normal loader. Mock data is all zeros. So mock mode does **not** run
  `analyze`, `display` or postprocessors, and no value fitted from mock data
  goes into a later cell.
- **Analysis code is checked on real saved files** (the pattern of
  `tests/test_mbr_analysis_golden.py` and `tests/test_analysis_is_offline.py`).
  Postprocessors are tested here too, with fits from real data.
- **A mock run changes nothing shared.** Data goes to `mock_data`. The vault
  skips mock (`log_measurement`). Config snapshots skip mock and return
  `CFG-XX-MOCK` (commit `5066eb0`).
- **Hardware runs** (`--mode hardware`) are judged by a person looking at the
  saved plots. The runner only reports crashes.

## Done

- `b43a8e1` mock mode reads `configs/soccfg_snapshot.json`, never the live board.
- `b41bce1` notebooks build `MultimodeStation` and `JobClient` directly. Two
  flags, `MULTIMODE_RUN_MOCK` and `MULTIMODE_RUN_USE_QUEUE`
  (`experiments/qsim/notebook_helpers/run_mode.py`).
- `5066eb0` mock stations do not write config snapshots.
- `5466268` the runner reports cells tagged `raises-exception` as xfail, and
  as XPASS if they ran clean.
- `d90f871` step 1: on a mock station, runner `execute()` defaults to acquire
  and save only.
- `436da60` step 2 for `floquet_calibration`, `multiphoton_calibration` and
  `floquet_displacement_kerr`: all pass `--mode mock` (smoke). Manual
  `expt.display()` calls moved into the runner (`show=`, `display_kwargs=`);
  acquire + fit cells split, fit/accept halves tagged `mock-skip`. Kerr
  `storage_reset` drops mode 6 (no pi time in the pinned M1 config).
- Step 3: `assert_reloads` in `tests/test_qsim_notebook_mock_acquisition.py`
  reloads each `CharacterizationRunner` / `SweepRunner` family's mock file and
  pins its array shapes. MBR files already reload in
  `tests/test_mbr_acquire_mock.py` (`load_job`).

Still open: step 2 for the four MBR notebooks, which waits on step 4.

## Work left

Do these in order. Commit after each step.

### 1. Runner mock defaults

In mock mode, make `CharacterizationRunner`, `SweepRunner` and `BatchRunner`
default to acquire and save only: `analyze=False`, `display=False`,
`postprocess=False`. `execute()` already defaults to local in mock mode, so
put the new defaults in the same place. An explicit argument from the caller
still wins.

- Check `run_local(postprocess=..., go_kwargs=...)` and `Experiment.go(save,
  analyze, display)` for the exact flags.
- `BatchRunner` aggregates its jobs (`_aggregate`). Check whether the
  aggregate calls `analyze`.
- Tests: `tests/test_characterization_runner.py` has mock tests that expect
  the postprocessor to run (`test_runner_postprocessor`). Those tests call
  `run_local(postprocess=True)` explicitly, so they should still pass. Check.

### 2. Notebook tags

For each measurement notebook, run it alone in mock mode and fix failures one
at a time:

    pixi run python tools/run_qsim_suite.py --mode mock --only floquet_calibration

Executed copies with tracebacks go to `.suite_runs/<timestamp>_measurement_mock/`.

- A cell that reads a fit result (`expt.data['fit_...']`, the accept cells,
  hand checks of fitted values): tag `mock-skip`. Later cells then build
  with the station's calibrated config values.
- A known code bug: tag `raises-exception`, with a one-line comment giving
  the reason. Only do this if later cells do not need that cell's output.
- Do not change physics code to make mock pass.

Failures seen on 2026-09-23 (mock, smoke, qick 0.2.291):

| Notebook | Cell | Error | Likely cause |
| --- | --- | --- | --- |
| `floquet_calibration` | "Source cell 72: one fixed +/-1 MHz span" | `RuntimeError: ('unsupported pulse parameter(s)', {'length'})` from `FloquetChevronProgram.core_pulses` (`experiments/qsim/floquet_chevron.py:15`) | Known bug (the user says xfail): sets `length` on an arb pulse. Tag `raises-exception`. |
| `multiphoton_calibration` | first `ErrorAmplificationExperiment` cell (about cell 37) | `ValueError: cannot convert float NaN to integer` in `us2cycles` (`experiments/single_qubit/error_amplification.py:102`) | An earlier fit on zeros gives NaN. |
| `floquet_displacement_kerr` | the 2D job cell | same NaN, through `active_reset` -> `man_stor_swap` -> `custom_pulse` (`experiments/MM_base.py:577`) | NaN from a fit, or an empty row in the M1 or storage-swap CSV. Check which. |
| `mbr`, `mbr_tomography`, `mbr_sff`, `mbr_disorder` | calibration cell | `RuntimeError: (N, 0, 0, 0, 0) has too few valid IQ points` in `MBRPhaseCorrectionExperiment.analyze_cycle_phase` | Analysis of mock zeros. See step 4. |

### 3. Reload check

After a mock acquisition, load the new file with the normal loader
(`from_h5file` / `experiments/saved_jobs.py`) and check the array shapes,
without fitting. This finds a change in the file layout that the real-data
analysis tests (with older files) cannot find. One pytest per experiment
family in `tests/test_qsim_notebook_mock_acquisition.py` is enough.

### 4. MBR calibration in mock

`experiments/qsim/notebook_helpers/mbr_campaign.py:acquire_calibration` runs
`calibration_expt.analyze()` on the data it acquired. In smoke runs the MBR
notebooks call it (`if RUN.smoke:`), and each notebook takes about 20 minutes
before it fails. Decide with the user:

- (a) in mock, acquire and save but skip the analysis. The later cells need a
  calibration, so they must then be tagged `mock-skip`.
- (b) in mock, load a real calibration by job ID from saved HDF5
  (`ensure_calibration`). This works only where the data are (pippin or a
  mounted data tree).

## Things to know

- Always `pixi run python` / `pixi run pytest`.
- On pippin, this worktree's `job_server/jobs.db` and `configs/versions` are
  links to the main checkout's. Anything that writes there writes to the
  shared store.
- Nine orphan config versions from 2026-09-23 are in that store (see
  `5066eb0`). The user has not yet decided whether to delete them.
- The shared `C:\python\qick` is used by every checkout. It must be at
  0.2.291 (`645f8905`) for this code. Another user may switch it for tProc v2
  tests. Check `qick.__version__` first if a station fails to start with a
  `KeyError` on the soccfg.
- Do not run `--mode hardware` unless the user asks. It drives the real
  device, and it takes the main worker lock.
