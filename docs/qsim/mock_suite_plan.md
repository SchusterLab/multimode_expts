# Qsim notebook suite: what checks what (revised 2026-09-23)

The qsim migration notebooks (`measurement_notebooks/202609_qsim_migration/`)
are working notebooks copied from the old source notebooks. The goal is basic
guardrails, so that bad code can be cut out without working blind. A rewrite
from the ground up is likely.

The first version of this plan tried to make every notebook pass in mock
mode. That was dropped: pytest already checks, in mock, the parts of these
notebooks that mock mode can check. So each layer has one check:

| Layer | Checked by | Runs where |
| --- | --- | --- |
| Library code: program build, ASM compile, mock acquisition, HDF5 save and reload | pytest mock tests (below) | anywhere, about 1 min |
| Analysis numerics | pytest on real saved files (`tests/test_mbr_analysis_golden.py`, `tests/test_analysis_is_offline.py`) | where the data are |
| Notebook wiring: imports and names between cells | `tests/test_qsim_notebooks_static.py` | anywhere, seconds |
| Physics | a hardware run of the notebook, judged by a person looking at the plots | pippin only |

## The rules

- **The notebooks are a suite for hardware runs.** `tools/run_qsim_suite.py
  --mode hardware` runs them; the runner reports only crashes.
- **Mock mode is an optional pre-flight**, only for `floquet_calibration`,
  `multiphoton_calibration` and `floquet_displacement_kerr` (about 1 min
  together). If a pre-flight breaks, do not spend time on it: drop the
  notebook from mock (`HARDWARE_ONLY` in the runner).
- **The MBR notebooks never run in mock** (`HARDWARE_ONLY`). Every cell after
  their phase calibration needs a real calibration: its phases go into the
  pulse configs of the later jobs.
- **When you cut code, its guardrail is the pytest next to it**, not a
  notebook. Before you remove or rewrite a module, check that a pytest covers
  it, and add one if not.
- **A mock run changes nothing shared.** Data goes to `mock_data`. The vault
  skips mock. Config snapshots skip mock and return `CFG-XX-MOCK`.

## The pytest mock tests

- `tests/test_qsim_notebook_mock_acquisition.py`: each non-MBR experiment
  family, built and acquired through its runner as in the notebook cell.
  `assert_reloads` loads the saved file with `from_h5file` and
  `saved_jobs.load_h5` and pins the array shapes. Also compiles the four MBR
  batch types, with zero phases in place of a calibration.
- `tests/test_mbr_acquire_mock.py`: every MBR stage (calibration, spectrum,
  propagator, orthogonality, SFF) built, compiled and acquired from committed
  pinned configs, and reloaded with `load_job`.
- `tests/test_runner_mock_defaults.py`: on a mock station, runner `execute()`
  only acquires and saves (no analyze, no postprocess).

The acquisition tests copy the notebook cell bodies, so they can drift from
the notebooks. That is accepted: the static check covers the notebook side.

## Done

- `b43a8e1` mock mode reads `configs/soccfg_snapshot.json`, never the live board.
- `b41bce1` notebooks build `MultimodeStation` and `JobClient` directly. Two
  flags, `MULTIMODE_RUN_MOCK` and `MULTIMODE_RUN_USE_QUEUE`
  (`experiments/qsim/notebook_helpers/run_mode.py`).
- `5066eb0` mock stations do not write config snapshots.
- `5466268` the runner reports cells tagged `raises-exception` as xfail, and
  as XPASS if they ran clean.
- `d90f871` on a mock station, runner `execute()` defaults to acquire and
  save only. An explicit argument from the caller still wins.
- `436da60` the three pre-flight notebooks pass mock (smoke). Manual
  `expt.display()` calls moved into the runner (`show=`, `display_kwargs=`);
  acquire + fit cells split, fit/accept halves tagged `mock-skip`. Kerr
  `storage_reset` drops mode 6 (no pi time in the pinned M1 config).
- `b990aed` `assert_reloads`: reload and shape checks for mock files.
- MBR notebooks marked `HARDWARE_ONLY`; `tests/test_qsim_notebooks_static.py`
  added.

## Known issues, not fixed

- `FloquetChevronProgram` sets `length` on an arb pulse
  (`experiments/qsim/floquet_chevron.py:15`), so the frequency chevron cannot
  build under `preload_flattop`. `floquet_calibration` cell 72 is tagged
  `raises-exception`; `test_floquet_chevron_only_accepts_the_legacy_flat_top`
  records it.
- `M1-S6` has no pi time in `CFG-M1-20260904-00014`
  (`test_storage_mode_6_has_no_calibrated_pi_length`).
- In the MBR notebooks, `merge_replacement_calibration` rebuilds the
  calibration from job IDs, so it cannot work on a local (no queue) run.
- A mock MBR calibration takes about 20 min, from slow code in the hot path
  (benchmarked on the laptop). Not a target for now.

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
