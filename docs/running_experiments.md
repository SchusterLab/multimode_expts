# Running experiments on the real device

Status: current (2026-10-29). This file: how the code and the device behave, and the rules
that follow from them. They are true for every user and every session. How we work together
(kernels, records, the vault, the order of a session) is in `docs/measurement_session_workflow.md`.

For a working example, follow a recent jupytext script in a measurement sandbox (e.g.
`measurement_notebooks/guan/agp_layers_hw_check.py`).

## The code path

1. **Code must be on main first.** The worker runs only `C:\python\multimode_expts` and
   reloads `experiments.*` for each job. So it also runs any half-edited files on main.
   Develop in a worktree, merge, then submit. Do not leave broken edits in main while jobs
   are queued.
2. **Never** run hardware code outside the queue while the worker is running.
3. **Station:** `MultimodeStation(user=..., experiment_name=..., project=...,
   log_measurements=True, hardware_config="CFG-HW-...", ...)`. Pin the config versions that
   the calibration you rely on was made with. An old snapshot has an old readout phase (see
   "Readout phase"). Without `log_measurements=True`, nothing goes to the vault.
4. **The station is stateful.** It holds the configs in memory (`station.hardware_cfg`,
   `station.multimode_cfg`, the datasets); postprocessors and your own edits change them.
   Keep one station per kernel and do not construct a new one to change modes:
   - `station.use_mock_instruments()`: mock QICK board and yokos, mock output paths. All
     in-memory configs stay. `station.use_real_instruments()` switches back.
   - Construct `MultimodeStation(mock=True)` only when there is no real station to start from
     (off prod, or a quick standalone check).
5. **Config branches** (`experiments/branch_manager.py`): a small version log that we made,
   **not git**. A branch is a name that points to the 4 config version IDs (hardware,
   multiphoton, man1, floquet). `BranchManager(station).commit(name, note)` saves the station's
   current configs as new versions and moves the branch to them; `checkout(name)` loads them
   back. The log is `branches.jsonl` in your sandbox folder. In these docs, "commit the config
   branch" means `BranchManager.commit`, never `git commit`.
6. **Test on mock first.** Switch the station to mock and run the same runner call: on a mock
   station `runner.execute(...)` runs locally, not on the queue. Mock data is all zeros, so by
   default the mock run does not analyze and does not postprocess (a mock fit must not write
   into your configs). To test the analysis code, pass `go_kwargs=dict(analyze=True)`; leave
   `postprocess` off. Mock runs are not logged to the vault.
7. **The mock does not check board resources** (envelope memory, readout buffer lengths). A job
   that exceeds them compiles in mock and fails at upload on the real board. Until the check
   exists (`docs/lab_workflow_tooling_spec.md`, item D), be careful with long pulses and many
   prepulse gates on one channel.
8. **Submit:** a job-server-aware runner, e.g. `CharacterizationRunner(station, ExptClass,
   default_expt_cfg, job_client=JobClient(), use_queue=True)`, then
   `runner.execute(overrides=[...], batch_size=...)`. Extra kwargs (such as `reps`,
   `relax_delay`) go into every job. The worker runs `go(analyze=True, save=True)`.
9. **Monitor:** use `JobClient` job status, not background stdout (it is buffered). The data
   file is `C:\experiments\<experiment_name>\data\JOB-<id>_<Class>.h5`.

## Shared hardware

- **Other users:** `JobClient().print_queue()` first. If you interleave with others, keep each
  job to a few minutes or use a small `batch_size` (2-3) so their jobs can run in between. Start
  with low reps and a short sweep to check the logic.
- **Coupler current (yoko):** the worker ramps the yoko to the set point in each job's
  `hardware_config`. We have seen no adverse effects from ramping between every job, if the set
  points are in the normal range (|current| < 1e-3 A; the config stores amps, e.g.
  `current: 0.00025`). `avoid_yoko=True` stops the ramp, but then the job runs at the current
  that the last user set, and the result is likely invalid. Use it only to bypass a wedged yoko.

## Readout phase

- All users' jobs must use the identical `device.readout.frequency`. When the readout channel
  frequency changes, the phase reference is lost for everyone. If you refit the readout
  frequency, tell all other users to pull the new value into their
  `hardware_config.device.readout`.
- Even at an identical frequency, the phase drifts over days, and the readout calibration is
  lost when the hardware restarts. So at the start of each day, and after any hardware restart,
  run a ge single shot
  (`HistogramExperiment` with `singleshot_postproc`, see
  `measurement_notebooks/guan/single_qubit_autocalibrate_v2.py`) for each set point you use.
  It writes `phase`, `threshold`, `Ig`, `Ie` into the station's hardware config; then commit the
  config branch.
- A wrong phase gives no error: the g and e blobs are still well separated, but the stored
  threshold reads the wrong state and active reset misfires. Check: in the histogram file, the
  direction of median(e) - median(g) in the raw I/Q plane must be near 0 degrees.
- A g/e/f single shot does not set the ge phase (its fitted angle is the e-f axis). Do not let
  it write into the config.

## The device

- **Modes:** one transmon qubit (T1 up to about 500 us, less when a TLS is near); one or two
  cavity modes M1, M2 (lifetime about 100-200 us); seven storage modes S1..S7 (about 1 ms).
- **Facts that hold for this device:**
  - The coupler is a SQUID that provides coupling between M1, M2 on one side and storage modes on the other.
    Sideband transitions are activated by flux drives. Usually it does not affect the qubit.
  - M1 and the storage modes do not affect the qubit readout resonator.
  - Photons in M1 shift the qubit line: the line with n photons is at f_ge + n chi_ge, with
    chi_ge < 0.
- **`relax_delay`** (us, passed to every QICK Program) is the wait between two shots, so that
  all excited modes relax to vacuum.
  - With no active reset: qubit or qubit + M: 2.5 ms; qubit + M + S: 8 ms.
  - Qubit active reset usually works. Reset of the M and S modes uses a dump mode and can fail.
    When all resets work: qubit only: 0.2 ms; qubit + M: 0.5 ms; qubit + M + S: 2.5 ms.
  - When in doubt, check with active reset off and `relax_delay` 8 ms. Can also tighten the waits
    if wait time becomes acquisition speed bottleneck. Run a quick characterization to trim fat.
- What changes from day to day (TLS positions, T1, the current calibration) is not in this
  file: it is in the vault day files.
