# Running experiments on the real device

When user asks for measurement runs on the real device,
follow a recent jupytext notebook in a measurement sandbox (e.g.,
`measurement_notebooks/guan/agp_layers_hw_check.py`). The steps:

1. **Code must be on main first.** The worker runs only `C:\python\multimode_expts` and
   reloads `experiments.*` for each job. So it also runs any half-edited files on main.
   Develop in a worktree, merge, then submit. Do not leave broken edits in main while
   jobs are queued.
2. **Test on a mock station:** Use `MultimodeStation(mock=True)` with the same
   config versions. Pass `go_kwargs=dict(analyze=True)` (a local run does not analyze by default,
   the worker always does).
3. **Station:** set `user`, `experiment_name`, `project`, and pin the config versions
   (`hardware_config="CFG-HW-..."` etc.) that the calibration you rely on was made with.
   But an old snapshot has an old readout phase: see item 7.
   Set `log_measurements=True`, or nothing goes to the Obsidian vault.
4. **Submit:** use one of the job-server-aware runners eg `CharacterizationRunner(station, ExptClass, default_expt_cfg,
   job_client=JobClient(), use_queue=True)`, then `runner.execute(overrides=[...],
   batch_size=...)`. Extra kwargs (such as `reps`, `relax_delay`) go into every job.
5. **Other users:** `JobClient().print_queue()` first. If interleaving, keep each job to a few minutes or
   use a reasonable `batch_size` so their jobs can run in between. Start with low reps and a short sweep to check the logic.
6. **Shared hardware state:** `avoid_yoko=True` prevents the worker from ramping the
   coupler, but then your job runs at the current the last user set and the results are likely invalid if current is different from setpoint in hardware_config.
   We have found no adverse effects of ramping yoko between every job so far as long as set points are within the normal range (|current| < 1e-3 A; the config stores amps, e.g. `current: 0.00025`).
   So the switch is mostly intended for bypassing a wedged yoko.
7. **Readout phase:** all users' jobs must use the identical `device.readout.frequency`.
   When the readout channel frequency changes, the phase reference is lost for everyone.
   If you refit the readout frequency, tell all other users to pull the new value into their
   `hardware_config.device.readout`. Even at an identical frequency the readout phase drifts
   over days, so at the start of each day run a single shot (`HistogramExperiment` with
   `singleshot_postproc`, see `measurement_notebooks/guan/single_qubit_autocalibrate_v2.py`)
   and update `phase`, `threshold`, `Ig`, `Ie` in your hardware config. A wrong phase gives no
   error: the g and e blobs are still well separated, but the stored threshold reads the wrong
   state and active reset misfires. Check: in the histogram file, the direction of
   median(e) - median(g) in the raw I/Q plane must be near 0 degrees. On 2026-10-23 the phase
   also jumped by about 210 degrees at a probable worker restart (about 10:20) with no
   frequency change; check again after a restart.
8. **Monitor:** use `JobClient` job status, not background stdout (it is buffered). The data
   file is in `C:\experiments\<experiment_name>\data\JOB-<id>_<Class>.h5`. The worker
   runs `go(analyze=True, save=True)`.
9. **Never** run hardware code outside the queue while the worker is running.
10. **Physics**: the device consists mostly of a qubit with T1<=500us (worse when TLS present), one or two cavity modes (M1, M2) with ~100--200us lifetime, and 7 more cavity modes (S1..S7) with ~1ms lifetimes.
   The `relax_delay` param in us always passed to every QICK Program sets the wait time between each Program execution to allow the excited modes to all relax back to vacuum.
   If no active reset, depending on which modes are used, the typical wait times are: (qubit or qubit + M: 2.5ms; qubit + M + S: 8ms).
   Qubit active reset typically works. Reset of M and S modes rely on dump mode and could fail sometimes. When everything works, typical waits: (qubit only: 0.2ms; qubit + M: 0.5ms; qubit+M+S: 2.5ms).
   When in doubt, sanity check with active reset off and 8ms relax_delay.
