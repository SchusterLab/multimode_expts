# 2026-11-13: `FStateHoldExperiment`, the |f,n> hold scan in one job (guan + Claude)

Purpose: check after the cooldown whether the |f,n> -> |g,n-1> decay of 10-28/29 is still
there (physics: vault report `Lab/guan/AGP/2026/10/2026-10-29_TLS_12GHz_report`). The
10-28/29 script (`measurement_notebooks/guan/yoko450ua_calibration.py`, cells 8-11) ran each
hold point as its own `HistogramExperiment` job: about 23 jobs of ~5 s for one set point.

## Done

- `experiments/qsim/f_state_hold.py`:
  - `FStateHoldProgram(HistogramProgram)`: one sweep point as the prepulse. A reference (g, e,
    f with the standard qubit pi pulses, as the `HistogramExperiment` references), or the Fock n
    ladder (`MM_base.prep_man_fock_state`), g_n-e_n, e_n-f_n, then `["wait", t]`. The same
    program as on 10-28/29, so the pulse timing is the same. The point keys go into the
    program's own copy of `cfg.expt`, not into the saved config.
  - `FStateHoldExperiment(QsimExperiment)`: one program per point in one job (references, then
    n-major, wait-minor), raw shots kept. `analyze()` writes plain arrays into `data` (blob
    centers, assignment matrix, raw and corrected populations, decay fit per n, omega_M1,
    nu_n). `display()`: one panel per n with the P(f) fit.
- `fitting/qsim/f_hold.py`: nearest-cloud assignment, 3x3 correction (least squares, so a
  singular matrix does not raise), exponential-plus-offset fit of P(f) with the share to g (the
  10-29 model), nu_n = s_n + 2 omega_M1 from calibrated frequencies.
- `tests/test_f_state_hold.py` (13 tests): estimators on synthetic shots and curves; a mock run
  that compiles every point (|f,2> plus a wait fits the envelope memory), checks the pulses of
  the last point, the saved config and the HDF5 file. Also compiled on the mock station with
  the configs of the notebook (24 points).
- `measurement_notebooks/guan/20261113_tls_recheck.py`: station, config branch `tls_0.45mA`,
  one hold job.

## Decisions

- guan: one job for the whole scan, not many short jobs, because jonginn interleaves
  ~2-minute jobs on the same queue.
- guan: use jonginn's 0.45 mA calibration of this cooldown (IDs in the notebook and the vault).
- Claude: no single shot before the hold job. The analysis assigns each shot to the nearest
  in-job reference cloud, so the readout phase and threshold do not enter.

## Found

- `Experiment.go()` calls `analyze()` before `save_data()`: an error in `analyze()` on the
  worker loses the data. `FStateHoldExperiment.analyze()` catches and prints errors; its
  estimators return NaN instead of raising. Other Experiments (for example
  `HistogramExperiment.analyze()`, which raises on invalid centers) have the same exposure.
- `CharacterizationRunner.execute()` on a mock station turns `analyze` off by default
  (`mock_run_defaults`; on purpose, `docs/running_experiments.md` item 6), but the worker
  always runs `go(analyze=True)`. A mock test of an analysis must pass
  `go_kwargs=dict(analyze=True)`.
- `experiments/qsim/readout_calibration.singleshot_runner` defaults to `avoid_yoko=True` and
  `use_queue=False`: wrong for a single shot at another user's set point unless both are passed.
- `SlabFile.add` stores `np.array(value)`: string values in `data` cannot be saved, so the
  analysis keeps only numeric arrays.

## Later on 11-13: f shots in the multiparity readout, and heralds in the M1 ladder

guan asked whether the 10-28 decay explains why empty modes read above 0 in
`FockPopulationExperiment` (multiparity readout), even after the confusion-matrix correction.
Physics and numbers: vault `Lab/guan/AGP/2026/11/2026-11-13` (Claude notes). Code:

- `fitting/qsim/f_postselect.py`: readout lanes from interleaved shots, n mod 4 decoding (as
  `classify_two_parity_readouts`), the shots where the qubit reads f (nearest g/e/f center), p(n)
  with a cut, the confusion matrix from shots with the same cut, corrected <n>.
- `experiments/qsim/fock_prep_herald.py`: `FockPrepHeraldProgram(FockPopulationProgram)` plays the
  M1 ladders step by step with `prep_herald` = none / end / each / each_reset (bare readout:
  `MM_base.post_selection_measure`; reset: qubit-only `MM_base.active_reset`, then a g herald).
  `readouts_per_shot` adds the prep lanes. `FockPrepHeraldExperiment` sweeps `ro_stor` at one prep
  and one mode per job (the lane count must be the same for all points of a job, because
  `QsimExperiment.acquire` takes `read_num` once). `shot_labels()` gives per-shot n, heralds and
  the f flags. Tests: `tests/test_fock_prep_herald.py` (12, mock runs of all four modes).
- `analysis_notebooks/guan/20261113_f_postselect.py`: the f cut on jonginn's 11-11/12 population
  jobs and the 0.45 mA quantum walk; vault entry by `log_analysis` from a mock station.
- `measurement_notebooks/guan/20261113_tls_recheck.py` cells 3-4: the herald jobs and their table.

Decisions: guan: keep the measurement and analysis code that is run in repo files, not in a
scratch folder, and log analyses in the vault.

Found: in the multiparity readout, a qubit in f reads (e, e), so n = 3, and the parity readout
uses only I, so these shots cannot be seen without Q. The confusion matrix does not correct this
error, because the error depends on the prep and not on the read mode.

Afternoon additions (code):
- `FockPrepHeraldExperiment.display()`: raw p(n mod 4) per read mode, all shots against the
  herald cuts, with the kept fractions. Before this, the vault showed the generic
  `QsimExperiment.display()` (averaged I/Q of the last readout against `ro_stor`), which does not
  mean anything here (guan found it misleading). The module docstring now lists the HDF5 fields.
- `labels_from_h5(path, centers)`: per-shot labels from a job file, for `FockPrepHeraldExperiment`
  and for `FockPopulationExperiment` run with `ExptProgram=FockPrepHeraldProgram` (the
  population analysis uses only the last two readouts, so it runs unchanged with herald lanes;
  the herald cut is then done offline). Tests 13-14 in `tests/test_fock_prep_herald.py`.
- Gap: the HDF5 files store no names for the readout lanes; their order is only in
  `prep_lanes()` and the docstring.

Clean-up for reuse (end of the day):
- `fock_prep_herald.prep_lanes()` returns early for "none", so `labels_from_h5()` reads any
  multiparity job file (also plain `FockPopulationExperiment` and
  `MultiParityConfusionMatrixExperiment` files, which have no `occupations` or no heralds).
- `fock_prep_herald.select(labels, herald=, after_reset=, no_f=)`: the keep mask for the cuts.
  `f_postselect.mean_n_with_error()`: raw <n> and its shot error per point. Tests 15-17.
- `analysis_notebooks/guan/20261113_f_postselect.py` rewritten on these (imports in cell 1, same
  cell numbers as the vault notes cite). Its vault cells run only with `LOG_TO_VAULT = True`, so
  running it again makes no duplicate entries. A full rerun gives the logged numbers.
- Summary for the other device users (physics): vault
  `Lab/guan/AGP/2026/11/2026-11-13_f_state_floor_report.md`.
- Not done: the herald cut inside `FockPopulationExperiment.analyze()`, and a confusion-matrix
  experiment that plays the herald (today the rows are `FockPrepHeraldExperiment` jobs).
  `fock_prep_herald` imports the private `qsim_base._herald_readouts`.
