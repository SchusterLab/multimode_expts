Status: current (2026-10-19, first version; not yet run on the device).

# Automated calibration (`experiments/autocal/`)

## Purpose

Replace the semi-manual calibration run through `measurement_notebooks/QEC/single_qubit_autocalibrate_v2.ipynb` and
`multiphoton_calibration_v2.ipynb` with one call that runs the same experiments with the same settings, checks every
fit before it changes the config, stops at the first doubtful fit, and leaves one review page per run.

## Design

- **Step** = one notebook experiment: the experiment class (resolved by name in `experiments`, as `meas.<name>`), the
  notebook's default `cfg.expt` (relax delay included), the run-time arguments (scan windows centred on the current
  config), the notebook's preprocessor, and an `evaluate` function: fit, check, and return the config changes.
  A step without an experiment is a config step (seeds).
- **Check** (`_checks.py`): the fitted value lies inside the scan (not at an edge), the feature stands out from the
  fit residuals (signal-to-noise), oscillations are resolved and in the unambiguous band, and the change from the
  old value is within a limit. Flat data never passes.
- **Run loop** (`_core.AutocalRun`): steps in order through the normal runners and the job queue (one job at a time,
  so other users interleave). A change is applied in memory only if its check passes. The run stops
  (`AutocalStop`) at the first failed check or job; the values after it are unchanged. `run(start_at=...)` resumes.
- **Persistence**: in memory during the run; at the end of each phase a config version with `update_main=False`
  (files and DB records written, the main pointer does not move). The user sets main after the review.
- **Review**: `<run folder>/index.html` (`AutocalRun.report`), rewritten after each step: status, reason, old/new
  values, check numbers, job ids, notebook cell, figure. `record.json` holds the same in machine form. The jobs are
  also logged to the lab notebook as usual when the station has `log_measurements=True`.
- **Time budget**: `AutocalRun.budget()` from the scan settings (points x reps x (relax + 30 us) + 25 s per job).
  The default plan: about 1.5 h of acquisition, 56 steps (2026-10-19 estimate; not yet measured).

## Default plan (`default_phases`)

| Phase | Steps | Source |
| --- | --- | --- |
| readout | resonator spectroscopy (report only, as the notebook), single shot | autocalibrate cells 14-20 |
| qubit | ge spec (100 MHz) -> Ramsey -> amplitude Rabi -> fine Ramsey -> T1; ef spec -> Ramsey -> Rabi -> fine Ramsey -> T1_ef | autocalibrate cells 23-56 |
| ladder | per rung g_n-e_n, e_n-f_n: seed -> Ramsey -> Rabi -> error amp. pi (freq, gain) [-> pi/2 (freq, gain) on g0-e0]; per sideband f_n-g_n+1: seed -> spec (1965-2010 MHz) -> chevron -> length Rabi -> error amp. freq -> length Rabi; up to \|2> | multiphoton cells 7-65 |
| manipulate | parity revival time -> parity spectroscopy -> cavity Ramsey -> gain to alpha | multiphoton cells 81-108 |
| storage | M1-S4 spectroscopy -> full-swap error amplification at fixed pi length: freq coarse -> gain -> freq fine -> gain fine -> freq fine -> convergence check | autocalibrate cell 110; transduction losssplit `calibrate_swap_conv` |

## Differences from the notebooks (on purpose)

- Fit checks before every change (the notebooks apply every fit).
- Seeds: a rung n > 0 starts from the new rung n-1 plus the start-of-run offset (keeps chi); photon 0 from the bare
  qubit. A storage or manipulate start point at the right coupler current comes from `seed_sections` and the
  station's `storage_man_file` (the values at another current are far off: M1-S4 872.4 MHz at 0.25 mA vs 878.3 at 0.30).
- pi/2 pulses only where used (g0-e0, the only multiphoton pi/2 of `prep_fock_state`); its seed keeps its own gain
  (other pulse shape); the Rabi sets the pi gain only.
- The M1 row of the storage dataset changes only from the f0-g1 sideband (the notebook's error-amp postprocessor
  also wrote g0-e0 photon-0 frequencies there).
- Error amplification analysed with `periodic=False` (the `analyze` default `periodic=True` is biased near scan edges).
- EF and M1 spectroscopy windows centred on the current value (the notebooks start at fixed frequencies).
- Not included: chi / Kerr gain sweep (hand-picked peaks), dump mode, storage chevron, storage modes other than S4,
  photon-number spectroscopy. Transduction checks (prep gate, eta) stay in the transduction code.

## Implementation notes

- Module names start with `_`: `import experiments` imports every module of every subpackage and copies all classes
  into the `experiments` namespace (a class name could replace another, and a broken file breaks every import); it
  skips names starting with `_`.
- Tests: `tests/test_autocal.py` (checks on synthetic data, run loop, default plan and mock acquisitions on a mock
  station built from the config archive).
