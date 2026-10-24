Status: current (2026-10-20; first device run 2026-10-19 after the cooldown, see docs/log/2026-10-20_cooldown-recal-night.md).

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
| ladder | per rung g_n-e_n, e_n-f_n: seed -> Ramsey -> Rabi -> error amp. pi (freq, gain) [-> pi/2 (freq, gain) on g0-e0]; per sideband f_n-g_n+1: seed -> spec (1965-2010 MHz) -> chevron -> length Rabi -> error amp. freq -> N = 1 map -> length Rabi; up to \|2> | multiphoton cells 7-65; N = 1 map from transduction losssplit |
| manipulate | parity revival time -> parity spectroscopy -> cavity Ramsey -> active-reset single shot -> gain to alpha (41 gains 0-4480, 2000 reps, width fit with the offset fixed) | multiphoton cells 81-108 |
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
- Error amplification analysed with `periodic=False` (`periodic=True` is biased near scan edges; since 2026-10-24
  False is also the `analyze` default).
- EF and M1 spectroscopy windows centred on the current value (the notebooks start at fixed frequencies).
- Ladder qubit rungs: error amplification only (seb 10-19). The notebook's Ramsey / amplitude Rabi measure the
  bare-frame values; the pulses use the driven-frame values (g0-e0: Ramsey 3561.53 vs error amp. 3562.09 MHz).
- Recovery rules (each at most one repeat, recorded): a clean error-amp. peak at the scan edge moves the centre; a
  coarse swap scan off centre (symmetry centre at the edge) moves to the sharpest product peak; a swap gain optimum
  outside the window moves the window to the end value. A failed repeat restores the old value and stops.
- Storage swap gain scans with n_step 1: with n_step 2 every row is a multiple of 4 pulses and every swap angle
  k pi/4 passes (10-19: the scan moved to 3 pi/4 after the cooldown). Verify a new swap with a chevron (one swap,
  frequency x length) when the gain moves by more than ~10 %.
- Not included: chi / Kerr gain sweep (hand-picked peaks), dump mode, storage chevron, storage modes other than S4,
  photon-number spectroscopy. Transduction checks (prep gate, eta) stay in the transduction code.

## Known limits (10-19 and 10-20 device runs)

- Sideband frequencies: neither the 20-pulse error amplification nor the N = 1 map centre gives the best preparation.
  10-20, |2> Wigner (P2) on a 5 x 5 grid of f0-g1 x f1-g2: f0-g1 best at +40..+80 kHz above the N = 1 centre
  (row means 0.955-0.957 vs 0.944 at the centre); f1-g2 flat over -40..+120 kHz; the error amplification was 80 kHz
  below the N = 1 centre that day (P2 0.909) and 128 kHz from the day before. The step should optimise the final
  state (P2 of |2>, or a fast proxy) instead; the same setting repeats within +/-0.013 in P2, so a grid or a fit is
  needed, not one point.
- Gain to alpha (10-20, B'): 2.558e-4 +/- 0.7 %, within 1 % of the photon-number Ramsey of a coherent state; the old
  full-grid runs (2.641 / 2.605e-4) were ~3 % high. A free-offset fit on the short grid gives offset ~+0.03 every time
  (possible non-Gaussian tail); the gate Wigners agree with the new scale (s ~ 0.975 on the old scale).
- Swap convergence limit 0.5 % gain is below the precision of the n_step 1 gain scan (scatter ~1.5 %); give the gain
  scans 20 pulses (n_pulses 10) or a 2 % limit.
- The EF Ramsey beats (f level split ~8-12 kHz, size drifts, centre fixed); the one-frequency fit biases f_ef by
  ~5 kHz. A two-frequency fit should be the EF Ramsey check.

## Implementation notes

- Module names start with `_`: `import experiments` imports every module of every subpackage and copies all classes
  into the `experiments` namespace (a class name could replace another, and a broken file breaks every import); it
  skips names starting with `_`.
- Tests: `tests/test_autocal.py` (checks on synthetic data, run loop, default plan and mock acquisitions on a mock
  station built from the config archive).
