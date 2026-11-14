# 2026-11-14: herald cut in the population analysis, and a heralded confusion matrix (Claude, for jonginn)

## Why

jonginn wants heralded AGP runs (`prep_herald="end"`) overnight. These are guan's two 11-13
"Next" items in `docs/status/measurement.md`: the herald cut inside
`FockPopulationExperiment.analyze()`, and a confusion-matrix experiment that plays the herald.
The third item, the f cut, is left for later (jonginn's choice). guan agreed to this work.
Done directly in main at jonginn's request; no job was running.

Before this, `FockPrepHeraldProgram` played the herald readouts and the files kept them, but the
population analysis used all shots, and the confusion matrix had no herald. guan's 11-13 data
show that a herald without the cut makes the empty-mode floor larger, not smaller.

## What changed

- `experiments/qsim/multiparity.py`: added `herald_keep(cfg, idata)`, the keep mask: every prep
  herald readout reads g. The herald readouts are the lanes between the active-reset readouts
  and the two parity readouts; the cut applies only to `prep_herald` "end" and "each".
  `classify_two_parity_readouts(keep=None)` and `analyze_multiparity()` use it: all results come
  from the kept shots. The analysis now also saves `shot_counts` (the shots used per point), and
  with heralds `kept_fraction`, `p_mod0_all` .. `p_mod3_all` and `nmod4_mean_all` (all shots, for
  comparison). Without heralds nothing changes.
- `experiments/qsim/fock_population.py`: `FockPopulationExperiment.analyze()` takes the kept
  counts for `n2_err`. `FockPopulationSetExperiment` refuses jobs with different `prep_herald`,
  stacks `kept_fraction`, and records `prep_herald` in its manifest parameters (so the AGP part
  manifests have it too).
- `experiments/qsim/fock_prep_herald.py`: added `HeraldedCalibrationProgram` (Fock 0..3 in M1
  through the population ladder, exactly one herald readout per shot; a bare one for Fock 0, so
  that all points of one job have the same readouts) and `HeraldedConfusionMatrixExperiment` (the
  multiparity confusion matrix as one job, from the kept shots; it also saves
  `multiparity_confusion_matrix_shot_counts`, the kept shots of each row).
- `experiments/qsim/agp.py`: `calibration_shots()` returns
  `multiparity_confusion_matrix_shot_counts` (one count per row) when the calibration file has
  it, so the calibration part of the AGP errors uses the kept shots. The name is not
  `calibration_shot_counts`: the sequential calibration already saves that, with another shape.
- `measurement_notebooks/jonginn/agp_measurement.py` (and its `.ipynb`, synced with jupytext):
  the `HERALD` switch. It selects the calibration class and the population program, adds
  `prep_herald="end"` to the defaults, adds `_herald` to the AGP names, and prints the kept
  fractions. With `HERALD = False` the notebook runs as before.
- Tests: 8 in `tests/test_fock_prep_herald.py`, 1 in `tests/test_fock_population_set.py`, and
  `tests/test_herald_cut_device_data.py` (the checks below on guan's job files; skipped on a
  machine without them).

## Checks

- guan's 11-13 walk (0.45 mA), the five "end" jobs JOB-20261113-00408, -00412, -00414, -00418,
  -00422: the new analysis keeps exactly the shots of his offline cut (`labels_from_h5` and
  `select(herald=True)`); kept fraction 0.89-0.91. With his "end" matrix (rows from
  JOB-20261113-00398 and -00402..-00404): empty storages 0.018-0.085 at cycle 0, sum of the five
  modes 3.23 at cycle 0, mean 2.88 over the cycles. These are his numbers in the vault entry
  ANA-20261113-142706.
- A job without heralds (JOB-20261113-00593) gives exactly its saved results.
- Mock station: the calibration job compiles and runs (5 readouts per shot at all 4 points); a
  population job for [1, 1, 1, 0, 0] has 7 (2 reset, 3 heralds, 2 parity). The mock returns I = 0
  in every readout (above the threshold), so the tests check the cut on synthetic shots.
- Full test suite: see the end of this entry.

## Known limits

- An existing heralded job that is analyzed again now gives cut results (guan's 11-13 walks;
  jonginn's walk-cell jobs with `prep_herald="end"`). The `*_all` keys keep the all-shot values.
- The heralded calibration prepares with the population ladder at cycle 0, not the prepulse
  path of `MultiParityConfusionMatrixExperiment`. Its Fock 0 row has one bare herald readout;
  guan's 11-13 matrices had none on Fock 0.
- "each_reset" is not cut. The f cut is still to do (one more keep mask in `analyze_multiparity`).

## Next

- First device run at 0.25 mA (guan's herald data are at 0.45 mA): the kept fraction of each
  calibration row and each occupation, the empty-mode floor, the AGP. Then set `AGP_REPS` to about
  reps / kept fraction for the same error bars.
- The f cut as an option of the multiparity analysis.
