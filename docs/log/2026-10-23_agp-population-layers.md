# 2026-10-23: AGP data in three layers, legacy conversion redone (guan + Claude, overnight)

## Why

guan looked at jonginn's 10-22 legacy AGP conversion (`analysis_notebooks/jonginn/convert_legacy_agp.py`,
run 10-22 19:27) and overrode its storage design. The converted per-W files were 386 MB and the
assembled `fine.h5` 4 GB, because every layer kept full copies of all shots (root stack plus
`legacy_jobs/` plus `children/*`). Commit `40a9e82` removed some copies, but the files on disk
were from before it. Decisions (guan, 2026-10-22):

- A job file keeps its shots. Every layer above keeps only its own arrays and a pointer to the
  layer below. Assembly is a convenience layer; with good provenance, shots are not needed there.
- The per-W layer keeps the 15 states x cycles moments; the scan layer keeps only the
  C_k(W, t) maps.
- Readout correction happens at the earliest moment: at acquisition (the job carries the
  matrix), or at conversion for old jobs. The files keep raw and corrected values side by side.
- One job is one state at one W (a few minutes), not one 2D (states x cycles) job per W, so
  other users' jobs can interleave. The old `CentralBosonAGPExperiment` was a 2D job
  (~75 min per W for the fine grid) and never ran on hardware.
- One-off conversions go in `tools/`; classes are dataset abstractions whose `acquire()`
  really acquires.
- Convert all the data sets looked at, then delete the redundant outputs (raw data untouched).
  The 10-19 stopgap `tools/backfill_fock_population.py` (never committed) can go once its data
  sets (the three quantum walks and the W=30 fine AGP) are covered.

## What was done (branch `agp-layers`, merged to `main`)

Branched from `main`, not `guan`: `guan` is 58 commits behind `main`, and jonginn's AGP work is
only on `main`. Developed in the worktree `C:\python\multimode_expts_agp` so that half-edited
files never reached the worker (it reloads `experiments.*` for every job) while seb's overnight
jobs ran; `main` was fast-forwarded only after the full suite passed (1482 passed; the known
`test_matrix_pencil_regression.py` file excluded).

- `experiments/qsim/fock_population.py`:
  - `FockPopulationExperiment` (job, `default_program = FockPopulationProgram`). analyze() adds
    the second moment, its shot error `n2_err` and `shot_counts`. `from_h5file(load_shots=False)`.
  - `FockPopulationSetExperiment` (`AssembledExperiment`): jobs on one cycle grid and one detuning
    set. A quantum walk (read modes change) or one W of an AGP scan (states change).
- `experiments/qsim/agp.py`: `CentralBosonAGPExperiment` is now that set for the direction
  sector, with C_1, C_2 and shot errors; `CentralBosonAGPDisorderScan` is an `AssembledExperiment`
  of one part per W (`from_pattern(...).acquire(runner)` or `from_children(parts)`).
  Both have `from_h5file(path)`: the arrays of that layer only, for a quick look.
- `tools/convert_fock_population_catalog.py` with `configs/datasets/fock_population_datasets.yaml`
  (the catalog: jonginn's job lists and the stopgap's walk lists). Replaces jonginn's notebook,
  which is deleted with its test.
- jonginn's "coarse" and "fine" scans are named by their cycle grids:
  `agp_n3_s7_cycles0to795step5` (cycles 0-795 by 5, 300 reps) and
  `agp_n3_s7_cycles0to399step1` (0-399 by 1, 500 reps). The disorder grid is not the difference.
- Tests: `test_fock_population_set.py`, `test_convert_fock_population_catalog.py`,
  `test_fock_population_acquire_mock.py` (real Program, mock station); `test_agp_experiment.py`
  and `test_agp_disorder_scan.py` rewritten (jonginn's operator-trace check kept).

## Conversion (pippin, 2026-10-23 00:14)

207 raw files (12 calibration, 15 walk, 75 + 105 AGP); SHA-256 unchanged. Each job's raw
probabilities agree with `fitting/qsim/agp.py`'s independent classifier.

| Data set | Top manifest (under `C:\experiments\agp_k44_first_scan\assembled_data\`) | HDF5 |
|---|---|---|
| walk_n3_20261017 | `261023_001438_FockPopulationSetExperiment.yaml` | 0.26 MB |
| walk_n2_20261017 | `261023_001439_FockPopulationSetExperiment.yaml` | 0.14 MB |
| walk_n1_20261017 | `261023_001440_FockPopulationSetExperiment.yaml` | 0.14 MB |
| agp_n3_s7_cycles0to795step5 | `261023_001448_CentralBosonAGPDisorderScan.yaml` | 52 kB (parts 0.3 MB each) |
| agp_n3_s7_cycles0to399step1 | `261023_001509_CentralBosonAGPDisorderScan.yaml` | 150 kB (parts 0.76 MB each) |

Checks against the old outputs, before they were deleted:
- C_1, C_2 (raw and corrected), per-state corrected <n> and the state order agree with
  jonginn's 10-22 `coarse.h5` / `fine.h5` to 2e-15.
- The three walks and the W=30 fine part agree exactly with the 10-19 stopgap outputs.
  The stopgap's n_err includes the calibration uncertainty; it is ~1% (median) above the
  new shot-only n_err.

Deleted (guan's approval): `converted_data\20261022_192721_agp_legacy\` (3.0 GB),
`assembled_data\20261022_192721_agp_legacy\` (4.5 GB), the five stopgap outputs
`assembled_data\261019_*_FockPopulationDataset.{h5,png,yaml}`, and
`tools\backfill_fock_population.py`. `converted_data\` is now 1.56 GB (195 jobs + 3
calibrations, with shots), `assembled_data\` 0.01 GB. Raw `data\` unchanged (487 files).

## Device check

`measurement_notebooks/guan/agp_layers_hw_check.py`, run as a script from the `main` checkout,
through the queue while seb's overnight Wigner jobs ran. A logic check, not physics:

- Station on the s4s7 config versions (HW `CFG-HW-20261022-00017`, M1 `CFG-M1-20261020-00016`,
  FL `CFG-FL-20261017-00034`, MP `CFG-MP-20260121-00001`). `avoid_yoko=True`: the coupler stayed
  at seb's 0.30 mA (`CFG-HW-20261022-00045`), so the worker did not ramp it between his jobs and
  ours. The s4s7 settings are for 0.25 mA.
- Readout confusion as a `FockPopulationSetExperiment` of 4 jobs (`JOB-20261023-00011..00014`,
  600 reps). C is not diagonal: prepared 1 reads mostly as 3 (0.80), 3 as 0 (0.79), 2 as 1 or 3;
  condition number 31. The 10-17 calibrations had 0.75-0.90 on the diagonal. Cause not
  investigated; the 0.25 mA settings at 0.30 mA are the first suspect.
- A two-W scan (`from_pattern([0, 60], the 10-17 pattern)`, N=2, direction S7, cycles 0-90 by
  10, 100 reps): 10 jobs `JOB-20261023-00015..00027`, in batches of 3. seb's jobs 00018, 00022
  and 00026 ran between the batches. All 14 jobs completed.
- Each worker-saved job file has raw and corrected p_mod, both moments, n_err, n2_err,
  shot_counts, the matrix and its source (the calibration set's manifest). Sizes: job 172 kB
  (with shots), part 23 kB, scan 11 kB. `from_manifest` rebuilds the scan with the same arrays.
- With this C the corrected values are not physical (<n> up to 4 at N=2). Expected here.

Scan manifest: `C:\experiments\261023_agp_layers_check\assembled_data\261023_010214_CentralBosonAGPDisorderScan.yaml`.

## Open

- jonginn: `measurement_notebooks/jonginn/agp_measurement.py` still submits
  `MultiparityQsimExperiment` jobs one by one. The new path is a runner with
  `ExptClass=FockPopulationExperiment` and `CentralBosonAGPDisorderScan.from_pattern(...).acquire(runner)`
  (see `measurement_notebooks/guan/agp_layers_hw_check.py`).
- The class names are jonginn's (`CentralBosonAGPExperiment`, `CentralBosonAGPDisorderScan`);
  the other assembled classes end in `Experiment`. Not renamed.
- The calibration uncertainty is not propagated (shot-only errors), as in jonginn's version.
  `fitting/qsim/agp.py` still has the propagation, unused by the classes.
