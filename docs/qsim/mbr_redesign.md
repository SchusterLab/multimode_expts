# MBR class redesign: target structure

Status: current spec, 2026-09-24. It supersedes `docs/arch_meeting_log.md`
(history and reasons only) and the MBR rows of `docs/qsim_refactor_surface_map.md`.
Where this file is silent, the implementer decides. Do not ask about ground-level choices.

## 1. Rules

1. **Every class has exactly one kind.**
   - A **job class** is sent to the worker. One job = one HDF5 = one instance.
     It never holds other Experiments.
   - An **assembled class** never goes to the worker. It holds child instances
     and nothing else from the hardware.
   - No class is both. No class uses keyword flags to switch between behaviors.
2. **One job is a 1D sweep**, over time *or* over final Fock state. All four
   phase combinations (prep 0/180 × analyzer 0/90) are inside the same job.
   After processing, the job holds one complex 1D array.
3. **No time chunks.** One time trace = one job.
4. **Dependencies point down.** An assembled class knows its child class and uses a
   runner. A job class knows nothing above it. The runner knows neither.
5. **No shims, no compatibility aliases.** Change the callers too.
6. **This stage is code organization, not physics.** Gates check that code loads and
   runs, not that the numbers are correct. The physics audit is a later, separate stage.
   Numerical baseline tests stay in the suite as `xfail(strict=False)` with the reason
   "physics audit pending". Do not delete them. They become blocking again in the audit.

## 2. Classes

| Class | Kind | Holds | Replaces (old code) |
|---|---|---|---|
| `MBRStarkCalExperiment` | job | time, one occupation, closed cycles | per-job part of `MBRPhaseCorrectionExperiment`; `EntireFloquetCyclePhaseCalibrationProgram` |
| `MBRCalibrationSetExperiment` | assembled | StarkCals over occupations; gives `phase_for(occupation)` | aggregate part of `MBRPhaseCorrectionExperiment` |
| `MBRTimeTraceExperiment` | job | time, one (init, final) pair; init ≠ final allowed | diagonal and off-diagonal branches of `spectroscopy_batch`; `NPhotonHamiltonianSpectroscopyProgram` |
| `MBRSpectrumExperiment` | assembled | diagonal TimeTraces over occupations | aggregate part of `MBRSpectrumExperiment` |
| `MBROrthoColumnExperiment` | job | final Fock, one init, at cycle count `q` (default 0) | per-job part of `MBROrthogonalityExperiment`; `EncodingOrthogonalityProgram`, `EncodingPropagatorProgram` |
| `MBROrthogonalityExperiment` | assembled | OrthoColumns over init, all at the same `q` (= the matrix M_q) | aggregate part of `MBROrthogonalityExperiment` |
| `MBRHamTomoExperiment` | assembled, `from_parts` | Orthogonality objects at q = 0, q, 2q, … | `MBRPropagatorExperiment`; its analysis moves without changes |
| `MBRDisorderEnsembleExperiment` | assembled, `from_parts` | Spectrum objects over disorder realizations | `DisorderSFFExperiment`, `notebook_helpers/mbr_disorder_*` (later phase) |

Notes:
- The long-time off-diagonal TimeTraces (init ≠ final) have no assembled class yet.
  TimeTrace must support them. Their assembly is out of scope.
- The old `MBRPhaseCorrectionExperiment`, `MBRSpectrumExperiment`,
  `MBROrthogonalityExperiment` and `MBRPropagatorExperiment` move without changes to
  `experiments/qsim/legacy_mbr.py`, so that the new classes can use the names. Code not
  yet ported imports them from there. This is a move, not a shim.
- `EncodingHamiltonianSpectroscopyExperiment` has no new counterpart. Move its shared
  loading code (`from_job_files`, `_saved_parameters`, `_quadrature`) to a plain helper
  module. Delete the class when nothing imports it.

## 3. Programs

- One Program per job class, in the same module as that job class.
- Names use the `MBR` prefix and describe the whole Ramsey sequence:
  `MBRStarkCalProgram`, `MBRTimeTraceProgram`, `MBROrthoColumnProgram`.
  Do not use "Encoding" or "Spectroscopy" in new names.
- Shared pulse code (today `NPhotonHamiltonianSpectroscopyProgram` and
  `SidebandScrambleDarkProgramNewNew`) becomes one base, for example `MBRRamseyProgram`.
- Remove the `_MOVED_TO` re-export entries in `floquet_dark_mode_readout.py` for the
  programs that are replaced.
- Out of scope: the Floquet calibration programs in `floquet_phase_calibration.py`,
  and all programs outside MBR.

## 4. Runner

- `CharacterizationRunner.execute(configs=None, batch_size=..., **kwargs)`.
  No `configs`: one job, as today. With `configs` (a list of override dicts):
  one job per dict, at most `batch_size` in the queue, returns a list of job instances.
- It supports queue and direct dispatch, real and mock, through the existing
  `run`/`run_local` code. Mock station + queue raises unless explicitly allowed.
- Delete `experiments/batch_runner.py`, `BatchRunner._aggregate` and `from_batch`.
  Move all their callers first, MBR and later-phase code alike (step 1).
- MBR does not use `SweepRunner`. Leave `SweepRunner` unchanged.

## 5. Assembled class interface

```python
spectrum = MBRSpectrumExperiment(occupations=..., cycles=..., calibration=cal)
spectrum.acquire(runner, batch_size=10)   # build configs -> runner.execute -> assemble
spectrum.save()                           # manifest YAML + assembled HDF5
spectrum.analyze(); spectrum.display()
spectrum = MBRSpectrumExperiment.from_manifest(path)   # re-assemble from raw job files
```

- `acquire(runner)` when one `execute` call does all the work (CalibrationSet,
  Spectrum, Orthogonality).
- `from_parts(...)` when acquisition takes hours or days with human decisions between
  parts (HamTomo, DisorderEnsemble). The notebook loops over the lower level.
- Not a slab `Experiment` subclass. Same method names: `acquire`, `analyze`,
  `display`, `save`, `from_manifest`.
- `save()` writes to `C:\experiments\<experiment_name>\assembled_data\`:
  - the manifest YAML: class, job IDs, raw HDF5 paths, calibration manifest, notes;
  - the assembled HDF5: assembled arrays (complex where possible), with the manifest
    path and code version.
- The manifest is the source of truth. The assembled HDF5 is a copy that can be rebuilt.
- Each job that uses a phase correction records the CalibrationSet manifest path in
  `cfg.expt`, beside the correction value.

## 6. Old data

- A migration script in `tools/` converts old job HDF5s to the new job layout and
  writes them to `C:\experiments\<experiment_name>\converted_data\`. Raw files are never changed.
- It handles: φ=0 and φ=90 in two files → one file; time chunks → one trace;
  old propagator jobs (one encoder × cycles × decoders) → one OrthoColumn per cycle.
- It works per dataset, not per file. Input: a hand-written list of the old job IDs
  of one dataset. If no list exists for a dataset, ask the user; they build it from the
  lab's OneNote logs. Do not guess the grouping. Output: the converted job files, plus
  the assembled output, written by the new class's own `save()`.
  Example: one old Spectrum = 70 files (35 occupations × φ=0/90) → 35 + 1 + 1:
  35 TimeTrace files in `converted_data\`, and one manifest YAML + one assembled HDF5
  in `assembled_data\`.
- Each converted file records the old job IDs and file paths it came from.
- Each converted file records the new job class name (for example `MBRTimeTraceExperiment`).
  Old files name `MBRSpectrumExperiment` etc., which is now an assembled class.
- New classes read only the new layout. Old data reaches them only through the script.

## 7. Order of work

Each step ends with the tests green (rule 6: numerical baselines do not block).

The gate for every step is a smoke test on converted fixture data:
- assembled classes: construct, `from_manifest`, `analyze`, `display` and `save`
  finish without error, and the outputs have the expected shape and dtype;
- job classes: the Program builds and compiles in mock mode
  (as in `tests/test_mbr_acquire_mock.py`).

The named baseline tests run beside the gate as `xfail(strict=False)`. If a baseline
changes during a step, write it in that step's commit message, so the physics audit
can find where it changed.

Each step also rewrites or deletes the tests of the code it removes (for example
`test_batch_runner.py` and `test_from_batch.py` in step 1; `test_mbr_stage_split.py`
and `test_no_stage_dispatch_remains.py` when the old stage classes go).

1. Move the four old MBR classes to `legacy_mbr.py` (section 2) and point all importers
   there. Then the runner merge (section 4). Move every `BatchRunner` and `from_batch`
   caller to `CharacterizationRunner.execute(configs=...)`, including the later-phase
   code (disorder, SFF, tomography helpers and notebooks). For later-phase code, change
   only the runner calls; its classes stay on `legacy_mbr.py`. This includes
   `experiments/qsim/mbr_campaign.py` (`run_stage`) and `notebook_helpers/mbr_campaign.py`.
2. StarkCal + CalibrationSet, with the migration script for their jobs.
   Baseline: `test_stark_cal_baseline_matches`.
3. TimeTrace + Spectrum, with the migration script for their jobs.
   Baseline: the complete-basis and quick-plot baselines in
   `tests/test_mbr_analysis_golden.py`.
4. OrthoColumn + Orthogonality, with the migration script for their jobs.
5. HamTomo. Baseline: `tests/test_propagator_dynamics.py`, ported to
   `MBRHamTomoExperiment` on its synthetic fixture.
6. Update `measurement_notebooks/202609_qsim_migration/mbr.py`,
   `analysis_notebooks/202609_qsim_migration/mbr.py`, `experiments/qsim/mbr_campaign.py`
   (`STAGES`) and `notebook_helpers/mbr_campaign.py` to the new classes.
   Delete the old MBR classes and programs that nothing imports. Delete
   `legacy_mbr.py` when nothing imports it.

Later phase, not now: DisorderEnsemble, and the disorder / SFF / sampling /
spectral-validation notebooks and their `notebook_helpers`. Until they are ported,
they keep the old classes they import (from `legacy_mbr.py`). Do not delete code they
still need. Exception: their runner calls move in step 1.

## 8. Progress

- Step 1 done (2026-09-24). `legacy_mbr.py` is at `experiments/qsim/deprecated/`,
  so the `experiments` namespace does not export it (`tests/test_meas_namespace.py`).
  Mock policy of `execute()`: a mock station runs locally unless the call passes
  `use_queue=True`; that raises unless `allow_queue_in_mock=True`.
- Step 2 done (2026-09-24). `experiments/qsim/mbr_stark_cal.py` (job + Program),
  `experiments/qsim/mbr_calibration_set.py` (assembled), `experiments/assembled_data.py`
  (manifest + assembled HDF5), `experiments/qsim/mbr_saved.py` (saved hardware
  parameters), `tools/migrate_mbr_jobs.py` (kind `stark_cal`). The old
  `EntireFloquetCyclePhaseCalibrationProgram` is now a subclass of `MBRStarkCalProgram`.
  `phase_unwrap_mode: odd_guide` has no new equivalent. `test_stark_cal_baseline_matches`
  runs on converted data and still matches.
