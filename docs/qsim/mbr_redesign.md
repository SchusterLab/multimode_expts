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

- `CharacterizationRunner.execute(overrides=None, batch_size=..., **kwargs)`.
  No `overrides`: one job, as today. With `overrides` (a list of override dicts,
  each going to the preprocessor like the kwargs of a single call, not a `cfg.expt`):
  one job per dict, at most `batch_size` in the queue, returns a list of job instances.
- It supports queue and direct dispatch, real and mock, through the existing
  `run`/`run_local` code. Mock station + queue raises unless explicitly allowed.
- Delete `experiments/batch_runner.py`, `BatchRunner._aggregate` and `from_batch`.
  Move all their callers first, MBR and later-phase code alike (step 1).
- MBR does not use `SweepRunner`. Leave `SweepRunner` unchanged.

## 5. Assembled class interface

```python
spectrum = MBRSpectrumExperiment(occupations=..., cycles=..., calibration=cal)
spectrum.acquire(runner, batch_size=10)   # build overrides -> runner.execute -> assemble
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
   caller to `CharacterizationRunner.execute(overrides=...)`, including the later-phase
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

7. Disorder / SFF (the "later phase" below). **Status: needs a concrete plan first.**
   Do not start porting before the user approves a written plan. It must settle at least:
   the `MBRDisorderEnsembleExperiment` design; an assembled class (or none) for
   off-diagonal TimeTraces, which the pairwise and D72 plans use; conversion of the old
   off-diagonal pair jobs (`offdiag_cycles`); what happens to `DisorderSFFExperiment`,
   which is its own hardware depth-sweep program and not a set of Ramsey traces; and the
   dataset ID lists (the user builds them). Scale: about 11k lines of
   `notebook_helpers/mbr_disorder*`, `mbr_sampling`, `mbr_spectral_validation`,
   `mbr_sff_campaign` plus `experiments/qsim/mbr_sff.py`. After step 7, delete
   `legacy_mbr.py`, `EncodingHamiltonianSpectroscopyExperiment` and the old programs.

Later phase (step 7): DisorderEnsemble, and the disorder / SFF / sampling /
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
- Step 3 done (2026-09-24). `experiments/qsim/mbr_ramsey.py` holds `MBRRamseyProgram`,
  the shared base (the old `NPhotonHamiltonianSpectroscopyProgram` body, which is now an
  empty subclass kept for the old jobs). `SidebandScrambleDarkProgramNewNew` is *not*
  merged into it: it is a standalone dark-mode program used by the Floquet calibration
  notebooks, so it stays as the base's parent. `experiments/qsim/mbr_time_trace.py`
  (job + Program; init != final allowed), `experiments/qsim/mbr_spectrum.py` (assembled),
  `AssembledExperiment` base in `experiments/assembled_data.py`, migration kind `spectrum`
  (joins time chunks; old off-diagonal pair jobs are refused until the disorder phase).
  The quick-plot and complete-basis baselines run on converted data and still match.
  `MBRRamseyProgram` inherits `SidebandScrambleProgram, DarkBaseProgram` directly; it does
  not use `SidebandScrambleDarkProgramNewNew` (that class adds only a dark-mode
  `core_pulses`, which MBR never plays). Untangling the big qsim mixins is out of scope here.
- Step 4 done (2026-09-24). `experiments/qsim/mbr_ortho_column.py` (job + Program; outer
  sweep `decoder_occupation`, per-decoder pulse correction `decoder_phase_per_cycle_deg`),
  `MBROrthogonalityExperiment` in `experiments/qsim/mbr_orthogonality.py` (assembled; square
  matrix, columns in the jobs' decoder order), migration kind `orthogonality`. Converted old
  propagator jobs that left their correction to analysis carry it as
  `analysis_phase_per_cycle_deg`; the assembled `matrix` applies it, `raw_matrix` does not.
  Fixture: `september_N3_orthogonality` in `tests/data/mbr_datasets.json` (35 jobs, given by
  the user). Gate: `tests/test_mbr_orthogonality.py`; the new matrix equals the old
  `reconstruct_orthogonality` (xfail baseline, XPASS).
- Step 5 done (2026-09-24). `experiments/qsim/mbr_ham_tomo.py` (`MBRHamTomoExperiment`,
  `from_parts`; its manifest lists the parts' manifests, through the new
  `AssembledExperiment._child_files` hook). The tomography is a plain function,
  `fitting/qsim/mbr_propagator.analyze_propagator_dynamics`, run only when a calibration
  set is given (it needs each occupation's q = 0 self-return). Migration kind `propagator`
  (old job -> one OrthoColumn per cycle -> one Orthogonality per cycle -> HamTomo).
  Fixture: `august_N1_propagator` (5 jobs, q = 0, 20; the user is not sure it is the set in
  the logged plots, so it gates code only). `tests/test_propagator_dynamics.py` now runs on
  synthetic OrthoColumn and StarkCal jobs through `MBRHamTomoExperiment`; it is exact on its
  synthetic data, so it stays blocking. Also fixed: `save()` no longer overwrites a set
  saved in the same second (`new_stem` adds `_2`, `_3`, ...).

- Step 6, measurement side (2026-09-24). `measurement_notebooks/202609_qsim_migration/mbr.py`
  and `mbr_tomography.py` (+ `notebook_helpers/mbr_tomography.py`) use the new classes.
  The stage/batch driver is deleted: `STAGES`, `_*_batch`, `build_stage`, `run_stage` in
  `experiments/qsim/mbr_campaign.py`, which keeps only config sets, mock stations,
  `mbr_defaults`, and `smoke()` (every product acquired in mock mode). Its tests
  (`asm_golden`, `test_mbr_acquire_mock`, `test_floquet_cycle_duration`, the MBR part of
  `test_qsim_notebook_mock_acquisition`) run on the new classes; the 16 ASM goldens were
  checked byte-identical to the old ones before the rename. `notebook_helpers/mbr_campaign.py`
  keeps `build_campaign`, `fixed_n_occupations`, the new `campaign_runner`, and the old-class
  `ensure_calibration` / `acquire_calibration` for the disorder/SFF notebooks.
  `MBRCalibrationSetExperiment.with_replacements` replaces `merge_replacement_calibration`.
  Deleted: `EncodingOrthogonalityProgram`. Kept, still used by step-7 code:
  `EncodingPropagatorProgram`, `EntireFloquetCyclePhaseCalibrationProgram`,
  `NPhotonHamiltonianSpectroscopyProgram`, `legacy_mbr.py`.
  Left on purpose (user decision): the `guan/` and `jonginn/` sandboxes and `dormant/`
  notebooks; they go when the new canonical notebooks replace them.

- Step 6b, analysis side, the ready half (2026-09-24). Six old datasets converted into the
  prod data tree with `tools/migrate_mbr_jobs.py` (`converted_data/` + `assembled_data/`):
  July N=1, N=2, N=2 supplement, N=3 and August N=3 under `260526_qsim_darkmode`, the
  August quickplot under `260814_qsim_encspec`. `analysis_notebooks/202609_qsim_migration/mbr.py`
  loads them by manifest (`data_root() / ...`); `mbr_replot` and the N=3 parts of
  `mbr_n3_reprocess` / `mbr_saved_reanalysis` use the new classes. Old and new paths were
  compared on the same raw files: spectra, returns and Matrix Pencil frequencies are
  identical. Still old-class, step 7: the disorder realizations (`load_disorder_*`), the
  saved off-diagonal batch, `reprocess_n3_spectroscopy` and
  `fit_self_kerr_from_peak_overlap` (shared with step-7 notebooks; the new
  `fit_self_kerr` serves the new class). The notebook runs in the analysis suite with four
  tagged known failures, all present before the port: cells reading `encspec_N3_*` (the
  source cell that built them is lost), the disorder theory check (rebuilt vs saved theory
  differ by up to 0.3 kHz), and the 5b disorder range (jobs 20260816-73..80 do not exist).
  The report's "N=4" sector was a second N=1 set (config `CFG-FL-20260717-00029`); dropped
  for now, and `analyze_sector` checks the photon number again.

### Handoff for steps 4-6

Patterns set in steps 2-3; follow them:

- **Job class**: subclass `DarkBaseExperiment`; default its Program in `__init__`
  (`program or MBRxxxProgram`); a `job_config(...)` staticmethod returns runner overrides;
  inner sweep `ramsey_phase` over `RAMSEY_PHASES` (`mbr_stark_cal.py`), outer sweep the 1D
  axis; `analyze()` builds `complex_return = Q_0 - i Q_90` and never raises except for
  `Ig == Ie` (a raise loses the job before save). Program subclasses `MBRRamseyProgram` and
  sets `spectroscopy_prep_phase`/`spectroscopy_analyzer_phase` from `ramsey_phase` in
  `initialize()`. See `mbr_time_trace.py`.
- **Assembled class**: subclass `AssembledExperiment` (`experiments/assembled_data.py`);
  implement `job_overrides`, `from_children`, `analyze`, `display`, `manifest_parameters`,
  `assembled_arrays`, `assembled_attrs`; `_from_manifest_kwargs` for linked sets. `acquire`
  checks `runner.ExptClass is child_class`. See `mbr_spectrum.py`.
- **Migration**: `tools/migrate_mbr_jobs.py`; reuse `merge_phase_jobs` and
  `write_converted`; converted files carry `converted_from`, `job_class`, `derived_params`.
- **Tests**: a gate file per step (mock acquire on both pinned config sets, plus
  `from_manifest`/`analyze`/`display`/`save` on data converted into `tmp_path`); ported
  baselines convert in `tmp_path` and are `xfail(strict=False, "physics audit pending")`;
  an ASM check that the new Program compiles to the old one's pulses.

Step 4 notes: the old programs are `EncodingOrthogonalityProgram` (`mbr_orthogonality.py`)
and `EncodingPropagatorProgram` (`mbr_propagator.py`); both sweep `cycle_decoder_analyzer`
and decode `spectroscopy_final_occupations`. The propagator program applies the decoder
correction on the pulse only when `phase_correction_location == "pulse"`; the old
orthogonality/propagator analysis is in `deprecated/legacy_mbr.py`
(`reconstruct_orthogonality`, `reconstruct_propagator`, `propagator_batch`).

Open items, not yet done:

- `EncodingHamiltonianSpectroscopyExperiment` still owns `from_job_files`, `_quadrature`
  and `_from_expts`; move what the later-phase code needs to a helper before deleting it.
- `_MOVED_TO` in `floquet_dark_mode_readout.py` still lists the old MBR programs; remove
  each entry when its program is deleted (step 6).
- Old off-diagonal pair jobs (`offdiag_cycles`) have no conversion (disorder phase).
- `measurement_notebooks/guan/mbramsey.py` uses `floquet_dark_mode_readout` without
  importing it (older than the redesign).
