# MBR redesign, step 7 plan: disorder / SFF

Status: approved by guan, 2026-09-24. jonginn reads it next (questions in section 7). It uses
`mbr_redesign.md` for the rules and patterns. Where this file is silent, that file applies.

## 0. Decisions (guan, 2026-09-24)

1. **Nothing is deleted in step 7.** guan and jonginn do the deletions together after step 7.
   Code that leaves the canonical path moves:
   - library code goes to `experiments/qsim/deprecated/`;
   - notebooks go to the `dormant/` folder next to them.
   It moves without changes, except for its import lines. Section 2 lists what goes where.
   Moved code is not maintained. If it breaks later, it stays broken, with a note in its header.
2. **Only diagonal time traces are canonical.** The off-diagonal disorder measurements
   ("pairwise" and "D72", section 1) are not ported and not converted. Their code moves to
   `deprecated/` and `dormant/`. Their raw data stays where it is. Reasons:
   - The Stark-shift calibration cannot calibrate an init ≠ final trace. It runs forward and
     backward Trotter steps, so for init ≠ final the return is about 0 and there is no phase
     winding to fit. D72 uses the *decoder* state's diagonal correction for each off-diagonal
     channel, and nothing validates that choice. Each channel gets its own frequency error.
     D72 merges levels from about 15 channels, so these errors become false gaps.
   - Off-diagonal elements add no new levels. Every level k is in the diagonals of a complete
     basis: Tr U(t) = Σ_k e^(−i E_k t). In the code, the only purpose of the off-diagonal
     channels is coverage when the allowed state set is cut down (`max_occupation`,
     `forbidden_states`). There is no test or comparison against a full diagonal basis.
3. **SFF (`DisorderSFFExperiment`) will be deleted.** It was an incomplete exploration: it has
   no data, it cannot run as written (section 1), and nobody reviewed it. It will be written
   again from the start later. In step 7 it moves to `deprecated/` (decision 1), marked
   "to be deleted".
4. **No `MBRChannelSetExperiment`.** DRAFT 1 proposed it only for D72. Without D72, no
   assembled class for off-diagonal traces is needed. `MBRTimeTraceExperiment` still accepts
   init ≠ final, as step 3 built it. Nothing canonical uses that.
5. **Shot sampling (`mbr_sampling`) is dormant code.** It is not ported. It moves to
   `deprecated/` and `dormant/`. It will be fixed or written again if it is needed.
6. **Arguments that functions ignore are not bugs to audit.** They came from notebook cells
   copied as whole blocks into named functions, with no data flow connected, because the
   code was to be reworked. The port connects the data flow (section 6).

## 1. Names and facts (from the survey)

The disorder code comes from section 7 of jonginn's `qsim_experiments.ipynb`
(cells 318-349), run about 2026-08-23 to 2026-09-11. It has three campaigns:

| Name in code | Notebook section | What it measures | Layout |
|---|---|---|---|
| "pairwise" (`disorder_*`) | 7 intro, cells 319-320 | preview: 10 (decoder, encoder) pairs chosen greedily, one realization | off-diagonal pair jobs |
| "7-1" / diagonal (`diagonal_disorder_*`) | 7-1, cells 323-334 | 10 diagonal occupations chosen by MILP per realization, dense time | diagonal traces |
| "D72" (`d72_*`) | 7-2, cells 338-348 | "occupation-constrained": about 15 (decoder, encoder) pairs chosen by a max-min visibility MILP, 0-80 µs | off-diagonal pair jobs (for *all* pairs, the diagonal ones too) |

"D72" means "disorder, section 7-2". It is not a physics name. The August realizations
(`august_disorder_r0..3`, `disorder_*` keys) are an older diagonal generation. The code that
made them no longer exists.

- **A realization** is one random zero-mean detuning vector over the modes
  (seed = master_seed + r). Only `detunings` and metadata change between realizations.
- **No existing disorder dataset is a complete basis.** The August and 7-1 sets have 10
  occupations each; D = 35 for N = 3. So their levels come from Matrix Pencil on each trace,
  not from Tr U.
- **Dataset lists that exist in code**:
  - D72: `mbr_disorder_h5.build_dataset_manifest` gives Sep01, Sep02, Sep02-04, Sep05, Sep07,
    Sep10, with timing overrides. The handoff note in `mbr_redesign.md` says these lists do not
    exist. That is wrong.
  - 7-1: `mbr_disorder_preview` (`diag_preview_job_ids`), realizations on `JOB-20260829/30`,
    calibration `JOB-20260828-00289..358`.
  - August: `tests/data/mbr_datasets.json`.
  - Pairwise: no list.
- **The saved off-diagonal batch** `JOB-20260823-00005..08` (analysis `mbr.py`) was recorded
  with `EncodingPropagatorProgram`. It is off-diagonal, so it goes with pairwise and D72.
- **SFF**: no job in `jobs.db` has an SFF class, and there is no file under `C:\experiments`.
  The depth-phase check `_prepare_pulse_with_phase_updated_by_depth` rejects negative phase
  steps, but the campaign's detunings are zero-mean, so almost every realization would stop
  with an error. It breaks rules 1 and 2 (the `sff_job_kind` flag; one job is
  realizations × occupations × depth × replica). jonginn's notes
  (`docs/archive/qsim/refactoring_floquet_dark_mode_readout.md`) say it was generated code
  that nobody read.

## 2. Where everything goes

"Port" = the code stays canonical and uses the new classes. "Move" = it goes to
`deprecated/` or `dormant/` without changes.

### Library code

| Now | Action | To |
|---|---|---|
| `notebook_helpers/mbr_disorder_campaign.py`: 7-1 functions (`DiagDisorderConfig`, `_diag_disorder_select_rows`, 7-1 plan/batch/analysis/plot) | port | stays; this file becomes 7-1 only |
| same file: pairwise and D72 functions (`build_pairwise_*`, `D72Config`, `_d72_*`, `build_d72_plans`, `preview_d72_jobs`, `check_d72_visibility`, `analyze_d72`) | move | `deprecated/mbr_disorder_offdiag.py` |
| `notebook_helpers/mbr_disorder_preview.py` (7-1 datasets) | port | stays |
| `notebook_helpers/mbr_disorder_h5.py` (D72 loader, `SavedSpectroscopyExperiment`, Sep05 decoder-clock correction) | move | `deprecated/mbr_disorder_h5.py` |
| `notebook_helpers/mbr_spectral_validation.py` | move, except one function | `deprecated/mbr_spectral_validation.py` |
| its `matrix_pencil_global_diagnostic` (uses only `A` and `time_us`) | move | `notebook_helpers/mbr_n3_reprocess.py` (a plotting diagnostic on N=3 data, like its neighbours there) |
| `notebook_helpers/mbr_sampling.py` (shot sampling, decision 5) | move | `deprecated/` |
| `notebook_helpers/mbr_saved_reanalysis.py`: `load_disorder_*` (August) | port | stays |
| `notebook_helpers/mbr_n3_reprocess.py`: `reprocess_n3_spectroscopy` | move | `deprecated/mbr_n3_reprocess_legacy.py` (has a new-class twin in analysis `mbr.py`) |
| same file: `fit_self_kerr_from_peak_overlap` (no callers) | move | `deprecated/mbr_n3_reprocess_legacy.py` |
| `notebook_helpers/mbr_loading.py` | move | `deprecated/` |
| `notebook_helpers/mbr_campaign.py`: `ensure_calibration`, `acquire_calibration`, `EncSpec` fields | port | use `MBRCalibrationSetExperiment` |
| `experiments/qsim/mbr_sff.py`, `notebook_helpers/mbr_sff_campaign.py` | move, "to be deleted" | `deprecated/` |
| `EncodingHamiltonianSpectroscopyExperiment` (`floquet_dark_mode_readout.py`) | move | `deprecated/` |
| `EncodingPropagatorProgram`, `EntireFloquetCyclePhaseCalibrationProgram`, `NPhotonHamiltonianSpectroscopyProgram` | move | `deprecated/` |
| `legacy_mbr.py` | stays in `deprecated/` | |
| SFF `_MOVED_TO` entries | remove (7a) | importers use the `deprecated/` path directly |
| MBR `_MOVED_TO` entries | remove (7e) | importers use the `deprecated/` path directly (keep `SidebandScrambleDarkProgramNewNew`) |

`floquet_phase_calibration.py` and `deprecated/single_photon_spectroscopy.py` subclass
`NPhotonHamiltonianSpectroscopyProgram`. Change their base name to `MBRRamseyProgram` (the old
class is an empty subclass of it), so that live code does not import `deprecated/`.

### Notebooks

| Now | Action | To |
|---|---|---|
| measurement `mbr_disorder.py`: section 7-1 | port | stays |
| same: intro (pairwise) and 7-2 (D72) | move | `measurement_notebooks/202609_qsim_migration/dormant/mbr_disorder_offdiag.py` |
| measurement `mbr_sff.py` | move | `measurement_notebooks/.../dormant/` |
| analysis `mbr_disorder.py`: section 1 (7-1 preview) | port | stays |
| same: section 2 (D72 HDF5 reprocessing) | move | `analysis_notebooks/.../dormant/mbr_disorder_offdiag.py` |
| analysis `mbr_spectral_validation.py` | move | `analysis_notebooks/.../dormant/` |
| its cell 188 (global Matrix Pencil on july_N3) | port | analysis `mbr.py` |
| analysis `mbr.py`: off-diagonal batch cell (`JOB-20260823-00005..08`) | move | `analysis_notebooks/.../dormant/mbr_disorder_offdiag.py` |
| analysis `mbr_sampling.py` | move | `analysis_notebooks/.../dormant/` |

### Rules for moved code
- A header says: moved on <date>, why (section 0), and "not maintained; may break when live
  code changes". If it breaks later, add a note to the header. Do not fix it.
- Live code must not import `experiments.qsim.deprecated` at the end of step 7. A new test
  checks this (7e).
- Tests of moved code move out of the blocking suite. The SFF mock tests
  (`tests/test_mbr_acquire_mock.py:161-243`) and the sampling tests (if any) go to one file,
  `tests/test_deprecated_mbr.py`. When one breaks, it gets
  `pytest.mark.skip(reason="moved to deprecated/, see docs/qsim/mbr_step7_plan.md")`.
- Moved notebooks are taken out of `tools/run_qsim_suite.py`.
- The old `dormant/`, `guan/` and `jonginn/` notebooks that import old classes are not changed.
  If a later deletion breaks them, they stay broken, with a header note.

## 3. Design

### 3.1 `MBRDisorderEnsembleExperiment` (new, assembled, `from_parts`)
- Parts: one `MBRSpectrumExperiment` manifest per realization, through `_child_files`.
  Spectrum already takes `detunings` and a subset of occupations.
- The manifest also holds the shared CalibrationSet manifest and one metadata record per
  realization: `seed`, `direction`, `strength_kHz`, `onsite_MHz`, `self_kerr_kHz`,
  `selected_occupations`. The August and 7-1 key generations map to this one record.
- `analyze()`:
  1. per part: Matrix Pencil, then theory, then level matching;
  2. then pooled gap ratios.
- Arrays: `levels` (R, D), `theory_levels` (R, D), `onsite_MHz` (R, M), and gap ratios
  (padded with NaN).
- 7-1 planning (the MILP selection) becomes a plain function that returns the constructor
  arguments of the parts. The acquisition itself does not change.
- The pure numerics of the diagonal path (gap ratios, Poisson/GOE, level matching,
  `wrap_frequency`) move without changes to a new `fitting/qsim/mbr_disorder.py`.
- Theory: call `fitting.qsim.mbr_spectrum.analyze_spectrum` directly. Do not use
  `EncSpec.analyze_spectrum` with a fake probe AttrDict.

### 3.2 Tr U / SFF / DOS
- When every part covers the complete basis, the ensemble also gives
  Tr U(t) = Σ_a A_aa(t), the SFF |Tr U|² averaged over realizations, and the DOS (FFT).
- For a part that is not a complete basis, these outputs are not made (no error; a note in
  `display`). No existing dataset is complete (section 1), so this is for new acquisitions.

## 4. Migration (`tools/migrate_mbr_jobs.py`)

- No new kind. The August and 7-1 realizations use kind `spectrum` (one Spectrum per
  realization), then one ensemble `save()`.
- The realization metadata keys go into `derived_params`.
- Move the 7-1 lists (from `mbr_disorder_preview`) into `tests/data/mbr_datasets.json`.
- Also move the D72 lists (from `build_dataset_manifest`) there, marked
  `"converted": false` with the reason (section 0.2). This keeps the raw data easy to find.
- `migrate_spectrum` keeps refusing off-diagonal pair jobs.

## 5. Order (each sub-step ends green, as in `mbr_redesign.md` section 7)

- **7a. Move.** Move the whole modules, functions and notebook sections of section 2 that have
  no live user, as one commit. Change imports only, and move the tests of moved code. The suite
  must be green after this step, before any port.
  The old classes and programs (`EncodingHamiltonianSpectroscopyExperiment`, the three old
  programs, the `_MOVED_TO` entries) still have live users until 7b-7d port them. They move in
  7e, when their live users are gone.
- **7b. Base work.**
  - `fitting/qsim/mbr_disorder.py`.
  - Replace the `EncSpec` theory and hardware calls in the live helpers.
  - `mbr_campaign.py` calibration on `MBRCalibrationSetExperiment`.
- **7c. Diagonal disorder.**
  - `MBRDisorderEnsembleExperiment`.
  - Convert August r0..3 and the 7-1 set.
  - Port the 7-1 part of `mbr_disorder_campaign`, `mbr_disorder_preview`, `load_disorder_*`,
    and section 1 of the analysis `mbr_disorder.py`.
  - Gate: the ensemble smoke test on converted fixtures.
  - Baseline (xfail): old vs new level statistics on the same raw files.
- **7d. Measurement notebook**, section 7-1:
  - It acquires one Spectrum per realization with `runner.execute(overrides=...)`, saves each
    one, and builds the ensemble with `from_parts`.
  - Gate: `tools/dryrun_qsim_notebook.py` on a mock station.
- **7e. Close.**
  - Move the old classes and programs (see 7a) to `deprecated/`, remove the MBR
    `_MOVED_TO` entries, and change the base name in `floquet_phase_calibration.py`.
  - Add the test "live code does not import `experiments.qsim.deprecated`" (tests are not
    live code).
  - Fix the handoff note in `mbr_redesign.md`.
  - Record there what is left in `deprecated/` and why.
  - `legacy_mbr.py` and the old programs stay in `deprecated/`. guan and jonginn delete them
    with the rest of the moved code, after step 7.

## 6. Connecting the data flow (ported code only)

Many helper functions are notebook cells copied as whole blocks, with no data flow connected
(decision 6). The port connects it:
- `globals()` reads in helper modules read the module's globals, not the notebook's. So
  `best_self_kerr_kHz` is never used, and the caches are always empty. They become arguments
  or return values.
- Hard-coded values overwrite arguments, for example: `load_preview_realization` always loads
  realization 12, and `analyze_every_realization` resets `edge_fraction` and `bins`. The
  argument wins; the hard-coded value becomes the default.
- Unused config unpacking (about 30 lines per function) goes.
- Old-vs-new comparisons use the values that the old blocks actually used. So the comparisons
  check the port and not a change of settings.
- Moved code keeps these problems. For example, six `mbr_spectral_validation` functions return
  None. Its header lists them.
- The known failures of step 6b stay tagged. Their causes are dataset or physics questions.

## 7. Questions for jonginn

1. Do you agree that D72 and pairwise leave the canonical path (section 0.2)? Is there a
   reason for the off-diagonal channels that the code does not show?
2. Is there a planned way to calibrate the phase of an init ≠ final trace? If yes, D72 can come
   back after that, with its own assembled class.
3. Are the 7-1 lists in `mbr_disorder_preview` correct and complete, and do you want them
   converted?
4. Do the `mbr_spectral_validation` methods also make sense on diagonal data (7-1, August)?
   If yes, they can be ported later on the ensemble.

## 8. Progress

- **7a done** (2026-09-24, `e9fdf85`). Moves only; see its commit message.
- **7b-7d done together** (2026-09-24). They depended on each other: the new calibration,
  the theory calls and the new acquisition all touch the same 7-1 functions.
  - New: `fitting/qsim/mbr_hamiltonian.py` (`fixed_n_hamiltonian`, taken out of
    `analyze_spectrum`; golden baselines unchanged), `fitting/qsim/mbr_disorder.py`
    (direction, row selection, cycle grid, gap ratios, level matching, pooled statistics,
    Tr U / SFF), `experiments/qsim/mbr_disorder_ensemble.py`
    (`MBRDisorderEnsembleExperiment`), migration kind `disorder` in
    `tools/migrate_mbr_jobs.py`, `tests/test_mbr_disorder_ensemble.py`.
  - Rewritten on the new classes: `notebook_helpers/mbr_disorder_campaign.py`
    (`plan_diagonal_disorder`, `realization_spectrum`, `analyze_diagonal_disorder`),
    `notebook_helpers/mbr_campaign.py` (no calibration fields; `build_campaign` takes no
    station or client), `load_disorder_*` in `notebook_helpers/mbr_saved_reanalysis.py`,
    the measurement and analysis `mbr_disorder.py`, section 5 of the analysis `mbr.py`.
  - Moved to `deprecated/` without changes: the old 7-1 functions
    (`mbr_disorder_campaign.py`), `mbr_disorder_preview.py`, the old campaign base
    (`mbr_campaign_legacy.py`), the old disorder loaders (`mbr_saved_reanalysis_legacy.py`).
  - Converted into the prod data tree (raw files unchanged):
    - August r0..r3: `C:\experiments\260526_qsim_darkmode\assembled_data\260924_195547_MBRDisorderEnsembleExperiment.yaml`,
      linked to the August N=3 calibration set of step 6b.
    - 7-1 r0..r18: `C:\experiments\260818_qsim_spectroscopy\assembled_data\260924_195637_MBRDisorderEnsembleExperiment.yaml`,
      with a new calibration set converted from `JOB-20260828-00289..358`. r19 is not
      converted: it has 11 of 20 jobs. The 7-1 jobs' provenance was exported to
      `tests/data/job_provenance.json` (read-only, 461 records added, none changed).
  - Dataset lists: `diagonal_disorder_71` and the six D72 lists (`d72_*`,
    `"converted": false`, with their archived timing) are in `tests/data/mbr_datasets.json`.
  - Checks:
    - The ensemble on converted 7-1 data gives the same poles, theory levels and pooled gap
      ratios as the old preview cells on the raw files (baseline, XPASS).
    - The planner picks the same occupations as the old cell-325 code on the same inputs
      (19 of 19). **For the physics audit:** the occupations recorded in the 7-1 jobs differ
      from a re-plan with the converted calibration's hardware in 14 of 19 realizations
      (onsite values and cycle grid are the same). So the campaign's planning inputs at
      acquisition time were different from what the calibration jobs saved.
    - Analysis suite: `mbr` ok (known xfail cells 30, 31, 48; cell 48 is the disorder theory
      check, still 0.3 kHz off), `mbr_disorder` ok (it failed before: no timing).
    - Mock dry run of the measurement `mbr_disorder.py` finishes; the Matrix Pencil cells
      are skipped on mock data (`MPM_MARKERS` in `tools/dryrun_qsim_notebook.py`).
    - pytest: 1058 passed, 1 skipped, 4 xfailed, 9 xpassed.
  - Behaviour changes, on purpose:
    - The pulse correction of a new 7-1 acquisition uses the calibration set's Kerr, not the
      campaign's self-Kerr. The manual-Kerr analysis undoes the played correction first, so
      the analyzed data do not depend on it.
    - Analysis `mbr.py` 5b: the source range `JOB-20260816-00011..80` cannot load (jobs
      73-80 do not exist, 11-12 are not spectroscopy jobs), so 5b now uses the 5a partition
      (the August ensemble). Whether the source meant another partition is for jonginn.
    - The notebooks now pass the values that the old cells hard-coded over their arguments
      (preview realization 12, excluded occupation (0, 3, 0, 0, 0), level plot r=14 at 0.5
      bins), so they print what the source printed.
- **7e next:** move the old classes and programs, remove the MBR `_MOVED_TO` entries, change
  the base name in `floquet_phase_calibration.py`, add the "no live import of
  `deprecated/`" test, fix the handoff note in `mbr_redesign.md`.
