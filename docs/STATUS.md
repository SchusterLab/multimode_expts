# Where things stand

**Last updated: 2026-09-25** (after MBR redesign step 8A, 8B, 8D). This file is overwritten at
the end of each work session; git keeps the old versions. What happened, and why, is in
`docs/log/` (newest first: `2026-09-25_mbr-step8.md`). Read this file first; read the log only
if you need the history.

## What is next

- **8C, the analysis notebook** (`docs/qsim/mbr_step8_plan.md`, question 3, guan + jonginn): a
  short canonical path (load manifest -> `analyze` -> `display`); the N=2 reprocessing, the
  FFT / peak-finder / MPM comparisons, the report replots and the Aug 15-17 reproduction move out.
- **`notebook_helpers` cleanup:** promote the MBR planners (`plan_diagonal_disorder`, the
  tomography plan, `fit_self_kerr`) into the classes and `fitting/qsim/`; merge the two copies of
  the MBR defaults (`build_campaign` vs `mbr_defaults`); move `run_mode` out.
- **jonginn:** `docs/qsim/mbr_step7_plan.md` section 7 (still open); and step 8's removal of the
  `'decoder'` mode and `floquet_phase_calibration.py` (moved to `deprecated/`).
- **guan + jonginn:** delete `experiments/qsim/deprecated/` and the `dormant/` notebooks, or
  rewrite what is still needed.
- **Physics audit** (a separate stage): the numerical baselines are `xfail(strict=False)`.
  Known open questions: the disorder theory differs by up to 0.3 kHz from the theory saved with
  the August jobs; the 7-1 jobs record other selected occupations than a re-plan gives in 14 of
  19 realizations.
- **Hardware:** the measurement suite has not run on the device since the redesign.

## Entry points: the notebooks

`measurement_notebooks/202609_qsim_migration/` and `analysis_notebooks/202609_qsim_migration/`.

| Notebook | State | Last check |
|---|---|---|
| meas `mbr.py` | new MBR classes | mock dry run passes (2026-09-25) |
| meas `mbr_tomography.py` | new MBR classes | mock dry run passes up to `analyze_tomography` (mock data gives a singular M_0; expected) |
| meas `mbr_disorder.py` | new classes, diagonal disorder (7-1) only | mock dry run passes (Matrix Pencil cells skipped on mock data) |
| meas `multiphoton_calibration.py`, `floquet_calibration.py`, `floquet_displacement_kerr.py` | not touched by the MBR redesign (one timing call in multiphoton) | static import test only |
| ana `mbr.py` | new classes | analysis suite passes (xfail cells 30, 31, 48 are old known failures) |
| ana `mbr_disorder.py` | `MBRDisorderEnsembleExperiment` | analysis suite passes |
| `dormant/` (both sides) | moved-out or old code; loads, not maintained | none |

How to check:
- `pixi run pytest` (about 1300 tests, under 2 min);
- `pixi run python tools/run_qsim_suite.py --suite analysis` (offline, on the prod data tree);
- `pixi run python tools/dryrun_qsim_notebook.py <measurement notebook>` (mock station);
- the measurement suite (`--suite measurement --hardware`) needs the real device; the redesign
  has not run it.

## Code map

### `experiments/qsim/` (37 live modules, about 13 kloc)
- **MBR, clean** (about 3.5 kloc; rules and patterns in `docs/qsim/mbr_redesign.md`):
  - job classes, each with its own Program: `mbr_stark_cal`, `mbr_time_trace`,
    `mbr_ortho_column`, on the shared `mbr_ramsey` (`MBRRamseyProgram` on
    `FloquetTrain` + `QsimBaseProgram`, and `MBRJobExperiment`; no dark-mode base since 8A);
  - the Floquet playback: `floquet_train` (shared with the dark-mode programs);
  - assembled classes: `mbr_calibration_set`, `mbr_spectrum`, `mbr_orthogonality`,
    `mbr_ham_tomo`, `mbr_disorder_ensemble`;
  - infra: `experiments/assembled_data.py` (manifest + assembled HDF5), `mbr_saved`,
    `mbr_campaign` (mock stations, pinned config sets, `smoke()`).
- **Not MBR, split out of the god module but not cleaned** (about 9 kloc): `dark_base`,
  `qsim_base`, `qsim_base_wigner`, `sideband_*`, `kerr`, `dark_mode_*`, `cooling`,
  `cavity_ramsey_flux_excursion`, ...
- `floquet_dark_mode_readout.py` (about 100 lines): only the `_MOVED_TO` re-exports that non-MBR
  notebooks still use.

### `experiments/qsim/notebook_helpers/` (about 4.6 kloc; about 20 kloc before step 7)
| Helper | State |
|---|---|
| `mbr_campaign`, `mbr_disorder_campaign`, `mbr_tomography`, `mbr_saved_reanalysis` | on the new classes |
| `mbr_n3_reprocess`, `mbr_replot`, `mbr_n2_spectroscopy` | new classes at the entry points; inside, still notebook cells copied as blocks |
| `multiphoton_calibration`, `floquet_calibration`, `floquet_bare_readout` | untouched (stage-2 copied blocks; not MBR) |
| `defaults`, `run_mode` | small infra |

### `experiments/qsim/deprecated/` (22 modules, about 15.9 kloc)
Moved, frozen, not maintained; each has a header note. Live code never imports it
(`tests/test_no_live_deprecated_imports.py`). Contents: the old MBR classes and programs
(`legacy_mbr`, `encoding_spectroscopy`, `mbr_nphoton_program`, `mbr_propagator`,
`mbr_phase_correction`); off-diagonal / D72 (`mbr_disorder_offdiag`, `mbr_disorder_h5`,
`mbr_spectral_validation`); SFF (`mbr_sff`, `mbr_sff_campaign`) and shot sampling
(`mbr_sampling`), both to be deleted or rewritten; the per-pulse phase calibrations for the
removed `'decoder'` mode (`floquet_phase_calibration`, step 8A4); the old versions of ported helpers; older
retired code.

### Numerics and data
- `fitting/qsim/` (about 2.9 kloc, pure functions): `matrix_pencil`, `mbr_spectrum`,
  `mbr_hamiltonian`, `mbr_phase`, `mbr_reconstruction`, `mbr_propagator`, `mbr_disorder`,
  `level_statistics`.
- Old jobs reach the new classes only through `tools/migrate_mbr_jobs.py`. Converted sets:
  `C:\experiments\<exp>\converted_data\` and `assembled_data\`. Dataset ID lists:
  `tests/data/mbr_datasets.json`.

## Docs: which are current

| Doc | Status |
|---|---|
| `docs/STATUS.md` | this file: current |
| `docs/log/` | dated session records; never edited after their day |
| `docs/qsim/mbr_redesign.md` | current spec for the MBR classes (steps 1-7 done; step 8 in its own plan) |
| `docs/qsim/mbr_step7_plan.md` | step 7 plan and record (done); its section 7 questions are open |
| `docs/qsim/mbr_step8_plan.md` | step 8 plan and record (8A, 8B, 8D done; 8C open) |
| `docs/qsim/mock_suite_plan.md` | notebook suite design (revised 2026-09-23) |
| `docs/qsim/stage2_notebook_map.md` | stage-2 instructions; history |
| `docs/qsim_refactor_surface_map.md`, `docs/arch_meeting_log.md` | history; `mbr_redesign.md` wins where they differ |
| `docs/archive/` | history |
