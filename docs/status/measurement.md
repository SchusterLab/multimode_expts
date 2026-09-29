# Where things stand: qsim measurement

**Last updated: 2026-09-29** (split out of `docs/STATUS.md`, on pippin). Theme: measurement code
(`experiments/`, `measurement_notebooks/`, calibration). Branch `guan`, worktree
`C:\python\multimode_expts_guan`; it goes to `main` when it passes validation. This file is
overwritten at the end of each work session on this theme; git keeps the old versions.
Cross-theme items and the branch rules are in `docs/STATUS.md`.

## What is next

- **The Program and Experiment tree refactor, before the merge** (`docs/qsim/program_tree_plan.md`,
  steps 10A-10G; approved by guan 2026-09-29). Decision (guan, 2026-09-29): do it on `guan`
  first, so the other users restart their kernels only once. Done: 10A (the nets: program
  golden, acquire golden, displacement-Kerr notebook in the notebook golden) and 10B (one
  template in `QsimBaseProgram`; the class names change in 10F) and 10C (the chain
  `QsimBaseProgram` -> `FloquetProgram` -> `DarkModeProgram`; `DarkBaseProgram` is an empty
  subclass kept until 10F for the deprecated modules) and 10D (the `readout` key; the old
  readout booleans raise; the notebooks on `guan` moved to the key), 10E (one sweep driver)
  and 10F (the renames of plan section 5, on `guan`: code, tests, the `202609_qsim_migration`
  and `guan/` notebooks, the shared `analysis_notebooks/qsim analysis.ipynb`). Next: 10G, the
  device check in local mode (calibration notebooks, then MBR), then the merge. At the merge,
  jonginn's and connie's notebooks move with `tools/rename_program_tree.py` and the readout
  table (plan section 4); two jonginn names wait for jonginn (plan section 5).
- **The `readout` key** (since 10D): `readout='qubit'` (default), `'parity'`, `'multiparity'`,
  `'wigner'`, `'slow_pi_ge'`; `postpulse` keeps its meaning (decoding, incl. f0-g1 for
  `'qubit'`). A qsim Program refuses `perform_wigner`, `parity_readout`,
  `multiparity_readout` and `slow_pi_ge_readout`, even when false, and
  `post_select_pre_pulse=True`. The notebooks of jonginn and connie still set them; they move at
  the merge (plan, 10F).
- **Flat `experiments` namespace** (guan, a separate session): it keeps the last class of a name
  without a warning. 16 older clashes outside qsim, one in qsim (`SidebandScrambleDarkProgram`,
  live vs deprecated; waits for jonginn). Details in `docs/log/2026-09-29_program-tree-plan.md`.
- **Analysis theme:** `tests/test_matrix_pencil_regression.py` fails 3 cases on `guan`
  (tolerance 1e-12; details in `docs/log/2026-09-29_program-tree-plan.md`).
- **Device:** a first run on the device on 2026-09-28 (guan, local mode) had no errors, but the
  data was not meaningful: the calibration steps need a manual check. After 10A-10F, run the
  calibration notebooks and then the MBR notebooks on the device (10G); then merge `guan` to
  `main`.
- **`main`:** `ErrorAmplificationExperiment.analyze` fits frequency and gain scans with
  `periodic=True` (biased near scan edges; `docs/qsim/mbr_step9_plan.md` 0.4). Fix the default
  on `main` for every user, then cherry-pick; then the Floquet error-amp postproc can drop it.
- **jonginn:** `docs/qsim/mbr_step7_plan.md` section 7 (still open); step 8's removal of the
  `'decoder'` mode and `floquet_phase_calibration.py`; step 9's consolidation of the Floquet
  calibration on the preloaded flat-top; the Stark-phase "not dividing" TODO.
- **Config database:** the mock-acquisition tests of the calibration notebooks need it (the API
  tunnel, or a `jobs.db` copy made on pippin; not the live file over SMB).
- **guan + jonginn:** delete `experiments/qsim/deprecated/` and the `dormant/` notebooks, or
  rewrite what is still needed.

## Entry points: the notebooks

`measurement_notebooks/202609_qsim_migration/`.

| Notebook | State | Last check |
|---|---|---|
| meas `mbr.py` | new MBR classes | mock dry run passes (2026-09-27) |
| meas `mbr_tomography.py` | new MBR classes | mock dry run passes up to `analyze_shared_step` (mock data gives a singular M_0; expected) |
| meas `mbr_disorder.py` | new classes, diagonal disorder (7-1) only | mock dry run passes (Matrix Pencil cells skipped on mock data) |
| meas `multiphoton_calibration.py` | autocalibrate pattern (step 9C) | dry run `--keep-going`: 9 mock/pinned-config failures, fewer than before |
| meas `floquet_calibration.py` | preloaded flat-top only, autocalibrate pattern (step 9D) | dry run `--keep-going`: 1 mock failure (chevron postproc, NaN), as before |
| meas `floquet_displacement_kerr.py` | not touched | dry run: cell 6 fails on mock data (pre-existing) |
| meas `dormant/` | moved-out or old code; loads, not maintained | none |

How to check:
- `pixi run pytest` (about 1500 tests);
- `pixi run python tools/dryrun_qsim_notebook.py <measurement notebook> [--keep-going]` (mock
  station; `--keep-going` lists every failing cell);
- the measurement suite (`--suite measurement --hardware`) needs the real device; first run
  2026-09-28 (see "What is next").

## Code map

### `experiments/qsim/` (38 live modules)
- **The stem** (steps 10B-10F, `docs/qsim/program_tree_plan.md`): one chain of Programs,
  `QsimProgram` (`qsim_base`: the template, the readout modes, `readouts_per_shot`) ->
  `FloquetProgram` (`floquet_train`) -> `DarkModeProgram` (`dark_mode_encoding`), plus
  `QsimRProgram` (RAverager) in `qsim_base`. One sweep driver, `QsimExperiment` (`qsim_base`),
  with `QsimWignerExperiment` (`qsim_base_wigner`) and `MBRJobExperiment` on it.
- **MBR, clean** (about 3.5 kloc; rules and patterns in `docs/qsim/mbr_redesign.md`):
  - job classes, each with its own Program: `mbr_stark_cal`, `mbr_time_trace`,
    `mbr_ortho_column`, on the shared `mbr_ramsey` (`MBRRamseyProgram` on `FloquetProgram`,
    and `MBRJobExperiment`);
  - assembled classes: `mbr_calibration_set`, `mbr_spectrum`, `mbr_orthogonality`,
    `mbr_ham_tomo`, `mbr_disorder_ensemble`;
  - infra: `experiments/assembled_data.py` (manifest + assembled HDF5), `mbr_saved`,
    `mbr_campaign` (the campaign base, mock stations, pinned config sets, `smoke()`).
- **Calibration support** (step 9): `multiphoton_swap` (N-photon swap sequences),
  `bare_readout_check`; `floquet_gain_chevron` fits the 2D chevron.
- **Not MBR: leaves on the stem, their insides not cleaned**: `sideband_*`, `kerr`,
  `dark_mode_*`, `cooling`, `cavity_ramsey_flux_excursion`, ... Eight of them no notebook or
  test uses (plan 2.4); their retirement waits for the other users.
- `floquet_dark_mode_readout.py` (about 100 lines): only the `_MOVED_TO` re-exports that non-MBR
  notebooks still use.

### `experiments/qsim/notebook_helpers/` (about 0.2 kloc; about 20 kloc before step 7)
Only `defaults` and `run_mode`: scaffolding for testing this refactor (step 9, decision 0.1).

### `experiments/qsim/deprecated/` (27 modules, about 18.4 kloc)
Moved, frozen, not maintained; each has a header note. Live code never imports it
(`tests/test_no_live_deprecated_imports.py`). Contents: the old MBR classes and programs
(`legacy_mbr`, `encoding_spectroscopy`, `mbr_nphoton_program`, `mbr_propagator`,
`mbr_phase_correction`); off-diagonal / D72 (`mbr_disorder_offdiag`, `mbr_disorder_h5`,
`mbr_spectral_validation`); SFF (`mbr_sff`, `mbr_sff_campaign`) and shot sampling
(`mbr_sampling`), both to be deleted or rewritten; the per-pulse phase calibrations for the
removed `'decoder'` mode (`floquet_phase_calibration`, step 8A4); the hooks of the old
all-envelope Floquet notebook (`floquet_calibration_hooks`, `floquet_bare_readout`, step 9D); the
old versions of ported helpers; older retired code.

## Docs: which are current

| Doc | Status |
|---|---|
| `docs/qsim/program_tree_plan.md` | plan for steps 10A-10G (qsim Program/Experiment tree, `readout` key); approved, 10A next |
| `docs/qsim/mbr_redesign.md` | current spec for the MBR classes (steps 1-7 done; step 8 in its own plan) |
| `docs/qsim/mbr_step7_plan.md` | step 7 plan and record (done); its section 7 questions are open |
| `docs/qsim/mbr_step8_plan.md` | step 8 plan and record (done) |
| `docs/qsim/mbr_step9_plan.md` | step 9 plan and record (done) |
| `docs/qsim/mock_suite_plan.md` | notebook suite design (revised 2026-09-23) |
| `docs/qsim/stage2_notebook_map.md` | stage-2 instructions; history |
| `docs/qsim_refactor_surface_map.md`, `docs/arch_meeting_log.md` | history; `mbr_redesign.md` wins where they differ |
