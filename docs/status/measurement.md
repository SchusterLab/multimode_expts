# Where things stand: qsim measurement

**Last updated: 2026-10-06** (issue 5: loaders, shim and renames; branches merged into `guan`; not deployed). Theme: measurement code
(`experiments/`, `measurement_notebooks/`, calibration). Branch `guan`, worktree
`C:\python\multimode_expts_guan`; it goes to `main` when it passes validation. This file is
overwritten at the end of each work session on this theme; git keeps the old versions.
Cross-theme items and the branch rules are in `docs/STATUS.md`.

## What is next

- **Issue 5 changes on `guan`, not on `main` yet** (`docs/log/2026-10-06_issue5-d72-and-loaders.md`).
  For acquisition code: `floquet_dark_mode_readout.py` (the compatibility shim) is deleted;
  import from the owning modules. Renamed modules: `dark_mode_scramble.py` (was
  `mbr_spectroscopy_program.py`), `broadband_ge_validation.py`, `man_stor_multiparity_chevron.py`.
  `MBRJobExperiment.from_h5file(..., load_shots=False)` exists; saved MBR jobs need their
  `derived_params` attribute (every job since 2026-09-17 has it). `measurement_notebooks/guan/mbramsey.py`
  is deleted. `guan` now also holds all of `qsim-analysis` and `job_id_sorting_out`.
  **Next:** when the device is idle and the `main` checkout is clean (it has other sessions'
  uncommitted edits, including `job_server/worker.py`), merge `guan` into `main`; users restart
  kernels once.
- **The Program and Experiment tree refactor (steps 10A-10G) is done and on `main`**
  (`docs/qsim/program_tree_plan.md`; record in `docs/log/2026-09-29_10g-device-check-and-merge.md`).
  10G on the device: every changed Program type ran with no errors (calibration notebooks, MBR
  StarkCal, TimeTrace, OrthoColumn). The Stark-phase sign changed from 09-28, but with the same
  config the Program ASM and the analysis are unchanged: the device drifted.
  After the merge, on `main`: the calibration notebooks and MBR section 0 ran locally with no
  errors and the same values; a worker started from SSH (18:44) ran MBR section 0 through the
  queue as `JOB-20260929-00087` (StarkCal) and `-00088` (TimeTrace), both completed. That
  worker is stopped: start the worker from the RDP desktop (Ctrl-C cancels a job there).
- **Jonginn notebook migration is complete locally** (Guan requested completion using
  Jonginn's issue 7 answers). Full fixed-N basis and RMS disorder are the new campaign
  defaults, with explicit custom subsets. Single-shot recalibration runs before each
  occupation by default, updates active-reset settings, and records calibration provenance
  in subsequent job HDF5 metadata. The five legacy notebooks, obsolete migration tool,
  decoder-mode N=2 analysis notebook and notebook-test exceptions are retired.
  Optional offline IQ refitting is preserved in modules; it cannot repair stale active reset.
  Decisions and destinations: `docs/qsim/notebook_migration_completion.md`.
  **Next:** review and merge `guan` into `main`, then check one interleaved single-shot/MBR
  pair on hardware. No prod files or config versions were changed in this session.
- **September branch sync is done:** main/guan were pushed and main merged into
  qsim-analysis on Sep 29 (`docs/log/2026-09-29_branch-sync-after-merge.md`). New local
  migration changes still need publication/deployment.
- **The `readout` key** (since 10D): `readout='qubit'` (default), `'parity'`, `'multiparity'`,
  `'wigner'`, `'slow_pi_ge'`; `postpulse` keeps its meaning (decoding, incl. f0-g1 for
  `'qubit'`). A qsim Program refuses `perform_wigner`, `parity_readout`,
  `multiparity_readout` and `slow_pi_ge_readout`, even when false, and
  `post_select_pre_pulse=True`. `tools/migrate_readout_flags.py` moves a notebook to the key.
- **Device physics:** the calibration steps need the manual check (data not meaningful on
  09-28 and 09-29: MBR orthogonality diagonal 0.04-0.15, off-diagonal power up to 2.7x). The
  full MBR smoke run is long: section 1 alone is 35 jobs of ~2 min.
- **Flat `experiments` namespace** (guan, a separate session): it keeps the last class of a name
  without a warning. 16 older clashes outside qsim, one in qsim (`SidebandScrambleDarkProgram`,
  live vs deprecated; waits for jonginn). Details in `docs/log/2026-09-29_program-tree-plan.md`.
- **Analysis theme:** `tests/test_matrix_pencil_regression.py` fails 3 cases
  (tolerance 1e-12; details in `docs/log/2026-09-29_program-tree-plan.md`).
- **Test setup:** a new worktree needs `configs\versions` (a junction to main's folder), or
  about 34 tests fail on the config archive.
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
| meas `mbr_disorder.py` | full basis / custom subsets, RMS, interleaved readout calibration | 2026-10-05: local planning, ordering, static and mock-acquisition checks pass; hardware pair pending |
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
- `floquet_dark_mode_readout.py`: deleted 2026-10-06 (issue 5); its users import from the
  owning modules.

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
