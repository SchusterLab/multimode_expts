# Where things stand

**Last updated: 2026-09-28** (pole finding: design calculator, on pippin). This file is overwritten at
the end of each work session; git keeps the old versions. What happened, and why, is in
`docs/log/` (newest first: `2026-09-28_pole-finding-diagnostics.md`, `2026-09-28_pole-finding-phase1.md`, `2026-09-28_pole-finding-design.md`, `2026-09-27_matrix-pencil.md`). Read this file first; read the log only
if you need the history.

## What is next

- **Pole finding** (guan; on pippin, `C:\python\multimode_expts_guan`): phases 1 and 2 of
  `docs/qsim/pole_finding.md` are done, fitters C and D sketched (`fitting/qsim/poles/`;
  notebooks in `analysis_notebooks/pole_finding/`). The limit is the data: the 7-1 window loses
  the small gaps for every fitter, and on the August sets row offsets (sigma 0.5-1 kHz) break
  A and B. **C** (B plus one offset per row, calibration prior) removes the offset loss on
  synthetic data (26 of 35 resolved at every sigma, A and B 7-9 at 1 kHz) and is best on real
  august_N3 (multiplet error 0.16; finds the level A and B miss); on august_disorder it gives
  P(r < 0.25) 0.09 vs the model's 0.18 (B's 0.15 was inflated by offset-made false poles).
  None of A-D is good enough on the real data, and the scores do not say why (guan): so first
  find the cause of each miss. Done: `display_pole_fit` (one figure for any fitter's
  `PoleFit`), and the per-level diagnosis (`diagnose_levels`, spec 5.5; notebook
  `analysis_notebooks/pole_finding/level_causes.py`): in the rows, resolvable (Cramér-Rao),
  held (refinement started at the model), found. Result (bounds at the data's decay, 0.005 per
  us): august_N3 is fitter-limited, and C finds all 10 levels; august_disorder/0 has 24 of 35
  levels resolvable (31 if the row offsets were known), of which C finds 19, B 18, A 15; 7-1/0
  is one unresolvable chain. The model is off by 1-2 kHz in places (assumed 0.3). **Next:
  step 3**: the design calculator (Cramér-Rao, spec 5.6; `fitting/qsim/poles/design.py`) says
  the largest gain is free: fit **real amplitudes** (a diagonal row's <b|P|b>), 24 -> 33 of 35
  resolvable on august_disorder/0 at T2 200 us, 19 -> 25 at 100 us. A complete basis and
  whole-number weights add little at equal time; offset calibration matters only with complex
  amplitudes; T2 dominates; the present window (about 2 T2) is right. Real amplitudes hold on the
  August data (the residual rises as on synthetic data), but C with a real polish finds no more
  levels: B merges the close pairs and C cannot add poles. **Next: a fitter that searches the
  pole count with real amplitudes** (split a pole, refit, keep if chi^2 drops), to reach the real
  bound (on synthetic August data 23-33 resolvable vs 24-30 found). Summed-trace pencil and TLS-ESPRIT only as checks inside it. Still open: C's analytic
  Jacobian (slow), C on benchmark 2 and the new registry entries (guan converts jonginn's
  logs); D rework or drop. Freezing a fitter and switching `MBRDisorderEnsembleExperiment` wait
  for that. Benchmarks run serially (parallel fitter-A workers crash on pippin, 0x80000003). The
  current Matrix Pencil's three known defects stay as strict xfails. Record:
  `docs/log/2026-09-28_pole-finding-diagnostics.md` (plan and reasons),
  `docs/log/2026-09-28_pole-finding-phase1.md`.
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
- **Physics audit** (a separate stage): the numerical baselines are `xfail(strict=False)`.
  Known open questions: the disorder theory differs by up to 0.3 kHz from the theory saved with
  the August jobs; the 7-1 jobs record other selected occupations than a re-plan gives in 14 of
  19 realizations.
- **Hardware:** the measurement suite has not run on the device since the redesign.

## Entry points: the notebooks

`measurement_notebooks/202609_qsim_migration/` and `analysis_notebooks/202609_qsim_migration/`.

| Notebook | State | Last check |
|---|---|---|
| meas `mbr.py` | new MBR classes | mock dry run passes (2026-09-27) |
| meas `mbr_tomography.py` | new MBR classes | mock dry run passes up to `analyze_shared_step` (mock data gives a singular M_0; expected) |
| meas `mbr_disorder.py` | new classes, diagonal disorder (7-1) only | mock dry run passes (Matrix Pencil cells skipped on mock data) |
| meas `multiphoton_calibration.py` | autocalibrate pattern (step 9C) | dry run `--keep-going`: 9 mock/pinned-config failures, fewer than before |
| meas `floquet_calibration.py` | preloaded flat-top only, autocalibrate pattern (step 9D) | dry run `--keep-going`: 1 mock failure (chevron postproc, NaN), as before |
| meas `floquet_displacement_kerr.py` | not touched | dry run: cell 6 fails on mock data (pre-existing) |
| ana `mbr.py` | N=3, self-Kerr, Aug 15-17 reproduction; canonical-flow cells (step 9B) | analysis suite passes, smoke profile (2026-09-27; xfail cell 18 is the known disorder-theory difference) |
| ana `mbr_disorder.py` | `MBRDisorderEnsembleExperiment` | analysis suite passes |
| `dormant/` (both sides) | moved-out or old code; loads, not maintained | none |

How to check:
- `pixi run pytest` (about 1300 tests; more than 2 min off-host, as some tests read the data
  mount). Matrix Pencil alone: `tests/test_matrix_pencil_*.py`, a few seconds, no mount;
- `pixi run python tools/run_qsim_suite.py --suite analysis` (offline, on the prod data tree);
- `pixi run python tools/dryrun_qsim_notebook.py <measurement notebook> [--keep-going]` (mock
  station; `--keep-going` lists every failing cell);
- the measurement suite (`--suite measurement --hardware`) needs the real device; the redesign
  has not run it.

## Code map

### `experiments/qsim/` (39 live modules, about 13 kloc)
- **MBR, clean** (about 3.5 kloc; rules and patterns in `docs/qsim/mbr_redesign.md`):
  - job classes, each with its own Program: `mbr_stark_cal`, `mbr_time_trace`,
    `mbr_ortho_column`, on the shared `mbr_ramsey` (`MBRRamseyProgram` on
    `FloquetTrain` + `QsimBaseProgram`, and `MBRJobExperiment`; no dark-mode base since 8A);
  - the Floquet playback: `floquet_train` (shared with the dark-mode programs);
  - assembled classes: `mbr_calibration_set`, `mbr_spectrum`, `mbr_orthogonality`,
    `mbr_ham_tomo`, `mbr_disorder_ensemble`;
  - infra: `experiments/assembled_data.py` (manifest + assembled HDF5), `mbr_saved`,
    `mbr_campaign` (the campaign base, mock stations, pinned config sets, `smoke()`).
- **Calibration support** (step 9): `multiphoton_swap` (N-photon swap sequences),
  `bare_readout_check`; `floquet_gain_chevron` fits the 2D chevron.
- **Not MBR, split out of the god module but not cleaned** (about 9 kloc): `dark_base`,
  `qsim_base`, `qsim_base_wigner`, `sideband_*`, `kerr`, `dark_mode_*`, `cooling`,
  `cavity_ramsey_flux_excursion`, ...
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

### Numerics and data
- `fitting/qsim/` (about 2.8 kloc, pure functions): `matrix_pencil` (settings are
  `MatrixPencilSettings`; the FFT of its fits is `mbr_spectrum.local_spectrum`), `mbr_spectrum`,
  `mbr_hamiltonian`, `mbr_phase`, `mbr_reconstruction`, `mbr_propagator`, `mbr_disorder`,
  `level_statistics`, `calibration` (step 9C); `poles/` (about 1.1 kloc): the pole-finding
  fitters and benchmarks (`docs/qsim/pole_finding.md`).
- Old jobs reach the new classes only through `tools/migrate_mbr_jobs.py`. Converted sets:
  `C:\experiments\<exp>\converted_data\` and `assembled_data\`. Dataset ID lists:
  `tests/data/mbr_datasets.json`.

## Docs: which are current

| Doc | Status |
|---|---|
| `docs/STATUS.md` | this file: current |
| `docs/log/` | dated session records; never edited after their day |
| `docs/qsim/pole_finding.md` | pole finding method and benchmark (draft; phase 1 done, 2026-09-28) |
| `docs/qsim/mbr_redesign.md` | current spec for the MBR classes (steps 1-7 done; step 8 in its own plan) |
| `docs/qsim/mbr_step7_plan.md` | step 7 plan and record (done); its section 7 questions are open |
| `docs/qsim/mbr_step8_plan.md` | step 8 plan and record (done) |
| `docs/qsim/mbr_step9_plan.md` | step 9 plan and record (done) |
| `docs/qsim/mock_suite_plan.md` | notebook suite design (revised 2026-09-23) |
| `docs/qsim/stage2_notebook_map.md` | stage-2 instructions; history |
| `docs/qsim_refactor_surface_map.md`, `docs/arch_meeting_log.md` | history; `mbr_redesign.md` wins where they differ |
| `docs/archive/` | history |
