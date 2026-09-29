# Where things stand: qsim analysis and numerics

**Last updated: 2026-09-29** (split out of `docs/STATUS.md`, on pippin). Theme: analysis,
numerics, fitting (`fitting/`, `analysis_notebooks/`, offline tools). Branch `qsim-analysis`,
worktree `C:\python\multimode_expts_qsim-analysis`. This file is overwritten at the end of each
work session on this theme; git keeps the old versions. Cross-theme items and the branch rules
are in `docs/STATUS.md`.

## What is next

- **Pole finding**: phases 1 and 2 of
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
  amplitudes; T2 dominates; the present window (about 2 T2) is right. Fitter F (pursuit with
  real, non-negative amplitudes) finds 1-2 more levels than C on synthetic August data; row
  alignment by rank does no better than C, and even with the true offsets the pencil resolves
  only 22-33 of 35. **The exploration plan `docs/qsim/pole_finding_explore.md`: T0 is done**
  (`gap_score.py`, `fixed_set.py`, `tools/pole_fixed_set.py`, notebook `gap_score.py`; log
  `docs/log/2026-09-29_pole-finding-t0.md`): 80 synthetic cases (August point; September point
  on its recorded grid, timing recovered: dt 0.4278 us, 468 samples, g 29.217 kHz) with
  Cramér-Rao gap bounds, C fitted on all, F on draw 0 / 10 rows. C misses levels (August T2 100
  us, 10 rows: 6-7 of 26 resolvable gaps) and biases P(r < 0.25) low (August 0.01-0.10 vs
  0.184); what it finds is near the bound. F beats C everywhere (mean +3.5 gaps, errors 2-3x
  smaller) but not in the hardest case. **Next:** in parallel T1 Hamiltonian fit, T2 the
  complete basis' weight structure, T3 a convex sparse fit, T4 convert the September
  complete-basis disorder data (9 x 35 occupations; the timing route is in the T0 log), T5
  speed; the rest of F on the set when pippin is free for hours. Summed-trace pencil and TLS-ESPRIT only as checks inside it. Still open: C's analytic
  Jacobian (slow), C on benchmark 2 and the new registry entries (guan converts jonginn's
  logs); D rework or drop. Freezing a fitter and switching `MBRDisorderEnsembleExperiment` wait
  for that. Benchmarks run serially (parallel fitter-A workers crash on pippin, 0x80000003). The
  current Matrix Pencil's three known defects stay as strict xfails. Record:
  `docs/log/2026-09-28_pole-finding-diagnostics.md` (plan and reasons),
  `docs/log/2026-09-28_pole-finding-phase1.md`.
- **Physics audit** (a separate stage): the numerical baselines are `xfail(strict=False)`.
  Known open questions: the disorder theory differs by up to 0.3 kHz from the theory saved with
  the August jobs; the 7-1 jobs record other selected occupations than a re-plan gives in 14 of
  19 realizations.

## Entry points: the notebooks

`analysis_notebooks/202609_qsim_migration/` and `analysis_notebooks/pole_finding/`.

| Notebook | State | Last check |
|---|---|---|
| ana `mbr.py` | N=3, self-Kerr, Aug 15-17 reproduction; canonical-flow cells (step 9B) | analysis suite passes, smoke profile (2026-09-27; xfail cell 18 is the known disorder-theory difference) |
| ana `mbr_disorder.py` | `MBRDisorderEnsembleExperiment` | analysis suite passes |
| ana `dormant/` | moved-out or old code; loads, not maintained | none |

How to check:
- `pixi run pytest` (about 1300 tests; more than 2 min off-host, as some tests read the data
  mount). Matrix Pencil alone: `tests/test_matrix_pencil_*.py`, a few seconds, no mount;
- `pixi run python tools/run_qsim_suite.py --suite analysis` (offline, on the prod data tree).

## Code map

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
| `docs/qsim/pole_finding.md` | pole finding method and benchmark (draft; phase 1 done, 2026-09-28) |
| `docs/qsim/pole_finding_explore.md` | pole finding: the plan of parallel exploration tasks (2026-09-29) |
