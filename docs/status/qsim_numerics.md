# Where things stand: qsim analysis and numerics

**Last updated: 2026-10-02** (data set catalog and the second conversion round, on pippin).
Theme: analysis, numerics, fitting (`fitting/`, `analysis_notebooks/`, offline tools). Branch
`qsim-analysis`, worktree `C:\python\multimode_expts_qsim-analysis`. This file is overwritten at
the end of each work session on this theme; git keeps the old versions. Cross-theme items and
the branch rules are in `docs/STATUS.md`.

## What is next

- **The data sets are in one catalog and converted** (`configs/datasets/mbr_datasets.yaml`;
  record `docs/log/2026-10-02_dataset-catalog-and-conversion-round.md`). 25 data sets, 20
  converted on pippin; the 5 off-diagonal D72 sets wait for an init != final calibration (step 7
  plan, decision 2 and questions 1-2 to jonginn). New since the September round: the Sep 10-14
  **complete-basis** disorder ensemble `sep10_full_K3p6_g29p2` (9 realizations x 35 occupations,
  the first data set that gives Tr U(t) and the spectral form factor), `august25_N3` (a second
  complete N=3 basis, quality concern), two N=1 sets, four orthogonality sets, two small
  disordered sets (quality concern), the September calibration set and the N=1 propagator.
  Both complete-basis sets are in `analysis_notebooks/pole_finding/registry.yaml`.
  **Next:** point `analysis_notebooks/202609_qsim_migration/mbr_disorder.py` at the Sep 10
  ensemble beside the 7-1 one and look at the form factor; run the pole benchmarks (T1T3, C, F)
  on the two new registry entries; decide the Matrix Pencil defaults for a 35-row complete basis
  (the default settings match 25-28 of 35 theory poles per realization, MAE 1.6-3.3 kHz).
  To convert a further data set: add its entry to the catalog, `pixi run python
  tools/convert_mbr_catalog.py --check <name>`, then without `--check`, then paste the printed
  manifest lines into `tools/build_mbr_dataset_catalog.py` (`CONVERTED_*`) and rebuild the YAML.
  jonginn still owes the `K_source` column and the two step 7 questions.
  Full suite after the round (2026-10-02, pippin): 1569 passed, 5 failed; the 5 are the known
  on-purpose failures listed in `docs/STATUS.md` (3 Matrix Pencil regression cases, 2 stage-dispatch
  notebooks), nothing new.
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
  smaller) but not in the hardest case. **T1 (Hamiltonian fit, `hamiltonian_fit.py`) and T3
  (convex sparse fit, `sparse_fit.py`, cvxpy) are done** (logs `..._t1.md`, `..._t3.md`,
  `..._t1t3.md`): T3 started from T1's offsets (T1T3, about 10 s per 10-row case, no C) equals
  T3 with the true offsets on the subset (draw 0, 10 rows): all resolvable gaps on September,
  17-26 of 20-27 on August, 0 false poles. The August P(r < 0.25) stays low because most small
  gaps there are not resolvable at all (a data limit). Real data: the recorded Kerr is 15-20 %
  too large (T1), and the model leaves 1.8-4x the noise. **Next:** T1T3 on august_disorder and
  august_N3, then on all 80 cases; T3's merge threshold at 35 rows; P(r < 0.25) from partly
  resolved spectra; T2 (weight structure) and T4 (September data; timing route in the T0 log)
  still open. Summed-trace pencil and TLS-ESPRIT only as checks inside it. Still open: C's analytic
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
  `configs/datasets/mbr_datasets.yaml`.

## Docs: which are current

| Doc | Status |
|---|---|
| `docs/qsim/pole_finding.md` | pole finding method and benchmark (draft; phase 1 done, 2026-09-28) |
| `docs/qsim/pole_finding_explore.md` | pole finding: the plan of parallel exploration tasks (2026-09-29) |
