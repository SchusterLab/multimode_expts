# Where things stand: qsim analysis and numerics

**Last updated: 2026-10-06** (all 25 data sets converted, the "D72" sets regrouped; issue 5: the library reads only the files; on `guan`).
Theme: analysis, numerics, fitting (`fitting/`, `analysis_notebooks/`, offline tools). Branch
`qsim-analysis`, worktree `C:\python\multimode_expts_qsim-analysis`; on 2026-10-06 `guan` took
all of it, so continue from `guan` (fast-forward `qsim-analysis` to it first). This file is overwritten at
the end of each work session on this theme; git keeps the old versions. Cross-theme items and
the branch rules are in `docs/STATUS.md`.

## What is next

- **Read in the lab vault, with figures** (2026-10-05; `docs/log/2026-10-05_pole-views-and-vault.md`):
  `G:\Shared drives\SLab\Multimode\Lab\guan\qsim_analysis\pole_finding\`, built by
  `analysis_notebooks/pole_finding/views_sep10.py`; guan comments with `> [!guan]` callouts (read
  them first). A fitter is judged by three views (`fitting/qsim/poles/views.py`): poles on the
  per-row FFT, the stick diagram, the r histogram. On Sep 10 the model is near Poisson (<r> 0.415);
  A, B, C show false repulsion (0.50-0.57); T1T3 matches (0.414). First feasibility map
  (`feasibility_map.py`; vault page `2026-10-05_feasibility_map.md`): at the Sep 10 hardware the
  resolution keeps the model's r shape only at large disorder (delta/g 6-10); the model's most
  chaotic cells reach <r> 0.50-0.52. Removing the row offsets (per-photon pattern) lifts the
  late form factor from 0.30 to 0.85 of the model. **Next:** check the bound stand-in against T1T3
  on synthetic ensembles at 3 points; a whole-shape test between two points; reachable K and
  delta (guan); the integer-weight penalty with per-photon offsets.
- **Jonginn's newer notebook work is migrated/retired** using his issue 7 decisions
  (`docs/log/2026-10-05_jonginn-notebook-migration.md`). The Sep10 full-basis ensemble
  now has a catalog-backed section in `analysis_notebooks/202609_qsim_migration/mbr_disorder.py`,
  beside the historical 7-1 reproduction. It shows spectra, level matches and the form factor.
  Optional `readout_refit=True` preserves the legacy IQ diagnostic in reusable modules and
  saves its fit provenance; it cannot repair active reset. Sep10's experimental Matrix Pencil
  settings were not promoted to global defaults. Five old Jonginn notebooks and the notebook
  test exceptions are removed. Shared code is on local `guan` and merged into `qsim-analysis`.
  **Next:** run the updated notebook with pippin data mounted, and publish/merge the branches
  for prod. Final focused checks: 209 passed. Saved-data checks here require the unavailable
  data tree; hardware was not accessed.
- **The survey of every pole-finding method on the Sep 10 ensemble is running unattended**
  (tmux `nb`, window `survey`; `analysis_notebooks/pole_finding/survey_sep10.py`; record
  `docs/log/2026-10-02_sep10-pole-survey.md`). Output:
  `C:\experiments\260818_qsim_spectroscopy\derived_data\pole_finding\sep10_survey\` (`survey.log`
  and CSVs). **Next:** read the results, write the tables into a log entry, and decide: the
  offset prior for this set (T1 finds row offsets of 4.7 kHz rms, up to 11.7 kHz, on r=0, against
  a 0.5 kHz prior), the T3 merge threshold at 35 rows, and the Matrix Pencil settings. Then T2's
  row-sum bound. `experiments/qsim/pole_data.py` now accepts realizations without a recorded Kerr.
- **All 25 data sets are converted** (`configs/datasets/mbr_datasets.yaml`; records
  `docs/log/2026-10-02_dataset-catalog-and-conversion-round.md` and
  `docs/log/2026-10-06_issue5-d72-and-loaders.md`). The five old "D72" (notebook section 7-2)
  sets were not off-diagonal: each realization mixes diagonal pairs (407 files) and
  off-diagonal pairs (188). Renamed `sepNN_pairs_*`, kind `disorder_pairs`: the diagonal pairs
  are a partial-basis `MBRDisorderEnsembleExperiment` (`manifest`), the off-diagonal ones one
  `MBRTimeTraceSetExperiment` per realization (`offdiag_manifests`, not analyzed; step 7
  decision 2). Their g15 timing is 0.409226 us (the archived 0.4135 was the old formula).
  The July sets were converted again with their analyzer sign (+1), so `legacy` is gone.
  **Next:** the 37 new partial-basis realizations match 1-20 of 35 theory poles with default
  settings: decide whether they are worth a pole benchmark. Execute the Sep10 section in
  `analysis_notebooks/202609_qsim_migration/mbr_disorder.py` and inspect the form factor; run
  the pole benchmarks (T1T3, C, F) on the two complete-basis registry entries; decide the
  Matrix Pencil defaults for a 35-row complete basis. To convert a further data set: add it
  to `tools/build_mbr_dataset_catalog.py`, `pixi run python tools/convert_mbr_catalog.py
  --check <name>`, then without `--check`, paste the printed manifest lines into the builder
  and rebuild the YAML. jonginn still owes the `K_source` column.
- **Loading (issue 5):** `from_manifest` loads each job file with `from_h5file` (no shots) and
  the timing from its `derived_params` attribute only. `experiments/saved_jobs.py` and
  `job_records` are now `tools/legacy_saved_jobs.py`, the raw reader of the frozen converter
  (`tools/migrate_mbr_jobs.py`, `tools/convert_mbr_catalog.py`); the deprecated classes reach it
  through the alias `experiments/qsim/deprecated/saved_jobs.py`. The sidecar `tests/data/job_provenance.json` stays as the job-ID -> path index.
- **Jonginn's decay investigation:** `measurement_notebooks/jonginn/decay_investigation.ipynb`
  now contains its JOB catalog, read-only HDF5 reconstruction, fitting, and plots. Per the
  user's request, the four companion modules and their module-dependent test file were
  removed. Code decreased from 1,408 to 753 lines; the fit core is 55 lines. It fits measured
  return power with a coherent-Hamiltonian curve times an exponential/Gaussian power
  envelope; only the selected envelope runs. Existing fit thresholds and tau are preserved:
  972 exponential traces and 98 sampled Gaussian traces match the previous cached results.
  Standard and interleaved real HDF5 traces also reproduce their prior arrays and fits;
  every plot renders. Original data/configs are only read; notebook execution writes no
  manifest, cache, export, or result file. Human review of the physical model remains open.
  (The cited record `docs/log/2026-09-30_decay-notebook-readability.md` was never committed.)
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
| ana `mbr_disorder.py` | historical 7-1 + complete-basis Sep10 ensemble, optional IQ refit | Oct 5 static checks pass; new Sep10 section needs mounted-data execution |
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
- Old jobs reach the new classes only through `tools/migrate_mbr_jobs.py` (frozen; all catalog
  sets are done). Converted sets:
  `C:\experiments\<exp>\converted_data\` and `assembled_data\`. Dataset ID lists:
  `configs/datasets/mbr_datasets.yaml`.

## Docs: which are current

| Doc | Status |
|---|---|
| `docs/qsim/pole_finding.md` | pole finding method and benchmark (draft; phase 1 done, 2026-09-28) |
| `docs/qsim/pole_finding_explore.md` | pole finding: the plan of parallel exploration tasks (2026-09-29) |
| `docs/qsim/pole_finding_recap.md` | pole finding: recap of every method, what it was tested on, performance; figures (2026-10-03) |
