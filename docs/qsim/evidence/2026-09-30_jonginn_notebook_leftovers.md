# What is left in jonginn's old notebooks

> **Status: tentative.** Written 2026-09-30 from an agent survey of `origin/main`
> at `c3bfeda`. Not yet read in full by guan or checked with jonginn. Cell
> indices are for the committed notebooks at that commit.

## Scope

The September refactor split two large notebooks into short jupytext entry points,
and moved the logic into modules:

- `measurement_notebooks/jonginn/qsim_experiments.ipynb` -> `measurement_notebooks/202609_qsim_migration/`
- `measurement_notebooks/jonginn/data_postprocess.ipynb` -> `analysis_notebooks/202609_qsim_migration/`

jonginn kept working in the old notebooks, so they were not deleted. This file lists
what the old notebooks have that the new tree does not.

## Why the two sides started from different states

- On prod, the main checkout (`C:\python\multimode_expts`) had jonginn's notebook
  edits **uncommitted** from about 09-06 (last commit `8d8dda5`) until 09-15.
  jonginn ran the Sep 10-14 measurements from that working copy.
- Branches and remote were in sync by commits: `origin/main` was `0e30936`
  (09-11), and prod `main` fast-forwarded to it on 09-14 13:26 with the notebook
  edits still uncommitted.
- The refactor on `guan` (`ea384a9` 09-11, `a8d8ea1` 09-14, `cb3f9f7` 09-15 01:06)
  read the **committed** notebook. So it never saw the Sep 06-15 edits.
- jonginn committed them on 09-15 12:09 (`b417076`, `f534dc2`) on top of `0e30936`.
  They reached `guan`/`main` only in the 09-29 merges.

## Base commits used for the diff

- `qsim_experiments.ipynb`: blob `05839a2` (at `0e30936`, what jonginn's edits start
  from) -> `143ff15` (origin/main, 352 cells). The migration itself read blob
  `90258ee` (after `a8d8ea1`). Cell indices are the same in both.
- `data_postprocess.ipynb`: `d45c71d` -> `68a7c90` (306 -> 494 cells).
- Commits after the snapshot: `b417076`, `f534dc2` (09-15); `1e4b03a`, `73609c1`
  (09-28); `aaac418`, `3184a6d`, `0321675`, `016d17e`, `866fc1f` (09-29, jonginn);
  `2858086` (09-29, 10F renames only).
- New since the migration: `measurement_notebooks/jonginn/data_recollecting.ipynb`
  (`aaac418` and later).
- None of the three runs on main: they use `BatchRunner` (gone, `86d0632`),
  `EncodingHamiltonianSpectroscopyExperiment` and `analyze(stage=...)` (gone in the
  MBR redesign).

## Inventory

Status: a = covered, b = can be absorbed, c = module only, d = one-off / obsolete,
e = needs jonginn.

| # | Where (cells, commit) | What | Status | Size |
|---|---|---|---|---|
| Q1 | qsim_experiments c2, b417076 | Config versions CFG-*-20260909 | b: `mbr.py:96`, `mbr_disorder.py:76` still use 20260904 (dataset choice) | small |
| Q2 | c4 deleted, b417076 | `station.ds_storage.df` | d | - |
| Q3 | c152 "Bare", b417076 | Dark scramble check, cycles 0..120 step 3 | a: `floquet_calibration.py:681-684` (scan value only) | small |
| Q4 | c292, b417076+f534dc2 | Calibration display | a: `mbr.py:208-212` | - |
| Q5 | c298, b417076 | Orthogonality over the full fixed-N basis | b: `mbr.py:273` uses a hand list | small |
| Q6 | c302, b417076 | Selected encoder/decoder pairs, cycles 0..1000 in 4 chunks | a: `mbr.py` section 4 (365-382); off-diagonal left out by decision 7-2 | small |
| Q7 | c315-326 (new section 7), f534dc2 | Full 35-state diagonal disorder campaign (see below) | b + e | large |
| Q8 | old 7-1/7-2 (c317-348), deleted in f534dc2 | Theory-selected 7-1; D72 7-2 | in `mbr_disorder.py` and `dormant/mbr_disorder_offdiag.py`. e: is 7-1 still the canonical flow? | - |
| P1 | data_postprocess c3 + 9 loader cells, 1e4b03a | `*_legacy` loaders for old HDF5 (August cycle-time arithmetic) | a/d: converted manifests in analysis `mbr.py`, `mbr_disorder.py`. e: does the cycle-time convention match? | small |
| P2 | c104, c107 (dark mode; analysis `dormant/dark_mode.py`), 1e4b03a | Dataset 20260905 #51-55, plot colours | d, or copy if dark mode comes back | small |
| P3 | c166, 1e4b03a | N=2 section marked "NOT NEEDED ... DISREGARD" | d: supports deleting `dormant/mbr_n2_decoder_mode.py` | - |
| P4 | c256-315 (HDF5 reprocessing; `dormant/mbr_disorder_offdiag.py`), f534dc2 | Sep10 dataset, readout refit, new plots (see below) | b, c | large |
| P5 | c293, 2858086 | `saved_readout_mode` rename | d | - |
| P6 | c316-493, 73609c1/aaac418 | One data cell per dataset (16) + 339-line HDF5 helper | d: duplicate of `data_recollecting.ipynb` | - |
| R1 | data_recollecting c6 | 853-line HDF5 helper: grouped figures of all realizations, job lists, printout of g | c: grouped display on `MBRDisorderEnsembleExperiment`; imports a removed class | medium |
| R2 | data_recollecting c7-56 | Job-range catalogue of 16 datasets with Floquet config versions | b: into `tests/data/mbr_datasets.json` | medium |
| H | `qsim_experiments_highkerr_untracked*.ipynb` | Copies of the Sep 6-7 notebook (see below) | d, archive | - |

### Q7: new section 7 (largest item)

- Measures all 35 diagonal N=3 states, or a custom list. Seeds from
  `master_seed = 20260912`. Manual-detuning path to re-measure one realization.
  Each occupation runs on its own, then a single-shot readout recalibration
  (`ss_runner`) and an FFT display. Provenance into the job config. Optional MILP
  state picker and cycle-grid picker. Matrix Pencil post-processing.
- Data: the Sep 12-14 part of "Sep10 K3.6 g29.2", realizations r=1..8. r=0
  (Sep 10-11) came from the short-lived 7-3 section in `b417076` (`d73_*` keys).
  For r=1: 70 spectroscopy + 34 single-shot jobs, 35 occupations, all diagonal.
- Missing in the new tree: `plan_diagonal_disorder` always picks 10 states; no
  single-shot step between occupations; ensemble `acquire` has no per-occupation hook.
- **Physics difference (e):** jonginn scales the random direction by its RMS.
  `fitting/qsim/mbr_disorder.disorder_direction` divides by the norm. For 4 modes
  norm = 2 x RMS, so "50 kHz" is a different disorder strength in the two codes.
- Work: add a full-basis / custom-state option, the normalization choice and a
  per-occupation hook to `experiments/qsim/mbr_disorder_ensemble.py`; then a
  section in `measurement_notebooks/202609_qsim_migration/mbr_disorder.py`.

### P4: HDF5 reprocessing (parked as dormant, but in use)

- Changes: Sep10 dataset and timing (T = 92/430.08 us, g = 1/(4*40*T)); loader
  accepts `d73_*` and unprefixed keys; offline readout refit (two-Gaussian IQ fit
  per job, `fit_readout_clouds`); Matrix Pencil tolerances 0.5 -> 1.5, sigma 1 -> 3,
  match 0.5 -> 2, pool merge 0.1 -> 1.0 kHz; `include_partial = True`; plots of
  Matrix Pencil on the summed trace, and coherent vs incoherent FFT sum.
- **Misfiled dataset:** Sep10 is diagonal (checked in the HDF5), not D72.
  `tests/data/mbr_datasets.json` has it as `d72_Sep10_K3p6_g29p2`, not converted,
  no timing. `docs/qsim/mbr_step7_plan.md` says no disorder dataset is a complete
  basis; Sep10 r=1..8 is one.
- Work: convert Sep10 with `tools/migrate_mbr_jobs.py` (kind disorder; skip the
  single-shot jobs, read unprefixed keys). Readout refit -> module (c). Matrix
  Pencil on the summed trace -> `fitting/qsim` (c), already listed in
  `docs/status/qsim_numerics.md:50`. Coherent FFT sum is covered (a) by
  `experiments/qsim/mbr_spectrum.py:492`, used at analysis `mbr.py:396`. Then a
  section in analysis `mbr_disorder.py`.

### H: the two high-Kerr notebooks

Copies of about the Sep 6-7 `qsim_experiments` ("Sep07 K52.3" run: config
`CFG-HW-20260906-00001`, modes [2,3,4,5], D72 seed 20260907, Kerr grid -56..-40 kHz).
`_untracked`: 355 of 372 cells match some version of `qsim_experiments`; the other
17 differ only in scan values. `_refactored`: same content with the API edits
`808fa75`, `8c700fc`, `2858086`. No unique logic.

## Verdict

After Q7, P4, R1 and R2 are absorbed, `qsim_experiments.ipynb`,
`data_postprocess.ipynb`, `data_recollecting.ipynb` and both high-Kerr notebooks
can be deleted (git keeps them). Blockers:

1. jonginn decides on section 7: full basis, the single-shot step, RMS vs norm.
2. Sep10 refiled in the dataset list and converted.
3. Readout refit moved into a module.
4. jonginn still edits the old notebooks (last commit 09-29), and prod's main
   checkout has uncommitted edits to `qsim_experiments.ipynb` (10F renames only).
5. Unsure: are Sep01/02/05/07 really D72? Only one Sep02 job was checked.

## Related tests and tools

- `tests/test_jonginn_notebook_migration.py`: 59 passed, 1 xfailed
  (`test_the_migration_script_has_nothing_left_to_do`, exempt via `KEPT_UNMIGRATED`).
- `tools/migrate_jonginn_notebooks.py` is obsolete: it targets the 09-14 stage
  classes (now `experiments/qsim/deprecated/legacy_mbr.py`) and cannot fix the
  `BatchRunner` / `EncodingHamiltonianSpectroscopyExperiment` removals. Delete it,
  the xfail test and `KEPT_UNMIGRATED` when the old notebooks go.
- `tests/test_no_stage_dispatch_remains.py`: 2 fail (`data_recollecting.ipynb`,
  `qsim_experiments_highkerr_untracked.ipynb`), as `docs/STATUS.md:33` says.
- Not examined: `Autocalibrate.ipynb` (changed in `f534dc2`, has uncommitted edits).
