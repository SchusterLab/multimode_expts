# 2026-10-06: issue closing day; the "D72" sets, and the end of the old loaders (issue 5)

Session with guan (gzhwang), on pippin. Branch `guan`.

## Issues and branches

- Closed issues 3 (BatchRunner folded into `CharacterizationRunner`, 86d0632 on `main`),
  6 (one YAML catalog, 36114c4; duplicate lists retired, 8ab8788) and 7 (Jonginn notebook
  migration, 0349bd8 / 15f5e33). guan: no follow-up issues.
- The device was busy (seb's Ramsey jobs) and the `main` checkout has uncommitted edits from
  other sessions (including `job_server/worker.py`), so nothing went to `main` today.
  Decision (guan): merge into one active branch. `guan` fast-forwarded to `qsim-analysis`
  (which held all of `guan`), then merged `job_id_sorting_out` (95ab8ad). Conflicts:
  `data_recollecting.ipynb` kept deleted (retired with Jonginn's approval, issue 7);
  `docs/status/qsim_numerics.md` kept guan's text plus the decay-notebook bullet. The record
  that bullet cites (`2026-09-30_decay-notebook-readability.md`) was never committed.

## The "D72" data sets were not off-diagonal

guan remembered checking them and finding diagonal data; the agents had always called them
off-diagonal. Finding: **both, in every set.** "D72" is the old notebook's section 7-2. Its
jobs (`EncodingPropagatorProgram`) each hold one (initial, decoder) pair, sweeping (cycle,
decoder, analyzer phase) with the preparation phase inside. A scan of all 595 files of the
five sets:

| Set (new name) | Realizations x pairs | Diagonal | Off-diagonal |
|---|---|---|---|
| `sep07_pairs_K52p3_g29p2` | 6 x 15 (2 time chunks) | 80 | 10 |
| `sep05_pairs_K3p6_g30` | 2 x 15 | 23 | 7 |
| `sep01_pairs_K44_g15` | 5 x 10 | 30 | 20 |
| `sep02_pairs_K20_g15` | 5 x 10 | 33 | 17 |
| `sep02to04_pairs_K3p6_g15` | 19 x 15 | 161 | 124 |

Config and data agree in every file: an init != final trace must start near 0 (orthogonal
Fock states), and the off-diagonal ones do (median |A(0)| 0.02, all < 0.09); the diagonal
ones start at 0.24-0.63 (median per set). The Sep01 files have no
`spectroscopy_final_occupations`; the decoder is in each sweep row, and their 20
off-diagonal rows also start near 0, so they are off-diagonal too. guan's memory is most
likely the Sep 10 set, which the old JSON called `d72_Sep10` and is fully diagonal
(corrected 2026-10-02).

Two more record corrections:

- Timing. The catalog's `archived_timing` gave 0.4135 us for the three g15 sets. That is the
  pre-0d7ea14 cycle formula (guan's commit of 2026-09-09 calls it "never right"); the
  resolver gives 0.409226 us = 176 tProc ticks at 430.08 MHz, as for every other converted
  set. Sep05/Sep07 agree (0.213914 us). `archived_timing` is dropped from the catalog.
- Phase correction. All section 7-2 jobs played no Stark correction on the pulse
  (`final_analyzer_phase_per_cycle_deg = 0`); `offdiag_decoder_phase_correction_deg` was
  applied by the old analysis afterwards. The converted files record the pulse truth; the
  old analysis value is kept in `converted_from.source_notes`.

## Conversion of the D72 sets (decision guan: group them properly, convert, catalog)

- `tools/migrate_mbr_jobs.py`: `migrate_pairs` / `merge_pair_jobs` turn the time chunks of
  one pair job into one `MBRTimeTraceExperiment`. Checked bit-identical against the raw
  averages and shots (also a new test).
- New `experiments/qsim/mbr_time_trace_set.py`: `MBRTimeTraceSetExperiment`, a plain set of
  TimeTraces (stack, display, manifest). It gives the off-diagonal traces a manifest. They
  are not analyzed (docs/qsim/mbr_step7_plan.md, decision 2, stands).
- `tools/convert_mbr_catalog.py`: kind `disorder_pairs`. Diagonal pairs -> one Spectrum per
  realization -> one `MBRDisorderEnsembleExperiment` (`manifest`); off-diagonal pairs -> one
  trace set per realization (`offdiag_manifests`). Sep05 reuses the `september_N3`
  calibration set.
- Converted on pippin, raw untouched (about 12 GB new). All 37 realizations analyze; with
  default settings 1-20 of 35 theory poles match per realization (partial bases; not judged).
- Catalog: renamed `sepNN_pairs_*`, former names in the notes. **All 25 data sets are now
  converted.**

## Issue 5: the library reads only the files

- `QsimExperiment.recorded_derived_params()`: live program, else the file's
  `derived_params`. `mbr_saved.saved_parameters` reads only that; no `prog` duck typing, no
  recomputation, no sidecar.
- `MBRJobExperiment.from_h5file(fname, load_shots=True)`; `AssembledExperiment.from_manifest`
  loads children with it (no shots). The `timing=` escape hatch is gone everywhere.
- `experiments/saved_jobs.py` and `job_paths.job_records` moved to
  `tools/legacy_saved_jobs.py` (guan: freeze the converter in tools/). Five deprecated modules
  (kept, guan's choice) and their tests import it through the alias
  `experiments/qsim/deprecated/saved_jobs.py`, which loads the tools file by path (deprecated
  code may import live code; the guard `test_no_live_deprecated_imports` forbids the reverse,
  which a first attempt broke). The sidecar
  stays as a job-ID -> path index in `job_paths` (infrastructure, not physics data) and for
  `mbr_campaign.config_set_for_job`.
- `legacy` removed (`postprocess_reconstruction`, Spectrum, ensemble, pole registry,
  notebooks). The July sets were converted again with their analyzer sign recorded. Code
  history: the program added the correction (+1) from 631a958 (07-21 22:44) to daff734
  (07-31 13:04), then subtracted (-1); it records the sign from f554dd3 (08-03). All five July
  sets ran on 07-22..23. The converter now fills the sign from that history
  (`ANALYZER_SIGN_FLIP`). The converted files keep their names, so the old July manifests
  also read the sign. The output field `legacy_analyzer_migration` stays (it reports a +1
  sign), so the golden baselines are unchanged.
- `mbr_phase.saved_correction` no longer reads the old `offdiag_*` keys.
- Shim `experiments/qsim/floquet_dark_mode_readout.py` deleted (guan). Users switched to the
  owning modules, including one import line in `jonginn/qsim_wigner_jonginn.ipynb`
  (`KerrWaitProgramDark` from `deprecated/dark_scramble_legacy.py`). The worker records
  `program.__module__`, so submissions never depended on it. Three deprecated functions that
  reload it no longer run (header notes).
- Renamed: `mbr_spectroscopy_program.py` -> `dark_mode_scramble.py`,
  `dark_mode_broadband_ge_validation.py` -> `broadband_ge_validation.py`,
  `dark_mode_multiparity_chevron.py` -> `man_stor_multiparity_chevron.py`; the `[DarkScramble]`
  / `[DarkT1]` log tags of the generic `FloquetProgram` are now `[Floquet]`.
- Deleted (guan): `analysis_notebooks/guan/MBR_analysis.py`, `measurement_notebooks/guan/mbramsey.py`.
- The offline tests ("no database, no HTTP") now load `august_quickplot` through
  `from_manifest`.

## Left, by decision

- `station_floquet_hardware` stays: it predicts the timing of a live station before
  acquisition, which is not the saved-data path issue 5 is about.
- The test fixtures in `tests/mbr_reference.py` still run the frozen converter on raw files;
  they test the tool.
- Jonginn's `decay_investigation.ipynb` has its own raw loader and cycle-time formula. guan:
  in scope for today, after the conversion, possibly another session.
- Nothing is on `main` yet; merge `guan` into `main` when the device is idle and the `main`
  checkout is clean, then move `qsim-analysis` to it.

## Later: coherence vs Kerr, and the decay notebook

- guan asked for a rough Kerr vs coherence check (low, medium, high Kerr). Matrix Pencil pole
  widths gave a flat ~50 us with 2-4x scatter: they mix decoherence with unresolved splitting,
  so they are not a coherence measure here. The StarkCal closed pairs (model-free) show the
  expected direction but reach only 13 us. The raw diagonal returns against the coherent
  theory show it plainly: the revivals at ~40 and ~78 us shrink with Kerr and with photons in
  M1, and vanish at |K| = 52 kHz.
- jonginn's `decay_investigation.ipynb` used the same model as the library (checked: same
  Hamiltonian and signs); its trend is real. It fed the model the rough catalog-heading Kerr,
  computed the cycle time with its own formula, and dropped 30% of traces as poor fits.
  Decision (guan): replace it with a short example on the library, and delete it.
- New `analysis_notebooks/202609_qsim_migration/kerr_coherence.py`: per diagonal trace,
  `|A| = c |A_theory| exp(-t/T) + floor` (linear in c and floor, scan in T), recorded Kerr,
  file timing, no trace dropped; section 2 plots the closed pairs and takes a longer run.
  Median T (us) by n_M1 = 0/1/2/3: |K| 3.8 kHz (g15) 217/184/153/116; 20.5 kHz 127/102/84;
  45 kHz 62/74/34; 52 kHz (g29) 30/26/15/14. A lower bound (model error also lowers overlap).
- Next (guan, tomorrow, after recalibration): a closed-pair run to 100-200 us at three Kerr
  points, n_M1 = 0-3.
- Deleted (guan): branch `job_id_sorting_out` (local and GitHub; every commit is in `guan`)
  and its worktree `C:\python\multimode_expts_jonginn`. The worktree had no uncommitted work;
  its ignored `job_server/jobs.db` was empty and its `configs/versions/` held local snapshots
  that no recorded job uses.
- One branch (guan): `qsim-analysis` retired, local and GitHub (0 commits not in `guan`), and
  `origin/mcp` deleted (2025-06, guan's). The `qsim-analysis` worktree is unregistered; its
  empty folder stays locked by an orphan `tail` (pid 4420, a dead session's watcher from
  2026-10-02), which the auto-mode policy did not let me stop. The idle `nb:1` (survey)
  shell was moved to the `guan` folder. seb's and jonginn's branches and `tests-cleanup` (WIP)
  are left alone (guan).
