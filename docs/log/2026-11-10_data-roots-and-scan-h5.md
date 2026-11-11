# One module for data paths; scan HDF5s that analyze without the job files

guan with Claude, branch `theory`. Not committed at the time of writing. Follows `2026-11-10_agp-summary-and-load-time.md`.

## Why

- Old experiment folders were moved from pippin's `C:\experiments` to the slabnas copy (`/Volumes/slab/Multimode/BF5_expts_autosync` on the laptop); new data still lands on pippin. The one data root of `job_paths.py` could not see both.
- Many places found files in their own way: `job_paths.data_root()` joined by hand in ~30 notebooks and tools, `job_paths._reroot`, `agp.local_path`, `registry.manifest_path(data_set, root)`, `pole_data.load_spectra(data_set, root)`, `fixed_set.set_folder(root)`, `transduction.process_tomo.DATA_ROOTS` (a glob of `C:\experiments`), and a hard-coded `/Volumes/slab-1/...` path in a notebook.
- The summary notebook took about 8 min because `load_scan` read all ~960 job files over SMB; the scan HDF5 could not be analyzed again.

## What changed

- `experiments/data_paths.py` replaces `experiments/job_paths.py` (the old module is gone, no alias). A data file is `<experiment>/data|assembled_data|derived_data/...` below a data root. `MULTIMODE_DATA_ROOTS` in `.env` lists the roots (separated by `os.pathsep`); unset, it is `C:/experiments`. `locate(path)` takes a path below the root, a path recorded on another machine (`C:\experiments\...`, `.../BF5_expts_autosync/...`) or a local path, and returns the first root that has it; roots that are not mounted are skipped, and the error names them. One folder check per (root, experiment) per kernel keeps a root without that folder cheap. `resolve_job_paths` reads the provenance sidecar, then walks `*/data` of each root for the rest. The `index` and `vault` backends and `MULTIMODE_PATH_BACKEND` are removed (nothing used them). `MULTIMODE_DATA_ROOT` (old name) raises an error that says to rename it. `JobPathError` is now `DataPathError`, a `FileNotFoundError`.
- Callers: `data_root() / x` is `locate(x)` everywhere; `manifest_path(data_set)`, `load_spectra(data_set)`, `set_folder()` lost their root argument; `process_tomo.find_wigner_h5` uses `resolve_job_path`; `agp.local_path` is gone (`locate`); `read_manifest` looks up a listed file in the data roots when it is not beside the manifest.
- Scan HDF5: `CentralBosonAGPDisorderScan.save` also writes each W's `analysis_inputs()` stacked over W as `inputs_<key>` (populations as stacked by `_collect_populations`, recorded couplings, Kerr, calibration shots) and the attributes `parameters`, `part_job_ids`, `notes`; the scan now stacks `coupling_mhz` too. `from_h5file` makes parts from them (`frozen_inputs`, no jobs) that `analyze()` with any settings; older files still give the arrays-only view. `load_scan(key)` uses the HDF5 when it has the inputs, else the manifest (`jobs=True` forces the manifest). `rewrite_h5()` writes the HDF5 of a saved scan again from its jobs and moves the old HDF5 to `<experiment>/assembled_data_old_<YYYYMMDD>` (guan's decision: retire, do not overwrite; a retired file is never overwritten); `tools/rewrite_scan_h5.py` does it for registered scans. A scan without jobs cannot be saved under a new name.
- `agp.agp_table(scans, mu)`: the summary table that was inline in the notebook (value, noise, DC term and total error, usable, closed system with and at K = 0, signal, start ratio, tau). `agp_summary_20261110.py` now only loads, analyzes, calls `agp_table` and lays out figures.
- `.env.example` and this laptop's `.env` use `MULTIMODE_DATA_ROOTS`. Pippin has no `.env`, so it uses the default.

## Checked

- Tests: new `tests/test_data_paths.py` (replaces `test_job_paths.py`); in `tests/test_agp_scans.py` a scan from its HDF5 alone gives every curve the job files give, at three settings other than the saved ones, and `rewrite_h5` makes an old-style file analyzable. `test_agp_disorder_scan` now expects the inputs in the HDF5. The 3 known failures of `test_convert_fock_population_catalog.py` (`save()` needs a name) are unchanged.
- Real data, laptop: all 17 registered data sets (AGP and pole registry) resolve from slabnas in 1.1 s. For `kerr_w0` and `261105_coarse_dir5`, an HDF5 with inputs (written to a scratch folder, not the NAS) loads in 0.01 s and `analyze(background_start_us=30, mu=[10, 20])` equals the job-file analysis exactly.

## Done on the data (laptop, 22:20-22:32)

- `tools/rewrite_scan_h5.py` ran on all 11 registered scans (about 10 min, from the job files over SMB). The new scan HDF5s are in place; the 11 old ones are in `agp_k44_first_scan/assembled_data_old_20261110/` and `261027AGPlowKerr/assembled_data_old_20261110/` on slabnas, unchanged. No manifest or part file was touched.
- `agp_summary_20261110.py` then ran top to bottom as a script in 40 s (was about 8 min) and wrote its 31 figures.

## Open

- On pippin only pippin's own data is found: pippin cannot reach slabnas.
