# AGP re-analysis of all scans, and why loading takes minutes

guan with Claude, branch `theory`. Not committed at the time of writing.

## What was done

- Five new entries in `analysis_notebooks/datasets/agp_scans.yaml` (in the folder
  `261027AGPlowKerr/assembled_data/`): `261105_S5_direction5`, `261105_coarse_dir5`,
  `261106_Storage_directions0`, `261106_Storage_directions5`, `261106_Storage_directions7`.
- New notebook `analysis_notebooks/guan/agp/agp_summary_20261110.py` (jupytext percent). It
  loads nine scans by key (`SCANS`), runs `scan.analyze(background_start_us=30.)`, saves the
  module display stages, prints one table per scan (value, total error, usable, closed-system
  theory, K = 0 theory, signal-to-noise, start ratio C1_corrected[0] / C1_theory[0], fitted
  tau), and draws four summary figures into `figures_20261110/` (git-ignored). The table and
  the four summary plots are written inline in the notebook; moving them into
  `experiments/qsim/agp.py` was offered, not done (guan to decide).
- The results went into a deck, "AGP update 11-10", on claude.ai:
  https://claude.ai/artifact/U3miLCq9jrxnrUJ4TSgjt8 (8 slides; read it with the Artifact tool).
  Its sources were in a session scratchpad under `/private/tmp`, which is temporary; the
  figures come again from the notebook. The physics results are in the deck, not in the vault
  day file yet.

## Found: the load is slow because it reads every job file

- `load_scan` (`CentralBosonAGPDisorderScan.from_manifest`) goes scan manifest -> part
  manifests -> `FockPopulationExperiment.from_h5file(raw, load_shots=False)` for each raw job
  file. Each job file is 6.8 MB (6.4 MB single shots, skipped). Over the slab SMB share the
  cost is about 0.5 s per file from latency (opens and many small reads), not from size. One
  50-job scan took 26.5 s; the nine scans have about 960 job files, so about 8 min.
- The aggregate `CentralBosonAGPDisorderScan_<name>.h5` (about 0.4 MB) loads in 0.5 s but has
  only the outputs of its save (C1, AGP, spectrum at the settings of that day) and no children.
  `analyze()` on it raises `IndexError`. So a re-analysis with new code or settings must read
  all job files.
- Proposed fix (AGENTS.md: "fix that on the data side"): store in the aggregate what
  `analyze_correlations` and `analyze` need (stacked populations and their readout correction,
  shot counts, confusion matrix, timing, recorded couplings and Kerr), with a loader that
  analyzes without the job files; then save the aggregates again on pippin. Not started; guan
  to decide between this and running the notebook on pippin.

## Small things seen

- The scan data does not stack `coupling_mhz`; the notebook reads it from
  `scan.children[i].data["coupling_mhz"]`.
- `mount-share slab status` said "not mounted" while `/Volumes/slab` was readable; one run
  failed with `OSError: Device not configured` when the share dropped, the next worked.

## Open, not code

- All scans, also those in the "low Kerr" folder, record K = -27.13 kHz. If the set point of the
  11-05..07 runs was different, the theory curves of those scans are wrong (guan to confirm).
- Uncommitted files that are guan's own work, not part of this entry: the diff in
  `analysis_notebooks/guan/agp/agp_scans.py`, `.prettierrc`,
  `analysis_notebooks/guan/agp/understanding/disorder_average.py`.
