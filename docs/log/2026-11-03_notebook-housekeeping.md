# 2026-11-03: move level_stats and pole_finding under analysis_notebooks/guan/

Branch `theory`. Asked by guan; done by Claude.

## What was done

- `analysis_notebooks/level_stats/` -> `analysis_notebooks/guan/level_stats/`, with its ignored
  `labbook/cache/` (2 MB of model results).
- `analysis_notebooks/pole_finding/` -> `analysis_notebooks/guan/pole_finding/`.
- The pole-finding data-set registry `pole_finding/registry.yaml` ->
  `analysis_notebooks/datasets/pole_registry.yaml`, because a module reads it
  (`fitting.qsim.poles.registry.REGISTRY`). Modules may read only the registries in `datasets/` (AGENTS.md).
- Path fixes: `fitting/qsim/poles/registry.py` (`REGISTRY`), `tools/pole_report.py` (`REPORT`),
  `views_sep10.py` (`../../../docs/qsim/figures/pole_finding`). Path text changed in the notebook comments
  and `Run:` lines, `mbr_datasets.yaml` and its generator `tools/build_mbr_dataset_catalog.py`,
  `docs/qsim/pole_finding.md`, `docs/qsim/pole_finding_explore.md`, `docs/status/qsim_numerics.md`.
  The older `docs/log/` entries keep the old paths.

- Deleted `postprocess_manifests/` (6 JSON files of 2026-09-09 and 09-15; guan agreed). No code read them:
  they were the optional report of the deprecated `experiments/qsim/deprecated/mbr_disorder_h5.py`, written by
  `dormant/mbr_disorder_offdiag.py`, which now has `manifest_directory = None`. Every job ID in them is in
  `analysis_notebooks/datasets/mbr_datasets.yaml`, except JOB-20260907-00394..00405: a partial realization 3
  (11 jobs) in a snapshot made while the Sep07 run was in progress. The registry's realization 3 starts at 00406.

## Checks

- Pole tests (`test_poles`, `test_pole_finding_spec`, `test_gap_score`, `test_hamiltonian_fit`,
  `test_pole_bench_ideal`): 53 passed, 2 xfailed. `load_registry()` reads 6 entries; `pole_report.REPORT` exists.

## Found

- A Windows folder rename fails with "Permission denied" when a process has its working folder in that folder.
  Here, an orphaned `tail -f idle_check.log | grep` monitor from the 2026-10-25 idle check kept
  `level_stats/labbook` open. guan stopped it. Five more orphaned `tail -f` processes on the same log
  remain (they do not hold a repo folder).
