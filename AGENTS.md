# AGENTS.md

Shared instructions for coding agents (Codex, Claude Code, Cursor, …) working in
this repo. Human contributors: see `README.md` for install/run, and `docs/` for
long-form design specs. This file is for the things we otherwise re-explain at
the start of every session.

## Python setup

- **Always `pixi run python`, not bare `python`/`python3`.** The repo lives in a
  pixi environment; a bare interpreter picks up the wrong packages. Same for
  `pixi run pytest`, `pixi run jupyter`, etc.
- Tasks are defined in `pyproject.toml` under `[tool.pixi.tasks]`.
- Imports: a top-level folder is importable only if it is listed under
  `[tool.setuptools.packages.find] include`.


## Repo map

| Path | What lives there |
| --- | --- |
| `experiments/` | Experiment classes. Top level modules are infra. Subpackages are themed by physics. Typically each module contains one QICK Program and its related Experiment class, but not a hard rule. |
| `fitting/` | Reusable numerical analysis + display. `fitting/qsim/` |
| `job_server/` | FastAPI job queue, worker, client, config versioning, dashboard. See `job_server/README.md` |
| `slab/` | Vendored legacy `slab` code, kept in its original state. |
| `configs/` | YAML hardware/experiment configs, calibration CSVs, `versions/` snapshots managed by a versioning system |
| `analysis_notebooks/`, `measurement_notebooks/` | Notebooks with per-user sandboxing. Not a library — do not import from outside |
| `simulation/` | Historical standalone QuTiP notebooks. Largely outdated and to be deprecated. |
| `tests/` | pytest. |
| `docs/` | Design specs, architecture notes, historical worklogs etc, not structured documentation yet |
| `tools/` | One-off maintenance scripts |


## Notes

- Outside infra, the core of this repo consists the physics modules under `experiments/`. The canonical design pattern follows the protocol laid out in `slab/experiments`: one physics experiment consists of data acquisition code and the (naturally coupled) analysis and display code. However, much of the existing code breaks this pattern, sometimes with good reasons, often not. Exercise judgment.
- The life cycle of a measurement typically starts from instantiating a `MultimodeStation`. See notebooks (ipynb or jupytext .py) under the measurement_notebooks sandboxes for details and submitting jobs to the job server using one of the runners. The canonical general pattern for measurement submission was first laid out in the single qubit autocalibrate notebooks: define default config, customize pre- and post-job functions if needed, execute with kwarg overrides. Keep this pattern, wrap complex behaviors in proper modules and leave the notebooks as clean as reasonably possible.
- The canonical output is HDF5 files. Experiment object pickles are available for ephemeral debugging only. Do not rely on them for data persistence or read out.
- Nor is the `jobs.db` database a good place for durable physics data or record keeping: those should belong in HDF5 metadata. The full db only lives on one machine. Relying on it makes data exchange (for collaborator analysis or eventual publication) impossible.
- As a default, it's preferable to save aggregated/processed data as new HDF5s with provenance and param persistence rather than overwriting raw data files.


## Environment

This repo has the origin on GitHub and checked out on various different machines.
One special machine is the measurement PC (`pippin-meas`) that has real hardware
connections and holds the live runtime files (data, job queue, config versions etc).
We can refer to this as prod or pippin. Pippin is a Windows machine. As of 2026, python code including this repo lives under `C:\python\` on this machine.
Data (HDF5 files, logs, etc) live under `C:\experiments\`.
There is further an Obsidian vault under Google Drive (`G:\Shared drives\SLab\Multimode\Lab\`) that stores automatically logged measurement summaries and plots.

On prod, we typically use a job server and worker for multi-user job submission. The worker monopolizes hardware connections and executes jobs in sequence.
However, there are mock modes offered in station to use mocked hardware on/off prod to test new or buggy measurement code. One can also choose to bypass the job queue and directly dispatch code execution to connected hardware *if one knows this to be safe*.

Other checkouts might have their own, local env files or memory notes explaining how to access various resources on prod.
Prod is typically accessed via SSH overall and sometimes RDP.
Users can tunnel various prod ports to their own machine for various purposes: Jupyter server, SMB share for data files, FastAPI job server for job queue monitoring, etc. Google drive can also sync the logging vault directly.
Ask the user how to access prod resources if this is needed and not instructions are found.

At any rate, remote job submission is not set up yet so off-prod runs are typically for data analysis, initial prototyping of new measurement code via mock station ASM compilers etc.
Even on prod, job execution is still coupled to the canonical main branch at `C:\python\multimode_expts`. Worker will only pick up code there, not in other checked out worktrees.

