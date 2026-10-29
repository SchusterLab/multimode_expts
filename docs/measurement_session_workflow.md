# How we run a measurement session (conventions)

Status: draft (2026-10-29), from the first session of this kind (2026-10-28). This file: how we
work together on this PC: who does what, kernels, records, the order of a session. Some of it
is different for each user. Some parts need tools that are not built yet
(`docs/lab_workflow_tooling_spec.md`); those parts say "(planned)".

How the code and the device behave (the queue, mock mode, config branches, readout phase,
relax delays) is in `docs/running_experiments.md`. Read it first.

## Roles

- **Human:** chooses what to measure and makes the decisions. Edits the script and the vault
  notes.
- **Agent:** writes the cells, tests them on mock, submits, analyzes, and writes notes.
  Asks before: a new kind of measurement, a change to shared configs, editing other users'
  files, a git commit.
- **Vault logger** (`station.log_measurement`, called by the runner): appends one entry per job
  to the user's vault day file, with the parameters and the `display()` figure.

## Records: physics to the vault, code to the repo

The device changes every day (a TLS can be there today and gone tomorrow). That is too much
churn for git, and a status file that we must keep current by hand is bad for forensics. So:

- **The vault day file is the lab book** for everything about the device and the
  measurements: results, device state, config branch names and version IDs, decisions about
  what to measure, and what is open. Path:
  `<vault_root>/Lab/<user>/<project>/YYYY/MM/YYYY-MM-DD.md`, figures in `figures/` next to it;
  `vault_root` is `data_management.vault_root` in the hardware config (the Obsidian vault on
  `G:\Shared drives\SLab\Multimode\`).
  - Under each entry (or the last entry of a group of jobs), a note that starts with the
    author, e.g. `(<user>)` or `(Claude)`: what was run (job-ID range, script cell), the result in
    numbers, and what it means.
  - A reading that turns out wrong is not deleted: add `[Update <time>: withdrawn, because
    ...]` and point to the new entry.
  - Analysis figures for a group of jobs: `station.log_analysis(...)` (planned); until then
    `station.log_measurement(experiment=None, fig=..., title=..., notes=...)`.
- **The repo records code, not device state.** This has been changing in the past but current
  decision is that `docs/STATUS.md`, `docs/status/*.md` and
  `docs/log/` hold code changes, code bugs found, and decisions about code and tools. No
  physics results, no device numbers.
  - At the end of a work session that changes the code state, **overwrite** the status file of
    your theme (update its date; change `docs/STATUS.md` only for cross-theme items) and **add**
    one `docs/log/YYYY-MM-DD_<topic>.md`: what was done, decisions and who made them, what was
    found. Do not edit a log entry after its day; write a new one. A newer entry wins over an
    older one.
  - Older status files and log entries also have physics in them, from before this rule. Leave
    them as they are; for the device state, the vault wins.
- Editing the vault day file: use the anchor-based note writer (planned). Until then: change
  one line at a time, read the file just before you write, and do not write while one of your
  jobs is about to finish (the logger appends then).
- Figures in chat (CC board) only when the human asks; the vault and the kernel output have
  them.

## Kernels

Keep the station in a long-lived kernel, not in a script that you run again and again: the
station is stateful (see `docs/running_experiments.md`).

- **Measurement kernel** (one per user and session): station, runners, job submission. It can
  block while a cell waits for the queue; that is normal.
- **Analysis kernel:** local work only (load HDF5, fits, plots). It never waits for the queue.
  Use it while the measurement kernel waits.
- **Blocked kernel:** if a kernel that you need is busy with a long run, start a new kernel
  yourself (an analysis kernel, or one for a long optimizer or a big fit). Do not interrupt or
  restart a kernel that someone else is using.

Per user:

- **guan:** all kernels live in the always-on Lab server (port 8765; `nb`). Any client
  (nvim/iron, a Lab console with "Show All Kernel Activity", the agent) attaches to a kernel by
  its session name. Lab console output streams only to the clients that are connected at that
  time. The agent: find the session by name with `GET /api/sessions` (the token is in the
  `jpserver-*.json`; never print it) and attach a `jupyter_client.BlockingKernelClient` to
  `kernel-<id>.json`. `tools/kernel_exec.py` (planned) will do this.
- **seb, jonginn:** VSCode, mostly. (To agree: how the agent reaches their kernels.)

## The script

One jupytext percent script per investigation in the user's sandbox
(`measurement_notebooks/<user>/<topic>.py`), with numbered cells. While we explore, it can grow
helpers and prose. At the end we deliver:

- a short notebook: config -> defaults -> pre/postprocessor -> `runner.execute`, per step;
- a "module gaps" cell: each inline helper we needed, and where in the modules it should go.
  This is the input for a later code session.

## Order of a session

1. Read `docs/running_experiments.md`. Read `docs/STATUS.md` and the theme status file for the
   code state. Read the last vault day files of the project for the device state.
2. Cell 1: `JobClient().print_queue()`; the station pinned to the config versions that the
   human gives; a config branch for each set point (`BranchManager(station).commit(...)`).
3. Readout baseline, every day, for every set point (see "Readout phase" in
   `docs/running_experiments.md`); commit the config branch.
4. For each new cell: `station.use_mock_instruments()`, run it with few reps, then
   `station.use_real_instruments()`.
5. Submit through the runner, choose appropriate `batch_size` (so that when interleaving,
   each user gets their turn), few reps first; one sweep cell at a time.
6. Analyze in the analysis kernel. Do a direct, model-light test before any model fit; label
   model results as such.
7. Write the notes in the vault as you go. At the end: the "Day summary" in the vault.
