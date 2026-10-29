# Spec: tooling for shared-kernel measurement sessions

Status: A-D implemented 2026-10-29 (`docs/log/2026-10-29_session-tooling.md`); B's schema is
guan's decision and may change after use in Obsidian. D: the mock checks only; no runner
preflight (guan: the mock at program composition catches these). Item E is for discussion only.
Original status: draft (guan + Claude, 2026-10-29). Origin: the 2026-10-28 session (`docs/log/2026-10-28_fock-prep-f-decay.md`),
where an agent and guan worked in the same kernel and the same vault day file.

Four tools (A-D) and one design note (E). Each one can be done and tested alone.

## A. `tools/kernel_exec.py`: run code in a named kernel

Why: on 10-28 the agent ran cells with awk/heredoc extraction. That merged two cells once, broke a
`\n` once and two Windows paths (`\2` became a control character). The helper lived in a session
scratchpad and was lost at a session restart.

- Server: the newest `jpserver-*.json` in `%APPDATA%\jupyter\runtime` with `port == 8765`
  (option `--port`). Read the token from it; **never print or log the token**.
- `list`: the server sessions (name, kernel id, state, connections, path).
- `start --name <n> --dir <checkout dir>`: `POST /api/sessions` with `path=<dir rel. to
  root>/<n>`, `type=console`, `kernel.name=pixi-cwd` (same as nvim `kernels.lua` `M.start`), then
  wait until the kernel answers. For an analysis kernel or a kernel for a long job.
- `run --kernel <n>` with one of: `--script X.py --cell <index | title prefix>`, `--file F.py`,
  `--code "..."`. Cells: jupytext percent format (`# %%` markers; `[markdown]` cells skipped).
  Stream stdout/stderr/results/tracebacks to the terminal and to a log file; `--timeout`.
  `--no-wait`: send and return (the output goes to the log only).
- `interrupt --kernel <n>`: `POST /api/kernels/<id>/interrupt`.
- Tests: the cell splitter on a fixture script (indices, titles, markdown cells, a cell that ends
  the file); `run --code` against a throwaway kernel started by the test (skip if no server).

## B. Vault day file: a schema, anchors, and a safe note writer

Why: the logger appends (safe), but notes were written by reading and rewriting the whole file.
A logger append between the read and the write would be lost; Drive sync from a laptop is a
second writer. `_add interpretation here_` is not unique, so notes had to be found by job ID.

- Schema (to agree with seb, who wrote the current format): front matter; one `##` section per
  entry. Entry kinds: `job` (today's entries), `group` (see E), `analysis` (see C). Each entry
  ends with a notes block between unique anchors, e.g.
  `<!-- notes:JOB-20261028-00272 -->` ... `<!-- /notes:JOB-20261028-00272 -->`
  (analysis and group entries get their own ids). Note lines start with the author:
  `(guan)`, `(Claude)`. A withdrawn reading is not deleted; it gets `[withdrawn <date>: why]`.
- `vault.add_note(day_file, anchor_id, text, author)`: read, check the anchor is there once,
  insert before the closing anchor, write to a temp file in the same folder and `os.replace`;
  before the replace, read the file again and retry if it changed (mtime or hash). Same path
  for the logger's own appends (append stays append).
- Check: the Drive client on pippin with `os.replace` on `G:` (it must keep the file id, or
  Obsidian on other machines sees delete + create).
- Tests: concurrent append during a note write (thread), a missing anchor, two notes in a row.

## C. `station.log_analysis(...)`: analysis entries in the vault day

Why: the most useful 10-28 figures (ladder populations, the NNLS unmixing, the 0.45 vs 0.25 mA
comparison) belong to groups of jobs, not to one job. They were inline in the kernel only, and a
client that was offline lost them.

- Thin layer on `station.log_measurement` mode B (`experiment=None`, `fig`, `title`,
  `parameters`, `notes`), which exists: `log_analysis(fig, title, job_ids, note="",
  parameters=None, data_files=None)`.
- Writes an `analysis` entry (B schema) at the end of the day file (append-only), with: the job
  IDs as a range and as links to their entries, the script and cell it came from (if known), the
  figure in the vault `figures/` folder, the parameters, and an empty notes block.
- Not in scope: a second plot folder. `station.plot_path` stays as it is.

## D. Board resource checks in the mock and before submit

Why: compiling against `soccfg` checks only what QICK checks when it builds the ASM. The resource
limits are checked only by the board drivers at upload, and `MockQickSoc` turns those calls into
no-ops. On 10-28 three jobs failed at upload (JOB-20261028-00775, -00777, -00780): envelope memory.

- One function, `check_program_resources(prog, soccfg)`, that replays the driver checks with the
  limits in `soccfg`:
  - per generator: sum of envelope lengths (`prog.envelopes[ch]['next_addr']`) <= `maxlen`
    (`qick/drivers/generator.py:131`);
  - mux generator: number of tones (`generator.py:416`);
  - readouts: accumulated and decimated lengths <= `avg_maxlen`, `buf_maxlen`
    (`drivers/readout.py:693, 756, 776`);
  - tProc: program and data memory <= `pmem_size`, `dmem_size`.
- Called by `MockQickSoc` (where the real soc would load), and by the runner before submit (a
  cheap preflight on the real station too).
- Error text names the channel, the envelopes on it and their sizes (on 10-28: ch 2 had 4992
  samples of standard pulses, 896 per prepulse gate, plus the probe).
- Tests: the 10-28 case: a Gaussian probe sigma 1.05 us after a Fock 1 prepulse fails; sigma 0.8 us
  after the |f,2> prefix (8 qubit gates + wait) passes (30784 of 32768 samples).
- Separate, larger change for the pulse-layer overhaul: identical envelopes are not shared between
  prepulse gates (`custom_pulse` adds one per gate).

## E. Design note (for discussion): one vault entry per job group

Why: `runner.execute(overrides=[...])` writes one vault entry per job (10-28: 10 for the ladder,
22 for the Rabi scan, 18 for the holds). The useful view is the group.

Options:
1. One `group` entry per `execute(overrides=...)` call: the job-ID range, the varied keys and
   their values as a table, the per-job figures as a small grid or folded callouts, one notes
   block. No per-job entries.
2. Per-job entries as now, under a `group` heading with its own notes block.
3. As now, plus a `group` entry at the end (the per-job entries stay).

Questions: what seb and jonginn use the per-job entries for; whether the group figure is the
runner's job (generic grid) or the caller's (via C).
