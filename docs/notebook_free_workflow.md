# Notebook-free measurement workflow

How we run measurements without `.ipynb` notebooks: plain `# %%` Python edited in
nvim, executed against a persistent Jupyter **kernel**, with matplotlib figures
shown in a browser tab over HTTP instead of inline. This doc is both a **user
guide** (how to drive it day to day) and a **reproduction guide** (how to stand
it up for a new user or a future rig).

It is **additive and opt-in**: teammates who stay on JupyterLab or VSCode-remote
are completely unaffected (see *Shared-rig safety* below).

## Why

`.ipynb` files are the friction in our loop: git-hostile JSON, browser-bound
editing, and output churn that buries the code. We keep the part we actually
like — the persistent kernel (shared state that survives disconnect) — and drop
the notebook document and its inline-output display surface. The persistence is a
property of the **kernel layer**, not the notebook UI, so the two separate cleanly.

## Shared-rig safety (read this first)

Everyone runs kernels **on the same Windows measurement PC, from the same working
directory, in the same pixi env, under the same Windows account.** So git-tracking,
"personal env files", and shared shell-profile variables are **not** valid
isolation boundaries — they would leak to everyone.

The boundary that *does* work is **which kernel a process is**: a Jupyter kernel
knows its own connection file. This workflow self-gates on the connection-file
name. Concretely:

- You always launch your kernel with a fixed connection file, e.g.
  `~/.molten/kernels/guan-meas.json`.
- A **committed, shared** IPython startup file checks the connection-file name and
  does nothing unless it matches your marker. Teammates' kernels (VSCode,
  JupyterLab, their registered kernelspecs like `multimode_direct`) never match,
  so the startup file is inert for them — no backend change, no import, nothing.

This is verified in `tests/test_figure_autoshow.py` (real kernels launched with
matching / non-matching connection files).

## Daily use

1. **Start your kernel** (in a persistent multiplexer / psmux window, or detached
   so it survives disconnect — see *Persistent kernels* below):
   ```powershell
   pixi run python -m ipykernel -f $HOME\.molten\kernels\guan-meas.json
   ```
   The startup file forces the **Agg** backend and starts the figure server on
   **http://localhost:8099**.

2. **Open the figure viewer** once:
   - On the workstation over RDP: just open `http://localhost:8099`.
   - From a laptop over SSH: `ssh -L 8099:localhost:8099 <host>` then open it.
   The page polls and updates only changed images in place — no flicker.

3. **Attach nvim/molten** to the kernel (attach-by-name, see below) and run cells.
   Anything that produces a figure shows up in the browser tab:
   - `expt.display()`
   - a bare `plt.subplots(...)` / `plt.plot(...)` cell with no `show()`
   - ad hoc `plt.pcolormesh(some_array)`

   No code changes — a `post_run_cell` hook captures whatever figure is open at
   the end of each cell (exactly what the inline backend does) and ships it.

### Overnight safety

Under the Agg backend, `plt.show()` is an inert no-op, so the `plt.show()` calls
scattered through the codebase **never block** — safe for long unattended sweeps.
(This is why we use Agg, not the WebAgg backend, whose `plt.show()` blocks the
kernel until you close the browser figure.)

### Runner-driven figures (the one manual step)

The characterization/sweep runners capture their figure, log it to the PNG/obsidian
pipeline, and then **close it** (`characterization_runner.py:330`,
`sweep_runner.py`) — so by cell end there is nothing open for the hook to catch.
To see a runner figure, **call `.display()` yourself** in the cell, e.g. uncomment
`man_spec.display()`. Under Agg this is the single render path (the runner's
internal `plt.show()` is a no-op), so there is no double-render — the reason that
line was originally commented out no longer applies.

## How it works

```
Windows measurement PC  (same account / CWD / pixi env for everyone)
┌──────────────────────────────────────────────────────────────────┐
│ persistent kernel:  python -m ipykernel -f .../guan-meas.json      │
│   └─ IPython startup file gates on connection-file name            │
│        └─ if "guan-meas":  MPLBACKEND=Agg  +  register hook        │
│                                                                    │
│ nvim + molten  ──attach──▶ the same kernel (execution + text out)  │
│                                                                    │
│ post_run_cell hook ──▶ Sink:                                       │
│   HttpImageSink (default, stdlib)  →  http://localhost:8099        │
│   VisdomSink (optional dashboard)  →  visdom server                │
└──────────────────────────────────────────────────────────────────┘
     ▲ RDP: view in workstation browser      ▲ SSH -L: view on laptop
```

The capture mechanism (`measurement_notebooks/figure_autoshow.py`) reproduces the
inline backend's "flush open figures at end of cell," but routes each figure to a
pluggable **sink** instead of rendering inline. The sink is the only thing that
decides *where* the figure goes; the hook is sink-agnostic.

- **`HttpImageSink`** (default) — a tiny stdlib `http.server` in a daemon thread.
  Serves a static page that polls `/state` and swaps only the images whose version
  changed (no full-page reload). Named windows overwrite in place; distinct names
  accumulate. Zero dependencies, nothing to launch separately.
- **`VisdomSink`** (optional) — ships figures to a visdom dashboard instead, if you
  want drag/resize window management. Requires a running `visdom.server` and the
  `visdom` package (installs via `pixi add visdom` / conda — **not** pip, whose
  source build fails on modern setuptools). Switching is one line in the startup
  file.

## Reproducing this for a new user / rig

1. **Pick a connection-file marker** unique to you, e.g. `alice-meas`. You will
   always launch your kernel with `-f ~/.molten/kernels/alice-meas.json`.

2. **Add a gated IPython startup file** at
   `~/.ipython/profile_default/startup/50-autoshow.py`. It must (a) check the
   connection-file name and return early if it doesn't match, and (b) only then
   set Agg + enable the hook. See the existing file for the exact shape; the gist:
   ```python
   # gate: only this kernel, inert for everyone else on the shared account
   if _connfile_stem() == "alice-meas":
       import matplotlib; matplotlib.use("Agg", force=True)
       import figure_autoshow as fa
       fa.enable(sink=fa.HttpImageSink(port=8099, host="127.0.0.1"))
   ```
   Pick a **distinct port** per concurrent gated kernel if more than one person
   uses this on the same box.

3. **Choose your sink** — `HttpImageSink` (default, nothing else to run) or
   `VisdomSink` (then also launch `pixi run python -m visdom.server`).

4. **View** at `http://localhost:<port>` (RDP) or via `ssh -L <port>:localhost:<port>`.

Implementation + tests: `measurement_notebooks/figure_autoshow.py`,
`tests/test_figure_autoshow.py`.

## Persistent kernels (survive disconnect)

Two complementary mechanisms on Windows:
- **psmux** holds nvim (and optionally the kernels) across **SSH** disconnect.
- **RDP** sessions persist server-side natively — apps keep running when you
  disconnect the remote desktop.

Most robust: launch kernels as **detached background processes** (survive even a
psmux/nvim restart) and run nvim inside psmux. Keep kernel *creation* out of nvim —
a molten-spawned kernel dies with nvim, defeating persistence. Wrap kernel launch
in a `start-measurement-kernels.ps1` so "start my session" is one command.

## Editor: nvim + molten, attach-by-name

Use molten for **execution + text output only** — no inline terminal images
(kitty graphics is poorly supported on Windows terminals / over SSH; display goes
over HTTP instead). Change the molten keybind to **attach** to the externally
started kernel rather than spawn one molten owns (a spawned kernel dies with nvim):

- Manual: `:MoltenInit <path-to-connection-file>.json` attaches without
  terminating the kernel when the buffer/nvim closes.
- Auto "match the script to its kernel" — a `BufEnter` autocmd on `*.py`:
  ```lua
  local name = vim.fn.expand("%:t:r")
  local cf = vim.fn.expand("~/.molten/kernels/" .. name .. ".json")
  if vim.fn.filereadable(cf) == 1 then
    vim.cmd("MoltenInit " .. cf)            -- attach, never spawn
  else
    vim.notify("No persistent kernel for " .. name .. " — launch it in psmux first")
  end
  ```
Keep a second binding (`MoltenInit python3` / `:MoltenRestart`) for a deliberate
clean slate.

## Known limitations

- **Runner figures need an explicit `.display()`** (see above) — the only manual
  step vs. inline.
- **Figures appear in a browser tab, not under the cell.** Fidelity is identical
  to inline (both static raster); the difference is purely where you look.
- **One gated kernel ↔ one fixed port.** If you run multiple gated kernels at once,
  give each its own port.
- **Archival is unchanged** — the existing PNG/obsidian pipeline remains the record
  of figures; the HTTP viewer is for live inspection, not storage.

## Why not the alternatives

- **WebAgg backend** — its `plt.show()` blocks the kernel until the browser figure
  is closed, which would freeze a shared persistent kernel and any overnight sweep.
  Rejected in favor of Agg + the off-terminal hook.
- **RTC / jupyter-collaboration** (bridging an agent into a live notebook) — the
  open disconnect/reconnect data-loss failure mode is exactly what an external
  client triggers. This workflow has no such fragility: code is a `.py` on disk,
  data is HDF5+PNG on disk, the kernel is a plain process — reattach and state is
  exactly where you left it.
