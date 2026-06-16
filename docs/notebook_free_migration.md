# Notebook-free measurement workflow — migration plan

Status: **proposal / not yet started**. Target host: the **Windows measurement PC**
(all persistent processes run there; a Mac/laptop is only a thin client).

## Why

`.ipynb` notebooks are the friction in our loop: JSON that's git-hostile, a
browser-bound editing UX, output churn that buries the code that actually
matters, and an AI-authoring story (Notebook Intelligence) that bills against
API tokens our team plan doesn't cover. We want a notebook-free Python workflow:
plain `# %%` scripts edited in nvim, run against a persistent Jupyter kernel,
with plots displayed over HTTP and figures already archived by our existing
PNG → obsidian pipeline.

Crucially, this is **additive and opt-in** — teammates who stay on JupyterLab
(web client or VSCode SSH-remote) are unaffected. See "Teammate compatibility".

## What we keep vs. drop

| Keep | Drop |
|---|---|
| The Jupyter **kernel** (persistent compute, shared state) | The `.ipynb` document + browser editing UX |
| Session persistence across disconnect | Reliance on inline-output as the record |
| Our figure-capture + PNG/obsidian logging (already object-based) | Inline cell rendering as the display surface |
| `jupyter_kernel_mcp.py` agent access (attaches to the same kernel) | Notebook Intelligence (API-token billing) |

The persistence we like is a property of the **kernel layer**, not the notebook
UI. Those layers are independent, so we keep the first and drop the second.

## Architecture (four layers)

```
Windows measurement PC
┌──────────────────────────────────────────────────────────────┐
│ psmux (persistent multiplexer) — survives SSH/RDP disconnect   │
│   ├─ window: persistent kernels (one per measurement context)  │
│   │     pixi run python -m ipykernel -f <fixed connection file>│
│   └─ window: nvim + molten  ──attach──▶ matching kernel        │
│                                                                │
│ Display servers (HTTP):                                        │
│   ├─ matplotlib WebAgg   — interactive inspection of a figure  │
│   └─ visdom              — live-monitor loop (named-window)    │
│                                                                │
│ jupyter_kernel_mcp.py ──attach──▶ same kernels (Claude/agents) │
└──────────────────────────────────────────────────────────────┘
        ▲ SSH (+ -L port-forward)            ▲ RDP
        │ view WebAgg/visdom in laptop       │ view WebAgg/visdom in the
        │ browser                            │ workstation's own browser
   Mac/laptop (thin client, VPN)
```

One persistent kernel can serve **molten + the MCP + a `jupyter console`**
simultaneously — Jupyter kernels accept multiple clients sharing one namespace.

## Layer 1 — Persistent kernels (Windows)

One long-lived kernel per measurement context, each pinned to a **fixed,
named connection file** so we can match it to a script by name (Layer 2). The
fixed path is what makes name-matching trivial — we avoid Jupyter's random
kernel-ids entirely.

Connection files live in a known dir, e.g. `%USERPROFILE%\.molten\kernels\`:

```powershell
# launched inside a dedicated psmux window (or as detached background
# processes via Start-Process pythonw, for survival across psmux restarts)
pixi run python -m ipykernel -f $HOME\.molten\kernels\qubit_cal.json
pixi run python -m ipykernel -f $HOME\.molten\kernels\cavity_spec.json
pixi run python -m ipykernel -f $HOME\.molten\kernels\readout.json
```

Wrap this in a `start-measurement-kernels.ps1` and/or a psmux/Windows-Terminal
layout so "start my session" is one command — this replaces "open N notebooks
in JupyterLab".

**Persistence model on Windows.** Two complementary mechanisms:
- **psmux** holds nvim (and optionally the kernels) across **SSH** disconnect.
- **RDP** sessions persist server-side natively — apps keep running when you
  disconnect the remote desktop. So RDP alone already gives "the server never
  dies" for anything launched in the desktop session.

Most robust combo: launch kernels as **detached background processes** (survive
even a psmux/nvim restart) and run nvim inside psmux. Keep kernel *creation*
out of nvim — a molten-spawned kernel dies with nvim, defeating persistence.

## Layer 2 — Editor: nvim + molten, attach-by-name

Today's keybind runs `MoltenInit python3`, which **spawns** a kernel molten owns
(dies with nvim). For persistence we change what the keybind *does*: **attach**
to the externally-started kernel instead.

- Manual: `:MoltenInit <path-to-connection-file>.json` attaches; the kernel is
  *not* terminated when the buffer/nvim closes
  (see molten docs: Initialization.md, the "shared"/external-kernel form).
- Auto "match the notebook name" — a `BufEnter` autocmd on `*.py`:

```lua
-- open qubit_cal.py  ->  attach to ~/.molten/kernels/qubit_cal.json
local name = vim.fn.expand("%:t:r")
local cf = vim.fn.expand("~/.molten/kernels/" .. name .. ".json")
if vim.fn.filereadable(cf) == 1 then
  vim.cmd("MoltenInit " .. cf)            -- attach, never spawn
else
  vim.notify("No persistent kernel for " .. name .. " — launch it in psmux first")
end
```

Keep **two** bindings, same muscle memory, two intents:
1. attach-to-persistent (new default), and
2. `MoltenInit python3` / `:MoltenRestart` for a deliberate clean slate.

**Windows note on molten:** use molten for **execution + text output only**.
We are *not* using inline terminal images (kitty graphics is poorly supported on
Windows terminals) — display goes over HTTP (Layer 3). This actually makes the
Windows story simpler, not worse.

## Layer 3 — Display over HTTP

Two distinct needs; two tools. (Archival is already solved by the PNG/obsidian
pipeline — don't rebuild it.)

- **matplotlib WebAgg** — interactive (pan/zoom) inspection of a *finished*
  figure. Native matplotlib, zero new deps: set `MPLBACKEND=webagg`. `plt.show()`
  serves the figure in a browser tab. Weak at continuous redraw (its `show()` is
  blocking), so not for live loops.
- **visdom (fossasia fork, maintained)** — the live-monitor loop. `vis.matplot(fig,
  win="live")` overwrites a named window in place — the robust, off-terminal
  version of today's `clear_output(wait=True)+display(fig)` hack.

**Viewing:**
- **RDP mode** (matches current habit): browser runs on the workstation, viewed
  through RDP — **no port-forwarding needed** (server and browser are the same
  machine). Simplest.
- **SSH mode** (from anywhere, lightweight): `ssh -L 8097:localhost:8097 ...`
  (visdom) / WebAgg port, view in the laptop browser. Windows OpenSSH client
  supports `-L`; use `ServerAliveInterval` (or an autossh-style reconnect loop)
  for resilience over VPN.

## Layer 4 — `live_update` dispatcher (the only real code change)

The single Jupyter-coupled idiom is `clear_output(wait=True)+display(fig)` in the
live-monitor path. Replace the call sites with a helper whose **default branch
reproduces today's inline behavior exactly**, so teammates are unaffected:

```python
def live_update(fig, win="live"):
    if _visdom_configured():          # opt-in via env — you
        _vis.matplot(fig, win=win)
    elif _in_ipython():               # teammates — unchanged path
        from IPython.display import clear_output, display
        clear_output(wait=True); display(fig)
    else:                             # plain script / webagg
        fig.canvas.draw_idle(); plt.pause(0.001)
```

### Code-change inventory

- **New:** `live_update` helper (location TBD — likely `slab/` or a small
  `display_utils.py`).
- **Swap 3 call sites** (`clear_output+display` → `live_update`):
  - `experiments/sweep_runner.py:232` (`_do_live_plot`)
  - `experiments/sequential_experiment_classes.py:135, 191, 365`
  - `fitting/wigner.py:608-609`
- **`.claude/jupyter_kernel_mcp.py`** — `RUNTIME_DIR` is hardcoded to macOS
  (`~/Library/Jupyter/runtime`). Make it platform-aware via
  `jupyter_core.paths.jupyter_runtime_dir()` before running the MCP on Windows.
- **No change** to the runner figure-capture logic
  (`characterization_runner.py:296-305`, `sweep_runner.py:281-294`) — it already
  captures `Figure` objects via `plt.get_fignums()` diffing and a `plt.show`
  no-op patch. This is backend-agnostic and *helps* under WebAgg (a live
  `plt.show()` would otherwise block).
- **No change** to `fit_display.py` / `fit_display_classes.py` /
  `Experiment.display()` contract.

## Teammate compatibility (non-breaking rules)

1. **No import-time backend coupling exists today** (verified: only function-
   scoped `matplotlib.use("Agg")` in the job-server path; nothing in
   `experiments/__init__.py`, `station.py`, `slab/`). Importing the package never
   changes anyone's backend.
2. **Backend is per-process.** Set `MPLBACKEND=webagg` only in *your personal,
   uncommitted* env (shell profile / gitignored env file) — **never** in a
   committed pixi/`.env` config, or teammates would get yanked off inline.
3. **`live_update` defaults to the inline path** under IPython, so JupyterLab and
   VSCode-remote users see byte-for-byte today's behavior. visdom/webagg branches
   are env-gated and new.

Net shared-code footprint: one new helper + three call-site swaps + one MCP path
fix. Everything else is personal env and editor config.

## Why this beats the RTC alternative

We considered bridging an external agent into the live notebook via
jupyter-collaboration / RTC. Rejected for now: the disconnect/reconnect
data-loss failure mode (jupyter-collaboration #219) is still open and is exactly
the scenario an external client triggers. This plan has no such fragility — code
is a `.py` on disk (git), data is HDF5+PNG on disk, the kernel is a plain
process. Reattach and state is exactly where you left it.

## Phased rollout (cheapest validation first)

1. **Validate the premise (no code changes).** On the Windows PC: set
   `MPLBACKEND=webagg` in your env, start one `ipykernel` with a fixed connection
   file, run an existing `experiment.display()` through it, confirm a figure
   renders in a browser (RDP-local first, then over `ssh -L`).
2. **Persistent kernel + molten attach.** Write `start-measurement-kernels.ps1`
   and the `BufEnter` attach autocmd; confirm open-script → auto-attach, and that
   the kernel survives nvim close + psmux detach/reattach.
3. **MCP on Windows.** Fix `RUNTIME_DIR`; confirm the kernel-MCP attaches to the
   same named kernels (agents + molten sharing one namespace).
4. **`live_update` dispatcher.** Add helper, swap the 3 call sites, verify both
   paths: your visdom/webagg render *and* the preserved inline path (run a cell
   under a JupyterLab kernel to confirm teammates are unbroken).
5. **Live-monitor on visdom.** Point `_do_live_plot` at visdom; validate a real
   sweep updates the named window live.
6. **Convert one real routine** end-to-end (e.g. a single-qubit calibration) to a
   `# %%` script and run a full session notebook-free before broadening.

## Open questions / risks

- **psmux maturity** for holding long-lived kernels on Windows — validate detach/
  reattach and process survival early (step 2). Fallback: detached background
  processes + RDP-session persistence.
- **visdom longevity** — maintained but not thriving; acceptable as the live sink,
  and replaceable (the `live_update` seam isolates it).
- **Multi-kernel daily friction** — explicit kernel launch is the one convenience
  tax vs. Jupyter's open-notebook-=-kernel. Mitigated by the one-command
  bootstrap script; re-evaluate after a week of real use.
- **`# %%` outputs aren't stored in-file** — accepted; the PNG/obsidian pipeline
  is the record. Consider `jupytext` pairing only if a teammate needs an `.ipynb`
  handoff.
