# Config branches & tags

`experiments/branch_manager.py` gives friendly, movable names to sets of config
versions, so you stop copy-pasting `CFG-...` IDs to track "which setup is which."
A **branch** is purely a mnemonic: a name → the 4 station config version IDs
(hardware / multiphoton / man1 / floquet). It stores no config data of its own —
only pointers into the existing `configs/versions/` store, which stays the source
of truth.

It is deliberately **local and non-intrusive**: the job server and worker never
see branches. They keep receiving full station YAML exactly as before. A branch is
a notepad for *you*, layered on top of the versioning system that already exists.

## Where the log lives (and why concurrency isn't an issue)

By default the backing file is `branches.jsonl` **in the current working
directory** — i.e. each user's sandbox notebook dir (`measurement_notebooks/<user>/`),
since a Jupyter kernel's CWD is the notebook's own folder. So every user gets their
own log and concurrent users **never contend for one file**. The config *version
IDs* inside still resolve against the shared version store + jobs DB, so the
pointers stay meaningful; only the index is per-workspace.

Consequences (intended for v1):

- **Branches are per-workspace.** User2 cannot `checkout` user1's branch — they
  read different files. For "each user tracks their own setups" that's the desired
  behavior, not a limitation.
- **One residual case:** the same user with two kernels open *in the same dir*
  shares one file. This is bounded and recoverable — the config data is already
  safe in the DB, and torn lines are skipped on read (see below) — so it's left
  unguarded in v1.

> **Cross-notebook rule.** A `checkout` only sees a branch that was committed from
> the *same working directory*. Two notebooks in the **same dir** share one
> `branches.jsonl`, so committing the branch in notebook A and then
> `checkout`-ing it in notebook B picks up A's new snapshots — this is the normal
> "calibrate in one notebook, use the values in another" flow and it just works.
> Two notebooks in **different dirs** have separate logs: B won't find A's branch
> (or may load a stale same-named one from its own log). The underlying snapshots
> are always shared via the version store/DB; it's only the branch *name → IDs*
> index that is per-directory.

To get a single lab-wide log instead, pass an explicit shared path:
`BranchManager(station, log_path=station.config_dir / "branches.jsonl")`. That
reintroduces cross-process contention on appends; revisit with a file lock
(`filelock` is available in the env) if it ever bites.

## Mental model: it's a reflog

The log is **append-only**. Every pointer move is one JSON line. The "current"
value of a branch is just its most recent line — nothing is ever overwritten, so
**history can't be lost**. This is the same idea as git's reflog: the current value
and the full history are the same structure, read two ways.

```
{"branch":"user1_coupler0.1","op":"commit","ids":{...},"note":"baseline","ts":"..."}
{"branch":"user1_coupler0.1","op":"commit","ids":{...},"note":"after retune","ts":"..."}
{"branch":"paper_v1","op":"branch","from":"user1_coupler0.1","ids":{...},"ts":"..."}
```

**Branch vs tag** is convention, not a separate type: a *tag* is a frozen entry
(`commit` refuses to advance it); a *branch* is one you keep committing onto. To
fork a tag, `branch()` off it.

## Usage

```python
from experiments.branch_manager import BranchManager

bm = BranchManager(station)            # uses configs/branches.jsonl by default

bm.commit("user1_coupler0.1", note="baseline")   # snapshot station -> advance pointer
# ... retune, run experiments (station.hardware_cfg gets edited) ...
bm.commit("user1_coupler0.1", note="after coupler retune")   # drag the pointer forward

bm.branch("paper_v1")                  # fork a new line off the current state
bm.tag("paper_v1_frozen")              # ...or freeze the current state immutably

bm.checkout("user1_coupler0.1")        # reload that setup into the station
bm.list()                              # {branch: latest_event} for all live names
bm.show("paper_v1")                    # the 4 current IDs for that name
bm.log("user1_coupler0.1")             # full history, oldest -> newest
bm.delete("paper_v1_frozen")           # tombstone (history stays in the log)
```

### What each verb does

| Verb | Effect |
|---|---|
| `commit(name, note="")` | Snapshot the 4 in-RAM configs (dedup makes unchanged ones free), append a `commit` event advancing `name`. Refuses if `name` is a tag. |
| `checkout(name, force=False)` | Reload the station's 4 configs from `name`'s latest IDs. Raises if the station has uncommitted edits unless `force=True`. |
| `branch(new, from_name=None)` | Create `new` pointing at `from_name`'s IDs (default: currently-loaded IDs); records a `from` lineage breadcrumb. Doesn't touch station RAM. |
| `tag(name, note="")` | Freeze the currently-loaded IDs under `name`; `commit` will later refuse it. |
| `delete(name)` | Append a tombstone; `list`/`show` stop resolving it but `log` keeps the history. |
| `list` / `show` / `log` | Read-only views (fold the log). |

## How it rides on the existing system

- `commit` calls `station.update_all_station_snapshots(update_main=False)` — the
  same snapshot path used elsewhere, just without touching `main`. Dedup by
  checksum means re-committing unchanged configs costs nothing.
- `checkout` calls the station's own loader (`_initialize_configs`), which already
  accepts `CFG-...` IDs and re-attaches the swap datasets.
- The dirty guard compares a cheap content fingerprint of the in-RAM configs
  against the state as loaded — no DB writes, just enough to avoid silently
  discarding un-committed edits.

## Relationship to `main`

Branches are orthogonal to the server's `main` pointer. `commit` never moves
`main`; the server keeps advancing `main` on its own as before. There is
intentionally no "promote my branch to `main`" verb — if that's ever wanted it's a
thin addition (`set_main_version` for each of the 4 IDs), kept opt-in.

## Limits (cheapest-thing-that-works; revisit if painful)

- **Single-machine scope.** `branches.jsonl` only resolves where the version store
  (`configs/versions/`) and jobs DB live. It's pointers, not data — emailing it to
  a collaborator won't resolve on their machine.
- **Private-method reach.** `checkout` uses `station._initialize_configs` directly.
  If this surface churns, extract a public `station.reload_configs(...)`.
- **No DAG / merge.** Lineage is a single `from` breadcrumb, not a full graph.
  Multiple independent moving pointers, no merge — by design.

Tests: `tests/test_branch_manager.py` (pytest, runs against a fake station, no
hardware). Run with `pixi run python -m pytest tests/test_branch_manager.py -q`.
