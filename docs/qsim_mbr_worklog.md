# MBR refactor worklog

Running handoff for the refactor in `docs/qsim_mbr_refactor.md`. Newest entry
last. Each entry says what landed, how to check it, and what is next, so a
session that starts cold can pick up from the bottom.

Read first: spec section 14 (working order), section 13 (execution
environment), appendix C (configuration traps).

## Standing facts

- Work happens on the acquisition workstation, in the `guan` worktree
  (`C:\python\multimode_expts_guan`). Spec section 13.1.
- **The offline track constructs no station, ever.** That is both the
  correctness rule (section 2.2) and the isolation rule (section 13.3): the
  worktree shares `configs/versions` and `job_server/jobs.db` with the live
  tree, so writes from here hit production.
- Run tests with `pixi run python -m pytest tests/ -q`.
- Re-bless the golden baseline only alongside a deliberate behavior change:
  `MBR_GOLDEN_BLESS=1 pixi run python -m pytest tests/test_mbr_analysis_golden.py`

## 2026-08-25 — safety net in place

**Landed**

- `experiments/job_paths.py` — `resolve_job_paths()`, the section 9.1 seam.
  Two backends: `index` (default, globs the local tree once and caches) and
  `vault` (scrapes `data_path` from the synced lab notebook, for slow remote
  mounts). Raises `JobPathError` naming the environment variable to set rather
  than skipping. Top-level module under `experiments/`, so the flattened
  exporter ignores it and it cannot collide.
- `tests/mbr_reference.py` — the reference analysis as a callable, plus
  `flatten_result()`. Holds the `_SavedJob`/`_SavedProgram` scaffolding, which
  exists only until aggregates load HDF5 themselves. Marked TODO in the module.
- `tests/test_mbr_analysis_golden.py` + `tests/data/mbr_spectrum_20260815.npz`
  — the characterization test. 4 tests, all passing.
- Spec section 14 (working order).

**Baseline contents**: 55 fields, 26 numeric arrays, 8611 elements, covering
`reconstruction.A`/`A_norm`, `acquired_reconstruction.*`, all 19 `spectrum.*`
entries, per-occupation phase corrections, and the resolved hardware block.

**Verified the net can fail**: perturbing `zero_padding` 1 to 2 flags six
fields. A golden that cannot fail is worse than none, so re-check this if the
comparison logic is ever touched.

**Behavior deliberately pinned as-is** (these are the section 2 defects; the
baseline locks them in on purpose):

- theory spectra are rescaled to match measured peaks (section 2.1);
- `_saved_parameters` adopts the first source with a program, no comparison
  (section 2.3);
- aggregate analysis writes `Pe` and `return_quadrature` back onto its sources
  (section 2.5), pinned by `test_source_mutation_is_pinned`.

**Next**: spine step 2, pure numerical extractions. Verified low-risk targets,
each with exactly one call site and no meaningful coupling to `self`:

| Method in `floquet_dark_mode_readout.py` | Target |
|---|---|
| `analyze_matrix_pencil` (already static) | `fitting/qsim/matrix_pencil.py` |
| `analyze_matrix_pencil_trace` (already static) | `fitting/qsim/matrix_pencil.py` |
| `analyze_matrix_pencil_occupation` | `fitting/qsim/matrix_pencil.py` |
| `analyze_level_statistics` | `fitting/qsim/level_statistics.py` |
| `analyze_sff` | `fitting/qsim/level_statistics.py` |

Pattern for each move: extract a pure function taking arrays, leave the method
as a thin wrapper that unpacks `self.data`, keep the golden green, one commit
per module.

## 2026-08-25 (later) — matrix pencil extracted

**Landed**: `fitting/qsim/matrix_pencil.py` (930 lines).
`floquet_dark_mode_readout.py` 8271 → 7381 lines.

- `analyze_matrix_pencil` and `analyze_matrix_pencil_trace` were already
  `@staticmethod` with pure array inputs, so they moved verbatim and stay
  reachable on the class via `staticmethod(...)` assignment.
- `analyze_matrix_pencil_occupation` keeps its signature and becomes a wrapper
  that supplies `self.data` then calls `refit_occupation`. The module is
  imported as `matrix_pencil_analysis` so the historical `matrix_pencil`
  argument name is not shadowed.

**Verification** — the golden alone was not enough, and this matters for every
later move:

The committed golden ran `spectrum_method="fft"`, which never executes the
Matrix-Pencil code. The move looked green while actually being broken:
`sliding_window_view` was imported in the god file and not carried across, so
the Matrix-Pencil path raised `NameError`.

Caught by running the Matrix-Pencil analysis on both sides — a throwaway
`git worktree add --detach <tmp> HEAD` for "before", the working tree for
"after" — and diffing flattened results: **1387 fields, 20354 elements,
bit-identical**.

So the golden now has **one baseline per spectrum method**
(`mbr_spectrum_20260815.npz`, `mbr_matrix_pencil_20260815.npz`), 5 tests.
The FFT baseline was byte-unchanged by re-blessing, which independently
confirms the move left that path alone.

**Lesson for the remaining moves**: check that the golden actually executes the
code being moved. If it does not, either extend the golden first or do the
temporary-worktree before/after diff. The worktree trick is cheap and is the
stronger check.

Also fixed: `flatten_result` raised on ragged object sequences (Matrix-Pencil
candidate lists). It now falls back to per-element recursion.

**Pre-existing, not ours**: `tests/test_branch_manager.py` has 3 failures
(`KeyError: 'hardware_config'`), present before this work started.

**Next**: `analyze_level_statistics` (line ~6906 pre-move) and `analyze_sff`
→ `fitting/qsim/level_statistics.py`. Same pattern. Check first whether the
golden reaches them — `merge_spectra` and the level-statistics path are
probably not on the `analyze(stage="spectrum")` route, in which case use the
worktree diff.
