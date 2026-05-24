# Mock-mode architecture

`MultimodeStation(mock=True)` and `station.use_mock_instruments()` route
instrument calls through in-process stubs while leaving the real qick library
in charge of program construction, ASM compilation, and the acquire control
loop. The result: the qick library's parameter validators fire normally on a
buggy program, but no bytes go to the FPGA and no job-queue round-trip happens.
Typical use is mid-session debug after a qick-program edit produces an error.

## Why this design works

QICK's library cleanly separates three phases:

1. **Build** — `Program(soccfg, cfg).__init__()` runs the user's `initialize()`
   and `body()`, building `self.prog_list`. Reads `soccfg`. **Where almost all
   parameter-validation errors fire.**
2. **Compile** — `program.compile()` turns `prog_list` into `binprog`. Catches
   ASM-encoding errors. Pure Python, doesn't touch soccfg.
3. **FPGA** — `program.acquire(soc, ...)` → `config_all(soc)` → `soc.start_*`,
   `soc.poll_data()`. First time the `soc` object is touched.

`soccfg` (a `QickConfig`) is a portable Python dict wrapper describing the
loaded FPGA bitstream. `soc` (a `QickSoc`) is the hardware driver — normally
reached from the PC over a Pyro4 RPC proxy via `slab.InstrumentManager`.
Replacing the Pyro4 proxy with a local no-op stub gives phases 1+2 unchanged
validation behavior with zero FPGA contact in phase 3.

This is the off-board mode the qick authors designed for. We are not inventing
a parallel mock framework; we use the existing PC-side split.

## What's mocked vs real

| Real (unchanged from production) | Mocked (no-op stub) |
|---|---|
| `qick.QickConfig` (fetched from live Pyro proxy at init) | `qick.QickSoc` proxy → `MockQickSoc` |
| Full `hardware_cfg`, `multimode_cfg` from version DB | `slab.InstrumentManager` → `MockInstrumentManager` (dict subclass holding `MockQickSoc`) |
| `ds_storage`, `ds_floquet` datasets | `YokogawaGS200` → `MockYokogawa` (no-op with action-trace prints) |
| The entire qick program-build + ASM-compile path | Output paths (redirected to `C:/experiments/mock_data/...`) |
| Postprocessors, analysis, plot rendering | |

## Public API

```python
# Real hardware (default)
station = MultimodeStation(user=user, hardware_config=..., ...)

# Mock at construction time (e.g. worker --mock)
station = MultimodeStation(mock=True)

# Mid-session swap. Preserves all in-memory state (hardware_cfg, multimode_cfg,
# datasets, soc, accumulated fits). Idempotent.
station.use_mock_instruments()    # → mocks installed, paths redirected
station.use_real_instruments()    # → restores from cache

# Current state
station.is_mock                    # True iff in mock mode right now
```

The `mock=True` constructor flag is semantically equivalent to a real init
followed by `use_mock_instruments()`, with one practical difference: the
constructor path skips claiming real yokos via VISA. That means `mock=True` at
construction provides no real-instruments cache, so a subsequent
`use_real_instruments()` raises. To switch a mock-constructed station to real,
reconstruct it.

## State preservation contract

`use_mock_instruments()` / `use_real_instruments()` are the only operations
that change `_is_mock`. They affect this set of attributes only:

| Swapped | Preserved |
|---|---|
| `im` | `hardware_cfg` |
| `yoko_coupler` | `multimode_cfg` |
| `yoko_jpa` | `ds_storage`, `ds_floquet` |
| `output_root` | `soc` (the QickConfig) |
| `experiment_path` | `experiment_name`, `user`, `project` |
| `data_path` | `vault_root` (but log writes are gated, see below) |
| `expt_objs_path` | everything else not in the left column |
| `plot_path` | |
| `log_path` | |
| `autocalib_path` | |

The path-swap keys live in `MultimodeStation._MOCK_SWAP_PATH_KEYS` as a single
source of truth. `tests/test_mock_mode.py::test_swap_methods_preserve_state_per_contract`
asserts this stays in sync with the table above.

## Behavior gates triggered by mock mode

### Runner dispatch (`SweepRunner`, `CharacterizationRunner`)

When `station.is_mock`, the `execute()` method auto-flips `use_queue` to False
so the experiment runs locally against the stub instead of being submitted to
a worker that might run against real hardware. Explicit `use_queue=True` from
the caller is allowed but emits a one-line `RuntimeWarning`.

### Lab-notebook vault writes

The Obsidian-vault logging path (`station.log_measurement`, called either
directly from notebooks or via the runners' `_maybe_log_measurement` helpers)
is suppressed in mock mode — fake measurements would otherwise pollute the
real lab notebook. Two guards:

- `_maybe_log_measurement` in both runners: early-return before the
  display-render scaffolding, avoiding wasted work.
- `log_measurement` itself: top-level guard, catches direct
  `station.log_measurement(...)` calls from notebooks (e.g.
  `coupler_systematic_study_v2.ipynb`).

There is no override flag. If you need to test the logging path itself, flip
out of mock mode briefly with `station.use_real_instruments()`.

## Constraints and known limitations

- **Mock-init still requires the Pyro proxy** at `192.168.137.26` to fetch a
  real `QickConfig` (the qick library's program-init validators need a
  complete soccfg dict). On the prod PC this is always reachable. Off-prod-PC
  mode would need a separate path (saved JSON snapshot) — flagged as out of
  scope.
- **Mock data path is hardcoded** to `C:/experiments/mock_data` in
  `_initialize_output_paths_mock`. Off-prod-PC mode will need its own path
  resolution.
- **`use_real_instruments()` requires a prior real init.** A station
  constructed with `mock=True` raises if you call this — reconstruct instead.
- **Worker `--mock`** continues to work without CLI change. Under the new
  design it produces a station whose `is_mock=True` is set by the constructor
  flag, with the same swap-disabled property as above.

## Out of scope / deferred

- **Off-prod-PC support** (Macbooks, dev machines without the config DB or
  `D:/` paths). The dormant `is_production_pc()` helper in `station.py`
  exists for the eventual code path; it is not currently called from
  `__init__`.
- **Soccfg JSON snapshot in repo.** Not needed on the prod PC; would unlock
  laptop validation runs.
- **`closed_loop/service.py` compatibility.** The service uses
  `MultimodeStation(mock=True)` as a config-holder; under the new design its
  `data_path` redirects into `mock_data/`. If that breaks the service, it's
  downstream of the infra and should be fixed there.
- **Renaming `station.soc` → `station.soccfg`** for clarity. Worth doing —
  the name has been misleading since the QickConfig/QickSoc split was added —
  but it touches every experiment.

## Verification

`tests/test_mock_mode.py` covers the durable contract:

- `MockQickSoc` shape/dtype contract (qick's `AcquireMixin.acquire` asserts
  `d[ii].shape[0] == new_points * reads_per_shot[ii]` and int64 dtype).
- The 19 soc methods qick's acquire / acquire_decimated / run_rounds call.
- `MockInstrumentManager` / `MockYokogawa` semantics including the print
  trace.
- `MultimodeStation` API surface — new methods exist, removed names stay
  removed.
- Runner mock-mode gates + vault gates exist in both layers.

Tests run in ~2 seconds, no hardware or qick imports beyond what's already
loaded by the package. Smoke-test a real qick program end-to-end manually
(e.g. fabricate a zero-length pulse and confirm `MockQickSoc + real
MMAveragerProgram` raises at the same line a real FPGA run would).

## File layout

- `experiments/mock_hardware.py` — `MockQickSoc`, `MockInstrumentManager`,
  `MockYokogawa`. Self-contained, only imports numpy.
- `experiments/station.py` — constructor `mock` flag, `_initialize_hardware_mock`,
  `_install_mock_instruments`, `use_mock_instruments`, `use_real_instruments`,
  `is_mock` property, `_MOCK_SWAP_PATH_KEYS`, vault-write guard in
  `log_measurement`.
- `experiments/sweep_runner.py` + `experiments/characterization_runner.py` —
  `execute()` auto-default, `_maybe_log_measurement` early-return.
- `tests/test_mock_mode.py` — contract tests.
