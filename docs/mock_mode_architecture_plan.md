# Mock-mode refactor — plan

## Goal

One switch on `MultimodeStation` controls whether instrument calls go to real
hardware or to in-process stubs. Everything else (configs, datasets, live
in-memory state, `station.soc`) is identical in both modes. The switch can be
flipped mid-session without re-creating the station, so a notebook can preserve
accumulated fits and calibrations while iterating on a buggy QICK program.

## Motivation

Today, when a notebook hits a QICK programming bug (bad parameter, malformed
pulse, channel mis-declared), the entire job-server round-trip fires before the
qick library's own validators reach the failing line — ~1 minute of overhead to
discover a five-character typo. The qick library actually catches almost all of
these in pure Python during program construction, before any FPGA byte is sent.
We exploit that by routing instrument calls to a stub during debug iterations.

The existing `mock=True` mode in `experiments/station.py` was an earlier
attempt at this. It never worked because `MockQickConfig` lacks fields the real
qick library reads at program-init time, so experiments crash before reaching
the stub's `acquire()`. This plan replaces that path with a working design.

## Architecture decision

QICK's library cleanly separates three phases:

1. **Build (Python only)** — `Program(soccfg, cfg).__init__()` calls user
   `initialize()` and `body()`, builds `self.prog_list`. Reads `soccfg`. Where
   most parameter-validation errors fire.
2. **Compile (Python only)** — `program.compile()` turns `prog_list` into
   `binprog`. Catches ASM-encoding errors.
3. **FPGA** — `program.acquire(soc, ...)` → `config_all(soc)` → `soc.start_*`,
   `soc.poll_data`. First time the `soc` object is touched.

`soccfg` (a `QickConfig`) is a pure-Python dict wrapper describing the FPGA
bitstream — portable, file-loadable, no hardware. `soc` (a `QickSoc`) is the
hardware driver, normally reached from the PC over a Pyro4 RPC proxy via
`InstrumentManager`. Replacing the Pyro4 proxy with a local no-op stub gives
phases 1+2 unchanged validation behavior, with zero FPGA contact at phase 3.

This is exactly the off-board mode the qick authors designed for. We are not
inventing a parallel mock framework; we are using the existing PC-side split.

## In scope

- New `MockQickSoc` class with the ~17-method stub surface the qick library
  actually calls during `acquire()` / `acquire_decimated()` / `run_rounds()`.
- `station.use_mock_instruments()` / `station.use_real_instruments()` for
  mid-session swap, preserving all in-memory state.
- `MultimodeStation(mock=True)` constructor flag retained; semantically
  equivalent to constructing real then calling `use_mock_instruments()` once.
  Worker `--mock` CLI flag continues to work without change.
- Auto-redirect `data_path`/`plot_path`/`log_path` to `mock_data/...` whenever
  mock instruments are active, so fake runs don't pollute real data dirs.
- `SweepRunner` and `CharacterizationRunner` auto-default `use_queue=False`
  when `station.is_mock_instruments` is True. Explicit override still works
  (one-line warning if user explicitly passes `use_queue=True` in mock mode).
- Delete `job_server/mock_hardware.py` (dead duplicate, no live importers).

## Out of scope (deferred)

- Laptop / non-prod-PC support. The config database and `D:/` paths aren't
  reachable from a Mac; we don't promise the station can even be constructed
  there. Auto-detect logic stays as is. Revisit if anyone actually needs it.
- A captured `soccfg.json` snapshot in the repo. Not needed on the prod PC
  since the live Pyro proxy provides `get_cfg()` once at station init, and
  that's already cached into `station.soc`.
- `closed_loop/service.py` compatibility. Service uses `MultimodeStation(mock=True)`
  as a config-holder; under the new design its `data_path` will redirect into
  `mock_data/`. If that breaks the service, it's downstream of the infra and
  should be fixed there.
- Renaming `station.soc` → `station.soccfg` for clarity. Worth doing, but
  separate task — touches every experiment.
- Improvements to the broader runner / Experiment scaffolding beyond the
  one auto-default.

## File changes

### `experiments/mock_hardware.py` — rewrite

- **Remove**: `MockQickConfig` (broken — incomplete cfg dict).
- **Replace**: `MockQickSoc` — old version had hardcoded Rabi data in
  `.acquire()`, which the real qick library never reaches. New version stubs
  the ~17 methods qick's `AcquireMixin` actually calls on `soc`:
  - No-ops: `start_src`, `stop_tproc`, `reload_mem`, `load_pulse_data`,
    `set_nyquist`, `set_mixer_freq`, `config_mux_gen`, `configure_readout`,
    `load_bin_program`, `config_avg`, `config_buf`, `start_tproc`,
    `set_tproc_counter`.
  - Stateful: `start_readout(total_count, counter_addr, ch_list, reads_per_shot)`
    records args; `get_tproc_counter(addr)` returns `total_count` to exit
    polling loops immediately.
  - Shaped returns (strictly zeros, no noise — noise looks too much like real
    data):
    - `poll_data()` → `[(total_count, ([zeros((total_count*nr, 2), int64) for nr in reads_per_shot], {}))]`
    - `get_decimated(ch, address, length)` → `np.zeros((length, 2))`
    - `get_accumulated(ch, address, length)` → `np.zeros((length, 2))`
- **Keep**: `MockYokogawa`, `MockInstrumentManager` unchanged in concept. The
  `MockInstrumentManager` continues to be a dict subclass holding `MockQickSoc`
  at the qick alias key.

### `experiments/station.py`

- `_initialize_hardware_mock` rewritten:
  - Build a real `QickConfig` for `self.soc`. On the prod PC, fetch from the
    live Pyro proxy at construction time (same call as the real path uses).
    If unavailable, raise — deliberately not falling back to a fake cfg dict,
    since that path was the source of all the existing problems.
  - Build `MockInstrumentManager` containing the new `MockQickSoc` at the qick
    alias, `MockYokogawa` instances for the yokos.
- New property `is_mock_instruments` (alias `is_mock` for back-compat).
- New `use_mock_instruments(self)`:
  - Cache real `im`, `yoko_coupler`, `yoko_jpa`, `data_path`, `plot_path`,
    `log_path` to private attrs.
  - Replace with stubs / `mock_data/...` paths.
  - Set `_is_mock = True`.
- New `use_real_instruments(self)`:
  - Restore from caches. Raise if never swapped.
  - Set `_is_mock = False`.
- Both methods preserve `hardware_cfg`, `multimode_cfg`, `ds_storage`,
  `ds_floquet`, `soc`, `experiment_name`, `user`.

### `experiments/sweep_runner.py` and `experiments/characterization_runner.py`

In both runners, `__init__` keeps `use_queue: bool = True` default. In
`execute()`, before dispatching, add:

```python
mode = use_queue if use_queue is not None else self.use_queue
if mode and self.station.is_mock_instruments:
    if use_queue is True:    # explicit override
        warnings.warn("use_queue=True with mock instruments — worker may run "
                      "against real hardware unless also in mock mode")
    else:
        mode = False         # auto-default
```

### `job_server/mock_hardware.py` — delete

374 lines, no live importers. The `MockStation` class it defines was
superseded by `MultimodeStation(mock=True)`. Remove the file and update the
one reference in `job_server/README.md`.

### Tests

- `tests/test_mock_station.py`, `tests/test_sweep_runner.py`,
  `tests/test_characterization_runner.py`: use in-file `MagicMock`-based local
  mocks, don't depend on `experiments/mock_hardware.py`. Leave alone unless
  they break.
- New `tests/test_mock_qicksoc.py`: instantiate a real `MMRAveragerProgram`
  (e.g., a small Rabi program) with a captured soccfg + `MockQickSoc`, call
  `.acquire()`, assert it completes and returns shaped data without errors.

## User-facing behavior

### Typical debug flow on the prod PC

```python
# normal day
station = MultimodeStation(user=user, hardware_config=..., ...)
# ... many cells of real measurements, fits accumulating in hardware_cfg ...

# bug surfaces in a new qick program — flip to mock to iterate
station.use_mock_instruments()
runner.execute(...)                  # local, against MockQickSoc, writes to mock_data/

# fix bugs, iterate

# back to real
station.use_real_instruments()
runner.execute(...)                  # back to real FPGA, real paths, queue
```

### Worker

`pixi run python -m job_server.worker --mock` continues to work; internally
the worker's station is now in `use_mock_instruments` mode after construction.
No CLI change. The worker cannot flip the switch mid-run because there's no
interactive surface — that's fine, the CLI flag is the only way to set it for
a worker process.

### Constructor flag

`MultimodeStation(mock=True)` retained as sugar. Semantically:

```python
station = MultimodeStation(mock=True)
# is equivalent to:
station = MultimodeStation(mock=False)
station.use_mock_instruments()
```

## State preservation contract

`use_mock_instruments()` / `use_real_instruments()` are the only two operations
that change `self._is_mock`. They affect this set of attributes only:

| Swapped | Preserved |
|---|---|
| `im` | `hardware_cfg` |
| `yoko_coupler` | `multimode_cfg` |
| `yoko_jpa` | `ds_storage` |
| `data_path` | `ds_floquet` |
| `plot_path` | `soc` |
| `log_path` | `experiment_name` |
| `_is_mock` flag | `user`, `project`, etc. |

If a future attribute is added that needs to swap, it must be added to both
methods explicitly. Any attribute not in this table is preserved across swap.

## Open implementation questions (decide while coding)

- Exact path layout under `mock_data/` (per-experiment subfolder vs flat).
- Whether to commit the deletion of `job_server/mock_hardware.py` in the same
  PR as the rewrite, or split for cleaner review.
- Naming nits inside the new `MockQickSoc` (e.g., method-arg ordering of the
  stateful methods).

## Verification

Smoke test by user before merge: pick a known-bug commit (or fabricate one,
e.g., set a pulse length to 0), run with `station.use_mock_instruments()`
active, confirm the failure raises at the same point a real FPGA run would
have. Confirms the qick library validators are reached unchanged.
