# Brief: Claude-driven closed-loop QOC on i9 desktop

You're a Claude session on a **Windows i9 desktop** at SLAC. The user is at Harmoniqs (Aaron). The goal is closed-loop quantum optimal control: you drive an **Intonato.jl** optimization (private repo `harmoniqs/Intonato.jl` — Robust QILC for Piccolo/Piccolissimo) that calls measurements on a remote QPU and iterates.

**No notebooks.** You orchestrate by writing Julia scripts and running them. Per-run state lives on disk (HDF5/JSON), so you can monitor, pause, and resume across Claude sessions.

## Topology

```
i9 desktop (this box)                        Lab box (fridge PC, DESKTOP-GONKTN3)
─────────────────────                        ───────────────────────────────────
Claude (you)                                  Claude (other session)
  └─ writes Julia scripts                       └─ owns expt_service.py + station
  └─ runs `julia script.jl`                     └─ owns the QPU
       └─ Intonato.jl QILC loop                          │
             └─ measure_wigner()      ───HTTP──>         │
                  POST /run_wigner    <──JSON───  hw measurement
                                                  (returns parity grid)
   tunnel: ssh -L 18765:127.0.0.1:18765 <fridge>
```

The service is at **`http://127.0.0.1:18765`** *via the SSH tunnel*. No tunnel = no service reachable.

## Files you should know about

These live on the **lab box** (in `multimode_expts/job_server/closed_loop/`) but their contracts are the user-visible API:

- **`service.py`** — FastAPI service. Endpoints:
  - `GET /` — health, returns `{station_attached, hostname, run_root, ...}`.
  - `POST /echo` — JSON round-trip smoke test.
  - `POST /run_wigner` — main measurement endpoint.
  - Launched as: `python -m job_server.closed_loop.service --hw` from the repo root.
- **`measure_wigner.jl`** — Julia helper that wraps the HTTP call. Pull from the same repo or copy into an Intonato project on this box.
- **`BRIEF.md`** — this file.

## The contract: IQ_table is in GHz

**Intonato emits envelopes in GHz Rabi rate units** — the same unit Piccolo natively emits and the same unit the existing npz files in `device.optimal_control` use. The service translates GHz → QICK gain registers via the calibrated π-pulse (mirrors `MM_base.get_gain_optimal_pulse`). The orchestrator never touches DAC concepts.

```json
"IQ_table": {
  "times": [...],            // μs — sets pulse duration only
  "I_c": [...], "Q_c": [...], // cavity drive Rabi (GHz), typical peak 0.001-0.01
  "I_q": [...], "Q_q": [...]  // qubit  drive Rabi (GHz), typical peak 0.001-0.01
}
```

Practical upper bound: peak amplitude that produces gain ≤ 32767 (QICK register max). With current π-pulse cal that's roughly 0.04 GHz (~40 MHz Rabi). The service computes gains and returns 400 if either channel overflows the register.

DAC sample overflow boundary (gross unit error check): peak ≤ 1.0 GHz. Far above typical values.

## Full /run_wigner request shape

```json
{
  "mode":        "hw",         // or "sim" for plumbing tests
  "IQ_table":    {...},        // see above
  "alphas":      [[re,im],...], // displacements for parity tomography
  "reps":        250,
  "pulse_ref":   [["optimal_control", "fock", "2", [0, 0]]],
  "qubits":      [0],
  "man_mode_no": 1,
  "knobs": {
    "displace_length":       0.05,
    "pulse_correction":      true,
    "active_reset":          false,
    "post_select_pre_pulse": false,
    "parity_fast":           false,
    "prepulse":              false,
    "gate_based":            false,
    "relax_delay":           2500.0
  },
  "gain_override": null       // optional: {"qb": N, "cav": M} to pin manually
}
```

Response:

```json
{
  "ok":         true,
  "alphas":     [[re,im],...],
  "parity":     [float, ...],
  "mode":       "hw",
  "iter_id":    "abcd1234",
  "shots_path": "D:/closed_loop_runs/.../iter_abcd1234.h5",  // server-side path
  "meta": {
    "wall_total_s": 2.1, "n_alphas": 64, "pulse_samples": 200,
    "iq_peak_ghz": 0.0025, "gain_qb": 1948, "gain_cav": 2204,
    "gains_source": "computed_from_ghz"
  }
}
```

## /calibrate_check endpoint

A reference measurement against a known-good pulse. Loads the IQ_table from the npz file referenced by `pulse_ref`, runs Wigner tomography at a single displacement (default α=0), compares measured parity to expected.

Default: Fock |1⟩ at α=0, expected parity = -1, tolerance 0.15. Auto-deduces expected parity for `["optimal_control", "fock", "<n>", ...]` references (Fock |n⟩ → (-1)^n at origin); for superposition keys pass `expected_parity` explicitly or get `null` back.

```json
{
  "pulse_ref":       [["optimal_control", "fock", "1", [0, 0]]],
  "alpha":           [0.0, 0.0],
  "reps":            500,
  "qubits":          [0],
  "man_mode_no":     1,
  "expected_parity": null,    // null → auto from pulse_ref
  "tolerance":       0.15
}
```

Response: `{measured_parity, expected_parity, residual, in_tolerance, ...}`. **Run this before any closed-loop session** to catch calibration drift early — if a known-good Fock-1 pulse doesn't return parity ≈ -1, the QILC won't converge against good physics.

## How to start a closed-loop run

### 1. Confirm tunnel is up

```powershell
Invoke-RestMethod http://127.0.0.1:18765/
```
Expect `hostname: "DESKTOP-GONKTN3"`, `station_attached: true`. If not, the user needs to (re-)open the SSH tunnel:
```powershell
ssh -L 18765:127.0.0.1:18765 <user>@<fridge-ssh-target>
```
Or set `LocalForward 18765 127.0.0.1:18765` in their `~/.ssh/config` for that host.

If `station_attached: false`, the service is in sim-only mode — ask the lab-box Claude to restart it with `--hw`.

### 2. Smoke-test the call from Julia

Write a tiny script that loads `measure_wigner.jl`, sends a simple constant-amplitude pulse, prints the response. If you get nonzero parity at α=0 and decaying parity for large |α|, the pipeline is alive.

### 3. Wire Intonato's measurement callback

In whichever Intonato.jl example you base the run on, replace the simulation-callback with `MeasureWigner.measure_wigner(IQ_table_MHz, alphas; reps=..., pulse_ref=...)`. Intonato's QILC loop calls this once per iteration.

### 4. Persist everything to disk

Per-iteration: write `(iter, IQ_table, alphas, parity, shots_path, fidelity, solver_state)` to a per-run HDF5 in `D:\closed_loop_runs\<run_id>\` on **this** box. The service also dumps raw shots HDF5 on the lab box at the `shots_path` it returns — those are server-side paths. Track them but you don't need to fetch immediately.

### 5. Drive the run

Run `julia --project=... run_optimization.jl > run.log 2>&1 &`. Poll `run.log` and the local HDF5 to monitor convergence. Surface a one-line status to the user every N iterations or on completion.

## Coordination with the lab-box Claude

- **Code** — push to shared git; the other side pulls. Don't paste long code blobs via Slack.
- **Service lifecycle** — only the lab-box Claude restarts `expt_service.py`. Don't try to manage it from here.
- **Hardware sanity** — if you see implausible parity (all zeros, NaN, sudden mode collapse), don't keep iterating. Pause, ask the lab-box Claude to inspect the latest `shots_path` HDF5.

## Things that will burn you

1. **Tunnel survival.** Laptop sleep / wifi drop kills it. Use `autossh` with `ServerAliveInterval=30`. Detect failure as a connection timeout in `measure_wigner` and pause the run (don't loop on a dead tunnel).
2. **Implicit gain assumptions.** The service uses the lab box's *current* π-pulse calibration to translate MHz → gain. If that drifts during a long run, your "optimized" pulse is partially compensating for the calibration. Periodic re-cal needed (we'll add `/calibrate_check` later).
3. **Sim mode is a smoke test, not physics.** Its `pulse_to_beta` is `-i·2π·Σ(I_c+iQ_c)·dt` — physically motivated but doesn't model anything except a coherent-state-displaced cavity. Don't tune hyperparameters against sim convergence and expect them to transfer to hw.
4. **Service mutex.** Only one job at a time on the QPU. The service serializes requests; your loop is naturally serial, so this is invisible.
5. **`pulse_ref`** must resolve to an existing node in `station.hardware_cfg.device.optimal_control`. If you reference `[["optimal_control","fock","2",...]]` but no such entry exists, 400. Ask the lab Claude what's available before assuming.

## What to do if you (Claude) get assigned this fresh

1. `Invoke-RestMethod http://127.0.0.1:18765/` — service reachable?
2. Check `harmoniqs/Intonato.jl` (private) is cloned and Pkg.instantiate'd on this box. Find its closed-loop example.
3. Read `D:\tmp\measure_wigner.jl` to see the call surface.
4. Ask the user what target state / fidelity / iteration budget. Default budgets if unspecified: 50 iters, target fidelity 0.95.
5. Run, monitor, report.

## Repos worth knowing

Private under `harmoniqs/`:
- `Intonato.jl` — start here
- `Piccolissimo.jl`, `Piccolo.jl` (public) — underlying QOC solver
- `nyu-bosonic-demo` — similar transmon-cavity architecture, look here for prior art
- `EcdControl-v0`, `ECDWarmStarts.jl` — cavity-gate warm starts
