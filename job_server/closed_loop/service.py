"""Closed-loop experiment service.

Two ways to run:

1) Standalone (Claude-driven, no notebook) — initializes its own station:

       python -m job_server.closed_loop.service --hw
       # or
       python job_server/closed_loop/service.py --hw

2) Embedded (notebook path) — reuse a warm `station` from a kernel:

       from job_server.closed_loop.service import start_service
       handle = start_service(station)   # uvicorn in daemon thread, 127.0.0.1:18765
       handle.stop()                     # graceful shutdown

Endpoints
---------
  GET  /             health (+ whether station is attached)
  POST /echo         echoes JSON body (connectivity smoke test, no station needed)
  POST /run_wigner   main measurement endpoint, mode="sim" or "hw"

Request shape (run_wigner)
--------------------------
{
  "mode": "sim" | "hw",
  "IQ_table": {
      "times": [...],          # μs, length N (sets pulse duration; DAC sample
                               #               grid is interpolated)
      "I_c": [...], "Q_c": [...],   # cavity drive Rabi (GHz), length N
      "I_q": [...], "Q_q": [...]    # qubit drive  Rabi (GHz), length N
  },
  "alphas": [[re, im], ...],   # displacements
  "reps":   int,
  # hw-only (ignored in sim):
  "pulse_ref":     [["optimal_control", "fock", "2", [0, 0]]],
  "gain_override": {"qb": int, "cav": int},   # optional; service computes from GHz
  "knobs":         {"displace_length": 0.05, "pulse_correction": true,
                    "active_reset": false, "post_select_pre_pulse": false,
                    "parity_fast": false, "prepulse": false,
                    "gate_based": false, "relax_delay": 2500},
  "qubits":      [0],
  "man_mode_no": 1
}

Response (run_wigner)
---------------------
{
  "ok": true,
  "alphas":  [[re,im], ...],
  "parity":  [float, ...],
  "mode":    "sim" | "hw",
  "iter_id": "hex8",
  "shots_path": "D:/closed_loop_runs/<run>/iter_xxxx.h5" | null,
  "meta":    { "iq_peak_ghz": ..., "gain_qb": ..., "gain_cav": ...,
               "gains_source": "computed_from_ghz" | "override", ... }
}

IQ_table values are in GHz Rabi rate — matches Piccolo's native unit and the
npz files in device.optimal_control. Typical peaks are 0.001-0.01 GHz (1-10
MHz Rabi). Service computes the QICK gain registers via the calibrated π-pulse
(same math as MM_base.get_gain_optimal_pulse). 400 if computed gain > 32767.

DAC sample overflow boundary: peak(|.|) ≤ 1.0 GHz (catches gross unit errors).
"""
from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================== module state ===============================

_station: Any = None             # MultimodeStation, set by start_service
_job_lock = threading.Lock()     # QPU is a singleton resource
_run_root: Optional[Path] = None # per-service-start root dir for shot dumps


# =========================== request/response ==============================

class IQTable(BaseModel):
    """Envelope waveforms for qubit and cavity drives — in GHz Rabi rate units.

    Contract: I_c, Q_c, I_q, Q_q are drive amplitudes in GHz (matches Piccolo's
    native unit and the npz files in device.optimal_control). Typical peaks are
    0.001-0.01 GHz (1-10 MHz Rabi). The service computes QICK gain registers
    via the same math as MM_base.get_gain_optimal_pulse (calibrated π-pulse +
    4σ Gaussian); see compute_gains_from_ghz().

    Practical upper bound: peak amplitude that produces gain ≤ 32767 (QICK gain
    register max). With current π-pulse cal that's roughly 0.04 GHz (~40 MHz
    Rabi); the service validates the computed gain after the fact and returns
    400 if either channel overflows.

    `times` is in microseconds and sets pulse duration only; the DAC sample
    grid is hardware-fixed and envelopes are linearly interpolated onto it.
    """
    times: list[float]
    I_c: list[float]
    Q_c: list[float]
    I_q: list[float]
    Q_q: list[float]

    def _check(self) -> None:
        n = len(self.times)
        for k in ("I_c", "Q_c", "I_q", "Q_q"):
            if len(getattr(self, k)) != n:
                raise ValueError(f"IQ_table.{k} length {len(getattr(self,k))} != times length {n}")

    def max_abs(self) -> float:
        m = 0.0
        for k in ("I_c", "Q_c", "I_q", "Q_q"):
            arr = getattr(self, k)
            if arr:
                m = max(m, max(abs(v) for v in arr))
        return m


class Knobs(BaseModel):
    displace_length:       float = 0.05
    pulse_correction:      bool  = True
    active_reset:          bool  = False
    post_select_pre_pulse: bool  = False
    parity_fast:           bool  = False
    prepulse:              bool  = False
    gate_based:            bool  = False
    relax_delay:           float = 2500.0
    pre_sweep_pulse:       list  = Field(default_factory=list)
    pre_gate_sweep_pulse:  list  = Field(default_factory=list)


class RunWignerRequest(BaseModel):
    mode:        Literal["sim", "hw"] = "sim"
    IQ_table:    IQTable
    alphas:      list[list[float]] = Field(..., description="N x 2 list of [Re, Im]")
    reps:        int = Field(default=1000, ge=1)
    pulse_ref:   Optional[list[list[Any]]] = None
    # Service derives gains from IQ_table (MHz) via the calibrated π-pulse.
    # gain_override lets you pin them manually for debugging; pass {"qb": N, "cav": M}.
    gain_override: Optional[dict[str, int]] = None
    knobs:       Knobs = Field(default_factory=Knobs)
    qubits:      list[int] = Field(default_factory=lambda: [0])
    man_mode_no: int = 1

    # sim-only escape hatch — useful for fixed targets
    sim_target_beta: Optional[list[float]] = None


class RunWignerResponse(BaseModel):
    ok: bool
    alphas: list[list[float]]
    parity: list[float]
    mode: str
    iter_id: str
    shots_path: Optional[str] = None
    meta: dict


class CalibrateCheckRequest(BaseModel):
    """Reference measurement against a known-good pulse from the config.

    Loads the IQ_table from the npz file referenced by `pulse_ref`, runs a
    single-displacement Wigner measurement, and compares to an expected parity.
    Default: Fock |1>, α=0, expected parity = -1.
    """
    pulse_ref: list[list[Any]] = Field(
        default_factory=lambda: [["optimal_control", "fock", "1", [0, 0]]]
    )
    alpha: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    reps: int = 500
    qubits: list[int] = Field(default_factory=lambda: [0])
    man_mode_no: int = 1
    expected_parity: Optional[float] = None      # auto-deduced for fock-N if None
    tolerance: float = 0.15                       # |measured - expected| <= tol → ok
    knobs: Knobs = Field(default_factory=Knobs)


class CalibrateCheckResponse(BaseModel):
    ok: bool
    pulse_ref: list[list[Any]]
    alpha: list[float]
    measured_parity: float
    expected_parity: Optional[float]
    residual: Optional[float]
    in_tolerance: Optional[bool]
    iter_id: str
    shots_path: Optional[str]
    meta: dict


# ============================== sim backend ================================

def pulse_to_beta(iq: IQTable) -> complex:
    """Toy coherent-state amplitude after driving cavity with iq (in GHz Rabi).

    In the rotating frame: dα/dt = -i·ε(t), with ε in rad/μs.
    For a GHz envelope (cycles/ns), ε = 2π·1e3·(I_c + i·Q_c) [rad/μs], so
        β(T) = -i · 2π · 1e3 · ∫(I_c + i·Q_c) dt   ≈   -i · 2π·1e3 · Σ·Δt
    Sim only — for sanity-testing the closed loop.
    """
    times = np.asarray(iq.times, dtype=float)
    Ic = np.asarray(iq.I_c, dtype=float)
    Qc = np.asarray(iq.Q_c, dtype=float)
    if len(times) < 2:
        return 0.0 + 0.0j
    dt = float(times[1] - times[0])  # μs
    integral = (Ic + 1j * Qc).sum() * dt
    return -1j * 2.0 * np.pi * 1e3 * integral


def coherent_state_parity(alphas_c: np.ndarray, beta: complex) -> np.ndarray:
    d = alphas_c - beta
    return np.exp(-2.0 * (d.real ** 2 + d.imag ** 2))


def run_wigner_sim(req: RunWignerRequest, alphas_c: np.ndarray) -> tuple[np.ndarray, dict]:
    if req.sim_target_beta is not None:
        beta = complex(req.sim_target_beta[0], req.sim_target_beta[1])
    else:
        beta = pulse_to_beta(req.IQ_table)
    parity = coherent_state_parity(alphas_c, beta)
    noise = np.random.normal(0.0, 1.0 / np.sqrt(req.reps), size=parity.shape)
    parity = np.clip(parity + noise, -1.0, 1.0)
    meta = {
        "beta_re":      beta.real,
        "beta_im":      beta.imag,
        "n_samples":    len(req.IQ_table.times),
        "iq_peak_ghz":  req.IQ_table.max_abs(),
    }
    return parity, meta


# ============================== hw backend ================================

class HardwareNotReady(RuntimeError):
    pass


def compute_gains_from_ghz(cfg: Any, iq: IQTable, man_mode_idx: int = 0) -> tuple[int, int]:
    """Translate peak GHz drive -> QICK gain register (qb, cav).

    Mirrors MM_base.get_gain_optimal_pulse (experiments/MM_base.py:2131).
    Source of truth lives there — change here if the calibration math drifts.

    The `*2π*1e3` factor converts GHz (cycles/ns) into angular rad/μs:
        GHz × 2π = rad/ns,  × 1e3 = rad/μs.
    """
    import scipy.special as sps

    def _peak(*arrs):
        m = 0.0
        for a in arrs:
            if a:
                m = max(m, max(abs(v) for v in a))
        return m

    max_q_GHz = _peak(iq.I_q, iq.Q_q)
    max_c_GHz = _peak(iq.I_c, iq.Q_c)
    # GHz -> rad/μs
    max_q = max_q_GHz * 2 * np.pi * 1e3
    max_c = max_c_GHz * 2 * np.pi * 1e3

    n = 4   # 4-sigma Gaussian
    gain_pi  = cfg.device.qubit.pulses.pi_ge.gain[0]
    sigma_pi = cfg.device.qubit.pulses.pi_ge.sigma[0]
    pi_type  = cfg.device.qubit.pulses.pi_ge.type[0]
    if pi_type != "gauss":
        raise ValueError(f"only gaussian π pulse supported (got {pi_type!r})")

    theta_to_gain_qb = np.pi / 2 / gain_pi
    drive_to_gain_qb = sigma_pi * np.sqrt(np.pi) / theta_to_gain_qb * sps.erf(n / 2)

    alpha_to_gain = cfg.device.manipulate.gain_to_alpha[man_mode_idx]
    sigma_cav     = cfg.device.manipulate.displace_sigma[man_mode_idx]
    drive_to_gain_cav = sigma_cav * np.sqrt(np.pi) / alpha_to_gain * sps.erf(n / 2)

    gain_qb  = int(round(max_q * drive_to_gain_qb))
    gain_cav = int(round(max_c * drive_to_gain_cav))
    return gain_qb, gain_cav


def _expected_parity_for_pulse(pulse_ref: list[list[Any]]) -> Optional[float]:
    """Auto-deduce expected parity at α=0 for known pulses.

    Fock |n> has parity (-1)^n at the origin. For superposition keys like
    '0+2' or 'encode'/'decode' we return None (caller must supply).
    """
    if not pulse_ref or not pulse_ref[0]:
        return None
    path = pulse_ref[0]
    if len(path) < 3:
        return None
    category, state = path[1], path[2]
    if category == "fock":
        try:
            n = int(state)
            return float(1 if n % 2 == 0 else -1)
        except (TypeError, ValueError):
            return None
    return None


def _load_iqtable_from_npz(filename: str) -> IQTable:
    """Load a Piccolo-style npz into the IQTable shape the service expects.

    npz convention (per the existing files in device.optimal_control):
      - times in nanoseconds       -> convert to μs by × 1e-3
      - I_c, Q_c, I_q, Q_q in GHz  -> passed through as-is
    """
    data = np.load(filename, allow_pickle=True)
    times_us = (np.asarray(data["times"], dtype=float) * 1e-3).tolist()
    return IQTable(
        times=times_us,
        I_c=np.asarray(data["I_c"], dtype=float).tolist(),
        Q_c=np.asarray(data["Q_c"], dtype=float).tolist(),
        I_q=np.asarray(data["I_q"], dtype=float).tolist(),
        Q_q=np.asarray(data["Q_q"], dtype=float).tolist(),
    )


def _resolve_pulse_conf(cfg_root: Any, pulse_ref: list[list[Any]]) -> dict:
    """pulse_ref = [["optimal_control", "fock", "2", [0,0]]] -> the conf node.

    cfg_root is any cfg-bearing object with .device.optimal_control (a station's
    hardware_cfg, or a deep-copied expt.cfg). Mutate the returned node freely;
    callers should pass a deepcopy when they want isolation.
    """
    if not pulse_ref or not pulse_ref[0]:
        raise ValueError("pulse_ref is required for hw mode")
    path = pulse_ref[0]
    if path[0] != "optimal_control":
        raise ValueError(f"pulse_ref[0][0] must be 'optimal_control', got {path[0]!r}")
    node = cfg_root.device.optimal_control
    for key in path[1:-1]:  # drill: fock, '2'
        node = node[key]
    return node


def _dump_shots(run_dir: Path, iter_id: str, data: dict, req: RunWignerRequest) -> Path:
    """Write raw shots + alphas + parity + request to HDF5. Lightweight."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"iter_{iter_id}.h5"
    try:
        import h5py
    except ImportError:
        # fall back to npz if h5py is somehow missing
        out = run_dir / f"iter_{iter_id}.npz"
        np.savez_compressed(
            out,
            i0=np.asarray(data.get("i0", [])),
            q0=np.asarray(data.get("q0", [])),
            alphas=np.asarray(data.get("alpha", [])),
            parity=np.asarray(data.get("parity", [])),
            request_json=json.dumps(req.model_dump()),
        )
        return out
    with h5py.File(out, "w") as f:
        for k in ("i0", "q0", "alpha", "parity", "pe"):
            v = data.get(k)
            if v is not None:
                f.create_dataset(k, data=np.asarray(v), compression="gzip")
        f.attrs["request_json"] = json.dumps(req.model_dump())
        f.attrs["iter_id"] = iter_id
        f.attrs["created"] = datetime.now().isoformat(timespec="seconds")
    return out


def run_wigner_hw(req: RunWignerRequest, station: Any, iter_id: str) -> tuple[np.ndarray, np.ndarray, Path, dict]:
    """Construct WignerTomography1ModeExperiment, run, return alphas + parity + shots path + meta."""
    if station is None:
        raise HardwareNotReady("no station attached. Call start_service(station) first.")
    if req.pulse_ref is None:
        raise ValueError("pulse_ref is required for hw mode")

    from slab import AttrDict
    from experiments.qubit_cavity.single_mode_wigner_tomography import (
        WignerTomography1ModeExperiment,
    )

    man_mode_idx = max(0, req.man_mode_no - 1)
    if req.gain_override is not None:
        gain_qb  = int(req.gain_override["qb"])
        gain_cav = int(req.gain_override["cav"])
        gains_source = "override"
    else:
        gain_qb, gain_cav = compute_gains_from_ghz(station.hardware_cfg, req.IQ_table, man_mode_idx)
        gains_source = "computed_from_ghz"

    if not (0 <= gain_qb <= 32767):
        raise ValueError(
            f"computed gain_qb={gain_qb} is outside QICK gain register range [0, 32767]. "
            f"Peak qubit drive too strong; reduce |I_q|/|Q_q| on the orchestrator side."
        )
    if not (0 <= gain_cav <= 32767):
        raise ValueError(
            f"computed gain_cav={gain_cav} is outside QICK gain register range [0, 32767]. "
            f"Peak cavity drive too strong; reduce |I_c|/|Q_c| on the orchestrator side."
        )

    expt = WignerTomography1ModeExperiment(
        soccfg     = station.soc,
        path       = str(station.data_path),
        prefix     = "WignerTomography1ModeExperiment",
        config_file= str(station.hardware_config_file),
    )
    expt.cfg = AttrDict(deepcopy(station.hardware_cfg))
    # re-inject the live dataset refs (CharacterizationRunner does the same)
    expt.cfg.device.storage._ds_storage = station.ds_storage
    expt.cfg.device.storage._ds_floquet = station.ds_floquet

    # Resolve on the deepcopy and inject gains there — do NOT mutate station.
    pulse_conf = _resolve_pulse_conf(expt.cfg, req.pulse_ref)
    pulse_conf["gain"][0] = gain_qb
    pulse_conf["gain"][1] = gain_cav

    iq = req.IQ_table
    iq._check()
    max_abs = iq.max_abs()
    # DAC sample overflow: maxv * |value| must fit in int16. Since maxv ~32767,
    # peak ≤ 1.0 (in GHz, Piccolo units) is the boundary. Far above typical
    # values (~0.005 GHz), so this catches gross unit errors only.
    if max_abs > 1.0:
        raise ValueError(
            f"IQ_table peak |value|={max_abs:.4f} GHz exceeds DAC sample range (≤1.0 GHz). "
            "Likely a unit error (MHz vs GHz)."
        )
    expt_cfg = AttrDict({
        "displace_length":       req.knobs.displace_length,
        "alpha_list":            req.alphas,
        "reps":                  req.reps,
        "rounds":                1,
        "prepulse":              req.knobs.prepulse,
        "pre_sweep_pulse":       req.knobs.pre_sweep_pulse,
        "pre_gate_sweep_pulse":  req.knobs.pre_gate_sweep_pulse,
        "qubits":                req.qubits,
        "pulse_correction":      req.knobs.pulse_correction,
        "post_select_pre_pulse": req.knobs.post_select_pre_pulse,
        "opt_pulse":             req.pulse_ref,
        "IQ_table":              iq.model_dump(),
        "active_reset":          req.knobs.active_reset,
        "parity_fast":           req.knobs.parity_fast,
        "gate_based":            req.knobs.gate_based,
        "man_mode_no":           req.man_mode_no,
    })
    expt.cfg.expt = expt_cfg
    expt.cfg.device.readout.relax_delay = [req.knobs.relax_delay]

    t_acq = time.time()
    data = expt.acquire(progress=False)
    t_an  = time.time()
    expt.analyze(data=data)
    t_done = time.time()

    run_dir = (_run_root or Path("D:/closed_loop_runs/_default"))
    shots_path = _dump_shots(run_dir, iter_id, data, req)

    alphas_c = np.asarray(data["alpha"])
    parity   = np.asarray(data["parity"])

    meta = {
        "wall_acquire_s":      t_an - t_acq,
        "wall_analyze_s":      t_done - t_an,
        "wall_total_s":        t_done - t_acq,
        "n_alphas":            int(len(alphas_c)),
        "pulse_samples":       len(iq.times),
        "pulse_duration_us":   float(iq.times[-1]) if iq.times else 0.0,
        "iq_peak_ghz":         float(max_abs),
        "gain_qb":             gain_qb,
        "gain_cav":            gain_cav,
        "gains_source":        gains_source,
    }
    return alphas_c, parity, shots_path, meta


# ============================== FastAPI app ================================

app = FastAPI(title="closed-loop experiment service", version="0.2.0")
START_TIME = datetime.now().isoformat(timespec="seconds")
HOSTNAME = socket.gethostname()


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "expt_service",
        "version": "0.3.0",
        "hostname": HOSTNAME,
        "started": START_TIME,
        "station_attached": _station is not None,
        "run_root": str(_run_root) if _run_root else None,
        "endpoints": ["GET /", "POST /echo", "POST /run_wigner", "POST /calibrate_check"],
    }


@app.post("/echo")
def echo(body: dict):
    return {
        "ok": True,
        "hostname": HOSTNAME,
        "received": body,
        "server_time": datetime.now().isoformat(timespec="seconds"),
    }


@app.post("/calibrate_check", response_model=CalibrateCheckResponse)
def calibrate_check(req: CalibrateCheckRequest):
    """Run a known-good pulse at one displacement and compare to expected parity.

    Loads the IQ_table from the npz referenced by pulse_ref, runs Wigner
    tomography at a single α (default 0+0i), and reports residual vs
    expected parity (auto-deduced for Fock states, otherwise caller-supplied).
    """
    if _station is None:
        raise HTTPException(503, "no station attached. Restart service with --hw.")
    iter_id = uuid.uuid4().hex[:8]

    expected = req.expected_parity
    if expected is None:
        expected = _expected_parity_for_pulse(req.pulse_ref)

    with _job_lock:
        try:
            pulse_conf = _resolve_pulse_conf(_station.hardware_cfg, req.pulse_ref)
            filename = pulse_conf["filename"]
            iq = _load_iqtable_from_npz(filename)
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise HTTPException(400, f"calibrate_check setup error: {e}")

        # build a RunWignerRequest using the loaded IQ_table + single alpha
        sub_req = RunWignerRequest(
            mode="hw",
            IQ_table=iq,
            alphas=[req.alpha],
            reps=req.reps,
            pulse_ref=req.pulse_ref,
            gain_override=None,
            knobs=req.knobs,
            qubits=req.qubits,
            man_mode_no=req.man_mode_no,
        )
        try:
            out_alphas, parity_arr, shots_path, meta = run_wigner_hw(sub_req, _station, iter_id)
        except HardwareNotReady as e:
            raise HTTPException(503, str(e))
        except (ValueError, KeyError) as e:
            raise HTTPException(400, f"hw setup error: {e}")

    measured = float(parity_arr[0])
    residual = None
    in_tol   = None
    if expected is not None:
        residual = measured - expected
        in_tol   = abs(residual) <= req.tolerance

    meta_out = dict(meta)
    meta_out.update({
        "filename":       str(filename),
        "tolerance":      req.tolerance,
        "n_pulse_samples": len(iq.times),
    })
    return CalibrateCheckResponse(
        ok=True,
        pulse_ref=req.pulse_ref,
        alpha=req.alpha,
        measured_parity=measured,
        expected_parity=expected,
        residual=residual,
        in_tolerance=in_tol,
        iter_id=iter_id,
        shots_path=str(shots_path) if shots_path else None,
        meta=meta_out,
    )


@app.post("/run_wigner", response_model=RunWignerResponse)
def run_wigner(req: RunWignerRequest):
    iter_id = uuid.uuid4().hex[:8]
    t0 = time.time()

    alphas_arr = np.asarray(req.alphas, dtype=float)
    if alphas_arr.ndim != 2 or alphas_arr.shape[1] != 2:
        raise HTTPException(400, f"alphas must be N x 2, got shape {alphas_arr.shape}")
    alphas_c = alphas_arr[:, 0] + 1j * alphas_arr[:, 1]

    with _job_lock:
        try:
            req.IQ_table._check()
        except ValueError as e:
            raise HTTPException(400, str(e))

        if req.mode == "sim":
            parity_arr, meta = run_wigner_sim(req, alphas_c)
            shots_path: Optional[Path] = None
            out_alphas = alphas_c
        elif req.mode == "hw":
            try:
                out_alphas, parity_arr, shots_path, meta = run_wigner_hw(req, _station, iter_id)
            except HardwareNotReady as e:
                raise HTTPException(503, str(e))
            except (ValueError, KeyError) as e:
                raise HTTPException(400, f"hw setup error: {e}")
        else:
            raise HTTPException(400, f"unknown mode {req.mode!r}")

    meta["wall_total_s"] = meta.get("wall_total_s", time.time() - t0)
    return RunWignerResponse(
        ok=True,
        alphas=[[float(a.real), float(a.imag)] for a in out_alphas],
        parity=[float(p) for p in parity_arr],
        mode=req.mode,
        iter_id=iter_id,
        shots_path=str(shots_path) if shots_path else None,
        meta=meta,
    )


# ============================== lifecycle ================================

class ServiceHandle:
    def __init__(self, server, thread):
        self.server = server
        self.thread = thread

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()


def start_service(
    station: Any = None,
    *,
    port: int = 18765,
    run_root: Optional[str | Path] = None,
) -> ServiceHandle:
    """Spin up the FastAPI service in a daemon thread and return a handle.

    Pass `station` to enable hw mode. Without it, only sim/echo/root work.
    """
    global _station, _run_root
    _station = station

    if run_root is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _run_root = Path(f"D:/closed_loop_runs/{ts}")
    else:
        _run_root = Path(run_root)
    _run_root.mkdir(parents=True, exist_ok=True)

    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="expt_service")
    thread.start()
    # give uvicorn a moment to bind before returning
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)
    return ServiceHandle(server, thread)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="expt_service: closed-loop QOC measurement endpoint")
    parser.add_argument("--hw", action="store_true",
                        help="Initialize MultimodeStation on startup (~30s) and enable hw mode.")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--experiment-name", default=None,
                        help="Passed to MultimodeStation. Defaults to yymmdd_closed_loop.")
    parser.add_argument("--user", default="claude",
                        help="Passed to MultimodeStation for config versioning tracking.")
    parser.add_argument("--hardware-config", default=None,
                        help="Pin the hardware config: filename or version ID (e.g. CFG-HW-20260514-00018). "
                             "Default: load the current main pointer from the DB.")
    parser.add_argument("--storage-man-file", default=None,
                        help="Pin the man1 storage-swap dataset: filename or version ID (e.g. CFG-M1-20260513-00023). "
                             "Default: load the current main pointer from the DB.")
    args = parser.parse_args()

    station = None
    if args.hw:
        import sys
        # repo root = .../multimode_expts (this file: .../job_server/closed_loop/service.py)
        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from experiments.station import MultimodeStation
        exp_name = args.experiment_name or f"{datetime.now().strftime('%y%m%d')}_closed_loop"
        print(f"[expt_service] initializing MultimodeStation (experiment_name={exp_name!r})...")
        station = MultimodeStation(
            experiment_name=exp_name,
            user=args.user,
            hardware_config=args.hardware_config,
            storage_man_file=args.storage_man_file,
        )
        print(f"[expt_service] station ready (mock={station.is_mock})")

    handle = start_service(station=station, port=args.port)
    mode_tag = "hw + sim" if station is not None else "sim-only"
    print(f"[expt_service] running on http://127.0.0.1:{args.port} ({mode_tag})")
    print(f"[expt_service] run dumps -> {_run_root}")
    print("[expt_service] Ctrl-C to stop")
    try:
        while handle.alive:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[expt_service] stopping...")
        handle.stop()
