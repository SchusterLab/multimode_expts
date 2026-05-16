"""Closed-loop experiment service — queue-client edition.

This service is a thin HTTP facade over the existing job queue. It owns NO
hardware. Every `/run_wigner` and `/calibrate_check` request becomes a
`WignerTomography1ModeExperiment` job submitted to the queue server; the queue
worker is the sole hardware owner. Parity comes back inline by loading the
worker's expt pickle when the job completes.

Lifecycle requirement: the queue server (port 8000) AND its worker must be
running. This service connects to the queue server as a client.

Run it
------
    pixi run python -m job_server.closed_loop.service \
        --hardware-config CFG-HW-20260515-00021 \
        --storage-man-file CFG-M1-20260513-00023

Endpoints
---------
  GET  /                health, includes pinned config IDs and queue reachability
  POST /echo            JSON round-trip smoke test
  POST /run_wigner      mode="sim" (in-process) or "hw" (queue-routed)
  POST /calibrate_check known-good fock-N reference, queue-routed

Request / response shapes — see RunWignerRequest / RunWignerResponse and
CalibrateCheckRequest / CalibrateCheckResponse below. They're unchanged from
the previous direct-station version, so existing i9 callers continue to work.

Gain handling
-------------
  /run_wigner       — Intonato-style pulse; service computes gain registers via
                      compute_gains_from_ghz against the pinned hardware_config.
                      gain_override pins {qb, cav} manually.
  /calibrate_check  — service uses the canonical pulse_conf["gain"] stored in
                      hardware_config alongside the npz (the cal context the
                      pulse was built against). gain_override pins manually.

In both cases the resolved gain pair is BAKED into a deepcopy of hardware_cfg
that's serialized into the job's station_config; the worker's station picks
this up via _update_station_from_job_config, so the experiment runs with the
gains we resolved here.

IQ envelopes are in GHz Rabi rate (Piccolo native unit, matches the npz
files in device.optimal_control). DAC sample overflow boundary: peak ≤ 1.0 GHz.
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

from job_server.client import JobClient


# ============================== module state ===============================
# Service is stateless w.r.t. hardware; we keep these to avoid reloading
# pinned configs and rebuilding the station_config template on every call.

_mock_station: Any = None                         # MultimodeStation(mock=True), config holder
_serializable_hw_cfg_base: Optional[dict] = None  # plain-dict snapshot of hardware_cfg
_station_data_template: Optional[dict] = None     # JSON-ready station_data sans hardware_cfg
_job_client: Optional[JobClient] = None
_run_root: Optional[Path] = None
_queue_url: str = "http://127.0.0.1:8000"

# Tuning for the wait_for_completion poll cadence. The worker has its own
# DB poll interval set at its startup; for low-latency QILC, launch the
# worker with `--poll-interval 0.2` and we'll pick that up implicitly.
_WAIT_POLL_S = 0.5


# =========================== request/response ==============================

class IQTable(BaseModel):
    """Envelope waveforms in GHz Rabi rate. See docstring at top of file."""
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

    Gain handling: by default uses the canonical pulse_conf["gain"] from the
    pinned hardware_config — those are the registers the npz was built against.
    Pass `gain_override` to pin {qb, cav} manually for debugging.
    """
    pulse_ref: list[list[Any]] = Field(
        default_factory=lambda: [["optimal_control", "fock", "1", [0, 0]]]
    )
    alpha: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    reps: int = 500
    qubits: list[int] = Field(default_factory=lambda: [0])
    man_mode_no: int = 1
    expected_parity: Optional[float] = None
    tolerance: float = 0.15
    knobs: Knobs = Field(default_factory=Knobs)
    gain_override: Optional[dict[str, int]] = None


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
    """Toy coherent-state amplitude after driving cavity with iq (in GHz Rabi)."""
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


# =========================== shared hw helpers =============================

class ServiceNotReady(RuntimeError):
    pass


def compute_gains_from_ghz(hw_cfg: Any, iq: IQTable, man_mode_idx: int = 0) -> tuple[int, int]:
    """Translate peak GHz drive -> QICK gain register (qb, cav).

    Mirrors MM_base.get_gain_optimal_pulse. Operates on any cfg-bearing object
    supporting attribute access (AttrDict / pydantic / etc); only reads, never
    mutates.
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
    max_q = max_q_GHz * 2 * np.pi * 1e3
    max_c = max_c_GHz * 2 * np.pi * 1e3

    n = 4   # 4-sigma Gaussian
    gain_pi  = hw_cfg.device.qubit.pulses.pi_ge.gain[0]
    sigma_pi = hw_cfg.device.qubit.pulses.pi_ge.sigma[0]
    pi_type  = hw_cfg.device.qubit.pulses.pi_ge.type[0]
    if pi_type != "gauss":
        raise ValueError(f"only gaussian π pulse supported (got {pi_type!r})")

    theta_to_gain_qb = np.pi / 2 / gain_pi
    drive_to_gain_qb = sigma_pi * np.sqrt(np.pi) / theta_to_gain_qb * sps.erf(n / 2)

    alpha_to_gain = hw_cfg.device.manipulate.gain_to_alpha[man_mode_idx]
    sigma_cav     = hw_cfg.device.manipulate.displace_sigma[man_mode_idx]
    drive_to_gain_cav = sigma_cav * np.sqrt(np.pi) / alpha_to_gain * sps.erf(n / 2)

    gain_qb  = int(round(max_q * drive_to_gain_qb))
    gain_cav = int(round(max_c * drive_to_gain_cav))
    return gain_qb, gain_cav


def _expected_parity_for_pulse(pulse_ref: list[list[Any]]) -> Optional[float]:
    """Auto-deduce expected parity at α=0. Fock |n> → (-1)^n; superpositions → None."""
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
    """Load a Piccolo-style npz: times in ns -> μs; I_c/Q_c/I_q/Q_q in GHz."""
    data = np.load(filename, allow_pickle=True)
    times_us = (np.asarray(data["times"], dtype=float) * 1e-3).tolist()
    return IQTable(
        times=times_us,
        I_c=np.asarray(data["I_c"], dtype=float).tolist(),
        Q_c=np.asarray(data["Q_c"], dtype=float).tolist(),
        I_q=np.asarray(data["I_q"], dtype=float).tolist(),
        Q_q=np.asarray(data["Q_q"], dtype=float).tolist(),
    )


def _resolve_pulse_conf_attr(cfg_root: Any, pulse_ref: list[list[Any]]) -> dict:
    """Attribute-style resolution (works on AttrDict and dict)."""
    if not pulse_ref or not pulse_ref[0]:
        raise ValueError("pulse_ref is required")
    path = pulse_ref[0]
    if path[0] != "optimal_control":
        raise ValueError(f"pulse_ref[0][0] must be 'optimal_control', got {path[0]!r}")
    node = cfg_root.device.optimal_control
    for key in path[1:-1]:
        node = node[key]
    return node


def _resolve_pulse_conf_dict(hw_cfg: dict, pulse_ref: list[list[Any]]) -> dict:
    """Plain-dict resolution. Mirrors _resolve_pulse_conf_attr for serialized hw_cfg."""
    if not pulse_ref or not pulse_ref[0]:
        raise ValueError("pulse_ref is required")
    path = pulse_ref[0]
    if path[0] != "optimal_control":
        raise ValueError(f"pulse_ref[0][0] must be 'optimal_control', got {path[0]!r}")
    node = hw_cfg["device"]["optimal_control"]
    for key in path[1:-1]:
        node = node[key]
    return node


# =================== station_config serialization ==========================

def _to_plain(obj, exclude_keys=None):
    """Recursively convert AttrDict/dict/numpy to JSON-safe primitives.

    Mirrors CharacterizationRunner._serialize_station_config's to_serializable_dict.
    """
    if exclude_keys is None:
        exclude_keys = set()
    if isinstance(obj, dict):
        return {k: _to_plain(v, exclude_keys) for k, v in obj.items() if k not in exclude_keys}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x, exclude_keys) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _build_station_data_template(station) -> tuple[dict, dict]:
    """Produce (template, hw_cfg_base).

    template: station_data dict ready for json.dumps after a hardware_cfg key
    is added. Includes multimode_cfg, storage_man_data, floquet_data, etc.

    hw_cfg_base: plain-dict snapshot of station.hardware_cfg (datasets stripped).
    Caller deepcopies and mutates this per-request to inject gain registers.
    """
    hw_cfg_base = _to_plain(station.hardware_cfg, exclude_keys={'_ds_storage', '_ds_floquet'})

    template: dict = {
        "experiment_name": station.experiment_name,
        "hardware_config_file": str(station.hardware_config_file),
    }
    if hasattr(station, "multimode_cfg") and hasattr(station, "multiphoton_config_file"):
        template["multimode_cfg"] = _to_plain(dict(station.multimode_cfg))
        template["multiphoton_config_file"] = str(station.multiphoton_config_file)
    if hasattr(station, "ds_storage"):
        df = station.ds_storage.df.copy()
        if "last_update" in df.columns:
            df["last_update"] = df["last_update"].astype(str)
        template["storage_man_data"] = df.to_dict(orient="records")
        template["storage_man_file"] = station.storage_man_file
    if hasattr(station, "ds_floquet") and station.ds_floquet is not None:
        df = station.ds_floquet.df.copy()
        if "last_update" in df.columns:
            df["last_update"] = df["last_update"].astype(str)
        template["floquet_data"] = df.to_dict(orient="records")
        template["floquet_file"] = station.floquet_file
    return template, hw_cfg_base


# =========================== hw via queue ==================================

def _resolve_gains(req: RunWignerRequest, hw_cfg_for_math: Any) -> tuple[int, int, str]:
    """Pick gain pair + source label for a /run_wigner request.

    Used by both /run_wigner and (indirectly) /calibrate_check via sub_req.
    """
    man_mode_idx = max(0, req.man_mode_no - 1)
    if req.gain_override is not None:
        gain_qb  = int(req.gain_override["qb"])
        gain_cav = int(req.gain_override["cav"])
        gains_source = "override"
    else:
        gain_qb, gain_cav = compute_gains_from_ghz(hw_cfg_for_math, req.IQ_table, man_mode_idx)
        gains_source = "computed_from_ghz"
    if not (0 <= gain_qb <= 32767):
        raise ValueError(
            f"computed gain_qb={gain_qb} outside QICK register range [0, 32767]. "
            f"Peak qubit drive too strong; reduce |I_q|/|Q_q| upstream."
        )
    if not (0 <= gain_cav <= 32767):
        raise ValueError(
            f"computed gain_cav={gain_cav} outside QICK register range [0, 32767]. "
            f"Peak cavity drive too strong; reduce |I_c|/|Q_c| upstream."
        )
    return gain_qb, gain_cav, gains_source


def submit_wigner_via_queue(req: RunWignerRequest, iter_id: str) -> tuple[np.ndarray, np.ndarray, Optional[str], dict]:
    """Build a WignerTomography1ModeExperiment job, submit, wait, extract parity."""
    if (_mock_station is None or _job_client is None
            or _serializable_hw_cfg_base is None or _station_data_template is None):
        raise ServiceNotReady(
            "service not initialized — start with --hardware-config and --storage-man-file."
        )
    if req.pulse_ref is None:
        raise ValueError("pulse_ref is required for hw mode")

    iq = req.IQ_table
    iq._check()
    max_abs = iq.max_abs()
    if max_abs > 1.0:
        raise ValueError(
            f"IQ_table peak |value|={max_abs:.4f} GHz exceeds DAC sample range (≤1.0 GHz). "
            "Likely a unit error (MHz vs GHz)."
        )

    gain_qb, gain_cav, gains_source = _resolve_gains(req, _mock_station.hardware_cfg)

    # Per-request hardware_cfg: deepcopy base, inject gain into pulse_conf
    hw_cfg = deepcopy(_serializable_hw_cfg_base)
    pulse_conf = _resolve_pulse_conf_dict(hw_cfg, req.pulse_ref)
    if "gain" not in pulse_conf or len(pulse_conf["gain"]) < 2:
        raise ValueError(
            f"pulse_conf for {req.pulse_ref} has no [qb, cav] gain slot to overwrite"
        )
    pulse_conf["gain"][0] = gain_qb
    pulse_conf["gain"][1] = gain_cav

    station_data = dict(_station_data_template)
    station_data["hardware_cfg"] = hw_cfg
    station_config_json = json.dumps(station_data)

    expt_config = {
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
        "relax_delay":           req.knobs.relax_delay,
    }

    t_submit = time.time()
    try:
        job_id = _job_client.submit_job(
            experiment_class="WignerTomography1ModeExperiment",
            experiment_module="experiments.qubit_cavity.single_mode_wigner_tomography",
            expt_config=expt_config,
            station_config=station_config_json,
            user="closed_loop",
            priority=5,
        )
    except Exception as e:
        raise ServiceNotReady(f"queue submit failed (is the queue server up at {_queue_url}?): {e}")

    result = _job_client.wait_for_completion(
        job_id,
        poll_interval=_WAIT_POLL_S,
        verbose=False,
        stream_output=False,
    )
    t_done = time.time()

    if not result.is_successful():
        raise RuntimeError(
            f"queue job {job_id} ended in status={result.status}: {result.error_message or 'no detail'}"
        )

    # Pull parity off the pickle the worker wrote. Same machine, same filesystem.
    try:
        expt = result.load_expt()
    except Exception as e:
        raise RuntimeError(f"could not load expt pickle for {job_id}: {e}")

    alphas_c = np.asarray(expt.data["alpha"])
    parity   = np.asarray(expt.data["parity"])

    meta = {
        "wall_total_s":      t_done - t_submit,
        "n_alphas":          int(len(alphas_c)),
        "pulse_samples":     len(iq.times),
        "pulse_duration_us": float(iq.times[-1]) if iq.times else 0.0,
        "iq_peak_ghz":       float(max_abs),
        "gain_qb":           gain_qb,
        "gain_cav":          gain_cav,
        "gains_source":      gains_source,
        "job_id":            job_id,
        "queue_url":         _queue_url,
        "hardware_config_version_id": result.hardware_config_version_id,
        "man1_storage_version_id":    result.man1_storage_version_id,
    }
    return alphas_c, parity, result.data_file_path, meta


# ============================== FastAPI app ================================

app = FastAPI(title="closed-loop experiment service (queue-client)", version="0.4.0")
START_TIME = datetime.now().isoformat(timespec="seconds")
HOSTNAME = socket.gethostname()


def _queue_reachable() -> bool:
    if _job_client is None:
        return False
    try:
        _job_client.health_check()
        return True
    except Exception:
        return False


@app.get("/")
def root():
    pinned = {}
    if _mock_station is not None:
        pinned["hardware_config_file"] = str(_mock_station.hardware_config_file)
        pinned["storage_man_file"] = getattr(_mock_station, "storage_man_file", None)
    return {
        "ok": True,
        "service": "expt_service",
        "version": "0.4.0",
        "mode": "queue_client",
        "hostname": HOSTNAME,
        "started": START_TIME,
        "queue_url": _queue_url,
        "queue_reachable": _queue_reachable(),
        "service_ready": _job_client is not None and _mock_station is not None,
        "pinned": pinned,
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


@app.post("/run_wigner", response_model=RunWignerResponse)
def run_wigner(req: RunWignerRequest):
    iter_id = uuid.uuid4().hex[:8]
    t0 = time.time()

    alphas_arr = np.asarray(req.alphas, dtype=float)
    if alphas_arr.ndim != 2 or alphas_arr.shape[1] != 2:
        raise HTTPException(400, f"alphas must be N x 2, got shape {alphas_arr.shape}")
    alphas_c = alphas_arr[:, 0] + 1j * alphas_arr[:, 1]

    try:
        req.IQ_table._check()
    except ValueError as e:
        raise HTTPException(400, str(e))

    if req.mode == "sim":
        parity_arr, meta = run_wigner_sim(req, alphas_c)
        shots_path: Optional[str] = None
        out_alphas = alphas_c
    elif req.mode == "hw":
        try:
            out_alphas, parity_arr, shots_path, meta = submit_wigner_via_queue(req, iter_id)
        except ServiceNotReady as e:
            raise HTTPException(503, str(e))
        except (ValueError, KeyError) as e:
            raise HTTPException(400, f"hw setup error: {e}")
        except RuntimeError as e:
            raise HTTPException(500, f"queue execution error: {e}")
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


@app.post("/calibrate_check", response_model=CalibrateCheckResponse)
def calibrate_check(req: CalibrateCheckRequest):
    """Run a known-good pulse at one displacement, compare to expected parity."""
    if _mock_station is None or _job_client is None:
        raise HTTPException(503, "service not initialized — pass --hardware-config / --storage-man-file at startup.")
    iter_id = uuid.uuid4().hex[:8]

    expected = req.expected_parity
    if expected is None:
        expected = _expected_parity_for_pulse(req.pulse_ref)

    # Resolve npz + canonical gains from the live (mock) station's hardware_cfg
    try:
        pulse_conf = _resolve_pulse_conf_attr(_mock_station.hardware_cfg, req.pulse_ref)
        filename = pulse_conf["filename"]
        iq = _load_iqtable_from_npz(filename)
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise HTTPException(400, f"calibrate_check setup error: {e}")

    if req.gain_override is not None:
        resolved_override = {"qb":  int(req.gain_override["qb"]),
                             "cav": int(req.gain_override["cav"])}
        gains_source_label = "override"
    else:
        try:
            canonical = pulse_conf["gain"]
            resolved_override = {"qb":  int(canonical[0]),
                                 "cav": int(canonical[1])}
        except (KeyError, IndexError, TypeError) as e:
            raise HTTPException(
                400,
                f"calibrate_check: no canonical gain in pulse_conf for "
                f"{req.pulse_ref}: {e}. Pass gain_override or fix the config."
            )
        gains_source_label = "canonical_from_config"

    sub_req = RunWignerRequest(
        mode="hw",
        IQ_table=iq,
        alphas=[req.alpha],
        reps=req.reps,
        pulse_ref=req.pulse_ref,
        gain_override=resolved_override,
        knobs=req.knobs,
        qubits=req.qubits,
        man_mode_no=req.man_mode_no,
    )

    try:
        _out_alphas, parity_arr, shots_path, meta = submit_wigner_via_queue(sub_req, iter_id)
    except ServiceNotReady as e:
        raise HTTPException(503, str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"hw setup error: {e}")
    except RuntimeError as e:
        raise HTTPException(500, f"queue execution error: {e}")

    measured = float(parity_arr[0])
    residual = None
    in_tol   = None
    if expected is not None:
        residual = measured - expected
        in_tol   = abs(residual) <= req.tolerance

    # submit_wigner_via_queue tagged gains_source as "override" because we
    # always pass gain_override; relabel to the higher-level provenance.
    meta_out = dict(meta)
    meta_out["gains_source"] = gains_source_label
    meta_out.update({
        "filename":         str(filename),
        "tolerance":        req.tolerance,
        "n_pulse_samples":  len(iq.times),
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


# ============================== lifecycle ==================================

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
    *,
    hardware_config: Optional[str] = None,
    storage_man_file: Optional[str] = None,
    queue_url: str = "http://127.0.0.1:8000",
    port: int = 18765,
    run_root: Optional[str | Path] = None,
    experiment_name: Optional[str] = None,
) -> ServiceHandle:
    """Initialize the service: load pinned configs into a mock station, build the
    station_config template, wire up the JobClient, then start uvicorn in a daemon thread.
    """
    global _mock_station, _serializable_hw_cfg_base, _station_data_template
    global _job_client, _run_root, _queue_url

    _queue_url = queue_url

    # 1) Load configs via a mock station — no hardware contention. The mock
    #    initialiser creates output dirs under D:/experiments/<exp_name>/.
    from experiments.station import MultimodeStation
    exp_name = experiment_name or f"{datetime.now().strftime('%y%m%d')}_closed_loop"
    print(f"[expt_service] loading pinned configs via MultimodeStation(mock=True, "
          f"hardware_config={hardware_config!r}, storage_man_file={storage_man_file!r}) ...")
    _mock_station = MultimodeStation(
        mock=True,
        experiment_name=exp_name,
        user="closed_loop",
        hardware_config=hardware_config,
        storage_man_file=storage_man_file,
    )

    # 2) Build the template + hardware_cfg base ONCE.
    template, hw_cfg_base = _build_station_data_template(_mock_station)
    _station_data_template = template
    _serializable_hw_cfg_base = hw_cfg_base

    # 3) JobClient — non-fatal if queue isn't up at startup. /run_wigner will 503
    #    later if the server is unreachable.
    _job_client = JobClient(server_url=_queue_url)
    try:
        h = _job_client.health_check()
        print(f"[expt_service] queue at {_queue_url} is reachable "
              f"(pending={h.get('pending_jobs')}, running={h.get('running_jobs')})")
    except Exception as e:
        print(f"[expt_service] WARNING: queue at {_queue_url} unreachable at startup: {e}")
        print(f"[expt_service]          /run_wigner requests will fail with 503 "
              f"until the queue server + worker are up.")

    # 4) Run root for any future direct shots dumps (currently the worker owns
    #    HDF5; this dir is just an artifact carrier for any future inline dumps).
    if run_root is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _run_root = Path(f"D:/closed_loop_runs/{ts}")
    else:
        _run_root = Path(run_root)
    _run_root.mkdir(parents=True, exist_ok=True)

    # 5) Boot uvicorn in a daemon thread.
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="expt_service")
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)
    return ServiceHandle(server, thread)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="closed_loop service (queue-client edition). "
                    "Forwards /run_wigner and /calibrate_check to the job queue."
    )
    parser.add_argument("--port", type=int, default=18765,
                        help="Port for this service (default 18765).")
    parser.add_argument("--queue-url", default="http://127.0.0.1:8000",
                        help="Job queue server URL (default http://127.0.0.1:8000).")
    parser.add_argument("--experiment-name", default=None,
                        help="MultimodeStation experiment name. Defaults to yymmdd_closed_loop.")
    parser.add_argument("--hardware-config", default=None,
                        help="Pin hardware config: filename or version ID (e.g. CFG-HW-20260515-00021).")
    parser.add_argument("--storage-man-file", default=None,
                        help="Pin man1 storage-swap dataset: filename or version ID (e.g. CFG-M1-20260513-00023).")
    args = parser.parse_args()

    # Make sure repo root is on sys.path (matches the old --hw path)
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    handle = start_service(
        hardware_config=args.hardware_config,
        storage_man_file=args.storage_man_file,
        queue_url=args.queue_url,
        port=args.port,
        experiment_name=args.experiment_name,
    )
    print(f"[expt_service] running on http://127.0.0.1:{args.port} (queue-client mode)")
    print(f"[expt_service]   queue url: {_queue_url}")
    print(f"[expt_service]   run root:  {_run_root}")
    print("[expt_service] Ctrl-C to stop")
    try:
        while handle.alive:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[expt_service] stopping...")
        handle.stop()
