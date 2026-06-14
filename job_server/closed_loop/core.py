"""Closed-loop measurement core — transport-agnostic.

This module owns all the logic that used to live inside the FastAPI handlers in
`service.py`: the request/response models, the sim backend, the GHz->gain
translation, the station_config assembly, queue submission, and parity
extraction. It owns NO transport. Callers reach it two ways:

  - `service.py`  — the legacy HTTP facade (`/run_wigner`, `/calibrate_check`).
  - `batch_runner.py` — the file-drop ("mailbox") runner that processes
    `pulses_*.zip` packages dropped into the shared Google Drive folder.

Both call `run_wigner_core(req)` / `calibrate_check_core(req)` after a one-time
`init_core(...)` that pins the hardware config + storage dataset and wires up
the JobClient. The functions raise plain exceptions (`ServiceNotReady`,
`ValueError`, `KeyError`, `RuntimeError`); the HTTP layer maps those to status
codes, the batch runner records them per-pulse.

Lifecycle requirement: the queue server (port 8000) AND its worker must be
running. This module connects to the queue server as a client.

Gain handling
-------------
  run_wigner_core       — Intonato-style pulse; gain registers computed via
                          compute_gains_from_ghz against the pinned hardware_config.
                          gain_override pins {qb, cav} manually.
  calibrate_check_core  — uses the canonical pulse_conf["gain"] stored in
                          hardware_config alongside the npz. gain_override pins
                          manually.

In both cases the resolved gain pair is BAKED into a deepcopy of hardware_cfg
serialized into the job's station_config; the worker's station picks it up via
_update_station_from_job_config, so the experiment runs with the gains we
resolved here.

IQ envelopes are in GHz Rabi rate (Piccolo native unit, matches the npz files
in device.optimal_control). DAC sample overflow boundary: peak <= 1.0 GHz.
"""
from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import qutip
from pydantic import BaseModel, Field

from fitting.state_tomography import (
    PX, PY, PZ, reconstruct_single_qubit, state_fidelity,
)
from fitting.wigner import WignerAnalysis
from job_server.client import JobClient


# ============================== module state ===============================
# Core is stateless w.r.t. hardware; we keep these to avoid reloading pinned
# configs and rebuilding the station_config template on every call.

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
    # active_reset accepts either a bool (back-compat: expands to {"active_reset": bool})
    # or a full dict like {'active_reset': True, 'pre_selection_reset': True,
    # 'ef_reset': True, ...}. The full dict is spliced into BOTH the
    # HistogramExperiment used for IQ-blob recalibration AND the WignerTomography
    # measurement, so the readout is calibrated under the same reset conditions
    # the Wigner runs under. See MM_base.get_active_reset_params for the full
    # set of keys honored by the experiment side.
    active_reset:          Union[bool, dict] = False
    post_select_pre_pulse: bool  = False
    parity_fast:           bool  = False
    prepulse:              bool  = False
    gate_based:            bool  = False
    relax_delay:           Optional[float] = None  # None -> pull from pinned config
    pre_sweep_pulse:       list  = Field(default_factory=list)
    pre_gate_sweep_pulse:  list  = Field(default_factory=list)
    # When True, run a HistogramExperiment (single-shot) just before the Wigner
    # job and inject the freshly-fit readout phase/threshold into the per-job
    # hw_cfg. Catches IQ-blob drift between back-to-back Wigner runs (see
    # GAIN_DERIVATION.md / batch_runner LAB_DEFAULTS).
    recalibrate_readout:   bool  = False
    recalibrate_reps:      int   = 2000
    # sigma_z probe: measure the post-prep transmon (P_g - P_e) alongside parity to
    # catch leakage out of |g> that the displaced-parity operator is blind to. The
    # collaborator boolean measure_sigma_z maps to the 'reset' tier (Tier 1: measure
    # then actively reset the ancilla to |g> before parity) unless sigma_z_mode is
    # given explicitly ('off' | 'reset' | 'postselect'). See MM_base.lane_layout.
    measure_sigma_z:       bool  = False
    sigma_z_mode:          Optional[str] = None
    # Wigner reconstruction: when True, run the MLE-style density-matrix
    # reconstruction (WignerAnalysis.wigner_analysis_results) on the measured
    # parity grid and return rho/populations/fidelity in the response. Default
    # off so existing parity-only callers keep their fast path.
    reconstruct:           bool  = False
    reconstruct_fock_dim:  int   = 5      # Hilbert-space cutoff for the reconstruction
    reconstruct_rotate:    bool  = False  # phase-rotation search to maximize fidelity


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

    # Explicit reference state for fidelity (only used when knobs.reconstruct).
    #   {"type": "fock", "n": 1}
    #   {"type": "ket", "amps": [[re, im], ...]}   # length == reconstruct_fock_dim
    # None -> return rho + populations only, fidelity = None.
    target_state: Optional[dict] = None


class RunWignerResponse(BaseModel):
    ok: bool
    alphas: list[list[float]]
    parity: list[float]
    mode: str
    iter_id: str
    shots_path: Optional[str] = None
    meta: dict
    sigma_z: Optional[float] = None
    sigma_z_raw: Optional[float] = None
    # Populated only when knobs.reconstruct is set (else None).
    rho:         Optional[dict] = None          # {"real": [[...]], "imag": [[...]]}
    populations: Optional[list[float]] = None   # diagonal of rho (photon-number pops)
    fidelity:    Optional[float] = None         # None unless target_state was given


class CalibrateCheckRequest(BaseModel):
    """Reference measurement against a known-good pulse from the config.

    Loads the IQ_table from the npz file referenced by `pulse_ref`, runs a
    single-displacement Wigner measurement, and compares to an expected parity.
    Default: Fock |1>, alpha=0, expected parity = -1.

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


class RunStateTomo1QRequest(BaseModel):
    """Single-qubit (transmon) state tomography of the state an optimized pulse
    prepares. Sibling of RunWignerRequest: same IQ_table/pulse_ref/gain plumbing,
    but instead of sweeping cavity displacements it measures the transmon in the
    Z/X/Y bases and reconstructs the 2x2 density matrix
    (experiments.single_qubit.state_tomography_1q.StateTomography1QExperiment).
    """
    mode:        Literal["sim", "hw"] = "sim"
    IQ_table:    IQTable
    pulse_ref:   Optional[list[list[Any]]] = None   # required for hw
    reps:        int = Field(default=1000, ge=1)
    rounds:      int = 1
    bases:       list[str] = Field(default_factory=lambda: ["Z", "X", "Y"])
    # Pre-rotation phases (deg) realizing the X/Y axes; None -> experiment default.
    tomo_phases: Optional[dict] = None
    # Optional gate-based prep composed AFTER the inline opt pulse (e.g. an extra
    # rotation). Usually None for the closed-loop path (the opt pulse is the prep).
    state_prep_seq: Optional[list] = None
    state_prep_postselect: bool = False
    recon_method:  Literal["fast", "cholesky", "linear"] = "fast"
    # Reference ket [[re,im],[re,im]] for fidelity; None -> fidelity omitted.
    target_state:  Optional[list[list[float]]] = None
    # 2x2 or flat-4 readout assignment matrix; None -> no confusion correction.
    confusion:     Optional[list] = None
    gain_override: Optional[dict[str, int]] = None
    knobs:       Knobs = Field(default_factory=Knobs)
    qubits:      list[int] = Field(default_factory=lambda: [0])
    man_mode_no: int = 1

    # sim-only escape hatch: simulate counts for this exact ket instead of |g>.
    sim_target_state: Optional[list[list[float]]] = None


class RunStateTomo1QResponse(BaseModel):
    ok: bool
    mode: str
    iter_id: str
    bases: list[str]
    counts: dict                              # {'Z': [n_g, n_e], ...}
    expectations: dict                        # {'Z': <Z>, ...}
    rho: Optional[dict] = None                # {"real": [[...]], "imag": [[...]]}
    fidelity: Optional[float] = None          # None unless target_state given
    azimuth_rad: Optional[float] = None       # present iff X and Y measured
    equatorial_contrast: Optional[float] = None
    shots_path: Optional[str] = None
    meta: dict


# ============================== sim backend ================================

def pulse_to_beta(iq: IQTable) -> complex:
    """Toy coherent-state amplitude after driving cavity with iq (in GHz Rabi)."""
    times = np.asarray(iq.times, dtype=float)
    Ic = np.asarray(iq.I_c, dtype=float)
    Qc = np.asarray(iq.Q_c, dtype=float)
    if len(times) < 2:
        return 0.0 + 0.0j
    dt = float(times[1] - times[0])  # us
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


def _ket_from_pairs(pairs: list[list[float]]) -> np.ndarray:
    """[[re,im], ...] -> normalized complex ket (1-D ndarray)."""
    v = np.array([complex(p[0], p[1]) for p in pairs], dtype=complex)
    n = np.linalg.norm(v)
    return v / n if n else v


def run_state_tomo_1q_sim(req: "RunStateTomo1QRequest") -> dict:
    """Toy single-qubit tomography: sample noisy Z/X/Y counts for a known state
    (sim_target_state, else target_state, else |g>), then reconstruct. Mirrors
    run_wigner_sim — plumbing test only, no pulse physics is modeled.
    """
    if req.sim_target_state is not None:
        psi = _ket_from_pairs(req.sim_target_state)
    elif req.target_state is not None:
        psi = _ket_from_pairs(req.target_state)
    else:
        psi = np.array([1.0, 0.0], dtype=complex)  # |g>
    rho_true = np.outer(psi, psi.conj())

    axis_op = {"X": PX, "Y": PY, "Z": PZ}
    counts: dict = {}
    expectations: dict = {}
    for b in req.bases:
        ev = float(np.real(np.trace(rho_true @ axis_op[b])))   # <A> in [-1, 1]
        p_g = float(np.clip((1.0 + ev) / 2.0, 0.0, 1.0))
        n_g = int(np.random.binomial(req.reps, p_g))
        n_e = req.reps - n_g
        counts[b] = (n_g, n_e)
        expectations[b] = (n_g - n_e) / req.reps if req.reps else 0.0
    meta = {"n_samples": len(req.IQ_table.times), "sim_state_re": psi.real.tolist(),
            "sim_state_im": psi.imag.tolist()}
    return {"counts": counts, "expectations": expectations, "meta": meta}


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
        raise ValueError(f"only gaussian pi pulse supported (got {pi_type!r})")

    theta_to_gain_qb = np.pi / 2 / gain_pi
    drive_to_gain_qb = sigma_pi * np.sqrt(np.pi) / theta_to_gain_qb * sps.erf(n / 2)

    alpha_to_gain = hw_cfg.device.manipulate.gain_to_alpha[man_mode_idx]
    sigma_cav     = hw_cfg.device.manipulate.displace_sigma[man_mode_idx]
    drive_to_gain_cav = sigma_cav * np.sqrt(np.pi) / alpha_to_gain * sps.erf(n / 2)

    gain_qb  = int(round(max_q * drive_to_gain_qb))
    gain_cav = int(round(max_c * drive_to_gain_cav))
    return gain_qb, gain_cav


def _expected_parity_for_pulse(pulse_ref: list[list[Any]]) -> Optional[float]:
    """Auto-deduce expected parity at alpha=0. Fock |n> -> (-1)^n; superpositions -> None."""
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


def _build_target_ket(spec: Optional[dict], fock_dim: int) -> Optional[qutip.Qobj]:
    """Build a reference ket for fidelity from an explicit request spec.

    None -> None (caller skips fidelity). Otherwise:
      {"type": "fock", "n": k}                 -> qutip.fock(fock_dim, k)
      {"type": "ket",  "amps": [[re,im], ...]} -> normalized ket of length fock_dim
    Raises ValueError on a malformed spec.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"target_state must be a dict with a 'type' key, got {spec!r}")
    kind = spec["type"]
    if kind == "fock":
        try:
            n = int(spec["n"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"target_state fock spec needs an integer 'n', got {spec!r}")
        if not (0 <= n < fock_dim):
            raise ValueError(
                f"target_state fock n={n} out of range for reconstruct_fock_dim={fock_dim}"
            )
        return qutip.fock(fock_dim, n)
    if kind == "ket":
        amps = spec.get("amps")
        if not amps or len(amps) != fock_dim:
            raise ValueError(
                f"target_state ket 'amps' must have length reconstruct_fock_dim={fock_dim}, "
                f"got {None if not amps else len(amps)}"
            )
        vec = np.array([complex(a[0], a[1]) for a in amps], dtype=np.complex128)
        return qutip.Qobj(vec.reshape(fock_dim, 1)).unit()
    raise ValueError(f"unknown target_state type {kind!r} (expected 'fock' or 'ket')")


def _reconstruct_from_parity(
    parity: np.ndarray,
    alphas_c: np.ndarray,
    fock_dim: int,
    target_ket: Optional[qutip.Qobj],
    rotate: bool,
) -> dict:
    """Run the MLE-style Wigner reconstruction on a measured parity grid.

    Reuses WignerAnalysis.wigner_analysis_results, which only needs the parity
    values, the complex displacements, and the Fock cutoff -- no raw IQ / config
    (passing threshold=0.0, config=None keeps GeneralFitting.__init__ off the cfg
    path). Returns JSON-serializable rho/populations/fidelity.
    """
    alphas_c = np.asarray(alphas_c)
    wa = WignerAnalysis(
        data={"alpha": alphas_c}, threshold=0.0, config=None,
        mode_state_num=fock_dim, alphas=alphas_c,
    )
    init = target_ket if target_ket is not None else qutip.fock(fock_dim, 0)
    res = wa.wigner_analysis_results(
        np.asarray(parity, dtype=float), initial_state=init, rotate=rotate,
    )
    rho = np.asarray(res["rho"])
    return {
        "rho": {"real": rho.real.tolist(), "imag": rho.imag.tolist()},
        "populations": [float(np.real(rho[i, i])) for i in range(fock_dim)],
        "fidelity": float(res["fidelity"]) if target_ket is not None else None,
    }


def _load_iqtable_from_npz(filename: str) -> IQTable:
    """Load a Piccolo-style npz: times in ns -> us; I_c/Q_c/I_q/Q_q in GHz."""
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


def _active_reset_dict(ar: Union[bool, dict]) -> dict:
    """Normalize Knobs.active_reset into a dict suitable for **-splicing into an
    experiment expt_config.

    Bool form  → {"active_reset": bool}                  (back-compat)
    Dict form  → defensive copy of the dict, passed through verbatim. The dict is
                 expected to carry 'active_reset' along with any
                 MM_base.get_active_reset_params keys (pre_selection_reset,
                 ef_reset, man_reset, ...).
    """
    if isinstance(ar, bool):
        return {"active_reset": ar}
    return dict(ar)


def _config_confusion_for_reset(active_reset: Union[bool, dict]) -> Optional[list]:
    """Readout confusion matrix from the pinned config, keyed on reset regime.

    active_reset on -> device.readout.confusion_matrix_with_active_reset,
    else            -> device.readout.confusion_matrix_without_reset.
    Returns the lab flat-4 list [Pgg, Pge, Peg, Pee] (consumed by
    fitting.state_tomography.as_confusion_matrix), or None if unavailable.
    """
    if _mock_station is None:
        return None
    ar_on = bool(_active_reset_dict(active_reset).get("active_reset", False))
    try:
        ro = _mock_station.hardware_cfg.device.readout
        cm = ro.confusion_matrix_with_active_reset if ar_on else ro.confusion_matrix_without_reset
        return [float(v) for v in cm] if cm is not None else None
    except Exception:
        return None


def _resolve_relax_delay(req: RunWignerRequest) -> float:
    """relax_delay precedence: explicit knob override -> pinned config -> 2500.

    Pulls device.readout.relax_delay[0] from the pinned hardware_config when the
    request doesn't override it, so whichever config is pinned governs it.
    """
    if req.knobs.relax_delay is not None:
        return float(req.knobs.relax_delay)
    if _mock_station is not None:
        try:
            return float(_mock_station.hardware_cfg.device.readout.relax_delay[0])
        except Exception:
            pass
    return 2500.0


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
    """Pick gain pair + source label for a run_wigner request.

    Used by both run_wigner_core and (indirectly) calibrate_check_core via sub_req.
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


def _recalibrate_readout(
    qubits: list[int],
    reps: int,
    hw_cfg_base: dict,
    station_data_template: dict,
    iter_id: str,
    active_reset_dict: dict,
) -> tuple[float, float, dict]:
    """Run a HistogramExperiment to re-fit the IQ-blob rotation + ge threshold.

    Returns (delta_phase_deg, new_threshold, meta). The convention (matches
    measurement_notebooks/.../single_qubit_autocalibrate_v2.ipynb:548-550):

        device.readout.phase[0]        := old + delta_phase_deg   (ADDITIVE)
        device.readout.threshold[0]    := new_threshold           (replace)
        device.readout.threshold_list  := [[new_threshold]]       (replace)

    The readout cfg used for this single-shot is the SAME pinned hw_cfg_base
    the Wigner will use (modulo phase/threshold), so the rotation it returns
    is the delta needed *from* that baseline.

    `active_reset_dict` is spliced into the HistogramExperiment cfg so the
    IQ-blob calibration sees the SAME reset conditions the Wigner will run
    under (call site passes `_active_reset_dict(req.knobs.active_reset)`).
    """
    if _job_client is None:
        raise ServiceNotReady("core not initialized — call init_core() first.")

    # Single-shot uses pinned hw_cfg as-is; no per-request mutation.
    station_data = dict(station_data_template)
    station_data["hardware_cfg"] = hw_cfg_base
    station_config_json = json.dumps(station_data)

    expt_config = {
        "qubits":       qubits,
        "reps":         int(reps),
        "rounds":       1,
        "check_e":      True,
        "check_f":      False,
        "prepulse":     False,
        **active_reset_dict,
    }

    t0 = time.time()
    try:
        job_id = _job_client.submit_job(
            experiment_class="HistogramExperiment",
            experiment_module="experiments.single_qubit.single_shot",
            expt_config=expt_config,
            station_config=station_config_json,
            user="closed_loop_recal",
            priority=0,  # same as ad-hoc notebook jobs -> FIFO, no queue-jumping
        )
    except Exception as e:
        raise ServiceNotReady(f"queue submit failed for recal (queue at {_queue_url}?): {e}")

    result = _job_client.wait_for_completion(
        job_id, poll_interval=_WAIT_POLL_S, verbose=False, stream_output=False
    )
    if not result.is_successful():
        raise RuntimeError(
            f"recal job {job_id} ended in status={result.status}: "
            f"{result.error_message or 'no detail'}"
        )

    expt = result.load_expt()
    angle_deg     = float(np.asarray(expt.data["angle"]).ravel()[0])
    thresholds    = np.asarray(expt.data["thresholds"]).ravel()
    new_threshold = float(thresholds[0])
    fids          = np.asarray(expt.data.get("fids", [np.nan])).ravel()
    fid_ge        = float(fids[0]) if fids.size > 0 else float("nan")
    # Fresh readout confusion matrix from this same single-shot, computed at the
    # just-fit angle/threshold under the same reset conditions the measurement
    # will run under -- flat-4 [Pgg, Pge, Peg, Pee], same convention as the config
    # (fitting/fit_display.py:223). Surfaced so the reconstruction can use it
    # instead of the (possibly stale) config matrix.
    _cm = expt.data.get("confusion_matrix")
    recal_confusion = [float(v) for v in np.asarray(_cm).ravel()] if _cm is not None else None

    meta = {
        "recal_job_id":      job_id,
        "recal_wall_s":      time.time() - t0,
        "recal_reps":        int(reps),
        "recal_angle_deg":   angle_deg,
        "recal_threshold":   new_threshold,
        "recal_fid_ge":      fid_ge,
        "recal_iter_id":     iter_id,
        "recal_confusion_matrix": recal_confusion,
    }
    return angle_deg, new_threshold, meta


def _normalize_iq_table_for_inline_path(iq: IQTable) -> tuple[IQTable, float, float]:
    """Normalize an IQ_table so each channel-pair has peak amplitude 1.0.

    `MM_base.load_opt_ctrl_pulse` (the inline-IQ_table playback path) multiplies
    the raw IQ_table values by `maxv` (the DAC full-scale) but does NOT
    self-normalize, unlike the npz path in `custom_pulse` which divides by
    `IQ_scale = max(|I|, |Q|)` before scaling by `maxv`. With un-normalized
    small linear-GHz values (typical peaks ~0.001-0.05), the envelope peak ends
    up at only `peak * maxv` DAC units (tens to hundreds out of 32767), and the
    gain register can only scale that — net output is hundreds of times weaker
    than `compute_gains_from_ghz` thinks. (See `experiments/MM_base.py` L542-568
    vs L905-922 for the asymmetry.)

    Fix: pre-normalize per channel-pair before sending. We compute the gain
    register from the original linear-GHz peaks (already done in _resolve_gains)
    and then send a normalized envelope so the gain register controls the
    actual DAC amplitude — matching what the npz path does internally.

    Returns (normalized_iq, peak_qb, peak_cav).
    """
    def _peak(arrs):
        m = 0.0
        for arr in arrs:
            if arr:
                m = max(m, max(abs(v) for v in arr))
        return m

    peak_qb  = _peak([iq.I_q, iq.Q_q])
    peak_cav = _peak([iq.I_c, iq.Q_c])
    sq = peak_qb  if peak_qb  > 0 else 1.0
    sc = peak_cav if peak_cav > 0 else 1.0
    return (
        IQTable(
            times=list(iq.times),
            I_c=[v / sc for v in iq.I_c],
            Q_c=[v / sc for v in iq.Q_c],
            I_q=[v / sq for v in iq.I_q],
            Q_q=[v / sq for v in iq.Q_q],
        ),
        peak_qb,
        peak_cav,
    )


def submit_wigner_via_queue(req: RunWignerRequest, iter_id: str) -> tuple[np.ndarray, np.ndarray, Optional[str], dict]:
    """Build a WignerTomography1ModeExperiment job, submit, wait, extract parity."""
    if (_mock_station is None or _job_client is None
            or _serializable_hw_cfg_base is None or _station_data_template is None):
        raise ServiceNotReady(
            "core not initialized — call init_core() with hardware_config and storage_man_file."
        )
    if req.pulse_ref is None:
        raise ValueError("pulse_ref is required for hw mode")

    iq = req.IQ_table
    iq._check()
    max_abs = iq.max_abs()
    if max_abs > 1.0:
        raise ValueError(
            f"IQ_table peak |value|={max_abs:.4f} GHz exceeds DAC sample range (<=1.0 GHz). "
            "Likely a unit error (MHz vs GHz)."
        )

    gain_qb, gain_cav, gains_source = _resolve_gains(req, _mock_station.hardware_cfg)

    # Normalize active_reset (bool back-compat or full dict) once; splice into
    # both the recal HistogramExperiment and the Wigner expt_config so the IQ-blob
    # fit and the parity measurement run under identical reset conditions.
    ar_dict = _active_reset_dict(req.knobs.active_reset)

    # Optional readout recalibration — fits IQ-blob rotation + ge threshold via
    # a HistogramExperiment, returns the delta we need to apply on top of the
    # baseline phase/threshold. Done BEFORE the per-job hw_cfg deepcopy so the
    # recal sees the same pinned baseline the Wigner will be measured against.
    recal_meta: dict = {}
    if req.knobs.recalibrate_readout:
        try:
            delta_phase_deg, new_threshold, recal_meta = _recalibrate_readout(
                qubits=req.qubits,
                reps=req.knobs.recalibrate_reps,
                hw_cfg_base=_serializable_hw_cfg_base,
                station_data_template=_station_data_template,
                iter_id=iter_id,
                active_reset_dict=ar_dict,
            )
        except ServiceNotReady:
            raise
        except Exception as e:
            raise RuntimeError(f"readout recalibration failed: {e}")
    else:
        delta_phase_deg, new_threshold = 0.0, None

    # Per-request hardware_cfg: deepcopy base, then inject (a) recalibrated
    # readout phase/threshold if we recal'd, (b) gain into the pulse_conf slot,
    # (c) override the opt_pulse carrier frequencies to the current bare
    # device.qubit.f_ge / device.manipulate.f_ge — the cached
    # optimal_control[*][*].frequency entries can be stale (calibrated against
    # an earlier resonance), so the inline-IQ_table playback would otherwise
    # play off-resonant from where the device actually sits today. The standard
    # Wigner displacement uses the bare f_ge values, so this also keeps our
    # cavity drive on the same resonance the tomography is calibrated against.
    hw_cfg = deepcopy(_serializable_hw_cfg_base)
    if req.knobs.recalibrate_readout:
        ro = hw_cfg["device"]["readout"]
        old_phase = float(ro["phase"][0])
        new_phase = old_phase + delta_phase_deg
        ro["phase"]          = [new_phase]
        ro["threshold"]      = [new_threshold]
        ro["threshold_list"] = [[new_threshold]]
        recal_meta["recal_phase_old_deg"] = old_phase
        recal_meta["recal_phase_new_deg"] = new_phase

    pulse_conf = _resolve_pulse_conf_dict(hw_cfg, req.pulse_ref)
    if "gain" not in pulse_conf or len(pulse_conf["gain"]) < 2:
        raise ValueError(
            f"pulse_conf for {req.pulse_ref} has no [qb, cav] gain slot to overwrite"
        )
    pulse_conf["gain"][0] = gain_qb
    pulse_conf["gain"][1] = gain_cav

    # Override frequencies to current bare f_ge values (see comment above).
    man_mode_idx = max(0, req.man_mode_no - 1)
    f_qb_bare  = float(hw_cfg["device"]["qubit"]["f_ge"][req.qubits[0]])
    f_cav_bare = float(hw_cfg["device"]["manipulate"]["f_ge"][man_mode_idx])
    f_old = list(pulse_conf.get("frequency", [None, None]))
    pulse_conf["frequency"] = [f_qb_bare, f_cav_bare]
    freq_override_meta = {
        "freq_qb_old_MHz":  f_old[0] if len(f_old) > 0 else None,
        "freq_qb_used_MHz": f_qb_bare,
        "freq_cav_old_MHz":  f_old[1] if len(f_old) > 1 else None,
        "freq_cav_used_MHz": f_cav_bare,
    }

    station_data = dict(_station_data_template)
    station_data["hardware_cfg"] = hw_cfg
    station_config_json = json.dumps(station_data)

    # Normalize the inline IQ_table to peak=1 per channel-pair so the
    # gain register actually controls the DAC output. See
    # `_normalize_iq_table_for_inline_path` for the full rationale.
    iq_to_send, peak_qb_sent, peak_cav_sent = _normalize_iq_table_for_inline_path(iq)

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
        "measure_sigma_z":       req.knobs.measure_sigma_z,
        "sigma_z_mode":          req.knobs.sigma_z_mode,
        "opt_pulse":             req.pulse_ref,
        "IQ_table":              iq_to_send.model_dump(),
        "parity_fast":           req.knobs.parity_fast,
        "gate_based":            req.knobs.gate_based,
        "man_mode_no":           req.man_mode_no,
        "relax_delay":           _resolve_relax_delay(req),
        **ar_dict,
    }

    t_submit = time.time()
    try:
        job_id = _job_client.submit_job(
            experiment_class="WignerTomography1ModeExperiment",
            experiment_module="experiments.qubit_cavity.single_mode_wigner_tomography",
            expt_config=expt_config,
            station_config=station_config_json,
            user="closed_loop",
            priority=0,  # same as ad-hoc notebook jobs -> FIFO, no queue-jumping
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
    sigma_z     = expt.data.get("sigma_z", None)
    sigma_z_raw = expt.data.get("sigma_z_raw", None)

    meta = {
        "wall_total_s":      t_done - t_submit,
        "n_alphas":          int(len(alphas_c)),
        "pulse_samples":     len(iq.times),
        "pulse_duration_us": float(iq.times[-1]) if iq.times else 0.0,
        "iq_peak_ghz":       float(max_abs),
        "iq_peak_qb_ghz":    float(peak_qb_sent),
        "iq_peak_cav_ghz":   float(peak_cav_sent),
        "iq_table_normalized": True,
        "gain_qb":           gain_qb,
        "gain_cav":          gain_cav,
        "gains_source":      gains_source,
        "job_id":            job_id,
        "queue_url":         _queue_url,
        "hardware_config_version_id": result.hardware_config_version_id,
        "man1_storage_version_id":    result.man1_storage_version_id,
    }
    if recal_meta:
        meta["readout_recal"] = recal_meta
    meta["freq_override"] = freq_override_meta
    return alphas_c, parity, result.data_file_path, meta, sigma_z, sigma_z_raw


def submit_state_tomo_1q_via_queue(req: "RunStateTomo1QRequest", iter_id: str) -> tuple[dict, dict, Optional[str], dict]:
    """Build a StateTomography1QExperiment job, submit, wait, extract counts.

    Shares all the gain/readout/hw_cfg plumbing with submit_wigner_via_queue
    (it reuses _resolve_gains, _recalibrate_readout, _resolve_pulse_conf_dict,
    _normalize_iq_table_for_inline_path, _resolve_relax_delay). The only
    differences are the expt_config (Z/X/Y bases instead of a displacement
    sweep) and the experiment class. Returns (counts, expectations, shots_path,
    meta); reconstruction/fidelity happen in run_state_tomo_1q_core.
    """
    if (_mock_station is None or _job_client is None
            or _serializable_hw_cfg_base is None or _station_data_template is None):
        raise ServiceNotReady(
            "core not initialized — call init_core() with hardware_config and storage_man_file."
        )
    if req.pulse_ref is None:
        raise ValueError("pulse_ref is required for hw mode")

    iq = req.IQ_table
    iq._check()
    max_abs = iq.max_abs()
    if max_abs > 1.0:
        raise ValueError(
            f"IQ_table peak |value|={max_abs:.4f} GHz exceeds DAC sample range (<=1.0 GHz). "
            "Likely a unit error (MHz vs GHz)."
        )

    gain_qb, gain_cav, gains_source = _resolve_gains(req, _mock_station.hardware_cfg)
    ar_dict = _active_reset_dict(req.knobs.active_reset)

    recal_meta: dict = {}
    if req.knobs.recalibrate_readout:
        try:
            delta_phase_deg, new_threshold, recal_meta = _recalibrate_readout(
                qubits=req.qubits,
                reps=req.knobs.recalibrate_reps,
                hw_cfg_base=_serializable_hw_cfg_base,
                station_data_template=_station_data_template,
                iter_id=iter_id,
                active_reset_dict=ar_dict,
            )
        except ServiceNotReady:
            raise
        except Exception as e:
            raise RuntimeError(f"readout recalibration failed: {e}")
    else:
        delta_phase_deg, new_threshold = 0.0, None

    hw_cfg = deepcopy(_serializable_hw_cfg_base)
    if req.knobs.recalibrate_readout:
        ro = hw_cfg["device"]["readout"]
        old_phase = float(ro["phase"][0])
        new_phase = old_phase + delta_phase_deg
        ro["phase"]          = [new_phase]
        ro["threshold"]      = [new_threshold]
        ro["threshold_list"] = [[new_threshold]]
        recal_meta["recal_phase_old_deg"] = old_phase
        recal_meta["recal_phase_new_deg"] = new_phase

    pulse_conf = _resolve_pulse_conf_dict(hw_cfg, req.pulse_ref)
    if "gain" not in pulse_conf or len(pulse_conf["gain"]) < 2:
        raise ValueError(
            f"pulse_conf for {req.pulse_ref} has no [qb, cav] gain slot to overwrite"
        )
    pulse_conf["gain"][0] = gain_qb
    pulse_conf["gain"][1] = gain_cav

    man_mode_idx = max(0, req.man_mode_no - 1)
    f_qb_bare  = float(hw_cfg["device"]["qubit"]["f_ge"][req.qubits[0]])
    f_cav_bare = float(hw_cfg["device"]["manipulate"]["f_ge"][man_mode_idx])
    f_old = list(pulse_conf.get("frequency", [None, None]))
    pulse_conf["frequency"] = [f_qb_bare, f_cav_bare]
    freq_override_meta = {
        "freq_qb_old_MHz":  f_old[0] if len(f_old) > 0 else None,
        "freq_qb_used_MHz": f_qb_bare,
        "freq_cav_old_MHz":  f_old[1] if len(f_old) > 1 else None,
        "freq_cav_used_MHz": f_cav_bare,
    }

    station_data = dict(_station_data_template)
    station_data["hardware_cfg"] = hw_cfg
    station_config_json = json.dumps(station_data)

    iq_to_send, peak_qb_sent, peak_cav_sent = _normalize_iq_table_for_inline_path(iq)

    expt_config = {
        "reps":                  req.reps,
        "rounds":                req.rounds,
        "qubits":                req.qubits,
        "opt_pulse":             req.pulse_ref,
        "IQ_table":              iq_to_send.model_dump(),
        "bases":                 list(req.bases),
        "state_prep_postselect": req.state_prep_postselect,
        "man_mode_no":           req.man_mode_no,
        "relax_delay":           _resolve_relax_delay(req),
        **ar_dict,
    }
    if req.tomo_phases is not None:
        expt_config["tomo_phases"] = req.tomo_phases
    if req.state_prep_seq is not None:
        expt_config["state_prep_seq"] = req.state_prep_seq

    t_submit = time.time()
    try:
        job_id = _job_client.submit_job(
            experiment_class="StateTomography1QExperiment",
            experiment_module="experiments.single_qubit.state_tomography_1q",
            expt_config=expt_config,
            station_config=station_config_json,
            user="closed_loop",
            priority=0,
        )
    except Exception as e:
        raise ServiceNotReady(f"queue submit failed (is the queue server up at {_queue_url}?): {e}")

    result = _job_client.wait_for_completion(
        job_id, poll_interval=_WAIT_POLL_S, verbose=False, stream_output=False,
    )
    t_done = time.time()

    if not result.is_successful():
        raise RuntimeError(
            f"queue job {job_id} ended in status={result.status}: {result.error_message or 'no detail'}"
        )

    try:
        expt = result.load_expt()
    except Exception as e:
        raise RuntimeError(f"could not load expt pickle for {job_id}: {e}")

    counts = expt.data.get("counts")
    expectations = expt.data.get("expectations")
    if counts is None:
        raise RuntimeError(
            f"job {job_id} produced no 'counts' in expt.data; analyze() may have failed."
        )

    meta = {
        "wall_total_s":      t_done - t_submit,
        "bases":             list(req.bases),
        "pulse_samples":     len(iq.times),
        "pulse_duration_us": float(iq.times[-1]) if iq.times else 0.0,
        "iq_peak_ghz":       float(max_abs),
        "iq_peak_qb_ghz":    float(peak_qb_sent),
        "iq_peak_cav_ghz":   float(peak_cav_sent),
        "iq_table_normalized": True,
        "gain_qb":           gain_qb,
        "gain_cav":          gain_cav,
        "gains_source":      gains_source,
        "job_id":            job_id,
        "queue_url":         _queue_url,
        "hardware_config_version_id": result.hardware_config_version_id,
        "man1_storage_version_id":    result.man1_storage_version_id,
    }
    if recal_meta:
        meta["readout_recal"] = recal_meta
    meta["freq_override"] = freq_override_meta
    return counts, expectations, result.data_file_path, meta


def _finalize_state_tomo_1q(counts: dict, req: "RunStateTomo1QRequest", *,
                            mode: str, iter_id: str, shots_path: Optional[str],
                            meta: dict, confusion: Optional[list] = None) -> "RunStateTomo1QResponse":
    """Shared sim/hw finalizer: from per-basis (n_g, n_e) counts, recompute
    expectations, reconstruct the density matrix (when Z/X/Y all present),
    compute fidelity to target_state, and the equatorial azimuth/contrast.

    `confusion` (a resolved readout assignment matrix, flat-4 or 2x2) overrides
    req.confusion when given — the hw path passes the config matrix keyed on the
    active-reset regime so the reconstruction is confusion-corrected by default.
    """
    counts = {b: (int(c[0]), int(c[1])) for b, c in counts.items()}
    expectations = {}
    for b, (n_g, n_e) in counts.items():
        tot = n_g + n_e
        expectations[b] = (n_g - n_e) / tot if tot else 0.0

    conf = confusion if confusion is not None else req.confusion
    rho_out = None
    fidelity_out = None
    if all(b in counts for b in ("Z", "X", "Y")):
        confusion_arr = np.asarray(conf, dtype=float) if conf is not None else None
        rho = reconstruct_single_qubit(counts, confusion=confusion_arr, method=req.recon_method)
        rho = np.asarray(rho)
        rho_out = {"real": rho.real.tolist(), "imag": rho.imag.tolist()}
        if req.target_state is not None:
            fidelity_out = float(state_fidelity(rho, _ket_from_pairs(req.target_state)))

    azimuth = contrast = None
    if "X" in expectations and "Y" in expectations:
        azimuth = float(np.arctan2(expectations["Y"], expectations["X"]))
        contrast = float(np.hypot(expectations["X"], expectations["Y"]))

    return RunStateTomo1QResponse(
        ok=True,
        mode=mode,
        iter_id=iter_id,
        bases=list(req.bases),
        counts={b: [c[0], c[1]] for b, c in counts.items()},
        expectations=expectations,
        rho=rho_out,
        fidelity=fidelity_out,
        azimuth_rad=azimuth,
        equatorial_contrast=contrast,
        shots_path=str(shots_path) if shots_path else None,
        meta=meta,
    )


# =========================== core operations ===============================

def run_wigner_core(req: RunWignerRequest) -> RunWignerResponse:
    """Run one Wigner-tomography measurement (sim or hw). Transport-agnostic.

    Raises:
        ValueError / KeyError — bad request (maps to HTTP 400).
        ServiceNotReady       — core/queue not ready (maps to HTTP 503).
        RuntimeError          — queue/execution failure (maps to HTTP 500).
    """
    iter_id = uuid.uuid4().hex[:8]
    t0 = time.time()

    alphas_arr = np.asarray(req.alphas, dtype=float)
    if alphas_arr.ndim != 2 or alphas_arr.shape[1] != 2:
        raise ValueError(f"alphas must be N x 2, got shape {alphas_arr.shape}")
    alphas_c = alphas_arr[:, 0] + 1j * alphas_arr[:, 1]

    req.IQ_table._check()

    sigma_z = sigma_z_raw = None
    if req.mode == "sim":
        parity_arr, meta = run_wigner_sim(req, alphas_c)
        shots_path: Optional[str] = None
        out_alphas = alphas_c
    elif req.mode == "hw":
        # A single job covers every tier: for 'measure' (Tier 3) the experiment
        # internally runs the native-parity sweep PLUS one alpha-independent
        # sigma_z measurement, returning both in expt.data -- no separate job.
        (out_alphas, parity_arr, shots_path, meta,
         sigma_z, sigma_z_raw) = submit_wigner_via_queue(req, iter_id)
    else:
        raise ValueError(f"unknown mode {req.mode!r}")

    meta["wall_total_s"] = meta.get("wall_total_s", time.time() - t0)

    # Optional full-pipeline step: reconstruct the cavity density matrix from the
    # measured parity grid. Off by default (knobs.reconstruct).
    rho_out = populations_out = fidelity_out = None
    if req.knobs.reconstruct:
        if not np.all(np.isfinite(np.asarray(parity_arr, dtype=float))):
            # Tier-3 sigma_z_mode='measure' runs carry NaN parity (no parity
            # grid); mirror the guard in the experiment's display().
            meta["reconstruct_skipped"] = "non-finite parity (no parity grid to reconstruct)"
        else:
            target_ket = _build_target_ket(req.target_state, req.knobs.reconstruct_fock_dim)
            rec = _reconstruct_from_parity(
                parity_arr, out_alphas, req.knobs.reconstruct_fock_dim,
                target_ket, req.knobs.reconstruct_rotate,
            )
            rho_out = rec["rho"]
            populations_out = rec["populations"]
            fidelity_out = rec["fidelity"]

    return RunWignerResponse(
        ok=True,
        alphas=[[float(a.real), float(a.imag)] for a in out_alphas],
        parity=[float(p) for p in parity_arr],
        mode=req.mode,
        iter_id=iter_id,
        shots_path=str(shots_path) if shots_path else None,
        meta=meta,
        rho=rho_out,
        populations=populations_out,
        fidelity=fidelity_out,
        sigma_z=float(sigma_z) if sigma_z is not None else None,
        sigma_z_raw=float(sigma_z_raw) if sigma_z_raw is not None else None,
    )


def run_state_tomo_1q_core(req: "RunStateTomo1QRequest") -> "RunStateTomo1QResponse":
    """Run one single-qubit state-tomography measurement (sim or hw).

    Sibling of run_wigner_core: same exception families (ValueError/KeyError ->
    400, ServiceNotReady -> 503, RuntimeError -> 500). Plays the optimized
    IQ_table pulse as state prep (hw), measures the transmon in the requested
    bases, and reconstructs the 2x2 density matrix + fidelity.
    """
    iter_id = uuid.uuid4().hex[:8]
    t0 = time.time()

    req.IQ_table._check()
    valid = {"Z", "X", "Y"}
    if not req.bases or any(b not in valid for b in req.bases):
        raise ValueError(f"bases must be a non-empty subset of {sorted(valid)}, got {req.bases!r}")

    confusion = req.confusion
    if req.mode == "sim":
        sim = run_state_tomo_1q_sim(req)
        counts, meta, shots_path = sim["counts"], sim["meta"], None
    elif req.mode == "hw":
        counts, _expectations, shots_path, meta = submit_state_tomo_1q_via_queue(req, iter_id)
        # Resolve the readout confusion matrix for the reconstruction, unless the
        # request supplied one explicitly. Precedence:
        #   1. the FRESH matrix from the readout recal that ran just before (it
        #      was fit at the same angle/threshold + reset conditions this job
        #      uses, so it matches the readout exactly), else
        #   2. the pinned-config matrix keyed on the active-reset regime
        #      (with_active_reset when active reset is on, else without_reset).
        if confusion is None:
            recal_conf = (meta.get("readout_recal") or {}).get("recal_confusion_matrix")
            if recal_conf is not None:
                confusion = recal_conf
                meta["confusion_source"] = "recal"
            else:
                confusion = _config_confusion_for_reset(req.knobs.active_reset)
                meta["confusion_source"] = (
                    "config:" + ("with_active_reset"
                                 if _active_reset_dict(req.knobs.active_reset).get("active_reset")
                                 else "without_reset")
                ) if confusion is not None else "none"
    else:
        raise ValueError(f"unknown mode {req.mode!r}")

    meta["wall_total_s"] = meta.get("wall_total_s", time.time() - t0)
    return _finalize_state_tomo_1q(
        counts, req, mode=req.mode, iter_id=iter_id, shots_path=shots_path,
        meta=meta, confusion=confusion,
    )


def calibrate_check_core(req: CalibrateCheckRequest) -> CalibrateCheckResponse:
    """Run a known-good pulse at one displacement, compare to expected parity.

    Raises the same exception families as run_wigner_core.
    """
    if _mock_station is None or _job_client is None:
        raise ServiceNotReady(
            "core not initialized — call init_core() with hardware_config / storage_man_file."
        )
    iter_id = uuid.uuid4().hex[:8]

    expected = req.expected_parity
    if expected is None:
        expected = _expected_parity_for_pulse(req.pulse_ref)

    # Resolve npz + canonical gains from the live (mock) station's hardware_cfg
    pulse_conf = _resolve_pulse_conf_attr(_mock_station.hardware_cfg, req.pulse_ref)
    filename = pulse_conf["filename"]
    iq = _load_iqtable_from_npz(filename)

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
            raise ValueError(
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

    (_out_alphas, parity_arr, shots_path, meta,
     _sigma_z, _sigma_z_raw) = submit_wigner_via_queue(sub_req, iter_id)

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

def init_core(
    *,
    hardware_config: Optional[str] = None,
    storage_man_file: Optional[str] = None,
    queue_url: str = "http://127.0.0.1:8000",
    run_root: Optional[str | Path] = None,
    experiment_name: Optional[str] = None,
) -> None:
    """Pin configs into a mock station, build the station_config template, and
    wire up the JobClient. Idempotent-ish: call once at process start.
    """
    global _mock_station, _serializable_hw_cfg_base, _station_data_template
    global _job_client, _run_root, _queue_url

    _queue_url = queue_url

    # 1) Load configs via a mock station — no hardware contention. The mock
    #    initialiser creates output dirs under C:/experiments/<exp_name>/.
    from experiments.station import MultimodeStation
    exp_name = experiment_name or f"{datetime.now().strftime('%y%m%d')}_closed_loop"
    print(f"[core] loading pinned configs via MultimodeStation(mock=True, "
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

    # 3) JobClient — non-fatal if queue isn't up at startup. hw calls will raise
    #    ServiceNotReady later if the server is unreachable.
    _job_client = JobClient(server_url=_queue_url)
    try:
        h = _job_client.health_check()
        print(f"[core] queue at {_queue_url} is reachable "
              f"(pending={h.get('pending_jobs')}, running={h.get('running_jobs')})")
    except Exception as e:
        print(f"[core] WARNING: queue at {_queue_url} unreachable at startup: {e}")
        print(f"[core]          hw requests will fail until the queue server + worker are up.")

    # 4) Run root for any future direct shots dumps (currently the worker owns
    #    HDF5; this dir is just an artifact carrier for any future inline dumps).
    if run_root is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _run_root = Path(f"C:/closed_loop_runs/{ts}")
    else:
        _run_root = Path(run_root)
    _run_root.mkdir(parents=True, exist_ok=True)


def is_ready() -> bool:
    return _job_client is not None and _mock_station is not None


def queue_reachable() -> bool:
    if _job_client is None:
        return False
    try:
        _job_client.health_check()
        return True
    except Exception:
        return False


def pinned_info() -> dict:
    info: dict = {}
    if _mock_station is not None:
        info["hardware_config_file"] = str(_mock_station.hardware_config_file)
        info["storage_man_file"] = getattr(_mock_station, "storage_man_file", None)
    return info


def queue_url() -> str:
    return _queue_url


def run_root() -> Optional[Path]:
    return _run_root
