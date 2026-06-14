"""Batch runner — the file-drop ("mailbox") front door for closed_loop.

Replaces the (now-dead) SSH-tunneled HTTP closed loop. External collaborators
drop a `pulses_*.zip` package into a shared Google Drive folder; this runner
picks it up, loops every pulse through `core.run_wigner_core`, and writes a
`results_<id>.zip` back into the same folder. Slack stays as the human nudge.

Mailbox layout (under --mailbox, default = the shared Drive `test` folder):

  mailbox/
    incoming/      <- collaborators upload pulses_*.zip here
    processing/    <- runner moves a package here while it runs (atomic claim)
    results/       <- runner writes results_<id>.zip here
    archive/       <- runner moves the input here when done

Modes
-----
    --mode sim   (default for first runs): uses core's toy coherent-state
                  parity, no QPU contention, no init_core needed.
    --mode hw    : real hardware via the job queue. Requires the queue
                   server + worker to be up; pass --hardware-config /
                   --storage-man-file to pin the configs.

Invocations
-----------
    pixi run python -m job_server.closed_loop.batch_runner once
    pixi run python -m job_server.closed_loop.batch_runner once --zip <path>
    pixi run python -m job_server.closed_loop.batch_runner once --mode hw \
        --hardware-config CFG-HW-...  --storage-man-file CFG-M1-...
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from job_server.closed_loop import core, pulse_io
from job_server.closed_loop.core import (
    IQTable, Knobs, RunWignerRequest, RunWignerResponse,
    RunStateTomo1QRequest, RunStateTomo1QResponse,
)
from job_server.closed_loop.validate import (
    PulseValidationError, validate_pulse_for_lab,
)


# Lab-side defaults applied per pulse unless the manifest overrides them.
# Confirmed with user:
#   - manipulate mode "M1" = man_mode_no 1 (idx 0)
#   - default reps 250
#   - relax_delay None → core pulls from pinned hardware_config (readout.relax_delay[0])
# Default pulse_ref reuses the existing optimal_control.fock.1 config slot as a
# gain placeholder + carrier-frequency source. This works for any Harmoniqs
# pulse targeting the same manipulate mode + qubit (we mutate the gain in
# core.submit_wigner_via_queue; the npz `filename` field is ignored when an
# IQ_table is passed inline). If/when collaborators target other modes, expose
# a per-pulse override via the manifest.
LAB_DEFAULTS: dict[str, Any] = {
    "man_mode_no": 1,
    "reps":        250,
    "pulse_ref":   [["optimal_control", "fock", "1", [0, 0]]],
    "alphas":      [[0.0, 0.0]],   # single-point sanity by default
    # relax_delay: not set -> Knobs.relax_delay=None -> core.resolve from config
    # IQ-blob drifts between back-to-back Wigners; re-fit phase/threshold before
    # every measurement via a HistogramExperiment. Default ON for batch runs.
    "recalibrate_readout": True,
    "recalibrate_reps":    2000,
    # Full active_reset configuration applied to BOTH the HistogramExperiment used
    # for IQ-blob recalibration AND the WignerTomography parity measurement.
    # Keeping both under the same reset regime is what makes the recal'd
    # threshold meaningful for the subsequent Wigner. Override per-pulse via
    # the manifest's `active_reset` key.
    # ef_reset is OFF: with the current in-sequence readout fidelity the ef
    # conditional-pi injects |e> on misread-g shots (net-harmful here); ge + the
    # pre-selection parity herald do the work.
    "active_reset": {
        "active_reset":         True,
        "ef_reset":             False,
        "storage_reset":        False,
        "coupler_reset":        False,
        "pre_selection_reset":  True,
        "man_reset":            False,
        "use_qubit_man_reset":  True,
        "pre_selection_parity": True,
    },
    # sigma_z probe default tier: 'measure' (Tier 3, non-invasive). The parity grid
    # is NATIVE and one extra alpha-independent transmon readout gives the sigma_z
    # scalar -- the parity is never perturbed/post-selected, so the QP-ILC optimizer
    # sees true parity. 'postselect' (Tier 2) gives cleaner near-vacuum parity by
    # heralding on transmon |g>, but its projection distorts states with residual
    # transmon-cavity entanglement (fock>=1, cats). Override per-pulse via the
    # manifest's `sigma_z_mode` ('off' | 'reset' | 'postselect' | 'measure').
    "sigma_z_mode": "measure",
    # Wigner reconstruction (the full tomography pipeline: parity grid -> density
    # matrix via WignerAnalysis). OFF by default -- most drops only want the
    # parity grid for the QP-ILC optimizer, and reconstruction needs a phase-space
    # alpha GRID (not the single-point sanity default) to be meaningful. Turn on
    # lab-wide here, or per-pulse via the manifest's `reconstruct` key. When on,
    # the fidelity target is auto-derived from each pulse's `target_state` meta
    # (Fock labels only); non-Fock targets still return rho + populations with
    # fidelity = null. A manifest `target_state` dict overrides the auto-derived one.
    "reconstruct":          False,
    "reconstruct_fock_dim": 5,
    "reconstruct_rotate":   False,
    # Which measurement to run per pulse: "wigner" (cavity parity tomography,
    # the default) or "tomography_1q" (single-qubit Z/X/Y state tomography of the
    # transmon the pulse prepares -- POST /run_tomography_1q / run_state_tomo_1q_core).
    # Override lab-wide here or per-pulse via the manifest's `measurement` key.
    "measurement":   "wigner",
    # --- 1Q tomography knobs (consumed only when measurement == "tomography_1q") ---
    "bases":         ["Z", "X", "Y"],   # subset OK; ["X","Y"] = azimuth-only (no rho)
    "recon_method":  "fast",            # "fast" | "cholesky" | "linear"
    # tomo_phases / confusion / target_state (a 2-level ket [[re,im],[re,im]] for
    # fidelity) / state_prep_postselect are taken from the manifest when present.
}

_MEASUREMENTS = ("wigner", "tomography_1q")

# Active-reset preset for the 1Q tomography path. Unlike the Wigner default
# (LAB_DEFAULTS["active_reset"], which heralds on a cavity PARITY measurement),
# a bare-transmon state-prep wants a plain transmon |g> herald -- so
# pre_selection_parity is OFF here. Confusion correction (keyed on active reset
# being on) is applied independently in core; the two address different error
# sources (reset reduces true |e> population, confusion corrects readout
# misassignment). A manifest `active_reset` entry overrides this.
TOMO_1Q_ACTIVE_RESET = {
    "active_reset":         True,
    "ef_reset":             False,
    "storage_reset":        False,
    "coupler_reset":        False,
    "pre_selection_reset":  True,
    "man_reset":            False,
    "use_qubit_man_reset":  True,
    "pre_selection_parity": False,
}


DEFAULT_MAILBOX = Path(r"G:\Shared drives\SLab\Multimode\optimal_control\test")


# ============================== mailbox ===================================

@dataclass
class Mailbox:
    root: Path

    @property
    def incoming(self)   -> Path: return self.root / "incoming"
    @property
    def processing(self) -> Path: return self.root / "processing"
    @property
    def failed(self)     -> Path: return self.root / "failed"
    @property
    def results(self)    -> Path: return self.root / "results"
    @property
    def archive(self)    -> Path: return self.root / "archive"

    def ensure(self) -> None:
        for d in (self.incoming, self.processing, self.results, self.archive, self.failed):
            d.mkdir(parents=True, exist_ok=True)

    def list_incoming_zips(self) -> list[Path]:
        return sorted(p for p in self.incoming.glob("*.zip") if p.is_file())

    def is_ready(self, zip_path: Path, stable_seconds: float = 5.0) -> bool:
        """Either a sibling <stem>.ready sentinel OR the size hasn't changed
        for `stable_seconds`. Errs on the side of waiting.
        """
        sentinel = zip_path.with_suffix(".ready")
        if sentinel.exists():
            return True
        try:
            s1 = zip_path.stat().st_size
            time.sleep(min(stable_seconds, 5.0))
            s2 = zip_path.stat().st_size
            return s1 == s2 and s1 > 0
        except FileNotFoundError:
            return False

    def claim(self, zip_path: Path) -> tuple[Path, str]:
        """Atomically move `zip_path` into processing/<id>/. Returns the new
        zip path + the claim id.
        """
        claim_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        dst_dir = self.processing / claim_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / zip_path.name
        shutil.move(str(zip_path), str(dst))
        sentinel = zip_path.with_suffix(".ready")
        if sentinel.exists():
            try: sentinel.unlink()
            except Exception: pass
        return dst, claim_id

    def archive_input(self, claim_dir: Path) -> Path:
        dst = self.archive / claim_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(claim_dir), str(dst))
        return dst

    def fail_input(self, claim_dir: Path) -> Path:
        """Move a claimed input to failed/<claim_id>/. Mirror of archive_input
        but for batches that hard-failed pre-flight validation (deterministic
        — retrying gains nothing). The caller is expected to have written a
        failure.json / failure.txt inside claim_dir before calling this.
        """
        dst = self.failed / claim_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(claim_dir), str(dst))
        return dst


# ============================ per-pulse plan ==============================

@dataclass
class PulsePlan:
    group:        str
    sampled_path: Path
    target_state: Optional[str]
    free_phases:  Optional[dict]
    iq_table:     IQTable
    request:      Union[RunWignerRequest, RunStateTomo1QRequest]
    manifest_entry: Optional[dict]
    # Which core entry point run_one dispatches to: "wigner" | "tomography_1q".
    measurement:  str = "wigner"
    # When pre-flight validation fails we still return a PulsePlan, but with
    # `validation_error` set. `run_one` short-circuits in that case so we
    # don't burn QPU time on a known-bad pulse.
    validation_error: Optional[dict] = None
    warnings:         list[dict]     = None


def _err(kind: str, message: str, suggestion: Optional[str] = None) -> dict:
    """Canonical structured-error dict used in PulseOutcome.error /
    PulsePlan.validation_error / results.json. Mirrors PulseValidationError.to_dict.
    """
    return {"kind": kind, "message": message, "suggestion": suggestion}


def _manifest_for(group: str, manifest: Optional[dict]) -> Optional[dict]:
    """Look up the per-pulse manifest entry by group name.

    `discover_pulses` derives `group` from the pulse subdirectory name, which
    is exactly what the manifest writes under `dir`. Older manifests used
    `group`; we accept both for backward compatibility.
    """
    if manifest is None: return None
    for entry in manifest.get("pulses", []):
        if entry.get("dir") == group or entry.get("group") == group:
            return entry
    return None


def _target_state_from_label(label: Optional[str]) -> Optional[dict]:
    """Map a pulse's free-form `target_state` meta string to the core
    target_state spec, for fidelity. Recognizes Fock labels only:
    'fock_1', 'fock 1', 'fock1', '|1>', 'n=1', or a bare '1' -> {"type":"fock","n":1}.
    Anything else (cats, superpositions, missing) -> None (fidelity skipped).
    """
    if not label:
        return None
    s = str(label).strip().lower()
    for tok in ("fock", "n", "=", "|", ">", "_", " "):
        s = s.replace(tok, "")
    if s.isdigit():
        return {"type": "fock", "n": int(s)}
    return None


def build_plan(
    pulse_entry: dict,
    *,
    manifest: Optional[dict] = None,
    mode: str = "sim",
    lab_defaults: Optional[dict] = None,
) -> PulsePlan:
    lab = {**LAB_DEFAULTS, **(lab_defaults or {})}

    # 1) Load — wrap `pulse_io.load_sampled_pulse`'s contract checks
    # (channel_order / units / frame) as a structured contract:* failure so
    # they ride the same failed/<id>/ routing as the rest of validation.
    try:
        sp = pulse_io.load_sampled_pulse(pulse_entry["sampled_path"])
    except (ValueError, KeyError) as e:
        # Don't try sampled_to_iq_table or anything else; we can't even read
        # the file. Build a minimal plan with the validation_error set so the
        # outer loop knows to skip run_one and route to failed/.
        return PulsePlan(
            group=pulse_entry["group"],
            sampled_path=pulse_entry["sampled_path"],
            target_state=None,
            free_phases=None,
            iq_table=IQTable(times=[], I_c=[], Q_c=[], I_q=[], Q_q=[]),
            request=None,  # type: ignore[arg-type]
            manifest_entry=None,
            validation_error=_err(
                kind="contract:missing_field",
                message=f"could not load {Path(pulse_entry['sampled_path']).name}: "
                        f"{type(e).__name__}: {e}",
                suggestion="check the sampled jld2 against test/SAMPLED_JLD2_CONTRACT.md",
            ),
            warnings=[],
        )

    iq = pulse_io.sampled_to_iq_table(sp)

    me = _manifest_for(pulse_entry["group"], manifest)
    free_phases = sp.meta.get("free_phases_rad") or (me.get("free_phases_rad") if me else None)

    # Per-pulse manifest entries override lab_defaults (which itself overrides
    # the module-level LAB_DEFAULTS). Only the keys we know how to consume.
    def pick(key, cast=lambda x: x):
        if me is not None and me.get(key) is not None:
            return cast(me[key])
        return lab[key]

    alphas      = pick("alphas")
    reps        = pick("reps", int)
    man_mode_no = pick("man_mode_no", int)

    # 2) Pre-flight validation. Geometry + manifest checks always run; gain
    # checks need hw_cfg, which `core.pinned_info()` exposes only in hw mode
    # (after init_core). In sim we skip the gain check.
    hw_cfg = None
    if mode == "hw":
        try:
            from job_server.closed_loop.core import _mock_station as _ms
            if _ms is not None:
                hw_cfg = _ms.hardware_cfg
        except Exception:
            hw_cfg = None

    validation_error: Optional[dict] = None
    warnings: list[dict] = []
    try:
        result = validate_pulse_for_lab(
            sp, iq,
            manifest_entry=me, hw_cfg=hw_cfg,
            man_mode_idx=max(0, man_mode_no - 1),
            mode=mode,
        )
        warnings = result.warnings
    except PulseValidationError as ve:
        validation_error = ve.to_dict()

    knobs = Knobs()
    if "relax_delay" in lab and lab["relax_delay"] is not None:
        knobs.relax_delay = float(lab["relax_delay"])
    if "recalibrate_readout" in lab:
        knobs.recalibrate_readout = bool(lab["recalibrate_readout"])
    if "recalibrate_reps" in lab:
        knobs.recalibrate_reps = int(lab["recalibrate_reps"])
    if "active_reset" in lab:
        # bool (back-compat) or dict (full granularity) — core normalizes via _active_reset_dict.
        knobs.active_reset = lab["active_reset"]

    # sigma_z probe: per-pulse manifest entry overrides lab default. measure_sigma_z
    # is the collaborator boolean; sigma_z_mode ('off'|'reset'|'postselect') is the
    # explicit override. Both flow into expt_config and on to MM_base.lane_layout.
    def pick_opt(key):
        if me is not None and me.get(key) is not None:
            return me[key]
        return lab.get(key)

    _msz = pick_opt("measure_sigma_z")
    if _msz is not None:
        knobs.measure_sigma_z = bool(_msz)
    _szmode = pick_opt("sigma_z_mode")
    if _szmode is not None:
        knobs.sigma_z_mode = str(_szmode)

    # Which measurement to run. The watcher routes here from the resolved
    # `measurement` key (LAB_DEFAULTS, overridable per-pulse via the manifest).
    measurement = pick_opt("measurement") or "wigner"
    if measurement not in _MEASUREMENTS and validation_error is None:
        validation_error = _err(
            kind="manifest:bad_measurement",
            message=f"measurement={measurement!r} not in {list(_MEASUREMENTS)}",
            suggestion="set measurement to 'wigner' or 'tomography_1q'",
        )

    if measurement == "tomography_1q":
        req = _build_tomo_1q_request(
            mode=mode, iq=iq, reps=reps, man_mode_no=man_mode_no,
            knobs=knobs, lab=lab, pick_opt=pick_opt, me=me)
    else:
        # Wigner reconstruction: opt-in full tomography pipeline. Fidelity target
        # is the manifest's explicit `target_state` dict if given, else
        # auto-derived from the pulse's target_state meta string (Fock labels).
        _recon = pick_opt("reconstruct")
        if _recon is not None:
            knobs.reconstruct = bool(_recon)
        _rfd = pick_opt("reconstruct_fock_dim")
        if _rfd is not None:
            knobs.reconstruct_fock_dim = int(_rfd)
        _rrot = pick_opt("reconstruct_rotate")
        if _rrot is not None:
            knobs.reconstruct_rotate = bool(_rrot)

        target_state = None
        if knobs.reconstruct:
            manifest_ts = me.get("target_state") if me is not None else None
            target_state = (manifest_ts if isinstance(manifest_ts, dict)
                            else _target_state_from_label(sp.meta.get("target_state")))

        req = RunWignerRequest(
            mode=mode,
            IQ_table=iq,
            alphas=alphas,
            reps=reps,
            pulse_ref=lab["pulse_ref"],
            man_mode_no=man_mode_no,
            knobs=knobs,
            target_state=target_state,
        )
    return PulsePlan(
        group=pulse_entry["group"],
        sampled_path=pulse_entry["sampled_path"],
        target_state=sp.meta.get("target_state"),
        free_phases=free_phases,
        iq_table=iq,
        request=req,
        manifest_entry=me,
        measurement=measurement,
        validation_error=validation_error,
        warnings=warnings,
    )


def _build_tomo_1q_request(*, mode, iq, reps, man_mode_no, knobs, lab, pick_opt, me) -> RunStateTomo1QRequest:
    """Assemble a RunStateTomo1QRequest from resolved lab/manifest fields.

    The optimized pulse plays as state prep (opt_pulse=pulse_ref + inline
    IQ_table); the transmon is then measured in `bases`. `target_state` (a
    2-level ket [[re,im],[re,im]]) is taken from the manifest for fidelity; with
    none, fidelity is omitted.
    """
    bases = list(pick_opt("bases") or ["Z", "X", "Y"])
    recon_method = pick_opt("recon_method") or "fast"
    tomo_phases = pick_opt("tomo_phases")
    confusion = pick_opt("confusion")
    state_prep_postselect = bool(pick_opt("state_prep_postselect") or False)
    # Use the 1Q transmon-herald reset preset (no cavity-parity herald) unless the
    # manifest explicitly overrides active_reset. The shared knobs block set
    # knobs.active_reset from the Wigner LAB_DEFAULTS, so override it here.
    me_ar = me.get("active_reset") if me is not None else None
    knobs.active_reset = me_ar if me_ar is not None else dict(TOMO_1Q_ACTIVE_RESET)
    # 1Q fidelity target is a ket (list of [re,im] pairs); a dict (Wigner-style
    # Fock target) doesn't apply here, so ignore it.
    manifest_ts = me.get("target_state") if me is not None else None
    target_state = manifest_ts if isinstance(manifest_ts, list) else None
    return RunStateTomo1QRequest(
        mode=mode,
        IQ_table=iq,
        pulse_ref=lab["pulse_ref"],
        reps=reps,
        bases=bases,
        tomo_phases=tomo_phases,
        recon_method=recon_method,
        target_state=target_state,
        confusion=confusion,
        state_prep_postselect=state_prep_postselect,
        man_mode_no=man_mode_no,
        knobs=knobs,
    )


# =============================== run loop ==================================

@dataclass
class PulseOutcome:
    group:           str
    target_state:    Optional[str]
    ok:              bool
    parity:          Optional[list[float]]
    alphas:          Optional[list[list[float]]]
    meta:            Optional[dict]
    # Soft warnings raised during validation (e.g. nonzero free_phases on a
    # cavity-cat pulse where virtual-Z isn't wired yet). Each entry is a
    # `{"kind", "message", "suggestion"}` dict.
    warnings:        list[dict]
    # Hard failure, if any. Same dict shape as PulseValidationError.to_dict().
    # `None` when ok=True. `kind` starts with "run:" for run-time failures
    # (queue blip, worker exception, etc.) and otherwise for pre-flight
    # validation failures (contract:*, pulse:*, gain:*, manifest:*).
    error:           Optional[dict]
    # Post-prep transmon sigma_z scalar (P_g - P_e), present iff measure_sigma_z /
    # sigma_z_mode was requested. `sigma_z` is confusion-corrected; `sigma_z_raw`
    # is the uncorrected 1 - 2*P_e. Both None when not requested or on failure.
    sigma_z:         Optional[float] = None
    sigma_z_raw:     Optional[float] = None
    # Full-pipeline reconstruction outputs, present iff knobs.reconstruct was set
    # (None otherwise). `rho` is {"real": [[...]], "imag": [[...]]}; `populations`
    # is the photon-number diagonal; `fidelity` is None unless a target_state
    # was resolved for this pulse.
    rho:             Optional[dict] = None
    populations:     Optional[list[float]] = None
    fidelity:        Optional[float] = None
    # Which measurement produced this outcome ("wigner" | "tomography_1q").
    measurement:     str = "wigner"
    # 1Q state-tomography outputs (measurement == "tomography_1q"), else None.
    # counts: {'Z': [n_g, n_e], ...}; expectations: {'Z': <Z>, ...}.
    counts:          Optional[dict] = None
    expectations:    Optional[dict] = None
    azimuth_rad:         Optional[float] = None
    equatorial_contrast: Optional[float] = None


def is_validation_failure(outcome: PulseOutcome) -> bool:
    """A `run:*` error is a transient run-time failure; everything else is
    a deterministic pre-flight validation failure (no point retrying).
    """
    if outcome.error is None:
        return False
    kind = outcome.error.get("kind", "")
    return not kind.startswith("run:")


def run_one(plan: PulsePlan) -> PulseOutcome:
    # Validation pre-empt — never burn QPU time on a known-bad pulse.
    if plan.validation_error is not None:
        return PulseOutcome(
            group=plan.group,
            target_state=plan.target_state,
            ok=False,
            parity=None, alphas=None, meta=None,
            warnings=list(plan.warnings or []),
            error=plan.validation_error,
            measurement=plan.measurement,
        )
    try:
        if plan.measurement == "tomography_1q":
            resp: RunStateTomo1QResponse = core.run_state_tomo_1q_core(plan.request)
            return PulseOutcome(
                group=plan.group,
                target_state=plan.target_state,
                ok=True,
                parity=None, alphas=None,
                meta=resp.meta,
                warnings=list(plan.warnings or []),
                error=None,
                rho=resp.rho,
                fidelity=resp.fidelity,
                measurement="tomography_1q",
                counts=resp.counts,
                expectations=resp.expectations,
                azimuth_rad=resp.azimuth_rad,
                equatorial_contrast=resp.equatorial_contrast,
            )
        resp: RunWignerResponse = core.run_wigner_core(plan.request)
        return PulseOutcome(
            group=plan.group,
            target_state=plan.target_state,
            ok=True,
            parity=resp.parity,
            alphas=resp.alphas,
            meta=resp.meta,
            warnings=list(plan.warnings or []),
            error=None,
            sigma_z=resp.sigma_z,
            sigma_z_raw=resp.sigma_z_raw,
            rho=resp.rho,
            populations=resp.populations,
            fidelity=resp.fidelity,
            measurement="wigner",
        )
    except Exception as e:
        return PulseOutcome(
            group=plan.group,
            target_state=plan.target_state,
            ok=False,
            parity=None, alphas=None, meta=None,
            warnings=list(plan.warnings or []),
            error=_err(
                kind="run:exception",
                message=f"{type(e).__name__}: {e}",
                suggestion=None,
            ),
            measurement=plan.measurement,
        )


# ============================ results packaging ============================

def _render_outcome(out: PulseOutcome, out_path: Path) -> Optional[Path]:
    """Dispatch to the right per-pulse plot for this outcome's measurement."""
    if out.measurement == "tomography_1q":
        return _render_state_tomo_1q(out, out_path)
    return _render_wigner_or_alpha_scan(out, out_path)


def _render_state_tomo_1q(out: PulseOutcome, out_path: Path) -> Optional[Path]:
    """Bar of Pauli expectations + Re(rho) heatmap. Best-effort, never fatal."""
    if not out.ok or not out.expectations:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        has_rho = out.rho is not None
        fig, axes = plt.subplots(1, 2 if has_rho else 1,
                                 figsize=(8.5 if has_rho else 4.5, 4.0),
                                 squeeze=False)
        ax = axes[0, 0]
        bases = [b for b in ("Z", "X", "Y") if b in out.expectations]
        ax.bar(range(len(bases)), [out.expectations[b] for b in bases], width=0.6)
        ax.set_xticks(range(len(bases))); ax.set_xticklabels(bases)
        ax.set_ylim(-1.05, 1.05); ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("expectation")
        ttl = f"{out.group} — 1Q tomography"
        if out.fidelity is not None:
            ttl += f"  (F={out.fidelity:.3f})"
        ax.set_title(ttl)
        if has_rho:
            axr = axes[0, 1]
            re = np.asarray(out.rho["real"])
            im = axr.imshow(re, cmap="RdBu", vmin=-1, vmax=1)
            axr.set_title("Re(rho)")
            axr.set_xticks([0, 1]); axr.set_yticks([0, 1])
            axr.set_xticklabels(["g", "e"]); axr.set_yticklabels(["g", "e"])
            for (r, c), v in np.ndenumerate(re):
                axr.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9)
            fig.colorbar(im, ax=axr, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception:
        return None


def _render_wigner_or_alpha_scan(out: PulseOutcome, out_path: Path) -> Optional[Path]:
    """If parity is on a 2D alpha grid, render a Wigner heatmap; otherwise
    render a parity-vs-index bar / line plot. Best-effort, never fatal.
    """
    if not out.ok or not out.parity or not out.alphas:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        alphas = np.asarray(out.alphas)
        parity = np.asarray(out.parity)
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        if len(alphas) == 1:
            ax.bar([0], parity, width=0.6)
            ax.set_xticks([0])
            ax.set_xticklabels([f"α=({alphas[0,0]:+.2f},{alphas[0,1]:+.2f})"])
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel("parity")
            ax.set_title(f"{out.group} — single-point parity")
        else:
            sc = ax.scatter(alphas[:, 0], alphas[:, 1], c=parity, cmap="RdBu_r",
                             vmin=-1, vmax=1, s=40)
            ax.set_xlabel("Re α")
            ax.set_ylabel("Im α")
            ax.set_title(f"{out.group} — parity (target {out.target_state})")
            plt.colorbar(sc, ax=ax, label="parity")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception:
        return None


def _summarize(outs: list[PulseOutcome], mode: str, claim_id: str, lab_defaults: dict) -> str:
    lines = [
        f"closed_loop batch results",
        f"run id     : {claim_id}",
        f"mode       : {mode}",
        f"finished   : {datetime.now().isoformat(timespec='seconds')}",
        f"lab defaults: {json.dumps(lab_defaults)}",
        f"pulses     : {len(outs)}",
        "",
    ]
    for o in outs:
        head = f"[{o.group}] target={o.target_state!r} "
        if o.ok:
            gains = ""
            if o.meta:
                gq = o.meta.get("gain_qb"); gc = o.meta.get("gain_cav")
                if gq is not None and gc is not None:
                    gains = f"  gains qb={gq}/cav={gc}"
            lines.append(head + f"OK [{o.measurement}]" + gains)
            if o.measurement == "tomography_1q":
                ex = o.expectations or {}
                ex_str = ", ".join(f"{b}={ex[b]:+.3f}" for b in ("Z", "X", "Y") if b in ex)
                lines.append(f"    expectations: {ex_str}")
                lines.append(f"    counts: {o.counts}")
                fid = "n/a" if o.fidelity is None else f"{o.fidelity:.4f}"
                lines.append(f"    fidelity: {fid}")
            else:
                p_str = ", ".join(f"{p:+.4f}" for p in (o.parity or []))
                a_str = ", ".join(f"({a[0]:+.2f},{a[1]:+.2f})" for a in (o.alphas or []))
                lines.append(f"    alphas: [{a_str}]")
                lines.append(f"    parity: [{p_str}]")
                if o.populations is not None:
                    pops = ", ".join(f"{p:.3f}" for p in o.populations)
                    fid = "n/a" if o.fidelity is None else f"{o.fidelity:.4f}"
                    lines.append(f"    reconstruction: fidelity={fid}  pops=[{pops}]")
        else:
            err = o.error or {}
            kind = err.get("kind", "unknown")
            msg  = err.get("message", "")
            sug  = err.get("suggestion")
            lines.append(head + f"FAILED  [{kind}] {msg}")
            if sug:
                lines.append(f"    suggestion: {sug}")
        for w in (o.warnings or []):
            lines.append(f"    !! [{w.get('kind','warning')}] {w.get('message','')}")
        lines.append("")
    return "\n".join(lines)


def pack_results(
    outs: list[PulseOutcome],
    *,
    out_zip: Path,
    workdir: Path,
    mode: str,
    claim_id: str,
    lab_defaults: dict,
    source_zip_name: str,
) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    # plots per pulse
    plots: dict[str, Path] = {}
    for o in outs:
        suffix = "tomo1q" if o.measurement == "tomography_1q" else "parity"
        png = workdir / f"{o.group}_{suffix}.png"
        rendered = _render_outcome(o, png)
        if rendered is not None:
            plots[o.group] = rendered

    summary_path = workdir / "summary.txt"
    summary_path.write_text(_summarize(outs, mode, claim_id, lab_defaults), encoding="utf-8")

    results_obj = {
        "run_id":      claim_id,
        "mode":        mode,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "source_zip":  source_zip_name,
        "lab_defaults": lab_defaults,
        "pulses": [
            {
                "group":         o.group,
                "target_state":  o.target_state,
                "ok":            o.ok,
                "measurement":   o.measurement,
                "parity":        o.parity,
                "alphas":        o.alphas,
                "sigma_z":       o.sigma_z,
                "sigma_z_raw":   o.sigma_z_raw,
                "rho":           o.rho,
                "populations":   o.populations,
                "fidelity":      o.fidelity,
                "counts":        o.counts,
                "expectations":  o.expectations,
                "azimuth_rad":   o.azimuth_rad,
                "equatorial_contrast": o.equatorial_contrast,
                "meta":          o.meta,
                "warnings":      o.warnings,
                "error":         o.error,
            }
            for o in outs
        ],
    }
    results_json = workdir / "results.json"
    results_json.write_text(json.dumps(results_obj, indent=2, default=str), encoding="utf-8")

    # Assemble the zip in a temp file, then atomic-move it into place — Drive
    # sync is happier with completed writes than with growing files.
    import zipfile
    tmp_zip = workdir / (out_zip.name + ".tmp")
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_path, arcname="summary.txt")
        zf.write(results_json, arcname="results.json")
        for group, png in plots.items():
            zf.write(png, arcname=png.name)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_zip), str(out_zip))
    return out_zip


# ================================ driver ===================================

def process_zip(
    zip_path: Path,
    *,
    mailbox: Mailbox,
    mode: str = "sim",
    claim_id: Optional[str] = None,
    lab_defaults: Optional[dict] = None,
    only: Optional[list[str]] = None,
) -> Path:
    """Process one pulses zip end-to-end. Returns the results zip path.

    `only`: optional list of pulse-group names to restrict the run to (e.g.
    ['fock1']). Names not present in the zip are ignored.
    """
    mailbox.ensure()
    lab = {**LAB_DEFAULTS, **(lab_defaults or {})}

    # claim
    if claim_id is None:
        claim_zip, claim_id = mailbox.claim(zip_path)
    else:
        # caller already moved it (e.g., out-of-mailbox testing); just use it
        claim_zip = zip_path
    claim_dir = claim_zip.parent
    print(f"[batch_runner] claimed {claim_zip.name} as {claim_id}")

    # extract + discover
    extract_dir = claim_dir / "extracted"
    pulses_root = pulse_io.unpack_request_zip(claim_zip, extract_dir)
    pulses = pulse_io.discover_pulses(pulses_root)
    if not pulses:
        raise RuntimeError(f"no pulses with *_sampled.jld2 found under {pulses_root}")
    if only:
        wanted = set(only)
        pulses = [p for p in pulses if p["group"] in wanted]
        if not pulses:
            raise RuntimeError(f"--only={list(only)} matched none of the pulses in the zip")
    print(f"[batch_runner] found {len(pulses)} pulse(s): "
          + ", ".join(p["group"] for p in pulses))

    # manifest (optional)
    manifest_paths = list(pulses_root.glob("manifest.json"))
    manifest = None
    if manifest_paths:
        try:
            manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
            print(f"[batch_runner] loaded manifest {manifest_paths[0].relative_to(pulses_root)}")
        except Exception as e:
            print(f"[batch_runner] WARNING: failed to parse manifest: {e}")

    # plan + run per pulse
    outcomes: list[PulseOutcome] = []
    for p in pulses:
        try:
            plan = build_plan(p, manifest=manifest, mode=mode, lab_defaults=lab)
        except Exception as e:
            # build_plan should normally NOT throw — contract/validation errors
            # are caught inside it and surface via plan.validation_error. Anything
            # that leaks here is an unexpected bug; tag it as run:exception so
            # it stays distinct from deterministic pre-flight failures.
            outcomes.append(PulseOutcome(
                group=p["group"], target_state=None, ok=False,
                parity=None, alphas=None, meta=None,
                warnings=[],
                error=_err(
                    kind="run:exception",
                    message=f"build_plan crashed: {type(e).__name__}: {e}",
                    suggestion=None,
                ),
            ))
            print(f"[batch_runner]   [ERR] {p['group']:28s}  build_plan crashed:\n{traceback.format_exc()}")
            continue
        t0 = time.time()
        out = run_one(plan)
        dt = time.time() - t0
        if out.ok:
            print(f"[batch_runner]   [ok ] {plan.group:28s}  ({dt:6.2f}s)")
        else:
            kind = out.error.get("kind", "unknown") if out.error else "unknown"
            msg  = out.error.get("message", "") if out.error else ""
            print(f"[batch_runner]   [ERR] {plan.group:28s}  ({dt:6.2f}s)  [{kind}] {msg}")
        outcomes.append(out)

    # package
    src_stem = claim_zip.stem  # e.g. "pulse_groups_metadata"
    workdir = claim_dir / "results_workdir"
    # Routing decision: if every pulse failed pre-flight validation (i.e.,
    # nothing actually hit the QPU and no transient run:* failures), the whole
    # batch is a deterministic failure and goes to failed/<claim_id>/.
    # Mixed batches (some ok, some validation-fail) and any-run:*-failure batches
    # still archive normally so the partial data isn't lost.
    all_failed_validation = (
        len(outcomes) > 0
        and all(is_validation_failure(o) for o in outcomes)
    )

    if all_failed_validation:
        # Write failure.json/txt alongside the results bundle so external readers
        # can see *why* in failed/<claim_id>/ without unpacking anything.
        _write_failure_artifacts(outcomes, workdir, mode=mode, claim_id=claim_id,
                                 lab_defaults=lab, source_zip_name=claim_zip.name)
        failed_dst = mailbox.fail_input(claim_dir)
        print(f"[batch_runner] all {len(outcomes)} pulse(s) failed pre-flight validation — "
              f"routed to {failed_dst}")
        # No results zip in results/ — there's nothing to report on QPU side.
        # The failure artifacts live inside failed/<claim_id>/.
        return failed_dst

    out_zip = mailbox.results / f"results_{src_stem}_{claim_id}.zip"
    pack_results(
        outcomes,
        out_zip=out_zip, workdir=workdir,
        mode=mode, claim_id=claim_id, lab_defaults=lab,
        source_zip_name=claim_zip.name,
    )
    print(f"[batch_runner] wrote {out_zip}")

    # archive input
    archived = mailbox.archive_input(claim_dir)
    print(f"[batch_runner] archived input to {archived}")
    return out_zip


def _write_failure_artifacts(
    outcomes: list[PulseOutcome],
    workdir: Path,
    *,
    mode: str,
    claim_id: str,
    lab_defaults: dict,
    source_zip_name: str,
) -> None:
    """Drop failure.json + failure.txt inside `workdir` for the failed/ routing.

    These mirror the shape of `results.json` / `summary.txt` so the same
    consumers can read either. The distinction is just *where* the file lives
    (`failed/<claim_id>/` vs `archive/<claim_id>/`).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id":      claim_id,
        "mode":        mode,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "source_zip":  source_zip_name,
        "lab_defaults": lab_defaults,
        "summary":     "all pulses failed pre-flight validation; nothing hit the QPU",
        "pulses": [
            {
                "group":         o.group,
                "target_state":  o.target_state,
                "warnings":      o.warnings,
                "error":         o.error,
            }
            for o in outcomes
        ],
    }
    (workdir / "failure.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (workdir / "failure.txt").write_text(
        _summarize(outcomes, mode, claim_id, lab_defaults), encoding="utf-8")


def _setup_core_if_hw(mode: str, args) -> None:
    if mode != "hw":
        return
    print("[batch_runner] mode=hw -> init_core ...")
    core.init_core(
        hardware_config=args.hardware_config,
        storage_man_file=args.storage_man_file,
        queue_url=args.queue_url,
        experiment_name=args.experiment_name,
    )
    if not core.queue_reachable():
        print("[batch_runner] WARNING: queue not reachable at startup; hw runs will 503 until it is.")


def cmd_once(args) -> int:
    mailbox = Mailbox(Path(args.mailbox))
    mailbox.ensure()
    _setup_core_if_hw(args.mode, args)

    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.is_file():
            print(f"--zip not found: {zip_path}"); return 2
        # if it's already inside the mailbox incoming/, claim normally;
        # otherwise treat it as out-of-mailbox and process in place (still
        # claim into processing/ for tidiness).
        if zip_path.parent == mailbox.incoming:
            process_zip(zip_path, mailbox=mailbox, mode=args.mode, only=args.only)
        else:
            dst = mailbox.incoming / zip_path.name
            shutil.copy2(zip_path, dst)
            process_zip(dst, mailbox=mailbox, mode=args.mode, only=args.only)
        return 0

    zips = mailbox.list_incoming_zips()
    if not zips:
        print(f"no zips in {mailbox.incoming}")
        return 1
    target = zips[-1]   # newest by name
    if not mailbox.is_ready(target, stable_seconds=args.stable_seconds):
        print(f"upload looks in-progress for {target.name}; not claiming this run")
        return 3
    process_zip(target, mailbox=mailbox, mode=args.mode, only=args.only)
    return 0


def cmd_watch(args) -> int:
    """Poll incoming/ and fire process_zip on each new drop. FIFO by mtime.

    Failures (count < max_failures): move the zip back to incoming/ and clean
    up the empty claim dir — transient hiccups (queue blip, network glitch)
    self-heal on the next poll. mtime is preserved so the same zip stays at
    the head of the FIFO.

    Persistent failure (count >= max_failures): leave the *current* zip in
    processing/ and exit. The poison-pill stays parked so a manual restart
    doesn't auto-retry it.
    """
    mailbox = Mailbox(Path(args.mailbox))
    mailbox.ensure()
    _setup_core_if_hw(args.mode, args)

    print(f"[watch] watching {mailbox.incoming}")
    print(f"[watch]   mode={args.mode}  poll={args.poll_interval}s  "
          f"stable={args.stable_seconds}s  circuit-breaker={args.max_failures}")

    seen_unstable: set[str] = set()   # names we logged as "settling"; throttle re-logs
    consecutive_failures = 0
    try:
        while True:
            zips = mailbox.list_incoming_zips()
            # FIFO by mtime (oldest first). A re-queued failed zip keeps its
            # original mtime so it stays at the head until it succeeds or
            # trips the circuit breaker.
            zips.sort(key=lambda p: p.stat().st_mtime)

            target: Optional[Path] = None
            for zp in zips:
                if not mailbox.is_ready(zp, stable_seconds=args.stable_seconds):
                    if zp.name not in seen_unstable:
                        print(f"[watch] {zp.name} still settling, deferring")
                        seen_unstable.add(zp.name)
                    continue
                target = zp
                break

            if target is None:
                time.sleep(args.poll_interval)
                continue

            seen_unstable.discard(target.name)
            # Claim ourselves so we know exactly where the file lives if process_zip
            # throws — needed to put it back in incoming/ on transient failures.
            original_mtime = target.stat().st_mtime
            claim_zip, claim_id = mailbox.claim(target)
            print(f"[watch] picking up {target.name} (claim {claim_id})")
            try:
                out_path = process_zip(claim_zip, mailbox=mailbox, mode=args.mode,
                                       only=args.only, claim_id=claim_id)
                # process_zip routes to mailbox.failed/ when EVERY pulse hard-failed
                # pre-flight validation. That's deterministic — no retry, no counter
                # bump. Otherwise it's a successful run (full or partial); reset the
                # circuit breaker.
                if out_path.parent == mailbox.failed:
                    print(f"[watch] validation failure -> {out_path}  (not retried)")
                else:
                    print(f"[watch] ok -> {out_path.name}")
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"[watch] FAILED {target.name} "
                      f"({consecutive_failures}/{args.max_failures}): "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
                if consecutive_failures >= args.max_failures:
                    print(f"[watch] {consecutive_failures} consecutive failures — "
                          f"poison pill left in processing/{claim_id}/. "
                          f"Inspect and restart manually.")
                    return 1
                # Transient: put the zip back in incoming/, preserve mtime, clean
                # up the empty claim dir. Next poll retries.
                back = mailbox.incoming / claim_zip.name
                try:
                    shutil.move(str(claim_zip), str(back))
                    import os
                    os.utime(back, (original_mtime, original_mtime))
                    shutil.rmtree(claim_zip.parent, ignore_errors=True)
                    print(f"[watch] returned {back.name} to incoming/ for retry")
                except Exception as move_err:
                    print(f"[watch] WARNING: failed to return {claim_zip.name} "
                          f"to incoming/: {move_err}. Left in processing/{claim_id}/.")
            # Re-scan immediately in case multiple zips queued up.
    except KeyboardInterrupt:
        print("\n[watch] interrupted, exiting cleanly.")
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="closed_loop batch runner (mailbox front door)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared args for any subcommand that may need to init core (hw mode).
    def add_hw_args(s):
        s.add_argument("--hardware-config", default=None,
                       help="(hw only) pin hardware config version id; omit to follow the main pointer")
        s.add_argument("--storage-man-file", default=None,
                       help="(hw only) pin man1 storage version id; omit to follow the main pointer")
        s.add_argument("--queue-url", default="http://127.0.0.1:8000",
                       help="(hw only) job queue server URL")
        s.add_argument("--experiment-name", default=None,
                       help="(hw only) MultimodeStation experiment name")

    o = sub.add_parser("once", help="process one drop now (manual kick)")
    o.add_argument("--mailbox", default=str(DEFAULT_MAILBOX),
                   help=f"mailbox root (default {DEFAULT_MAILBOX})")
    o.add_argument("--zip", default=None,
                   help="explicit zip path; if outside mailbox/incoming we copy it in first")
    o.add_argument("--mode", choices=["sim", "hw"], default="sim",
                   help="sim = toy parity (no QPU); hw = real measurement via queue")
    o.add_argument("--stable-seconds", type=float, default=5.0,
                   help="size-stability wait (s) when no .ready sentinel is present")
    add_hw_args(o)
    o.add_argument("--only", action="append", default=None,
                   help="restrict the run to one pulse group (e.g. --only fock1). Repeatable.")
    o.set_defaults(func=cmd_once)

    w = sub.add_parser("watch", help="poll incoming/ and fire process_zip on each new drop")
    w.add_argument("--mailbox", default=str(DEFAULT_MAILBOX),
                   help=f"mailbox root (default {DEFAULT_MAILBOX})")
    w.add_argument("--mode", choices=["sim", "hw"], default="hw",
                   help="sim = toy parity (no QPU); hw = real measurement via queue (default hw)")
    w.add_argument("--poll-interval", type=float, default=5.0,
                   help="seconds between incoming/ scans (default 5)")
    w.add_argument("--stable-seconds", type=float, default=5.0,
                   help="size-stability wait (s) when no .ready sentinel is present")
    w.add_argument("--max-failures", type=int, default=3,
                   help="pause watcher after this many consecutive failures (default 3)")
    add_hw_args(w)
    w.add_argument("--only", action="append", default=None,
                   help="restrict each run to one pulse group. Repeatable.")
    w.set_defaults(func=cmd_watch)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
