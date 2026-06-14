"""Pre-flight validation for closed_loop pulse drops.

The mailbox runner calls `validate_pulse_for_lab` on every pulse after it's
been loaded by `pulse_io` and before any QPU time is spent. Hard failures are
raised as `PulseValidationError` with a machine-parseable `kind`, a
human-readable `message`, and an optional `suggestion`. Soft warnings are
returned alongside; the caller can decide whether to surface them.

Validation classes (round 1):
  contract:*    — schema/shape problems in the sampled jld2. (Most contract
                  checks already live in `pulse_io.load_sampled_pulse` and
                  raise plain ValueError; we wrap and re-raise here so they
                  carry a kind.)
  manifest:*    — schema problems in the per-pulse manifest entry.
  pulse:*       — duration / sample-count vs hardware envelope memory.
  gain:*        — computed DAC gain exceeds the 16-bit register limit.

Soft warnings:
  free_phases:nonzero_cavity — cat-state pulses need virtual-Z; not yet wired.

NOT included (intentionally, see conversation 2026-05-29):
  carrier:* — runtime always overrides cached frequencies with live
              `device.qubit.f_ge[0]` / `device.manipulate.f_ge[0]`. The
              manifest's `omega_q_GHz` / `omega_c_GHz` are documentation.
  run:timeout — dropped; queue-unreachable / job-stalled handled separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from job_server.closed_loop.core import IQTable, compute_gains_from_ghz
from job_server.closed_loop.pulse_io import SampledPulse


# QICK signal-generator envelope memory, in samples. Both qubit gens (DAC tile
# 2, fs=6389.76 MHz) and the manipulate gen (DAC tile 2, ch 3, same fs) hold
# 32768 samples ⇒ ~5.128 us. TODO: replace with a soccfg lookup when we have
# a non-mock station handle here. For now, hardcoded with a small margin.
QICK_ENVELOPE_SAMPLES   = 32768
QICK_GEN_FS_MHZ_DEFAULT = 6389.76   # qubit / manipulate gens on this box
DURATION_LIMIT_US       = 5.0       # < 5.128 us, leaves ~130 ns margin

# 16-bit signed register; +1 bit for sign ⇒ +/- 32767. We hard-fail at the
# limit (the QICK wrapper already clamps softly; better to refuse the run).
GAIN_REGISTER_LIMIT = 32767

PULSE_DURATION_MIN_NS = 10.0   # sanity floor; short single-qubit gates (~40 ns
                               # Hadamard) are legitimate, but < 10 ns almost
                               # always means a unit error / truncated times[].


class PulseValidationError(Exception):
    """Pre-flight validation failed. Carries a structured `kind` (machine-
    parseable) + `message` (human) + optional `suggestion`. `to_dict()` is
    the canonical serialization for results.json / failure.json.
    """

    def __init__(self, kind: str, message: str, suggestion: Optional[str] = None):
        self.kind = kind
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"[{kind}] {message}" + (f"  ({suggestion})" if suggestion else ""))

    def to_dict(self) -> dict:
        return {
            "kind":       self.kind,
            "message":    self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Returned by `validate_pulse_for_lab` on success. `warnings` is a list
    of (kind, message) tuples for non-fatal observations the caller may want
    to surface in the results (e.g., free_phases warning).
    """
    warnings: list[dict] = field(default_factory=list)


def _check_contract(sp: SampledPulse) -> None:
    """Shape/length sanity. `pulse_io.load_sampled_pulse` already enforces
    channel_order / units / frame; this catches the rest.
    """
    n = len(sp.times_ns)
    for name, arr in (("Omega_I", sp.Omega_I), ("Omega_Q", sp.Omega_Q),
                      ("epsilon_I", sp.epsilon_I), ("epsilon_Q", sp.epsilon_Q)):
        if arr.ndim != 1:
            raise PulseValidationError(
                kind="contract:shape_mismatch",
                message=f"{name} must be 1D; got ndim={arr.ndim} shape={arr.shape}",
            )
        if len(arr) != n:
            raise PulseValidationError(
                kind="contract:shape_mismatch",
                message=f"{name} length {len(arr)} does not match times length {n}",
            )
    if sp.times_ns.ndim != 1:
        raise PulseValidationError(
            kind="contract:shape_mismatch",
            message=f"times must be 1D; got ndim={sp.times_ns.ndim}",
        )
    if not np.all(np.diff(sp.times_ns) > 0):
        raise PulseValidationError(
            kind="contract:shape_mismatch",
            message="times must be strictly increasing",
        )


def _check_pulse_geometry(sp: SampledPulse) -> None:
    """Duration vs envelope memory + duration vs sanity floor."""
    duration_ns = sp.duration_ns
    duration_us = duration_ns * 1e-3

    if duration_ns < PULSE_DURATION_MIN_NS:
        raise PulseValidationError(
            kind="pulse:too_short",
            message=f"pulse duration {duration_ns:.1f} ns is below "
                    f"{PULSE_DURATION_MIN_NS:.0f} ns floor (probably a bug)",
            suggestion="check that times[] is in ns and that the pulse wasn't truncated",
        )

    if duration_us > DURATION_LIMIT_US:
        # The QICK signal generator stores up to 32768 envelope samples on the
        # gen's sample grid (fs ~= 6.39 GS/s for both qubit and manipulate
        # channels on this box) — that's the hard ceiling.
        raise PulseValidationError(
            kind="pulse:too_long",
            message=f"pulse duration {duration_us:.3f} us exceeds the "
                    f"{DURATION_LIMIT_US:.2f} us QICK envelope memory limit "
                    f"(gen sample grid at fs={QICK_GEN_FS_MHZ_DEFAULT:.0f} MHz, "
                    f"{QICK_ENVELOPE_SAMPLES} samples per channel)",
            suggestion=f"shorten the pulse below {DURATION_LIMIT_US:.2f} us "
                       f"or split it into stitched segments",
        )


def _check_gains(iq: IQTable, hw_cfg: Any, man_mode_idx: int) -> None:
    """Compute the would-be DAC gain registers and fail if they would clip.

    Mirrors `core.submit_wigner_via_queue`'s gain stamping path; if `core` ever
    rescales differently this will need to stay in sync.
    """
    gain_qb, gain_cav = compute_gains_from_ghz(hw_cfg, iq, man_mode_idx=man_mode_idx)
    if abs(gain_qb) > GAIN_REGISTER_LIMIT:
        peak_q = max(max((abs(v) for v in iq.I_q), default=0.0),
                     max((abs(v) for v in iq.Q_q), default=0.0))
        scale = GAIN_REGISTER_LIMIT / abs(gain_qb)
        raise PulseValidationError(
            kind="gain:qb_overflow",
            message=f"qubit DAC gain {gain_qb} > {GAIN_REGISTER_LIMIT} "
                    f"(peak |Omega| = {peak_q:.6f} GHz)",
            suggestion=f"reduce qubit drive amplitude by factor >= {1.0/scale:.3f} "
                       f"(scale upstream Omega by <= {scale:.3f})",
        )
    if abs(gain_cav) > GAIN_REGISTER_LIMIT:
        peak_c = max(max((abs(v) for v in iq.I_c), default=0.0),
                     max((abs(v) for v in iq.Q_c), default=0.0))
        scale = GAIN_REGISTER_LIMIT / abs(gain_cav)
        raise PulseValidationError(
            kind="gain:cav_overflow",
            message=f"cavity DAC gain {gain_cav} > {GAIN_REGISTER_LIMIT} "
                    f"(peak |epsilon| = {peak_c:.6f} GHz)",
            suggestion=f"reduce cavity drive amplitude by factor >= {1.0/scale:.3f} "
                       f"(scale upstream epsilon by <= {scale:.3f})",
        )


def _check_manifest_entry(me: Optional[dict]) -> None:
    if me is None:
        return
    alphas = me.get("alphas")
    if alphas is not None:
        if not isinstance(alphas, list):
            raise PulseValidationError(
                kind="manifest:bad",
                message=f"manifest 'alphas' must be a list (got {type(alphas).__name__})",
            )
        for i, a in enumerate(alphas):
            if not (isinstance(a, (list, tuple)) and len(a) == 2):
                raise PulseValidationError(
                    kind="manifest:bad",
                    message=f"manifest 'alphas[{i}]' must be a 2-element [re, im] "
                            f"pair; got {a!r}",
                )
            for j, v in enumerate(a):
                if not isinstance(v, (int, float)):
                    raise PulseValidationError(
                        kind="manifest:bad",
                        message=f"manifest 'alphas[{i}][{j}]' must be a number; "
                                f"got {v!r}",
                    )
    reps = me.get("reps")
    if reps is not None and (not isinstance(reps, int) or reps <= 0):
        raise PulseValidationError(
            kind="manifest:bad",
            message=f"manifest 'reps' must be a positive integer (got {reps!r})",
        )
    man_mode_no = me.get("man_mode_no")
    if man_mode_no is not None and (not isinstance(man_mode_no, int) or man_mode_no < 1):
        raise PulseValidationError(
            kind="manifest:bad",
            message=f"manifest 'man_mode_no' must be a positive integer "
                    f"(got {man_mode_no!r})",
        )
    measure_sigma_z = me.get("measure_sigma_z")
    if measure_sigma_z is not None and not isinstance(measure_sigma_z, bool):
        raise PulseValidationError(
            kind="manifest:bad",
            message=f"manifest 'measure_sigma_z' must be a bool "
                    f"(got {measure_sigma_z!r})",
        )
    sigma_z_mode = me.get("sigma_z_mode")
    if sigma_z_mode is not None and sigma_z_mode not in ("off", "reset", "postselect", "measure"):
        raise PulseValidationError(
            kind="manifest:bad",
            message=f"manifest 'sigma_z_mode' must be one of "
                    f"'off' | 'reset' | 'postselect' | 'measure' (got {sigma_z_mode!r})",
        )
    measurement = me.get("measurement")
    if measurement is not None and measurement not in ("wigner", "tomography_1q"):
        raise PulseValidationError(
            kind="manifest:bad",
            message=f"manifest 'measurement' must be 'wigner' or 'tomography_1q' "
                    f"(got {measurement!r})",
        )
    bases = me.get("bases")
    if bases is not None:
        if not isinstance(bases, list) or not bases or any(b not in ("Z", "X", "Y") for b in bases):
            raise PulseValidationError(
                kind="manifest:bad",
                message=f"manifest 'bases' must be a non-empty subset of "
                        f"['Z','X','Y'] (got {bases!r})",
            )


def _collect_warnings(sp: SampledPulse, manifest_entry: Optional[dict]) -> list[dict]:
    warnings: list[dict] = []
    # free_phases: cats with phi_2_cavity != 0 need a virtual-Z that we don't
    # apply yet. fock targets have phi_2 = 0 and are unaffected.
    free_phases = sp.meta.get("free_phases_rad") or (
        manifest_entry.get("free_phases_rad") if manifest_entry else None)
    if free_phases:
        nz = {k: v for k, v in free_phases.items() if abs(float(v)) > 1e-9}
        if nz:
            warnings.append({
                "kind":    "free_phases:nonzero",
                "message": f"pulse carries nonzero free_phases {nz!r}; "
                           f"virtual-Z is NOT yet applied. Measured parity "
                           f"reflects F_fix, not F_free. Cats need this; "
                           f"fock targets have zero phases and are unaffected.",
                "suggestion": None,
            })
    return warnings


def validate_pulse_for_lab(
    sp: SampledPulse,
    iq: IQTable,
    *,
    manifest_entry: Optional[dict] = None,
    hw_cfg: Optional[Any] = None,
    man_mode_idx: int = 0,
    mode: str = "hw",
) -> ValidationResult:
    """Run all enabled pre-flight checks.

    `mode='hw'` runs the hardware-dependent checks (gain, geometry) that need
    `hw_cfg`. `mode='sim'` runs only the schema/manifest checks. Raises
    `PulseValidationError` on first hard failure; returns a `ValidationResult`
    with soft warnings on success.
    """
    _check_contract(sp)
    _check_pulse_geometry(sp)   # geometry depends only on the jld2, not hw_cfg
    _check_manifest_entry(manifest_entry)

    if mode == "hw":
        if hw_cfg is None:
            raise PulseValidationError(
                kind="contract:missing_field",
                message="hw_cfg is required for hw-mode validation but was None",
            )
        _check_gains(iq, hw_cfg, man_mode_idx)

    return ValidationResult(warnings=_collect_warnings(sp, manifest_entry))
