"""Pulse I/O for the closed_loop file-drop ("mailbox") runner.

Reads the rotating-frame `*_sampled.jld2` files Harmoniqs produces (Piccolo /
Intonato), converts their `rad/ns` channels into the linear-GHz amplitudes the
existing `core.IQTable` / gain pipeline expects, and renders a preview matching
their `*_IQ.png` (top panel = Ω drive; bottom panel = α displacement) for
visual validation of channel ordering / sign.

Sampled-jld2 contract (Harmoniqs side, confirmed via manifest.json):
  channel_order = [Omega_I_qubit, Omega_Q_qubit, epsilon_I_cavity, epsilon_Q_cavity]
  channel_units = "rad/ns" for all four
  time_units    = "ns"
  frame         = "rotating"
  qubit drive:  H = Omega_I·X̂ + Omega_Q·P̂  ;  Rabi rate (rad/ns) = 2|Omega|
  cavity drive: epsilon(t) = i·alpha_dot(t)   (Eickbusch 2022 closed-system, k=0)
  free_phases_rad : Rz(phi_1) on transmon + Rz(phi_2) on cavity AFTER pulse
                    (virtual-Z; deferred — cats with nonzero phi_2 measure
                    F_fix not F_free).

Unit conversion to our IQTable (linear GHz Rabi rate; service multiplies by
2π·1e3 in `compute_gains_from_ghz` to recover angular rate in rad/μs):

  Qubit:  angular Rabi (rad/ns) = 2|Omega| ⇒ linear (GHz) = |Omega|/π
          ⇒ I_q[GHz] = Omega_I / π ;  Q_q[GHz] = Omega_Q / π

  Cavity: scale factor TBD-and-validated in M2 (cross-check vs the existing
          `device.optimal_control.fock.1` npz). For M1 we default to /π — the
          same convention as the qubit — and surface the choice so M2 can
          calibrate it without code changes downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from job_server.closed_loop.core import IQTable


# Conversion factors. See GAIN_DERIVATION.md for the full derivation.
#
# The codebase uses H_drive,q = Ω·(q+q†) = Ω·σ_x (no factor 1/2), so the
# formula's `max_q` is the σ_x coefficient in rad/μs — the same thing as
# Harmoniqs' Ω, just unit-converted from rad/ns. Similarly `max_c` is the
# cavity drive coefficient ε in rad/μs. Both conversions are therefore plain
# unit math, /(2π).
#
# An earlier version of this file used qubit_factor = 1/π under the assumption
# `max_q` was the angular Rabi rate Ω_R = 2|Ω| under standard RWA. That was
# wrong (off by 2) — it doubled the effective σ_x coefficient and pushed
# Harmoniqs fock1 over the QICK register limit. See §5–§8 of GAIN_DERIVATION.md.
QUBIT_RAD_PER_NS_TO_GHZ  = 1.0 / (2.0 * np.pi)  # I_q[GHz] = Ω   / (2π)
CAVITY_RAD_PER_NS_TO_GHZ = 1.0 / (2.0 * np.pi)  # I_c[GHz] = ε / (2π)

# Back-compat alias (the _DEFAULT name was used earlier in development).
CAVITY_RAD_PER_NS_TO_GHZ_DEFAULT = CAVITY_RAD_PER_NS_TO_GHZ

# Expected conventions in the sampled jld2 — we hard-fail if these drift.
EXPECTED_CHANNEL_ORDER = (
    "Omega_I_qubit",
    "Omega_Q_qubit",
    "epsilon_I_cavity",
    "epsilon_Q_cavity",
)
EXPECTED_CHANNEL_UNIT = "rad_per_ns"
EXPECTED_TIME_UNIT    = "ns"
EXPECTED_FRAME        = "rotating"


@dataclass
class SampledPulse:
    """Raw rotating-frame envelopes as Harmoniqs ship them, plus minimal meta.

    Arrays are 1-D and same length. Units: Omega/epsilon in rad/ns; times in ns.
    `meta` carries the manifest-relevant scalars (target, free phases, source
    file, nominal carriers) so downstream code doesn't re-open the jld2.
    """
    times_ns: np.ndarray
    Omega_I:  np.ndarray
    Omega_Q:  np.ndarray
    epsilon_I: np.ndarray
    epsilon_Q: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.times_ns)
        for name in ("Omega_I", "Omega_Q", "epsilon_I", "epsilon_Q"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} length {len(getattr(self, name))} != times length {n}"
                )

    @property
    def duration_ns(self) -> float:
        return float(self.times_ns[-1] - self.times_ns[0])

    @property
    def dt_ns(self) -> float:
        return float(self.times_ns[1] - self.times_ns[0])


def _decode_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _decode_str_array(arr) -> list[str]:
    out = []
    for v in np.asarray(arr).ravel().tolist():
        out.append(_decode_str(v))
    return out


def load_sampled_pulse(path: str | Path) -> SampledPulse:
    """Load a `*_sampled.jld2` file. Raises ValueError if conventions diverge."""
    import h5py
    path = Path(path)
    with h5py.File(str(path), "r") as h:
        # convention guards — fail loudly if Harmoniqs changes the contract
        ch_order = tuple(_decode_str_array(h["channel_order"][()]))
        if ch_order != EXPECTED_CHANNEL_ORDER:
            raise ValueError(
                f"{path.name}: unexpected channel_order {ch_order}, "
                f"expected {EXPECTED_CHANNEL_ORDER}"
            )
        ch_units = tuple(_decode_str_array(h["channel_units"][()]))
        for u in ch_units:
            if u != EXPECTED_CHANNEL_UNIT:
                raise ValueError(
                    f"{path.name}: channel_units {ch_units} include {u!r}, "
                    f"expected all {EXPECTED_CHANNEL_UNIT!r}"
                )
        time_units = _decode_str(h["time_units"][()])
        if time_units != EXPECTED_TIME_UNIT:
            raise ValueError(
                f"{path.name}: time_units={time_units!r}, expected {EXPECTED_TIME_UNIT!r}"
            )
        frame = _decode_str(h["frame"][()])
        if frame != EXPECTED_FRAME:
            raise ValueError(
                f"{path.name}: frame={frame!r}, expected {EXPECTED_FRAME!r}"
            )

        times_ns = np.asarray(h["times"][()], dtype=float)
        Omega_I  = np.asarray(h["Omega_I"][()], dtype=float)
        Omega_Q  = np.asarray(h["Omega_Q"][()], dtype=float)
        epsilon_I = np.asarray(h["epsilon_I"][()], dtype=float)
        epsilon_Q = np.asarray(h["epsilon_Q"][()], dtype=float)

        meta: dict = {
            "source_file": _decode_str(h["source_file"][()]) if "source_file" in h else None,
            "target_state": _decode_str(h["target_state"][()]) if "target_state" in h else None,
            "frame": frame,
            "sampling_dt_ns": float(h["sampling_dt_ns"][()]) if "sampling_dt_ns" in h else None,
            "sampling_n_points": int(h["sampling_n_points"][()]) if "sampling_n_points" in h else len(times_ns),
            "omega_q_GHz": float(h["omega_q_GHz"][()]) if "omega_q_GHz" in h else None,
            "omega_c_GHz": float(h["omega_c_GHz"][()]) if "omega_c_GHz" in h else None,
            # free phases live inside a Julia kvvec — best-effort decode; if it
            # fails we fall through and let the manifest supply them instead.
        }
        try:
            fp = h["free_phases_rad"][()]
            # numpy void with one object-ref field 'kvvec'
            ref = fp["kvvec"]
            kv = h[ref]
            kvals = np.asarray(kv[()]).ravel()
            # JLD2 NamedTuple: alternating (key, value) refs. Best-effort:
            phases: dict[str, float] = {}
            for i in range(0, len(kvals), 2):
                try:
                    k = _decode_str(h[kvals[i]][()])
                    v = float(np.asarray(h[kvals[i+1]][()]).ravel()[0])
                    phases[k] = v
                except Exception:
                    pass
            meta["free_phases_rad"] = phases or None
        except Exception:
            meta["free_phases_rad"] = None

    return SampledPulse(
        times_ns=times_ns,
        Omega_I=Omega_I, Omega_Q=Omega_Q,
        epsilon_I=epsilon_I, epsilon_Q=epsilon_Q,
        meta=meta,
    )


def sampled_to_iq_table(
    sp: SampledPulse,
    *,
    qubit_factor: float = QUBIT_RAD_PER_NS_TO_GHZ,
    cavity_factor: float = CAVITY_RAD_PER_NS_TO_GHZ_DEFAULT,
) -> IQTable:
    """Convert a SampledPulse to our IQTable (times µs; amplitudes in 'linear GHz').

    Channel mapping: their qubit-first ordering -> our IQ_table fields:
      I_q = Omega_I * qubit_factor
      Q_q = Omega_Q * qubit_factor
      I_c = epsilon_I * cavity_factor
      Q_c = epsilon_Q * cavity_factor
    """
    times_us = (sp.times_ns * 1e-3).tolist()
    return IQTable(
        times=times_us,
        I_c=(sp.epsilon_I * cavity_factor).tolist(),
        Q_c=(sp.epsilon_Q * cavity_factor).tolist(),
        I_q=(sp.Omega_I  * qubit_factor).tolist(),
        Q_q=(sp.Omega_Q  * qubit_factor).tolist(),
    )


def compute_alpha_trajectory(
    sp: SampledPulse,
    *,
    alpha0: complex = 0.0 + 0.0j,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ε(t) -> α(t) per ε = i·α_dot ⇒ α_dot = -i·ε.

    Returns (alpha_I, alpha_Q) on the sampled time grid. Uses cumulative
    trapezoid for stability and to match the smooth-looking displacement
    panels in their preview PNGs.
    """
    eps = sp.epsilon_I + 1j * sp.epsilon_Q
    # alpha_dot = -i * eps
    alpha_dot = -1j * eps
    # cumulative trapezoid in real and imag parts
    t = sp.times_ns
    if len(t) < 2:
        return np.array([alpha0.real]), np.array([alpha0.imag])
    # integrate alpha_dot
    re = np.concatenate(([0.0], np.cumsum(0.5 * (alpha_dot.real[1:] + alpha_dot.real[:-1]) * np.diff(t))))
    im = np.concatenate(([0.0], np.cumsum(0.5 * (alpha_dot.imag[1:] + alpha_dot.imag[:-1]) * np.diff(t))))
    return alpha0.real + re, alpha0.imag + im


def render_pulse_preview(
    sp: SampledPulse,
    out_path: str | Path,
    *,
    title: Optional[str] = None,
) -> Path:
    """Render a two-panel preview of the rotating-frame drives that will hit
    hardware. Top = qubit drive Ω; bottom = cavity drive ε.

    This is what we send to the AWG, so it's the right thing to diff against
    Harmoniqs' *_IQ.png Ω panel. Their bottom panel is the displaced-frame α
    trajectory (the optimizer's primary variable, derived from ε via
    ε = i·α_dot in their frame); we don't reproduce α here because the
    integration depends on their displaced-frame convention.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    t = sp.times_ns
    Om_env  = np.sqrt(sp.Omega_I  ** 2 + sp.Omega_Q  ** 2)
    eps_env = np.sqrt(sp.epsilon_I ** 2 + sp.epsilon_Q ** 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    ax1.plot(t, sp.Omega_I, color="C0", label=r"$\Omega_I$", lw=1.0)
    ax1.plot(t, sp.Omega_Q, color="C1", label=r"$\Omega_Q$", lw=1.0)
    ax1.plot(t, Om_env, color="k", ls="--", lw=0.8, label=r"$|\Omega|$")
    ax1.set_ylabel("Amplitude (rad/ns)")
    ax1.set_title("Qubit drive ($\\Omega$, rotating frame)", loc="left", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, sp.epsilon_I, color="C0", label=r"$\varepsilon_I$", lw=1.0)
    ax2.plot(t, sp.epsilon_Q, color="C1", label=r"$\varepsilon_Q$", lw=1.0)
    ax2.plot(t, eps_env, color="k", ls="--", lw=0.8, label=r"$|\varepsilon|$")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Amplitude (rad/ns)")
    ax2.set_title("Cavity drive ($\\varepsilon$, rotating frame)", loc="left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================== zip I/O ====================================

_IGNORE_PREFIXES = ("__MACOSX/",)
_IGNORE_NAMES    = (".DS_Store",)


def _is_junk(name: str) -> bool:
    n = name.replace("\\", "/")
    if any(n.startswith(p) for p in _IGNORE_PREFIXES):
        return True
    base = n.rsplit("/", 1)[-1]
    return base in _IGNORE_NAMES


def unpack_request_zip(zip_path: str | Path, extract_to: str | Path) -> Path:
    """Extract a `pulses_*.zip` drop, skipping macOS junk.

    Returns the root of extracted contents (the inner folder, e.g. `pulse_groups/`,
    if the zip wraps everything in one). Otherwise returns `extract_to` itself.
    """
    import zipfile
    zip_path   = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if _is_junk(info.filename):
                continue
            zf.extract(info, extract_to)
    # Detect single-root-folder wrap (the common Mac-zipped case)
    top = [p for p in extract_to.iterdir() if not p.name.startswith(".")]
    if len(top) == 1 and top[0].is_dir():
        return top[0]
    return extract_to


def discover_pulses(root: str | Path) -> list[dict]:
    """Walk an extracted drop and return one entry per pulse subfolder.

    Each entry: {group, dir, sampled_path, source_path, lab_path, preview_path}.
    The `sampled_path` is the file we feed to load_sampled_pulse; others are
    informational. `preview_path` is the collaborator's `*_IQ.png` if present.
    """
    root = Path(root)
    out: list[dict] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        sampled  = sorted(sub.glob("*_sampled.jld2"))
        if not sampled:
            continue
        lab     = sorted(sub.glob("*_lab.jld2"))
        previews = sorted(sub.glob("*_IQ.png"))
        # The "source" spline jld2 is whichever .jld2 isn't tagged sampled/lab.
        all_jld2 = sorted(sub.glob("*.jld2"))
        sampled_set = {p.name for p in sampled}
        lab_set     = {p.name for p in lab}
        sources = [p for p in all_jld2 if p.name not in sampled_set and p.name not in lab_set]
        out.append({
            "group":        sub.name,
            "dir":          sub,
            "sampled_path": sampled[0],
            "source_path":  sources[0] if sources else None,
            "lab_path":     lab[0] if lab else None,
            "preview_path": previews[0] if previews else None,
        })
    return out


# =============================== CLI =======================================

def _cli_preview(args) -> int:
    """Render our previews for every pulse in a drop folder/zip. Print a table
    of peak amplitudes per channel so ordering/sign is human-checkable side by
    side with their *_IQ.png.
    """
    import tempfile, shutil
    src = Path(args.input)
    if src.is_file() and src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="closed_loop_preview_"))
        try:
            root = unpack_request_zip(src, tmp)
            return _render_all(root, Path(args.out))
        finally:
            if not args.keep:
                shutil.rmtree(tmp, ignore_errors=True)
    elif src.is_dir():
        return _render_all(src, Path(args.out))
    else:
        print(f"input must be a .zip or a directory: {src}")
        return 2


def _render_all(root: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pulses = discover_pulses(root)
    if not pulses:
        print(f"no pulse subfolders with *_sampled.jld2 found under {root}")
        return 1
    print(f"{'group':28s}  {'n':>5s}  {'peak |Ω| (rad/ns)':>20s}  {'peak |ε| (rad/ns)':>20s}  {'duration (ns)':>14s}  preview")
    for p in pulses:
        try:
            sp = load_sampled_pulse(p["sampled_path"])
        except Exception as e:
            print(f"{p['group']:28s}  ERROR: {e}")
            continue
        Om_env = float(np.max(np.sqrt(sp.Omega_I ** 2 + sp.Omega_Q ** 2)))
        eps_env = float(np.max(np.sqrt(sp.epsilon_I ** 2 + sp.epsilon_Q ** 2)))
        out_png = out_dir / f"{p['group']}_ours.png"
        title = f"{p['group']} — target={sp.meta.get('target_state')} | dt={sp.meta.get('sampling_dt_ns')}ns"
        render_pulse_preview(sp, out_png, title=title)
        their = p["preview_path"].name if p["preview_path"] else "(none)"
        print(f"{p['group']:28s}  {len(sp.times_ns):5d}  {Om_env:20.4f}  {eps_env:20.4f}  {sp.duration_ns:14.2f}  ours={out_png.name}  theirs={their}")
    return 0


def _cli_validate_units(args) -> int:
    """Round-trip the existing canonical pulse_fock_1.npz through the gain
    pipeline as a sanity check, then dry-run the gains a Harmoniqs *_sampled.jld2
    pulse would produce with the current QUBIT_/CAVITY_ conversion factors.

    Reads hardware_config.yml directly (no station boot needed). Prints a
    side-by-side so unit drift is visually obvious.
    """
    import yaml
    from job_server.closed_loop.core import (
        _load_iqtable_from_npz, compute_gains_from_ghz, IQTable,
    )

    cfg = yaml.safe_load(open(args.hw_config))

    # tiny attr-style proxy so compute_gains_from_ghz works without a station
    class _Attr:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, _Attr(v) if isinstance(v, dict) else v)
    hw = _Attr(cfg)

    # 1) Existing canonical pulse round-trip
    oc1 = cfg["device"]["optimal_control"]["fock"]["1"]
    npz_path = args.fock1_npz or oc1["filename"]
    print(f"--- existing canonical: {npz_path}")
    iq_ref = _load_iqtable_from_npz(npz_path)
    gq_canon, gc_canon = oc1["gain"][0], oc1["gain"][1]
    gq_round, gc_round = compute_gains_from_ghz(hw, iq_ref, man_mode_idx=0)
    peak_q = max((abs(v) for v in iq_ref.I_q + iq_ref.Q_q), default=0.0)
    peak_c = max((abs(v) for v in iq_ref.I_c + iq_ref.Q_c), default=0.0)
    print(f"  peak (linear-GHz, from npz):  qubit={peak_q:.5f}  cavity={peak_c:.5f}")
    print(f"  computed gains:               qb={gq_round:5d}  cav={gc_round:5d}")
    print(f"  canonical gains in config:    qb={gq_canon:5d}  cav={gc_canon:5d}")
    if gq_round != gq_canon or gc_round != gc_canon:
        print(f"  WARNING: round-trip mismatch (delta qb={gq_round-gq_canon:+d}, "
              f"cav={gc_round-gc_canon:+d})")

    # 2) Dry-run a Harmoniqs sampled pulse
    if args.sampled:
        sp = load_sampled_pulse(args.sampled)
        iq_h = sampled_to_iq_table(sp)
        peak_q_h = max((abs(v) for v in iq_h.I_q + iq_h.Q_q), default=0.0)
        peak_c_h = max((abs(v) for v in iq_h.I_c + iq_h.Q_c), default=0.0)
        peak_Om = float(np.max(np.sqrt(sp.Omega_I**2 + sp.Omega_Q**2)))
        peak_ep = float(np.max(np.sqrt(sp.epsilon_I**2 + sp.epsilon_Q**2)))
        try:
            gq_h, gc_h = compute_gains_from_ghz(hw, iq_h, man_mode_idx=0)
            in_range = (0 <= gq_h <= 32767) and (0 <= gc_h <= 32767)
            tag = "OK" if in_range else "OUT OF RANGE"
        except Exception as e:
            gq_h, gc_h, tag = -1, -1, f"ERROR: {e}"
        print(f"\n--- Harmoniqs sampled: {args.sampled}")
        print(f"  target_state = {sp.meta.get('target_state')!r}")
        print(f"  raw peaks (rad/ns):           |Ω|={peak_Om:.5f}  |ε|={peak_ep:.5f}")
        print(f"  conversion factors:           qubit /(2π) = {QUBIT_RAD_PER_NS_TO_GHZ:.5f}"
              f"   cavity /(2π) = {CAVITY_RAD_PER_NS_TO_GHZ:.5f}")
        print(f"  converted peaks (linear-GHz): qubit={peak_q_h:.5f}  cavity={peak_c_h:.5f}")
        print(f"  computed gains:               qb={gq_h:5d}  cav={gc_h:5d}   [{tag}]")
        print(f"  ratio vs existing canonical:  qb x{(gq_h/max(gq_canon,1)):.2f}  "
              f"cav x{(gc_h/max(gc_canon,1)):.2f}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="closed_loop pulse_io tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("preview", help="render our previews for every pulse in a drop")
    pv.add_argument("input", help="path to a pulses_*.zip OR an extracted pulse_groups/ dir")
    pv.add_argument("--out", default="./previews_ours", help="output dir for our PNGs")
    pv.add_argument("--keep", action="store_true", help="keep extracted temp dir (if input is a zip)")
    pv.set_defaults(func=_cli_preview)

    vu = sub.add_parser("validate-units", help="round-trip existing fock_1 npz + dry-run a Harmoniqs jld2 through the gain pipeline")
    vu.add_argument("--hw-config", default=r"c:\python\multimode_expts\configs\hardware_config.yml",
                    help="path to hardware_config.yml (defaults to repo copy)")
    vu.add_argument("--fock1-npz", default=None,
                    help="override path to pulse_fock_1.npz (defaults to the one named in hw-config; H: path may need substitution for G:)")
    vu.add_argument("--sampled", default=None,
                    help="path to a Harmoniqs *_sampled.jld2 to dry-run alongside")
    vu.set_defaults(func=_cli_validate_units)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
