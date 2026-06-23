"""Generate DEVICE_PARAMETERS.md from the live (main) hardware config.

The optimal-control contract Harmoniqs reads (`DEVICE_PARAMETERS.md` in the
shared Drive mailbox) used to be hand-typed, so it drifted from the config the
runner actually plays against. This regenerates it from the same config source
of truth `core.init_core` loads, so the two can't disagree.

Lean by design: the optimizer works in the rotating frame, so it only needs the
model Hamiltonian parameters and the carrier frequencies. Hardware-side
translation (gain_to_alpha, π-pulse gain, α_scale, ...) is the lab box's job and
is deliberately left out of the contract.

All emitted parameters are in MHz (f = ω/2π), straight from the config fields:

  α/(2π)  = device.qubit.f_ef[q]  − device.qubit.f_ge[q]   (anharmonicity)
  Kc/(2π) = device.manipulate.kerr[m]                       (cavity self-Kerr)
  χ/(2π)  = device.manipulate.chi_ge[m]                     (dispersive shift)
  f_q     = device.qubit.f_ge[q]                            (carrier)
  f_c     = device.manipulate.f_ge[m]                       (carrier)

Usage
-----
    pixi run python -m job_server.closed_loop.gen_device_parameters            # write to mailbox
    pixi run python -m job_server.closed_loop.gen_device_parameters --stdout   # preview only
    pixi run python -m job_server.closed_loop.gen_device_parameters \
        --hardware-config CFG-HW-20260619-00053   # pin a specific version
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from job_server.closed_loop.batch_runner import DEFAULT_MAILBOX
from job_server.config_versioning import ConfigType, ConfigVersionManager
from job_server.database import get_database

CONFIG_DIR = "configs"


def resolve_config_path(hardware_config: str | None) -> Path:
    """Path to the hardware-config YAML the contract should mirror.

    Default (None) follows the DB `main` pointer — the same config
    `MultimodeStation(mock=True, hardware_config=None)` resolves, which is what
    batch_runner pins by default. A version id pins that specific snapshot.
    """
    cm = ConfigVersionManager(CONFIG_DIR)
    if hardware_config:
        p = Path(CONFIG_DIR) / "versions" / "hardware_config" / f"{hardware_config}.yml"
        if not p.exists():
            raise FileNotFoundError(f"no such hardware-config version: {p}")
        return p
    db = get_database()
    with db.session() as s:
        return Path(cm.get_main_config_path(ConfigType.HARDWARE_CONFIG, s))


def render(cfg: dict, version_id: str, *, qubit: int, man_mode_no: int) -> str:
    m = man_mode_no - 1
    q = cfg["device"]["qubit"]
    man = cfg["device"]["manipulate"]

    f_q   = float(q["f_ge"][qubit])
    f_ef  = float(q["f_ef"][qubit])
    anharm = f_ef - f_q
    f_c   = float(man["f_ge"][m])
    Kc    = float(man["kerr"][m])
    chi   = float(man["chi_ge"][m])

    chi_sign = "down" if chi < 0 else "up"
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""# Device parameters for optimal-control pulse design

**Auto-generated from `{version_id}` on {today}** — do not hand-edit; rerun
`pixi run python -m job_server.closed_loop.gen_device_parameters` after a
recalibration. Source of truth is the multimode QPU's main hardware config (the
same one the mailbox runner plays your pulses against). Values for qubit {qubit},
manipulate mode {man_mode_no}.

All parameters are in MHz (i.e. quoted as `f = ω/(2π)`).

## Model Hamiltonian (rotating frame at bare ω_q, ω_c — closed system, κ_c = 0)

```
H/ℏ  =  (α/2) · b† b† b b              # transmon anharmonicity
      + (Kc/2) · a† a† a a             # cavity self-Kerr
      +  χ    · b† b · a† a            # dispersive coupling
      +  Ω(t) · b†  +  Ω*(t) · b       # qubit drive
      +  ε(t) · a†  +  ε*(t) · a       # cavity drive
```

with

- `b, b†` : transmon (Duffing oscillator)
- `a, a†` : storage cavity (manipulate mode)
- complex drive envelopes
  `Ω(t) = Ω_I(t) + i Ω_Q(t)` and  `ε(t) = ε_I(t) + i ε_Q(t)`

Equivalently for the drive (the form in your manifest):

```
H_drive/ℏ = Ω_I (b + b†) + Ω_Q · i (b† − b)
          + ε_I (a + a†) + ε_Q · i (a† − a)
```

**No `1/2` in front of the drive** — this is the convention your manifest
already uses ("Rabi rate (rad/ns) = `2|Ω|`"), and it's what the gain pipeline
on the hardware side maps into. Consequences:

- Angular Rabi rate on the g–e transition: `Ω_R(t) = 2 |Ω(t)|`
- Cavity displacement (closed system, κ_c = 0): `α̇(t) = −i ε(t)`

## Model parameters

| symbol | value (MHz) | notes |
|---|---|---|
| `α/(2π)`  | **{anharm:.3f}**   | transmon anharmonicity (= `f_ef − f_ge`). **Negative** by convention. |
| `Kc/(2π)` | **{Kc:.6f}**  | cavity self-Kerr |
| `χ/(2π)`  | **{chi:.6f}**     | dispersive shift, `χ = χ_ge`. **{'Negative' if chi < 0 else 'Positive'}** — cavity shifts *{chi_sign}* with qubit excited. |

## Hardware carriers (for playing the rotating-frame envelope on resonance)

Not part of the Hamiltonian above (the model is in the rotating frame), but
you need these to know the carrier frequencies the AWG should emit the
envelope at on this device:

| symbol | value (MHz) | source |
|---|---|---|
| `f_q = ω_q/(2π)`  | **{f_q:.4f}** | `device.qubit.f_ge[{qubit}]`         |
| `f_c = ω_c/(2π)`  | **{f_c:.4f}** | `device.manipulate.f_ge[{m}]`    |

A pulse designed *in the rotating frame* against the parameters above will play
on-resonance: the hardware side overrides the (stale) carriers cached in its
`optimal_control` config entries with these bare-`f_ge` values automatically.

## What is **not** included in this model (i.e. not yet captured for you)

- `κ_c`, `T1_cavity`, `T2_cavity` — open-system cavity loss. We don't have
  freshly-measured numbers; if you need realistic loss in the optimizer,
  ping us and we'll measure.
- `T2 / Tφ` of the qubit (only `T1 = {float(q['T1'][qubit]):.0f} µs` is in the pinned config).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware-config", default=None,
                    help="hardware-config version id to pin (default: DB main pointer)")
    ap.add_argument("--qubit", type=int, default=0)
    ap.add_argument("--man-mode-no", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help=f"output path (default: {DEFAULT_MAILBOX / 'DEVICE_PARAMETERS.md'})")
    ap.add_argument("--stdout", action="store_true",
                    help="print to stdout instead of writing the file")
    args = ap.parse_args()

    path = resolve_config_path(args.hardware_config)
    version_id = path.stem
    cfg = yaml.safe_load(path.read_text())
    doc = render(cfg, version_id, qubit=args.qubit, man_mode_no=args.man_mode_no)

    if args.stdout:
        print(doc)
        return

    out = Path(args.out) if args.out else DEFAULT_MAILBOX / "DEVICE_PARAMETERS.md"
    out.write_text(doc, encoding="utf-8")
    print(f"[gen_device_parameters] wrote {out}  (from {version_id})")


if __name__ == "__main__":
    main()
