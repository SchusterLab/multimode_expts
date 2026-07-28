---
type: spec
date: 2026-07-16
session_id: "e3dfcc03-f686-47a0-b129-29407ed941d0"
status: draft
priority: P2
platform: bosonic
tags: [spec, qicklab, bosonic, multimode, qick, tproc-v2, schuster-lab, multimode_expts, pulse-ir, device-model, fluent-builder, config-versioning, single-source-of-truth, active-reset, mock-testing, fitting, lmfit, job-server, data-lifecycle]
linked_plan: null
---

# `qicklab` — a first-principles QICK tProc-v2 experiment framework (first target: the Schuster multimode device)

> **What this is.** A design-of-record for a **fresh, standalone Python package**, `qicklab` — a
> general **QICK tProc-v2 experiment framework** (device model + pulse IR + compiler + experiment/fitting
> lifecycle + a raw-qick escape hatch). Its **first target** is the *core functionality* of Stanford's
> [`SchusterLab/multimode_expts`](https://github.com/SchusterLab/multimode_expts)
> on **QICK tProc v2** — from first principles, **not** as a port-in-place. It keeps
> the physics (qubit + manipulate cavity + storage modes + coupler, with f0g1,
> beamsplitter swaps, parity, active reset) and throws out the architecture: the
> 2745-line `MM_base` god-mixin, the positional 7-row pulse descriptor, the
> `eval`-dispatched string pulse-creator, the contradictory magic-int channel maps,
> and the multi-sourced config.
>
> **Status.** Draft for review. The design was worked out interactively with Aaron;
> every major fork below records that decision. One genuine feasibility risk
> (measurement-conditional feedback on tProc v2 — "active reset") is flagged for the
> QICK collaboration (Sho Uemura).
>
> **Relationship to existing work.** Sibling to `expt_service` (the coarse board-side
> 3-verb service for the closed-loop optimizer) and `IntonatoQICK.jl`. This package is
> a general QICK experiment framework whose **first target is the native Stanford multimode stack** on
> v2 — a much richer surface than the 3-verb contract. It does not replace them; if a closed-loop path is later wanted, one adapter
> maps GHz-Rabi → `Pulse.gain`.

## 1. Context & motivation

`multimode_expts` is a tProc **v1** codebase (confirmed: zero `asm_v2` usage anywhere).
Its own TODO reads *"Migrate to tProcv2 at some point."* The stack today is four layers:

1. **`Experiment` classes** (**~83** ending in `Experiment` in `single_qubit/` alone; **~140** across
   `experiments/`) — expand cfg, instantiate a Program, `prog.acquire(soc)`, unpack
   `(xpts, avgi, avgq)`, `analyze()`/`display()`.
2. **Three program bases** — `MMAveragerProgram` / `MMRAveragerProgram` /
   `MMNDAveragerProgram` = v1 `AveragerProgram`/`RAveragerProgram`/`NDAveragerProgram`
   **+ `MM_base` mixin**, each with hand-written `initialize()`/`body()`.
3. **`MM_base`** (2745 lines) — every pulse primitive (`custom_pulse`,
   `register_long_pulse`, `active_reset`, `measure_wrapper`, `reset_and_sync`, parity,
   dual-rail) written against v1's **register-level** API (`setup_and_pulse`,
   `freq2reg`/`deg2reg`/`us2cycles`, `sync_all`, `add_gauss`).
4. **`prepulse_creator2`** — builds the pulse descriptor `[[freq],[gain],[len],[phase],[ch],[shape],[sigma]]`, dispatched via literal `eval(f"creator.{channel_name}(...)")`.

### 1.1 The messes a redesign fixes (all found in-tree)

- **God-mixin + pre-converted registers.** `parse_config` dumps ~50 attributes onto
  `self` and pre-converts physical values to register units at parse time — a pure
  v1-ism, since v2 `add_pulse` takes physical units (MHz/µs/deg) directly.
- **Contradictory magic-int channel model — *three* disagreeing legends.** `channel_table` says
  `4→man, 6→storage`; `custom_pulse`'s docstring says `4→storage, 6→manipulate`; and
  `prepulse_creator2.__init__` says `3→manipulate, 4→storage` — and none matches the actual config DACs.
- **Positional untyped descriptor.** 7 parallel object-dtype lists indexed by physical row;
  string-typed shapes (`"gaussian"`/`"gauss"`/`"g"`/`"flat_top"`/`"f"`/…).
- **Multi-sourced config (no single source of truth).** See §7.1 — f0g1 defined in two files,
  `man()` duplicates `multiphoton('fn-gn+1')`, qubit ge freq in 3+ places, channel map both in
  config and hardcoded in code.
- **High-friction extension.** Adding one pulse type touches a config location (one of *5*
  schemas), a `prepulse_creator2` method, maybe a `custom_pulse` branch, and
  `channel_assign`/`channel_table` — four edits, four desync chances.

### 1.2 Assets to reuse (conceptually — reimplemented in the new package)

- **`MockQickSoc` strategy** — the real qick library runs through build→compile→acquire; only
  the leaf `soc.*` calls are stubbed. This validates programs (v2 param validators fire) with no
  FPGA. Directly reusable for v2 (same three-phase split).
- **`soccfg_snapshot.json`** — committed `QickConfig` snapshot for off-board work.
- **`BranchManager`** — a git-reflog-style config-versioning idea (good model; re-homed onto git, §7.3).

## 2. Locked decisions (from the design session)

| # | Question | Decision |
|---|---|---|
| D1 | Migration goal | **First-principles greenfield**, not port-in-place. Keep core functionality, free to redesign architecture. |
| D2 | Functional surface of first slice | **Broad multimode core**: single-qubit (ge/ef) + f0g1 + storage swaps + parity + active_reset. |
| D3 | Where it lives | **Fresh standalone package** (`qicklab`), outside `multimode_expts`. |
| D4 | Fitting/data-processing | **New `lmfit`-based core fitters** (Rabi/T1/T2/parity) behind a typed result seam, **validated against the current `fitting.py` as a physics oracle** (§9). Full fit-layer refactor is a separate follow-on. |
| D13 | Data lifecycle | **Keep the high-level `Experiment` contract** (`go`/`acquire`/`analyze`/`display`/`save`) but **separate the internals** into distinct collaborators — Runner (acquire) / Analyzer (lmfit fitters) / Display (plot). The v1 conflation lives in the *implementation*, not the interface (§9). |
| D5 | Architecture | **Approach A — Pulse-IR + compiler**, thin base class; logic in bounded, independently-testable modules. |
| D6 | Escape hatch | **Every abstraction is escapable** (Tenet 0). `QickLabProgram` *is* an `AveragerProgramV2`; raw qick reachable at three levels. |
| D7 | Channel abstraction | **Model the device, not the FPGA** (Tenet 1). Two layers: device language on top, swappable DAC wiring underneath. |
| D8 | Sequence surface | **Fluent builder** `Seq().play(...).wait(...).raw(...).measure()` as primary. |
| D9 | Config versioning | **OPEN — to circulate (§14.1).** Tension between append-only immutable snapshots (conflict-free by construction; the *current* model) and git-native (simpler, but race-prone under the shared worker + concurrent users). The earlier lean toward git-native is reopened given the operational reality. |
| D10 | Config integrity | **Single source of truth** for every physical quantity; nothing derived is stored; low-friction extension. |
| D14 | Job-server / multi-user | **OPEN — to circulate (§14.2).** How qicklab relates to the existing FastAPI + SQLite + single-worker queue: compatible-with vs a clean re-take on the runner vs reimplement. Aaron to circulate. |
| D11 | Pulse API | **Typed factory functions**, no string keys, no `eval`; pulses are *computed from calibration*, not a stored bank keyed by strings. |
| D12 | `dev` | The one loaded `Device` object (wiring + calibrations + soccfg **snapshot** + datasets); factory namespaces hang off it. The compiler builds the live `QickConfig` from the snapshot, keeping `device.py` qick-free. |
| D15 | Generality / horizontal scope | **Design-for-extension, implement-one** (§15). Split into a modality-agnostic `core` + per-modality `device packs`; multimode is pack #1. Formalize the seam + a **readout-kind extension point** now (cheap, good regardless); validate against spinQICK's identical layering as a design oracle; **do not build a 2nd pack (spin/tweezer) until a real 2nd user exists.** |
| D16 | Config schema | **Pydantic** for the config layer (wiring, device, pack calibration) — validation + JSON round-trip for the §14.2 job blob, matching spinQICK. Hot-path IR (`Pulse`/`Delay`/`Seq`) stays plain dataclasses. |
| D17 | Optimal-control / Piccolo seam | **Drop `dev.optimal_control`** → address by operation (`dev.prepare(target)` / `dev.gate(op)`; realization analytic *or* OC). First-class **`from_solution(...)` import boundary** (artifact → normalized-ARB `Segment`, drive→line map, physical→DAC gain calib, resample, provenance) + **`Segment`/simultaneity** in the IR — **in the minimal core** (§6.5). **Catalog auto-retrieval deferred.** |
| D18 | Calibration-stack relationship | **qicklab is the open substrate at the bottom of the QILC stack** (Intonatissimo→Intonato→IntonatoQICK→qicklab→board); it stays **strategy-agnostic** (QILC method stays in Intonatissimo). **`expt_service` unifies as a thin adapter over qicklab** (direction locked; execution to circulate with Sho — §16). `from_solution` (§6.5) is the QILC-iterate-on-hardware seam; readout reducers emit Intonato's measurement vectors (§16). |

## 3. Package architecture (Approach A)

Dependency arrows point **down**; only `core/compiler`/`core/program`/`core/mock` import qick.

**The package is split into a modality-agnostic `core` and per-modality `device packs`** (§15). The
`core` knows nothing about superconducting/spin/atom physics; a *pack* supplies the device model,
pulse library, readout kind, calibration schema, and experiments for one platform. Multimode is
**pack #1**; the seam is validated against spinQICK's identical layering (§15).

```
qicklab/
  core/                       ─────────────── modality-AGNOSTIC (no physics knowledge) ───────────────
    channels.py     Wiring: line/generator → DAC binding (Pydantic)                              (pure)
    pulses.py       Pulse / Delay / Raw / Segment / Seq IR + fluent builder + from_solution import  (pure)
    sweeps.py       Sweep axis → QickSweep1D lowering                                            (pure)
    readout.py      ReadoutKind + Discriminator PROTOCOLS (IQ / PSB / counting impls live in packs)(pure)
    compiler.py     lower a Seq/Pulse onto a live AveragerProgramV2; quantize(); feedback         (qick)
    program.py      QickLabProgram(AveragerProgramV2): _initialize/_body, play/play_seq, acquire  (qick)
    result.py       MeasurementResult typed seam                                                  (pure)
    fitters.py      generic lmfit fit primitives (decaying sinusoid, exponential, …)              (pure)
    device.py       Device protocol + load(): wiring + calib + soccfg_snapshot                    (pure*)
    mock.py         MockQickSocV2 no-FPGA harness                                                 (qick)
  devices/
    multimode/                ─────────────── device PACK #1 (Schuster multimode) ───────────────
      model.py      device elements (Qubit/Manipulate/Storage/Coupler/Readout) → lines           (pure)
      calibration.py  Pydantic calib schema + swap-dataset accessors (get_pi/poly models)         (pure)
      library/      factory namespaces: qubit, sideband, coupler, manipulate, readout, composite  (pure)
      readout.py    DispersiveIQ readout kind + discriminator (implements core protocol)          (pure)
      experiments/  Rabi / T1 / T2 / parity / active_reset (compose + sweep + measure)            (pure)
  tests/
```

`*pure` = importable and unit-testable with **no qick, no board** — the bulk of the logic. qick is
confined to `core/compiler`, `core/program`, `core/mock`. **Config-layer types (`channels`,
`device`, pack `calibration`) are Pydantic models** — typed validation *and* clean JSON round-trip,
which the job-server config blob (§14.2) needs. The hot-path IR (`Pulse`/`Delay`/`Seq`) stays plain
dataclasses (runtime objects, not serialized config).

> **`soccfg` and purity.** `Device` holds the **JSON soccfg snapshot** (a plain dict — pure, no qick
> import), *not* a live `qick.QickConfig`. The **compiler/program** constructs the `QickConfig` from
> that snapshot (or receives a live one from the board), and `quantize()` (§6.1) takes the built
> `QickConfig`. So `device.py` stays qick-free; only the qick-facing modules touch `QickConfig`.

Each unit answers "what does it do / how do I use it / what does it depend on" in isolation. The
Pulse IR is the spine: experiments never touch qick; only the compiler does.

## 4. Tenet 0 — every abstraction has an escape hatch

`QickLabProgram` **subclasses** `AveragerProgramV2`; the device layer only *adds* helpers, never walls
qick off. Three escape levels, always available:

1. **Raw op mid-sequence** — `.raw(lambda p: p.delay_auto(0.01))` splices any qick call into a `Seq`
   (invoked with the live program `p` at that point).
2. **Raw method on the program** — inside a hand-written `_body`, `self.<any AveragerProgramV2 method>`.
3. **Bypass the builder entirely** — hand-write `_body` with raw qick while still calling library
   factories for the pulses you don't want to hand-roll.

> Design rule: no abstraction ships without one of these. "You must be able to hack one thing at the
> very bottom and revert to raw qick at any moment while still using the abstractions."

## 5. Tenet 1 — the device model (model the device, not the FPGA)

Two layers. **`dev` is the one `Device` object** loaded from the versioned config (mechanism is §14.1);
everything resolves against it.

**Layer 1 — what you write (device language, zero DACs):**

```python
dev.qubit.ge_pi(gain=amp)      # π on qubit g↔e
dev.qubit.ef_pi()              # π on e↔f
dev.sideband.f0g1(photon=0)    # |f,0⟩↔|g,1⟩ exchange (qubit ↔ manipulate cavity)
dev.coupler.swap("M1-S3")      # beamsplitter swap manipulate ↔ storage 3
dev.manipulate.displace(...)   # direct cavity drive
dev.readout.measure()
dev.parity(man=1)              # composite → a sub-sequence
dev.prepare(Fock(1))           # state prep → analytic OR optimal-control waveform (§6.5)
```

Five drive-namespaces — `qubit`, `sideband`, `coupler`, `manipulate`, `readout` — matching "things
driving qubit, manipulate, coupler," plus readout. `sideband` (f0g1 / multiphoton ladder) is its own
namespace even though it physically enters the qubit port, because it is a distinct generator.
*(Minor open point: `qubit.f0g1()` vs `sideband.f0g1()` grouping — kept as `sideband` by default.)*

**Layer 2 — the wiring (plain config data, `channels.py`, the ONLY channel map).**

Built from the v1 `hardware_config.yml → hw.soc.dacs/adcs` (this is the **v1 snapshot**; the target
v2 board's map is regenerated from its own config at implementation time — R2). A **line** owns one or
more **generators** (role → DAC); one line can have several DACs:

```
LINE        GENERATOR(role → DAC)                       notes
qubit       drive    → DAC 2 (nyq 1, full)              ge/ef rotations
            sideband → DAC 0 (nyq 1, full)              f0g1 / multiphoton ladder — a DIFFERENT DAC,
                                                          combined into the qubit port (same LINE)
coupler     low/high → DAC 1 (nyq 1, full)              beamsplitter swaps; band-gated in general,
                                                          but flux_low.ch == flux_high.ch == 1 HERE,
                                                          so both bands collapse to one generator
manipulate  main     → DAC 3 (nyq 2, full)              direct cavity drive
storage     main     → DAC 6 (nyq 2)                    direct storage drive (rarely used) — see gen_type note
readout     dac      → DAC 5 (full) + ADC 0             (a second ADC `cavity_out` = ch 1 exists)
```

- **There is no separate "sideband" *line*.** `sideband` is a **generator role under the qubit line**
  (DAC 0), distinct from the qubit `drive` generator (DAC 2) but feeding the same physical port. The
  Layer-1 `dev.sideband.*` namespace (§5, above) is an *API grouping*; its factories emit
  `LineRef("qubit", "sideband")`. "Different DAC" and "same line/port" are both true and consistent.
- **Coupler band-gate:** the v1 code picks flux_low vs flux_high by `freq < 1800`, but in this snapshot
  both resolve to DAC 1, so the choice is a no-op *here*. The schema still models the low/high split as
  two band-gated generators (a different board may separate them); when they collapse to one DAC, the
  compiler declares it once and both bands route to it.
- **Storage `gen_type` — a YAML-vs-firmware discrepancy to resolve.** `hardware_config.yml` records
  `storage_in: type: full, nyquist: 2`, but the v1 `MM_base` comment ([:275-279]) reports the actual
  firmware block is `axis_sg_mux4_v2` (which `declare_gen` rejects without `mixer_freq`/`mux_freqs`),
  and v1 therefore *excludes* storage from auto-declare. The schema supports `gen_type="mux4_v2"` as a
  capability; the correct value for the target v2 board must be read from its config (R4), not assumed.

**Wiring schema** (`channels.py`) — this is the seam an implementer copies, so it is typed, not prose:

```python
@dataclass(frozen=True)
class Generator:
    role: str                       # "drive" | "sideband" | "low" | "high" | "main" | ...
    dac_ch: int
    nyquist: int = 1
    gen_type: str = "full"          # "full" | "int4" | "mux4" | "mux4_v2"
    band_hz: tuple[float,float] | None = None   # freq gate; None = always eligible
    mixer_freq: float | None = None             # required for int4 / mux*
    mux_freqs: list[float] | None = None        # required for mux4 / mux4_v2 (e.g. storage)

@dataclass(frozen=True)
class Line:
    name: str                       # "qubit" | "coupler" | "manipulate" | "storage"  (NOT "sideband")
    generators: list[Generator]     # >1 when band-gated (coupler low/high) or multi-source
                                    #   (qubit line owns drive@DAC2 + sideband@DAC0)

@dataclass(frozen=True)
class Readout:
    dac_ch: int; adc_chs: list[int]
    mux_freqs: list[float] | None = None        # freq→lane map for multiplexed readout
    lengths_us: list[float] = ...; thresholds: list[float] = ...; trig_offset: int = ...

class LineRef(NamedTuple):           # what a Pulse carries
    line: str; role: str            # e.g. ("qubit","drive"), ("qubit","sideband"), ("coupler","low")

class Wiring:
    lines: dict[str, Line]; readout: Readout
    def resolve(self, ref: LineRef, freq_mhz: float) -> Generator:
        """Pick the generator for (line, role, freq): role match + band_hz gate.
        Raises if no eligible generator (e.g. a swept freq leaves the band — see §6.4)."""
```

The compiler's declare step reads `Generator` to emit `declare_gen(ch, nqz, mixer_freq=…,
mux_freqs=…)` — including the `mux4_v2` storage args, so the R4 quirk is handled by data, not a comment.

> **Config types are Pydantic.** `Generator`/`Line`/`Readout`/`Wiring` are shown as dataclasses for
> brevity but are implemented as Pydantic `BaseModel`s — typed validation **and** clean JSON round-trip
> for the §14.2 job-server config blob (matching spinQICK's Pydantic config, §15). `LineRef` stays a
> lightweight `NamedTuple`: it rides on the runtime `Pulse`, not the serialized config.

Consequences:
- **f0g1 = a second generator on the qubit line.** `QubitLine` has `{drive, sideband}`; `ge_pi` uses
  `drive`, `f0g1` uses `sideband` — same line, different generator.
- **The `freq<1800`/`freq<1000` magic thresholds die** — they become the coupler's
  generator-selection rule in the wiring, where "which DAC realizes this frequency" belongs.
- **`channel_table` and `channel_assign` are deleted**; all code reads the wiring.
- **Re-patch a cable / move a drive to a different DAC = edit one line in Layer 2.** Every experiment
  and pulse factory is untouched — they only speak Layer-1 device language.

A `Pulse` targets a `(line, generator-role)` — resolved to a DAC by the wiring at compile — never a
bare integer.

## 6. The pulse layer

### 6.1 Pulse IR (`pulses.py`) — typed, physical-unit, sweep-aware

```python
class Shape(Enum): GAUSSIAN; FLAT_TOP; CONST; ARB

@dataclass
class Pulse:
    line: LineRef                       # (namespace, generator-role) — resolved to DAC at compile
    freq_mhz:      float | Sweep
    gain:          float | Sweep        # see gain-units note below — NOT GHz-Rabi
    phase_deg:     float | Sweep = 0.0
    shape:         Shape = Shape.GAUSSIAN
    length_us:     float | Sweep | None = None   # flat portion (const / flat_top)
    ramp_sigma_us: float | None = None
    idata: np.ndarray | None = None     # ARB: NORMALIZED [-1,1] envelope (× gain at lowering)
    qdata: np.ndarray | None = None     #   — computed by a factory OR imported (§6.5), same field
    start_us: float | None = None       # absolute start; None ⇒ sequential/auto. Enables simultaneity.
    label: str | None = None

@dataclass
class Delay:  duration_us: float | Sweep
@dataclass
class Raw:    fn: Callable[["QickLabProgram"], None]   # escape hatch element
@dataclass
class Segment: pulses: list[Pulse]      # a set of pulses that start together (multi-line, same t)

Seq = fluent builder over [Pulse | Delay | Raw | Segment]
```

- **Simultaneity is first-class.** Real experiments (and every OC pulse, §6.5) drive **multiple lines
  at once**. A `Pulse.start_us` (absolute) or a `Segment` (pulses sharing a start) expresses this; the
  builder exposes `.play_parallel([...])`. The compiler lowers to v2's absolute/relative `t` on
  `pulse()`. Sequential `.play()` (the common case) leaves `start_us=None` and the compiler chains.

- **Physical units throughout.** All `freq2reg`/`us2cycles`/`deg2reg` conversion moves *out* of
  config-parsing and *into* the compiler — the single biggest v1 simplification.
- **Gain units — a real v1→v2 semantics change (do not skip).** v1 `set_pulse_registers`/`add_pulse`
  take an **integer DAC gain code** (0..32766); the whole calibration DB stores these ints
  (e.g. `pi_ge.gain = 3289`). QICK **v2** `add_pulse` takes a **normalized float** fraction of full
  scale (≈ −1..1), *not* the v1 int code. Decision: the calibration schema keeps storing what the lab
  measures (**int codes**, preserving continuity with the existing DB), and the **compiler performs
  the int→fraction conversion** (`gain_frac = gain_int / 32766`, verified against the target qick's
  `AbsPulse`/`maxv` convention) alongside `freq2reg`/`us2cycles`. `Pulse.gain` therefore carries the
  **int code** end-to-end and only becomes a v2 fraction at lowering. A unit test asserts a known int
  code maps to the intended fraction. *(Verify the exact v2 gain convention against the qick version
  Sho targets; no `../qick` fork is present in this environment.)*
  - **Two input regimes, one IR.** The **analytic library stays DAC-native** (int codes from the
    calibration DB). The **OC / Piccolo import path (§6.5) takes physical units** (GHz-Rabi / rad·s⁻¹,
    Piccolo's native output) and **calibrates them to DAC gain** via the π-pulse reference — the same
    physical→gain adapter as the `expt_service` contract. Both regimes produce the same `Pulse.gain`;
    physical-unit input is confined to the import boundary, not smeared across the library.
- **Quantization is queryable, IR stays pure.** `quantize(pulse, qickcfg) → Realized(freq_mhz_actual,
  length_us_actual, gain_frac, cycles, …)` gives exactly what the DAC will play (for phase/evolution-time
  math) without polluting the requested-value IR. It is a side-effect-free query that *takes* a built
  `QickConfig` (from the compiler layer, per the purity note in §3) — the IR itself never holds one.

### 6.2 Pulses are computed factory functions, not a data bank (`library/`)

Each pulse type is a typed function that *computes* its parameters from calibration — no string keys,
no `eval`. Factory namespaces are bound to `dev`.

**Uniform override surface (required for sweeps).** Every factory accepts keyword overrides for
*every* physical field — `freq`, `gain`, `phase`, `length` — where `None` ⇒ take the calibrated value
and a scalar-or-`Sweep` ⇒ override. This is what lets `dev.qubit.ge_pi(freq=f, gain=a)` (a chevron)
type-check: any field a factory produces can be swept.

```python
Num = float | Sweep | None      # None ⇒ from calibration; scalar/Sweep ⇒ override

# library/qubit.py    → dev.qubit.*
def ge_pi(dev, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse: ...  # gaussian, calib.qubit.pi_ge
def ge_hpi(dev, *, freq=None, gain=None, phase=0.0) -> Pulse: ...
def ef_pi(dev, *, freq=None, gain=None, phase=0.0) -> Pulse: ...

# library/sideband.py → dev.sideband.*   (identity args positional; physical overrides keyword-only)
def f0g1(dev, kind="pi", photon=0, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse: ...  # DEFAULT f0g1
def multiphoton(dev, transition, kind="pi", photon=0, *, freq=None, gain=None, phase=0.0) -> Pulse:  # ladder

# library/coupler.py  → dev.coupler.*
def swap(dev, mode="M1-S1", kind="pi", *, freq=None, gain=None, phase=0.0, length=None) -> Pulse: ...  # ds_storage
def floquet_swap(dev, mode, pi_frac=1.0, *, freq=None, gain=None, phase=0.0) -> Pulse: ...             # ds_floquet

# library/composite.py → dev.parity(...), dev.prepare(...), dev.gate(...), dev.storage_write(...)
def parity(dev, man=1, *, second_phase=180, fast=False) -> Seq: ...   # hpi · wait(revival) · hpi
def prepare(dev, target, *, via=None) -> Seq: ...   # state prep; target=Fock(1)/GKP("Z")/…
def gate(dev, op, *, via=None) -> Seq: ...          # gate; op="CZ" / Unitary(U)
```

- **Address OC pulses by operation, not method.** `dev.prepare(Fock(1))` / `dev.gate("CZ")` replace v1's
  `optimal_control("fock","1")`. The realization — an **analytic sequence** (multiphoton ladder) *or* an
  **optimal-control waveform** (§6.5) — is the device's choice; `via="oc"`/`via="analytic"` forces it.
  Addressing by *what it does* is what makes analytic and synthesized pulses interchangeable.
- **"default f0g1 vs `f0g1_multiphoton` are different functions"** → `f0g1()` is the common case,
  `multiphoton(transition=…)` the general one; each with its own explicit signature, no overloaded string.
- **Adding to the library = writing one function.** No registration ceremony to *call* one (plain
  typed functions, IDE-discoverable). Optional `@pulse_factory` decorator only feeds a catalogue for
  `list_pulses()` introspection.
- **Arb factories may compute samples** (from a synthesis routine) or **import a solved pulse from the
  Harmoniqs stack** (§6.5) rather than load a bare `.npy` — the IR just holds normalized `idata/qdata`.

### 6.3 Composition — the fluent builder (`Seq`)

```python
seq = (Seq()
   .play(dev.qubit.ge_pi(gain=amp))
   .wait(0.5)
   .play(dev.sideband.f0g1())
   .play(dev.coupler.swap("M1-S3", phase=180))
   .play(dev.parity(man=1))              # .play accepts a Pulse OR a sub-sequence
   .raw(lambda p: p.delay_auto(0.01))    # escape hatch
   .measure())
```

`Seq()` is a plain container (no `dev` — pulses are already device-resolved). Composing existing
library pulses into larger programs and adding pulses to the library are the two first-class daily
workflows. **Simultaneous multi-line play** uses `.play_parallel([...])` (a `Segment`) or explicit
`start_us` — e.g. `seq.play_parallel(dev.prepare(Fock(1)))` drives qubit + manipulate together (§6.5).

### 6.4 Sweeps and generator selection (edge cases the compiler must handle)

- **A frequency `Sweep` that straddles a generator band boundary is rejected.** A v2 generator is bound
  to one DAC for the whole program, so a coupler-swap frequency `Sweep` whose `[start, stop]` spans the
  low/high gate cannot switch DACs mid-loop. `Wiring.resolve` is evaluated over the sweep's full range;
  if `[start, stop]` maps to more than one generator, the compiler raises a clear diagnostic ("sweep
  spans coupler low/high boundary at f=…; split into per-band programs"). (In the v1 snapshot both bands
  are DAC 1, so this never triggers there — but a separated board would.)
- **Field-width / quantization limits are caught, not silently wrapped.** A swept `length` (or const
  length) exceeding v2's timing field width, or a step below DDS/cycle quantization, surfaces as a v2
  validator error via the mock (§10) — the same line a real board would fail on. The compiler does not
  clamp or wrap silently. (v1's long-pulse chunking — `register_long_pulse`/`play_long_pulse*` — is a
  known pattern to port here if long constant drives are needed; out of the minimal core.)

### 6.5 Optimal-control / Piccolo import boundary (D17) — the Piccolo × QICK seam

Optimal-control pulses come from **your stack** (Piccolo/Piccolissimo → `extract_pulse`; the
armonissima pulse catalog). This is a headline seam, not a corner case — so the import is a **typed,
lossless conversion**, replacing v1's `np.load(.npz)` + hand-`interp1d` + `maxv`-scale in `custom_pulse`:

```python
# core/pulses.py — modality-agnostic (a spin/atom pack imports its own solutions the same way)
def from_solution(
    solution,              # Harmoniqs pulse artifact: extract_pulse output / NamedTrajectory / catalog entry
    *, drive_map,          # named solution drive → qicklab line ('qubit'→dev.qubit, 'cavity'→dev.manipulate)
    gain_calib,            # physical amplitude (GHz-Rabi / rad·s⁻¹) → DAC gain, via π-pulse reference
    dac_rate,              # resample the control grid → DAC sample grid
) -> Seq: ...              # → Segment of normalized-ARB Pulses (one per driven line, same start) + provenance
```

What this nails that the `.npz` loader fudged:

1. **Drive→line mapping is explicit**, resolved through the wiring (§5) — not the hardcoded
   `qubit_ch`/`man_ch` the v1 opt_cont path assumed.
2. **Physical units → DAC gain via calibration** (§6.1) — Piccolo controls are physical; this is *the*
   place physical-unit input belongs.
3. **Resampling** onto the DAC grid is owned here (typed `dac_rate`), not a buried `interp1d`.
4. **Multi-line simultaneity** → a `Segment` (§6.1), played with `.play_parallel(...)`.
5. **Provenance travels with the pulse** — source solve, target, fidelity, sample rate, units — feeding
   config versioning (§14.1) and reproducibility.

`dev.prepare(target)` / `dev.gate(op)` (§6.2) are the operation-addressed front doors; when they resolve
to an OC realization they call `from_solution(...)` under the hood. **The same path is the closed-loop
calibration hardware seam** (§16.1): an Intonato QILC iterate → PythonCall → `from_solution → acquire →
reduce → Vector{Measurement}`.

**In the minimal core (D17):** `from_solution(...)` importing an artifact you hand it, + `prepare`/`gate`
naming, + `Segment`/simultaneity. **Deferred:** **catalog auto-retrieval** — `dev.prepare(Fock(1))`
looking the solved pulse up in the armonissima catalog by *(device system, target)* and importing it
automatically. That closes the full Piccolo×QICK loop but belongs partly in the catalog layer; land it
once the import boundary is proven.

## 7. Config: single source of truth (versioning mechanism is §14.1)

### 7.1 Single-source-of-truth violations today (concrete)

| Same fact, multiple places | Evidence |
|---|---|
| f0g1 / multiphoton ladder defined twice | `hardware_config.yml → device.multiphoton` **and** `configs/multiphoton_config.yml` (loaded but its reader is commented out — dead duplicate) |
| `man()` ≡ `multiphoton('fn-gn+1')` | both read `device.multiphoton.pi['fn-gn+1']`; comments admit it |
| Channel map in code *and* config | `hw.soc.dacs.*` shadowed by `channel_table` + `channel_assign`, which contradict |
| Qubit ge freq in 3+ places | `device.qubit.f_ge`, `device.multiphoton.pi['gn-en'].frequency[0]`, `self.f_ge`/`self.f_ge_reg` |
| `flux_low.ch == flux_high.ch == 1` | low/high split is calibration-only; wiring collapses them — proves the DAC map must be *data* |

### 7.2 The single-source design (`device.py` + `calibration.py` + `channels.py`)

One typed **`Device`**, loaded once, with:
- **`wiring` (Layer 2)** — the only channel map (§5).
- **`calib` (Layer 1)** — one *uniform* schema for "a calibrated operation," organized by family
  (qubit rotations, sideband ladder, storage swaps, manipulate, optimal-control, readout,
  active-reset). Each physical quantity appears **once**.
- **Rule: nothing derived is stored.** Register values computed at compile from soccfg; shared
  physical values (ge freq) *referenced*, never copied.

Result — **adding functionality is one edit each**: new pulse type → one calibration row (uniform
schema) + one factory; no `channel_table` edit (wiring is data), no `custom_pulse` branch (compiler
lowers shapes generically), no second config file. `multiphoton_config.yml` **deleted**; `man()`
**deleted**; the 169 KB `experiment_config.yml` keyed-by-classname blob becomes per-experiment typed
defaults co-located with each experiment.

### 7.3 Versioning — OPEN design decision (see §14.1)

Single-source-of-truth (§7.2) is settled; the *versioning mechanism* is **not**. The earlier lean
toward git-native reopens once the operational reality is admitted (single shared checkout, one
long-lived worker reading configs off disk, concurrent submitters — §14.2). `Device.load(...)` accepts
a version handle regardless of which mechanism wins; §14.1 lays out the tension for circulation.

## 8. `QickLabProgram` interface

```python
class QickLabProgram(AveragerProgramV2):          # IS a qick v2 program
    def __init__(self, device, seq=None, sweeps=(), reps=..., rounds=..., final_delay=...): ...

    def _initialize(self, cfg):
        self._auto_declare()                   # declare gens/readout + add envelopes referenced by
        for s in self.sweeps:                  #   seq/sweeps from wiring; manual declare still allowed
            self.add_loop(s.name, s.count)

    def _body(self, cfg):
        self.play_seq(self.seq)                # DEFAULT: run the declarative Seq

    # device-aware helpers
    def play(self, pulse_or_seq): ...          # lower onto self (add_pulse + pulse / delay / raw)
    def play_seq(self, seq): ...
    def measure(self, ...): ...
    def active_reset(self, ...): ...           # ⚠ wraps v2 feedback — see §11 risk
    def reset_and_sync(self): ...
    # + every AveragerProgramV2 method (escape hatch)
```

**Two `_body` modes:** declarative (hand it a `Seq`, never write `_body`) or imperative (subclass,
write `_body`, mix device helpers + raw qick).

**Sweeps as first-class values.** A `Sweep` is a named axis; any pulse/delay parameter may be a scalar
*or* a `Sweep`. The compiler lowers a `Sweep` to v2's `QickSweep1D` and auto-registers `add_loop`. The
entire v1 `RAverager`/`NDAverager` register-update machinery disappears.

```python
# Amplitude Rabi (1D over gain)
amp = Sweep("amp", 0, 30000, 51)
rabi = QickLabProgram(dev, reps=1000, sweeps=[amp],
    seq=Seq().play(dev.qubit.ge_pi(gain=amp)).measure())
res = rabi.acquire(soc)                           # -> MeasurementResult (see below)
res.avgi.shape                                    # (n_ro, 51); res.xpts == amp axis

# T1 (1D over an idle delay)
t = Sweep("t", 0, 500, 101)                      # µs
t1 = QickLabProgram(dev, reps=1000, sweeps=[t],
    seq=Seq().play(dev.qubit.ge_pi()).wait(t).measure())

# Chevron (2D nested; list order = loop nesting)
f = Sweep("freq", 3550, 3575, 51); a = Sweep("amp", 0, 30000, 51)
res = QickLabProgram(dev, sweeps=[f, a],
    seq=Seq().play(dev.qubit.ge_pi(freq=f, gain=a)).measure()).acquire(soc)
res.avgi.shape                                    # (n_ro, 51, 51)
```

**`QickLabProgram.acquire(soc, ...)` returns a `MeasurementResult`** (§9), *not* v1's `(xpts, avgi, avgq)`
tuple and not v2's raw native structure. It wraps `AveragerProgramV2.acquire`'s native return (which
differs from v1) into the typed seam, attaching the sweep axes as `xpts`. A `.as_tuple()` convenience
returns `(xpts, avgi, avgq)` for anyone porting v1 analysis code incrementally.

**Nothing v2 is hidden** — multiplexed/decimated/DDR4 readout (`declare_readout`/`trigger(ros=…)`),
`delay`/`delay_auto`, per-pulse `phase`/`phrst`, `QickParam` arithmetic, feedback control-flow: all on
`self`, with device helpers only for the common path.

## 9. Data lifecycle — run / analyze / fit / display (D13, D4)

### 9.1 Keep the high-level Experiment contract, split the internals

The slab `Experiment` base already exposes the right seams — `go(save, analyze, display)` →
`acquire()` / `analyze()` / `display()` / `save_data()`. That contract is **respected and kept**. The
v1 problem is in the *implementation*: each concrete `analyze()` **inlines** `fitter.fitdecaysin(...)`
*and prints* derived quantities, and `display()` **inlines** matplotlib — run, fit, analyze, and plot
are conflated on one class. `qicklab` keeps the Experiment as the user-facing orchestrator but
delegates to distinct, independently-testable collaborators:

```
Experiment (thin orchestrator; keeps go/acquire/analyze/display/save)
  ├─ Runner    — builds QickLabProgram from device+params, acquires → MeasurementResult   (no fitting, no plotting)
  ├─ Analyzer  — MeasurementResult → typed FitResult via lmfit fitters                 (no plotting, no I/O)
  └─ Display   — FitResult + MeasurementResult → figure                                (no fitting, no I/O)
```

Each collaborator answers "what does it do / how used / what depends on" alone: the Runner never
fits, the Analyzer never plots, the Display never re-fits. Derived-quantity printing (v1's inline
`print(f'Pi gain …')`) becomes typed fields on `FitResult`, not stdout side-effects.

### 9.2 `result.py` — `MeasurementResult` (the seam D4 protects)

Typed carrier of `xpts` (sweep axes), per-readout `avgi/avgq` (loop/channel axes), **raw per-shot data
+ herald-lane indices** (for post-selection), and metadata (device ref, sweep axes, timestamp). Exposes
`.as_tuple()` (v1-compat `(xpts, avgi, avgq)`) and `.postselect(...)` (ANDs enabled heralds per R1(ii)).
It does **not** inherit the 8.3 kloc fitting mess.

### 9.3 `fitters.py` — clean `lmfit` fitters + the current code as a physics oracle (D4)

- **Built on `lmfit`** (already a dependency — `slab/dsfit.py`, `experiments/qsim/utils.py`). Each core
  fit is an `lmfit.Model` with named `Parameters`, bounds, and returned uncertainties — replacing the
  v1 `def xfunc(x,*p)` + `def fitx(...)` scipy pattern and the derived-quantity printing.
- **Core set only:** **Rabi** (decaying sinusoid → π/π-2 gain), **T1** (exponential), **T2 Ramsey**
  (decaying sinusoid ± detuning), **parity** (contrast/phase). Take a `MeasurementResult`, return a
  typed `FitResult` (params + stderr + derived quantities).
- **Oracle validation.** The new fitters are cross-checked against the **current `fitting.py`**
  (`fitdecaysin`, `fitexp`, …) on the *same* data: a test asserts the new lmfit params agree with the
  legacy fit within tolerance. The trusted old code confirms the new code before the old code is retired.
- The full `fit_display*` / `fit_display_classes` / `wigner` / `noise_psd` refactor (≈8.3 kloc) is an
  explicit **follow-on spec**, not this one.

## 10. Testing & acceptance

- **`mock.py` — `MockQickSocV2`.** Reimplement the proven strategy: real qick runs build→compile→acquire;
  only leaf `soc.*` calls are stubbed (correctly-shaped zeros, right dtypes). v2's param validators fire
  on a buggy program with no FPGA. (v2's acquire path calls a slightly different soc surface than v1 —
  the mock covers v2's.)
- **Committed soccfg snapshot** for off-board runs (regenerate the v2 shape once from the target board).
- **Three acceptance gates.** Structural alone is too weak for a reimplementation — a shapes-only mock
  would pass the int-vs-float gain bug (§6.1) unnoticed:
  1. **Structural gate (mock).** Program builds + compiles on `MockQickSocV2`; readout shapes match the
     declared sweep/readout axes; v2 validators fire on malformed programs. (Per `expt_service` §5.4.)
  2. **Numeric golden-equivalence gate (offline correctness).** For each core operation, compile the
     v1 program and the `qicklab` program and assert the **realized physical values match**: register
     frequencies, **DAC gain codes / v2 gain fractions**, cycle counts, and envelope samples — via
     `quantize()` / the emitted ASM. The cheap middle gate between "it builds" and "it works on the
     board," and exactly what catches the gain-units / unit-conversion class of bugs.
  3. **Physics-agreement gate (on-board, the real acceptance).** On the actual board (with the
     collaboration): run **T1 / T2-Ramsey / amplitude-Rabi**, fit with the new `lmfit` fitters, and
     require (i) **agreement with physics** (sensible τ₁, τ₂, π-gain; Ramsey detuning matches the set
     detuning) and (ii) **agreement with the current stack** — the same raw data fit by the legacy
     `fitting.py` oracle (§9.3) yields matching parameters. This is the criterion that says the migration
     actually works, not just that it compiles.
  Gates 1–2 are mock/offline (no FPGA); gate 3 needs the board and is run with the QICK collaboration.
- **Unit tests** hit the pure layers directly (no qick): channel/wiring resolution (incl. band-gate
  rejection §6.4), factory→Pulse correctness against calibration, Seq composition, sweep→loop lowering
  (assert shapes), quantize() round-trips, int→fraction gain conversion, fitters against synthetic data.

## 11. Minimal core deliverable & sequencing (tracer bullets)

Each slice is end-to-end (device → factory → Seq → compile → mock-acquire → result → fit) and
independently demoable.

1. **Skeleton + wiring + one pulse, one readout.** Establishes the `core` / `devices/multimode` split
   (§3, §15): `Device.load`, `core.channels`, a `Pulse`, compiler lowers `ge_pi` + `measure` onto
   `QickLabProgram`, runs on `MockQickSocV2`, returns a `MeasurementResult`. *Proves the whole vertical.*
2. **Amplitude Rabi** — `Sweep`-as-value + `add_loop` + Rabi fitter. *Proves sweeps + fitting.*
3. **T1 + T2 Ramsey** — swept delay, phase, `wait`. *Proves timing/phase.*
4. **f0g1 + storage swap** — `sideband`/`coupler` namespaces, freq-gated wiring, `ds_storage`. *Proves
   multi-line device model.*
5. **Parity** — composite factory (`dev.parity`), revival-time calibration.
6. **Optimal-control import (the Piccolo × QICK seam, §6.5).** `from_solution(...)` on a Piccolo /
   `extract_pulse` artifact → a `Segment` of normalized-ARB pulses across qubit + manipulate,
   `.play_parallel`, physical→DAC gain calibration, provenance; `dev.prepare(Fock(1))` as the front door.
   *Proves simultaneity + the physical-unit import path.* Validated by numeric-golden vs a known solution
   (§10 gate 2).
7. **Active reset** — split per R1: **(7a)** herald / post-selection data path in the result layer
   (no v2 risk) + host-side post-selection fallback; **(7b)** in-sequence v2 feedback (⚠ gated on Sho).
   `active_reset` is in the D2 core *surface*; sequencing it last is a schedule choice, not a scope cut —
   7a lands regardless, 7b lands when the v2 feedback API is confirmed.

Slices 1–6 have no v2-feasibility risk (pulses, delays, sweeps, readout, arb import are well-defined v2 API).
Only slice **7b** carries real uncertainty; 7a and everything before it do not.

> **Out of the minimal core (deferred, but modeled):** **catalog auto-retrieval** for `dev.prepare/gate`
> (§6.5), `dev.coupler.floquet_swap`, and `register_long_pulse`-style long constant drives appear in the
> design so it accommodates them, but are **not** in the tracer bullets above — they follow once the core
> is green. *(The OC import boundary itself IS in-core, slice 6; only the catalog auto-lookup is deferred.)*

## 12. Risks & open questions

- **⚠ R1 — active reset on tProc v2 (the one real risk).** This has **two separable parts**; only the
  first carries feasibility risk:
  - **(i) In-sequence measurement-conditional feedback.** v1 uses low-level ASM: `safe_regwi` /
    `read(0,0,"lower",reg)` / `condj(page,reg,"<",thresh,label)` / `label(...)` with hand-assigned
    register pages. tProc v2 expresses this through a **different control-flow model**; how much the
    target `qick.asm_v2` exposes (and its exact API) must be confirmed with **Sho Uemura**. Quarantined
    to one `active_reset` helper + `compiler.py`.
  - **(ii) Herald / post-selection data path.** v1 `active_reset` also emits up to **two independent
    post-selection herald rounds** (bare-|g⟩ and parity-mapped), each its own readout lane, and does
    **per-shot filtering** into `avgi/avgq` gated on `_pre_selection_filtering`. This is *not* a feedback
    problem — it is a **result-layer** concern: `MeasurementResult` must carry raw per-shot data + the
    herald lane indices, and expose a `postselect(...)` that ANDs the enabled heralds. This part has no
    v2 risk and can land independent of (i).
  - **Fallback if v2 feedback is unavailable/insufficient:** **host-side post-selection** — run without
    in-sequence reset, herald in software from raw shots — so slice 6 degrades gracefully instead of
    being all-or-nothing. (Reset *quality* differs, but the experiment surface stays usable.)
  Slices 1–5 do not depend on any of this.
- **R2 — exact v2 API names.** `AveragerProgramV2`, `_initialize`/`_body`, `declare_gen`, `add_gauss`,
  `add_pulse`, `add_loop`, `QickSweep1D`, `pulse`, `delay`/`delay_auto`, `trigger`/`readout` are used
  from recollection; **verify against the `../qick` fork / version the collaboration is targeting** before
  implementation. The `storage_in` = `mux4_v2` generator quirk (rejects `declare_gen` without mixer/mux)
  must be handled explicitly in the wiring/compiler.
- **R3 — multi-qubit generality.** The core is single-qubit-centric (`qTest`). The device model
  (`Qubit` element, per-line wiring) generalizes, but the first slices assume one qubit. Multi-qubit is
  out of scope for the minimal core; the model should not preclude it.
- **R4 — storage line vs coupler-addressed storage.** Storage *swaps* go through the coupler flux line;
  a separate `storage_in` DAC (mux4_v2) exists for direct storage drive (rarely used). Model both; the
  core exercises the coupler path.
- **R5 — optimal-control provenance.** `.npy`/`.npz` waveforms currently reference Windows share paths
  (`H:\Shared drives\...` in `device.optimal_control[...]['filename']`). The arb-factory should accept a
  configurable waveform source (or compute), not hardcode paths.
- **R6 — envelope naming / dedup (carry the v1 lesson).** v1 `add_gauss` is **not idempotent** — it
  re-allocates an envelope on every call, so per-gate naming blows up the waveform buffer with sequence
  length (v1 added an opt-in `dedupe_waveforms` flag as a patch). The v2 compiler's `_auto_declare` must
  add each distinct envelope (keyed by `(generator, shape, sigma, length)`) **once** and reference it by
  name from multiple pulses — dedup by construction, not as an afterthought.

## 13. Appendix — pulse-type catalogue (what backs each, from `prepulse_creator2` + configs)

| v1 verb | Physical op | Data source | Shape | Line (Layer 1) | new home |
|---|---|---|---|---|---|
| `qubit('ge'/'ef','pi'/'hpi')` | qubit rotation | `calib.qubit.{pi,hpi}_{ge,ef}` | gaussian | qubit | `dev.qubit.*` |
| `qubit('ge_broadband')` | broadband ge | own row | gaussian | qubit | `dev.qubit.ge_broadband` |
| `qubit('ge','parity_Mx')` | parity wait | `manipulate.revival_time[x]` | delay | qubit | folded into `dev.parity` |
| `multiphoton(gn-en/en-fn/fn-gn+1,kind,n)` | multiphoton ladder | `calib.multiphoton[kind][trans]` by photon n | gauss/flat_top | qubit or sideband | `dev.sideband.multiphoton` |
| `man('M1','pi'/'hpi')` | f0g1 (legacy, M1) | `multiphoton.pi['fn-gn+1']` | flat_top | sideband | **deleted** → `dev.sideband.f0g1` |
| `storage('M1-Sx','pi'/'hpi')` | beamsplitter swap | `ds_storage` (+ poly freq↔gain) | flat_top | coupler (freq-gated) | `dev.coupler.swap` |
| `floquet('M1-Sx',pi_frac)` | Floquet swap | `ds_floquet` (`get_pulse_envelope`) | flat_top/gauss | coupler | `dev.coupler.floquet_swap` |
| `optimal_control(enc,state)` | arb IQ (2-ch) | `.npz` → **`from_solution` import** (Piccolo/catalog, §6.5) | arb `Segment` | qubit + manipulate | `dev.prepare(...)` / `dev.gate(...)` |
| `buffer(t)` / `wait(t)` | idle gap | — | const/delay | — | `Seq.wait` / `Delay` |

## 14. Open design decisions (to circulate before locking)

These two are deliberately **not decided** in this spec. Aaron to circulate with the team; the answers
shape the plan but not the core module design (§3–§9), which stands either way.

### 14.1 Config versioning mechanism (D9)

Single-source-of-truth (§7.2) is settled. The *versioning mechanism* is the open question, because the
operational reality (§14.2) — one shared checkout, a long-lived worker reading configs off disk,
concurrent submitters — makes the naive choice race-prone.

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. Append-only immutable + pointer** (the *current* model) | Content-addressed snapshots (`CFG-…` ids, SHA256 dedup, write-once files) + one movable `main` pointer; writes serialized (SQLite WAL). | **Conflict-free by construction**; a new version can never disturb an in-progress read; reproducibility is permanent (an id resolves forever to byte-identical files); no textual merge of numeric calibration data. | Bespoke store (SQLite + `configs/versions/`); "not just git"; currently gitignored/per-machine. |
| **B. Git-native** | Config dir *is* a git repo; branches/tags/merges are git's. | Zero bespoke code; familiar; portable history. | **Reintroduces the races** the current model avoids: line-merge of YAML/CSV can silently corrupt structure; `checkout`/`merge`/`reset` rewrite files **in place, non-atomically, under the live worker's open reads**; one shared working tree defeats per-user branches; rebase/gc/force-push make old runs unreproducible. |
| **C. Hybrid (immutable-in-git)** | Append-only immutable snapshots, but **stored/synced via git that only ever ADDS files** — never in-place edits, never content merges; the pointer is a single serialized writer. | Keeps "configs handled by git" (backup, sync, portability) **while preserving conflict-freedom** (immutability, not merge semantics, is what makes it safe). | Needs discipline/tooling to guarantee git stays additive; two systems (git + pointer). |

**Recommendation:** A or C — **immutability is the load-bearing property**; git, if used, must be
additive-only. Pure B is not safe under the current shared-worker/multi-user topology. *(The verified
race analysis is in the session's job-server exploration; happy to attach it to the circulation.)*

**Bearing of later decisions (added after §15 / D16):**
- **This is now a framework-wide decision, not a multimode one.** The versioning mechanism lives in
  `core` and versions whatever config a *pack* defines (§15); a spin/atom lab using qicklab wants the
  same mechanism. Choose it for the framework, not just for Stanford.
- **Pydantic (D16) strengthens A and C.** Canonical, typed serialization yields deterministic content
  hashes, so content-addressed dedup and diffs are robust across machines.
- **New tension the generality goal surfaces.** Option A *as literally reusing the existing store*
  hard-couples config versioning to Stanford's **`job_server` SQLite + `ConfigVersionManager`** —
  lab-specific infra a general framework shouldn't require. So even if immutability + pointer wins,
  `core` should expose versioning behind a small **store interface** (default: a standalone
  file-snapshot + pointer implementation with **no queue/DB dependency**; Stanford's
  `ConfigVersionManager` becomes *one* backend). This keeps A/C viable without importing the job server.

### 14.2 Job-server / multi-user integration (D14)

**Current model (verified):** FastAPI HTTP server + SQLite queue + a **single polling worker that is
the sole hardware owner** (PID-locked) + Pyro4 to the RFSoC. A user process **serializes the full
station config to JSON** and submits `experiment_class` / `experiment_module` **strings** (+ optional
`program_class`); the worker **dynamically imports** the class, applies the shipped config blob (rebuilding
`ds_storage`/`ds_floquet` from CSV records), and runs `expt.go()` → `acquire(soc)` where `soc` is a
**Pyro4 proxy**. Result = HDF5 + a pickled expt object; the user postprocesses in their own process.
Exclusivity is structural (one worker, serial FIFO+priority queue, SQLite WAL) — there is no per-user
device lock.

**Hard constraints this places on `qicklab` regardless of the decision** (bake into slice 1 so the
compatible-with path stays open):
- Experiments must be **worker-importable by `module:class` path**.
- Experiments must be **constructable from the serialized config blob** — no reliance on live objects
  the worker doesn't have (the worker rebuilds datasets from CSV records).
- `QickLabProgram` must run under a **Pyro4 `soc` proxy** in the worker (not just an in-process soc/mock).
- Results must be **serializable** (HDF5 + picklable `MeasurementResult`).

| Option | How | Trade-off |
|---|---|---|
| **A. Be compatible, don't reimplement** | qicklab experiments satisfy the contract above; the existing queue/worker is untouched. | Smallest; keeps multi-user working immediately; no queue rewrite. Inherits the current runner/serialization ergonomics. |
| **B. Clean re-take on the runner** | qicklab ships a tidy submit/runner layer (re-doing `CharacterizationRunner`/`SweepRunner`) that talks to the *existing* queue. | Better ergonomics/serialization; larger surface; must stay wire-compatible with the worker. |
| **C. Out of scope for now** | Direct in-process `acquire` + mock only; queue integration later. | Fastest to first light; but defers the multi-user reality the lab runs on daily. |

**Recommendation:** **A** for the minimal core (keeps the lab's multi-user workflow working), with **B**
as a natural follow-on once the core is proven. **Aaron to circulate** (this is a shared-infra decision
touching everyone who submits jobs).

## 15. Generality / horizontal scope (D15, D16)

**Ambition:** `qicklab` as a general QICK tProc-v2 framework for experimental labs — usable by
practitioners across modalities (superconducting, spin qubits, neutral-atom, sensing), not just the
Schuster multimode device. This section records how far that reaches and the disciplined path to it.

### 15.1 Evidence the architecture already generalizes

**spinQICK** (HRL Laboratories, shipped 2025-07) is an open QICK framework for *electrostatic spin
qubits* that **independently converged on the same layering** `qicklab` designs — and it targets tProc
**V2** too. Module-for-module:

| spinQICK (spins) | qicklab (multimode) | qicklab layer |
|---|---|---|
| `qick_code_v2/` (tProc-V2 API + asm) | `core/compiler`, `core/program` | **core** (agnostic) |
| `Core/dot_experiment` | `Experiment` orchestrator (§9.1) | **core** |
| `helper_function/analysis`, `plot_tools` | `core/fitters` + Display (§9) | **core** |
| `Models/` (**Pydantic** config) | `core/channels`, `core/device` + pack calib | **core schema + pack contents** |
| `Core/awg_pulses, eo_pulses, ld_pulses` | pack `library/` | **device pack** |
| `Core/readout_v2, psb_setup` (PSB) | pack `readout` (DispersiveIQ) | **pack (readout kind differs)** |
| `Experiments/` (charge-stability, exchange) | pack `experiments/` (parity, dual-rail) | **device pack** |

The ecosystem already builds per-modality frameworks on the shared QICK substrate: **spinQICK** (spins),
**QICK-DAWG** (NV/defect sensing), **Qibosoq** (circuits). That is both the market signal and the
competitive reality.

### 15.2 What generalizes for free vs. what a new modality costs

- **Generic by construction (`core`, ~70–80%):** Pulse IR, compiler, wiring, sweeps, fluent builder,
  program, `MeasurementResult`, lmfit fit primitives, mock, escape hatch. A spin or atom experiment
  compiles pulses onto DACs, sweeps parameters, acquires, and fits T1/T2/Ramsey the same way.
- **Per-pack (modality-specific):** device model, pulse library, calibration schema, experiments, and
  — the one genuinely new engineering surface — the **readout kind**. Dispersive IQ (superconducting)
  vs PSB/charge-sensing (spin) vs fluorescence/photon counting (atoms/sensing) are structurally
  different. `core/readout.py` therefore defines `ReadoutKind` + `Discriminator` **protocols**; each
  pack implements them (multimode = `DispersiveIQ`). This is where generality costs real design work;
  everything else is packaging. The reducers must also emit the measurement vectors the QILC loop
  consumes (populations / wigner / tomography, keyed by knot index — §16.2).

### 15.3 Stance — design-for-extension, implement-one (D15)

- **Factor the `core` / `devices/*` seam now** (§3). It is good design regardless — the same seam that
  makes multimode clean — so it costs little and is not speculative.
- **Use spinQICK as a design oracle.** Sanity-check on paper: can `core` express `ld_pulses`/`eo_pulses`
  + PSB readout + a charge-stability sweep without contortion? If yes, the seam holds; if not, that is
  where `core` is secretly multimode-specific — fix it there.
- **Do not build a second pack until a real second user exists.** Prove generality with ≥2 real packs,
  not speculation (rule of three: one real modality in hand risks baking its assumptions into `core`).
  This keeps the Stanford deliverable on schedule while making the horizontal play a cheap later option.

### 15.4 Caveats

- **Incumbents + ownership.** spinQICK/QICK-DAWG/Qibosoq have users; "better core, they'll switch" is
  not a plan. A general framework is a *product* (docs, support, governance) — a Harmoniqs
  horizontal-bet decision, not a code decision.
- **Tweezers are the weakest fit.** No established QICK tweezer framework was found, and optical-tweezer
  control is a *different paradigm* — continuous multi-tone AOD RF + real-time rearrangement/feedback,
  not "compile a pulse program → acquire → fit." QICK's multi-tone DACs suit AOD combs, but the
  experiment lifecycle here fits spins far better than tweezers. Treat tweezer as a speculative, later
  validation target — not a near-term pack.

## 16. Relationship to the Intonato/Intonatissimo calibration stack (D18)

qicklab is the **open Python measurement/execution substrate at the bottom of the QILC calibration
stack.** It is *driven by* the calibration layers; it does not contain the calibration method.

```
Intonatissimo   PRIVATE — QILC method / ILCStrategy
  Intonato       PUBLIC — strategy-generic QILC chassis; solve!(PulseTuningProblem):
                          extract_pulse → run_experiment(::AbstractExperiment,::AbstractPulse)→Vector{Measurement}
    IntonatoQICK.jl  PUBLIC — AbstractHardwareBackend: upload_pulse! / trigger! / readout (PythonCall)
       qicklab   OPEN — this spec: from_solution → compile → acquire → reduce
          board
```

### 16.1 Three alignments (the design already fits)

1. **`expt_service` ⊂ qicklab — unify (D18).** expt_service's 3-verb contract — `upload_pulse`
   (GHz-Rabi IQ → gain via `PiPulseReference`) / `trigger` (assemble `AveragerProgramV2` + acquire) /
   `readout` (reduce to a measurement vector; kinds `iq`/`wigner`/`tomography_1q`) — is a strict subset
   of qicklab (`from_solution` §6.5 → compiler → `acquire` → readout-reduce). `gain.py`'s
   `PiPulseReference` **is** qicklab's `gain_calib`. **Direction locked: the 3-verb contract becomes a
   thin adapter/server over qicklab** (one substrate). *Execution to circulate with Sho* — it touches
   the live IntonatoQICK collaboration.
2. **`from_solution` (§6.5) = the QILC-iterate-on-hardware seam.** Each `solve!` iteration extracts an
   `AbstractPulse`/`NamedTrajectory` and calls `run_experiment(pulse)`. On hardware that is
   Intonato's `HardwareExperiment(pulse -> run)` → PythonCall → qicklab
   `from_solution → compile → acquire → reduce → Vector{Measurement}`. §6.5 is not just a Piccolo
   convenience — it is the closed-loop calibration hardware path.
3. **π-reference closure.** qicklab's *experimental* calibration (amplitude Rabi → π-gain, slice 2)
   **produces** the `PiPulseReference` (`pi_rabi_ghz`, `pi_gain`) that the physical→DAC gain conversion
   (§6.1) depends on. The framework calibrates the very reference its optimal-control import needs.

### 16.2 What the QILC loop requires of qicklab

Intonato's `MeasurementModel` consumes measurement vectors keyed by knot index (`Measurement.data`,
`.index`) — **populations, wigner, displaced_parity, tomography** (its `measurement_functions/`), which
are exactly expt_service's readout kinds. So qicklab's **readout-kind seam (§15.2)** must emit those
reduced vectors from discriminated IQ, indexed by knot — its `ReadoutKind`/`Discriminator` protocols
should cover `populations`/`wigner`/`tomography` one-for-one with Intonato's measurement functions
(not just raw IQ).

### 16.3 The boundary to hold (public/private)

qicklab stays **strategy-agnostic**: it exposes *import-pulse / acquire / discriminate / reduce*; the
**QILC method stays in Intonatissimo**. Same open-core split as IntonatoQICK (public interface) vs
ILCStrategy (private method) — qicklab is the open layer the private strategy ultimately runs on. No
ILC/QILC logic ever lands in qicklab. (This also keeps qicklab honestly general for non-Harmoniqs
practitioners, §15 — a spinQICK-style user gets the substrate without the calibration method.)

---

*End of spec. §14 items are open pending circulation; §16's expt_service **execution** circulates with
Sho (direction locked). §15 records the generality stance. Next steps per the brainstorming workflow:
spec review → user review → `writing-plans`.*
