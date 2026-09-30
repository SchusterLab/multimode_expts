---
type: review
date: 2026-07-18
reviews: spec-20260716-183446-qicklab-tproc-v2-first-principles.md
status: draft
tags: [review, stress-test, qicklab, tproc-v2, refactor-plan, program-plan, migration-staging, frame-tracking, pulse-ir]
---

# Stress-test review of the `qicklab` spec — findings + proposed restructure

> **What this is.** A design stress-test of `spec-20260716-…-qicklab-tproc-v2-first-principles.md`,
> run as a sequence of interrogation passes against the actual v1 code and the real tProc-v2 source
> (`../qick`). Every claim below is grounded in `file:line` evidence gathered by reading code, not the
> spec's description of it. Passes run: (1) optimal-control / uploaded-AWG path; (2) full measurement
> lifecycle (Station → config → job server → result → analysis); (3) device / multi-qubit / new-PC
> extensibility; (4) config versioning; (5) an IR-expressiveness sweep of the ~140-experiment catalog;
> (6) a deep dive on active-reset/feedback and phase/frame tracking; (7) a ground-truth read of the v2
> `asm_v2` sweep + escape-hatch model; (8) the qsim Trotter/Floquet family as a hardware-loop case study.
>
> **Headline.** The spec is **architecturally strong on the pulse→compile→acquire→fit spine it spent
> most of its words on**, and for the majority of experiments it largely *formalizes what
> `MM_base.custom_pulse` already does*. Three things change how it should be packaged:
> **(1)** the IR spine needs **first-class frame tracking + a frame-aware loop + a two-tier sweep model +
> feedback nodes** — most of which appear nowhere in the spec, and one of which (frames) is both a huge
> ergonomic win and the key to a *hardware-loop lowering that removes a program-memory ceiling the lab
> actively hits*; **(2)** an entire **lifecycle/session layer** (Station, instruments, calibration
> write-back) is left as an open bracket though it's the layer a user touches first; and **(3)** the
> document operates at **four altitudes at once** and should be split into a program plan + a scoped
> core spec + follow-ons. It also lacks a **migration journey** for a system that runs daily.

---

## Part 0 — Verdict at a glance

| Axis | Verdict |
|---|---|
| Core IR / compiler / program / mock (§3–§6, §8, §10) | **Solid, buildable.** Largely formalizes existing `custom_pulse` practice. |
| IR *expressiveness* vs the real catalog | **Biggest gap.** ~55–65% fit cleanly; ~25–30% need a richer-but-declarative IR; ~10–15% need the escape hatch. Needs frames, a frame-aware loop, a two-tier sweep model, and feedback nodes. |
| Phase / frame tracking | **Absent from spec, high-value.** A static `phase_deg` can't express the 5 v1 phase mechanisms. A frame model is the right primitive *and* unlocks hardware loops. |
| Optimal-control / uploaded-AWG (§6.5) | **Right instinct, under-specified.** `from_solution` needs per-drive gain, explicit frame/sign, co-quantized timing, content-hash dedup, a memory guard, a preload/re-fire mode. |
| Lifecycle / session layer | **Largely missing / deferred.** The layer a new user lives in is an open §14.2 bracket. |
| Multi-qubit / device extensibility | **"Should not preclude" ≠ "designed for."** Singleton baked into the daily-use API surface. |
| Config versioning (§14.1) | **Over-weighted as open.** Keep the mechanism; the real win is Pydantic schema/validation/migration. Spec contradicts itself (git vs immutable). |
| Migration *journey* for a live lab | **Absent.** The spec designs a destination for a system that can't be turned off. |
| Cost-model transparency (glass box) | **Missing guarantee.** Escape hatch (write raw) ≠ inspectable output (see/cost what compiled). |
| Document structure | **One flat P2 doc at four altitudes.** Split it. |

**Decisions cheap now / expensive later — make before slice 1:** (A) elements **collection-first**
(multi-qubit); (B) a **frame / phase-accumulator model** in the IR.

---

## Part A — The gap between where we are and where we want to be (the "plan plan")

The spec is really a *core-framework + multimode-pack design* wearing the clothes of a whole-program
plan. Separate the work into self-contained workstreams with explicit maturity and dependencies, and
add the thing the spec omits entirely: a staged migration for a **live** lab.

### A.1 Workstream map

| WS | Owns | Depends on | Maturity in current spec |
|----|------|-----------|--------------------------|
| **WS1 Core framework** | Pulse IR, compiler, program, mock, wiring, sweeps, result seam | — | **Ready to build.** *Must also close the multi-qubit + frame-model decisions up front.* |
| **WS2 Multimode pack** | device model, pulse library, calib schema, experiments | WS1 | Mostly ready; blocked on WS1 collection/frame decisions |
| **WS3 Lifecycle / session** | Session/Station, non-QICK instruments, calibration write-back loop, rig-config, cold-start | WS1, WS4 | **Open / missing** (spec §14.2 bracket) |
| **WS4 Config & versioning** | Pydantic schema + validation + migration; keep immutable-snapshot mechanism; 4→1 handle; BranchManager UX | — | Near-settled (Part B5); spec over-weights it |
| **WS5 Fitting / analysis** | core lmfit fitters (in-core) + full `fit_display*` refactor (follow-on) | WS1 result seam | Core carved (D4/§9.3); rest explicitly deferred |
| **WS6 Job-server** | compatible-with now → clean runner later | WS1, WS3 | Open (§14.2) |
| Positioning (non-gating) | horizontal modalities (§15), QILC/Intonato (§16), OC/Piccolo strategy | — | Strategic; should not gate the build |

**The scoped core spec should be WS1 + WS2**, with the frame model + the four IR extensions (B1) and the
OC lowering contract (B2) folded in, and the collection/frame decisions made. Everything else moves to
the program plan or a follow-on.

### A.2 The migration journey (the biggest omission)

D1 says "first-principles greenfield, not port-in-place." But the lab runs **daily** on v1 (~140
experiment classes, a live single-worker queue, real users). You cannot ship only a destination.

- **Strangler-fig, not big-bang.** qicklab experiments run *alongside* v1 through the **same** job
  server (§14.2 Option A: worker imports by `module:class`, so a qicklab experiment is just another
  importable class — `worker.py:421-464`). Migrate experiment-by-experiment; starve v1 over time.
- **Bridge analysis.** `MeasurementResult.as_tuple()` (§8) lets v1 analysis consume qicklab results
  during the transition; keep it until WS5 lands.
- **Coexisting config.** v1 configs and the qicklab Pydantic `Device` must both be loadable; the
  write-back loop (B3) updates whichever the running experiment uses.
- **Suggested cutover order:** linear spectroscopy/Rabi/T1 (highest IR fit, B1) → sidebands/swaps →
  parity/dual-rail (needs the frame model) → Trotter/Floquet (needs the frame-loop, B1) → RB/feedback.

This staging is program-management work the current spec does not contain; the program plan must.

---

## Part B — Specific gaps in the existing spec

### B1 — IR expressiveness: the spine needs frames, a frame-loop, a two-tier sweep model, and feedback nodes *(highest priority)*

The catalog sweep (sampling ~120 program classes) shows the IR is a clean fit for the majority and is
essentially what `MM_base.custom_pulse` (`experiments/MM_base.py:433-529`) already does. Rough split:
**~55–65%** fit "compose factories + one sweep + measure" cleanly; **~25–30%** need a richer-but-still-
declarative IR; **~10–15%** legitimately need the escape hatch. Four patterns must become first-class or
the escape hatch becomes the norm rather than the exception.

#### B1.1 Frame tracking — the missing primitive (biggest single win)

`Pulse.phase_deg` (a static float, or an *independent* `Sweep`) cannot express the phase bookkeeping the
catalog actually needs. v1 has **five** distinct phase mechanisms:

- **Phase coupled to the swept axis:** Ramsey/T2 advance `r_phase2 += deg2reg(360·ramsey_freq·step)` in
  lockstep with the delay register (`t2_ramsey.py:195-198`, `t2_echo.py:196-198`).
- **AC-Stark parity phase:** `θ₂ = second_phase + 360·revival_stark_shift·revival_time`
  (`MM_base.py:2043-2048`), written mid-sequence via `safe_regwi`.
- **RB virtual-Z frame tracker:** per-mode `vz` accumulator + time-dependent idle term + a **7×7
  `idling_phase` overhead matrix** (`MM_rb_base.py:523-637`; `rb_ziqian.py:174-206`).
- **Dual-rail accumulators:** AC-Stark idle rate on every wait + a joint-parity cross-phase matrix
  (`compute_dr_phase_corrections`, `MM_base.py:1182-1234`).
- **Per-generator phase registers + `phrst` frame reset:** f0g1/flux/BS each carry a phase register set
  relative to a shared origin established by `reset_and_sync` (`MM_base.py:375-416`).

These are all the *same concept*: a **rotating-frame / frame** whose phase accrues as `φ₀ + 360·f·t`,
with explicit frame ops (`shift_phase`/`set_phase`/`reset`) — the primitive Qiskit Pulse / OpenPulse
settled on. Every v1 mechanism maps onto it:

| v1 mechanism | frame op |
|---|---|
| `advance_qubit_phase` constant offset (`MM_base.py:436`) | `frame.shift_phase(c)` |
| Ramsey/T2 `φ=360·f·τ` (`t2_ramsey.py:195`) | drive frame **runs at `ramsey_freq`**; final π/2 phase falls out of elapsed time |
| AC-Stark parity (`MM_base.py:2045`) | frame accrues at Stark-shifted rate during the revival `wait` |
| RB virtual-Z + 7×7 `idling_phase` (`MM_rb_base.py:604`) | a gate **shifts other lines' frames** — frame *coupling* (phase crosstalk) |
| `phrst` / `reset_and_sync` (`MM_base.py:375`) | `frame.reset()` — the shared zero origin |

**Recommendation:** add a **frame abstraction** to the IR. Pulses play *in* a frame; virtual-Z, Ramsey
phase, AC-Stark parity phase, and f0g1/BS phase crosstalk become declarative frame ops the *factories*
emit — the experiment author writes `dev.qubit.ge_hpi()` twice with a `wait` and the Ramsey phase is
correct **by construction**, no manual `deg2reg`. **So much of current experiment-writing time is spent
hand-working-out these frame updates; this is the highest-leverage ergonomic change in the whole
redesign.**

**Who owns the computation (the mechanism):** the factory declares frame membership + intrinsic phase;
the **compiler's frame pass** owns accumulation, reading a *declarative frame-coupling model* (the
crosstalk CSVs re-expressed as "a swap on A shifts B's frame by constant X"); the hardware does the
actual per-iteration increment. The frame pass must **report which lowering it chose** (host-side static
phase vs affine hardware sweep vs register-math) — "trust me, the phases are right" is not acceptable for
this. **Migration cost to be honest about:** moving the crosstalk from imperative experiment code
(`swap_stor_phases[j] += get_phase_from(...)`, `sideband_scramble.py:109`) into a declarative coupling
model the compiler applies is real work; anything the model can't express falls to the escape hatch.

#### B1.2 A frame-aware loop (`repeat`) — removes a real program-memory ceiling

**Case study (flagship): qsim Trotter/Floquet scramble.** `SidebandScrambleProgram.core_pulses`
(`sideband_scramble.py:95-111`) unrolls `floquet_cycle × len(swap_stors)` beam-splitter pulses **flat**
at compile time (~400 pulses at depth 200) — and the lab has hit the resulting program-memory ceiling.
Ground truth from the code: the body is **strictly periodic** (freq/gain/length/waveform fixed, built
once before the loop; only `phase` rewritten — `:95-99`), and the phase is **provably affine**:
`swap_stor_phases[j] += get_phase_from(M1-Sj, M1-Sstor)` is a constant per period (7×7 static CSV
`floquet_storage_swap_dataset.csv`), so `phase_j(n) = phase_j(0) + n·Δ_j (mod 360)`. The code even has a
closed-form accumulator (`_accumulate_scramble_phases`, `floquet_dark_mode_readout.py:946-955`) that just
repeats the constant addition — proof the increment is constant.

This maps **directly** onto a native v2 hardware loop: body = one period (2 pulses), each pulse's phase =
an affine `QickParam(start=phase_j(0), step=Δ_j)`, `add_loop(N)`; v2's `CloseLoop` increments each phase
per iteration (`asm_v2.py:683-706`), and the mod-360 wrap is free. **Program length becomes independent
of Trotter depth — the memory ceiling goes away.** The palindrome variant
(`floquet_dark_mode_readout.py:1853-1857`) is period-2 → loop body covers two steps, still affine.

**Honest caveats on the *bigger* prize.** Today the depth sweep is *also* a host-side recompile loop —
a fresh program per depth (`qsim_base.py:406-410`). Collapsing the whole depth scan into one program is
bounded by (a) **destructive measurement** (a depth scan re-preps each point → physically independent
shots; you can't just measure every loop iteration) and (b) **v2 loop trip counts are build-time
constants** (a variable inner count needs the `cond_jump`+counter escape hatch). So: **per-program memory
ceiling — solved cleanly**; whole-scan collapse — possible via the escape hatch, worth prototyping, not
worth promising.

**Recommendation:** add a **frame-aware `repeat`/loop IR node** (distinct from a parameter `Sweep`): the
user writes one period + `.repeat(N)`; the frame pass tracks frames over one period, computes
`(φ_j(0), Δ_j)`, and lowers to a hardware loop with affine `QickParam` phases (or `exec_after` register
math if non-affine), with a diagnostic saying which. Distinct from the RB/CPMG structure-sweep (B1.4):
RB structure *varies* (must generate-per-point); Trotter *repeats* (hardware-loopable).

#### B1.3 The two-tier sweep model (the spec's §8 claim is only half-true)

§8 says the v1 RAverager/NDAverager machinery "disappears" into one `QickSweep1D` per axis. Ground truth
on both sides:

- **v2 `QickParam`** (there is no `QickSweep` class — spec R2 recollection error) is **affine-only,
  uniform steps** (`asm_v2.py:135-143, 276-291`). Arithmetic is affine: `QickParam + QickParam` merges
  spans, `QickParam * scalar` works, but **`QickParam * QickParam` = `NotImplemented`** (`asm_v2.py:191`).
- **Consequence (good):** the Ramsey `φ = 360·f·τ` case *is* declaratively expressible — `f` is a
  constant, so `360*f * τ_sweep` is scalar × swept and phase can *share the delay's loop axis*. The IR
  just needs to expose **affine arithmetic over a shared sweep axis** (a `phase` field that is
  `360*f * the_delay_sweep`), not only "field = independent `Sweep`." §6.2 shows the latter.
- **Consequence (real losses):** arbitrary-list / log / non-uniform sweeps (Wigner α-grid
  `single_mode_wigner_tomography.py:331-386`; power/volt sweeps) and products of two swept axes **cannot**
  be a `QickParam`. In practice these are already **Python outer recompile loops** — an entire existing
  tier (`sweep_runner.py`, `qsim_base.py:406-410`) the spec doesn't acknowledge. Only **3 files** use v2
  `QickParam`-style sweeps today.

**Recommendation:** state the **two-tier sweep model** explicitly (in-program affine `QickParam` +
outer recompile/table driver), and add a `Sweep.table(values)` → data-memory + `read_dmem` escape for
arbitrary-point sweeps (`asm_v2.py:1124`).

#### B1.4 Feedback as a first-class node (bounded; good news)

~40 files use `active_reset` and variants (`MM_base.py:1699-1898`, `parity_active_reset:1339`,
`joint_parity_active_reset:2179`, `slow_ge_pulse_active_reset:1392`). It's control flow
(`read → condj(<thresh) → conditional π → label`), not decoration. **But it's bounded and portable:**
v2 has turnkey `read_and_jump`/`cond_jump`/`read_input` (`asm_v2.py:1152-1204`), and crucially the
hardware branch only *resets* — **all statistical post-selection is host-side numpy** on per-shot data
(`collect_shots`, `MM_base.py:2336`; filtering in each experiment's `analyze`, e.g.
`dual_rail_sandbox_v2.py:465-517`). So bless ~4 feedback/herald primitives as first-class IR nodes;
host-side filtering ports as-is. This validates the spec's 7a/7b split and strengthens "7a lands
regardless."

**Bounded escape-hatch cases (legitimately raw):** TOF/mixer/nyquist calibration
(`rfsoc_tof_calibration.py:52,99`), decimated/ring-down (`t1_ring_down.py`), single-shot histogram
buffers (`single_shot.py:36-48`), int4 freq|phase bit-packing (`t2_ramsey.py:165-168`). Small, bounded.

#### B1.5 Glass box, not just escape hatch (cost-model transparency)

The lab lives on low-level optimization — e.g. the hand-tuned `sync_all(10)` guarding tProc
instruction-processing underflow (~13 register writes per v1 `set_pulse_registers`); when 10 feels high
you read the ASM and trim. An abstraction must preserve a **visible cost model**, not just correct
behavior. Tenet 0 gives an *escape hatch* (you can *write* raw qick); it does **not** guarantee you can
*see and cost* what the abstraction *generated*. Those are different.

- v2 preserves inspectability: `QickLabProgram` *is* an `AveragerProgramV2`, so `prog.asm()` returns
  printable ASM (`asm_v2.py:1976`) and `prog.prog_list` is the instruction list (`asm_v2.py:1962-1968`).
- The specific `sync_all(10)` is v1-flavored and **must be re-characterized on v2 anyway** (different
  timing model — waveform-memory params, `delay`/`delay_auto` vs `sync_all`).
- A compiler can *beat* hand-tuning: knowing instruction counts per pulse, it can insert the *minimal*
  safe gap and report the budget — but only if the gap is **(a) computed, (b) surfaced in a timing
  report, (c) override-able** (`sync=` knob or `.raw(lambda p: p.delay(...))`).

**Recommendation:** add a **glass-box tenet** — compiled ASM, per-pulse instruction count, timing budget,
and waveform-memory usage are always inspectable (`prog.asm()`, `prog.timing_report()`); performance-
critical knobs (inter-pulse gap, envelope dedup) are surfaced and override-able, not buried.

### B2 — Optimal-control / uploaded-AWG lowering contract (§6.5)

The `from_solution` instinct is right and fixes real v1 messes. But its 4-arg signature under-specifies
what v1 does (`custom_pulse` OC branch `MM_base.py:532-629`; `load_opt_ctrl_pulse:890-952`;
`optimal_control` creator `:2532-2560`; config `hardware_config.yml:346-459`):

- **Frame/carrier/sign must be explicit.** v1 plays a rotating-frame baseband envelope on a nonzero DDS
  carrier (`frequency[0/1]`, e.g. 3569.7/4979.5 MHz) and **negates Q** (`Q_mhz = -Q`, `MM_base.py:556`).
  Wrong frame or dropped sign = wrong Hamiltonian, silently. Carry frame + carrier + I/Q sign as
  validated provenance. *(This is the same frame concept as B1.1 — unify them.)*
- **`gain_calib` must be per-drive.** Config has separate `gain[0]`(qubit)/`gain[1]`(cavity); the physics
  differs (π-pulse Rabi vs `gain_to_alpha`/`displace_sigma`, `MM_base.py:2272-2334`). Add a
  peak-amplitude/headroom overflow check.
- **Co-quantized timing.** v1 resamples qubit and cavity independently with per-channel `samps_per_clk`
  (`MM_base.py:918-934`) → the two OC drives may not co-terminate. Co-quantize the `Segment` to a shared
  T; carry the solver's interpolation model, not a generic `interp1d`.
- **ARB dedup + memory guard.** R6's `(generator, shape, sigma, length)` key is wrong for ARB (no sigma;
  distinct solutions collide) — use a **content hash**. And there is **no** guard against the 32768-sample
  buffer today (`state_tomography_1q.py:186` docstring only); add one.
- **Reproducibility + transport.** The immutable snapshot stores the `H:\…` path, not the bytes
  (`hardware_config.yml:358` copied verbatim into `configs/versions/…`) — §14.1's "byte-identical
  forever" is already violated for OC. Content-address the samples. Reconcile with §14.2 JSON/Pyro4
  transport of large waveforms.
- **Preload / re-fire mode.** v1's `waveform_preload` (fire-by-name) exists so the closed loop doesn't
  re-upload each iteration; the spec's fresh-program-per-acquire has no analog — needed for §16.

### B3 — Lifecycle / session layer (the layer a user touches first)

`Device` is deliberately qick-free and passive. But `MultimodeStation` (`experiments/station.py:133`) is
what the notebook workflow orbits: live `QickConfig` + `InstrumentManager`, **two `YokogawaGS200` flux
sources**, output paths, `log_measurement`→Obsidian, all `snapshot_*` config write-back, mock mode, and
global registration (`Experiment._active_station`). Nothing in the spec plays this role.

- **Add a first-class `Session`/`Station`** distinct from `Device`: `Device` = pure serializable config;
  `Session` = live object holding `Device` + soc (Pyro/local/mock) + instruments + submit/retrieve. This
  is where §14.2 Option B should be *designed*, not deferred.
- **Non-QICK instruments are in the critical path** (the worker ramps the coupler yoko before every run,
  `worker.py`). Decide scope explicitly — a QICK-only framework can't own the operating point.
- **Model the calibration write-back loop** as first-class: `preprocessor` (span/center→start/step +
  snapshot) → run → fit → `postprocessor` writes result back into `Device` calib → version. Today:
  `resspec_postproc` writes `expt.data['fit'][0]` into `station.hardware_cfg…readout.frequency`;
  `ChevronFitting → station.ds_floquet.update_freq → snapshot`. This is the wire between §9 and §7/§14.1,
  and the human-scale version of the §16 QILC loop. Absent from spec.
- **Typed result round-trip** — worker emits a serializable `MeasurementResult`; notebook reconstructs
  the typed result and runs Analyzer/Display locally, replacing v1's "pickle the whole live `expt`"
  (`client.py:86-108`). Specify the HDF5↔`MeasurementResult` symmetry.
- **Mock is broader than `MockQickSocV2`** — off-hardware runs also need `MockInstrumentManager` +
  `MockYokogawa` (`station.py:449-499`). If instruments are out of scope, so is the mock station, and the
  de-facto onboarding path breaks.
- **Cold-start (design piece only; daemons are QoL):** a **Site/Rig config** layer separating machine
  plumbing (addresses, paths, board id — today hardcoded in `station.py` + inlined in notebooks) from
  the portable/versioned `Device`; a **genesis-config** path (how the first `Device` for a new chip is
  authored); a **slice-0 loopback/first-light** tracer.

### B4 — Multi-qubit / device extensibility

R3 ("should not preclude" multi-qubit) is a weak bar, cleared only where it was already easy.

- **Generalizes free:** wiring (`Wiring.lines: dict[str,Line]`) and readout (already list-shaped:
  `mux_freqs`/`thresholds`/`adc_chs`).
- **Singleton baked in (daily-use surface):** Layer-1 language (`dev.qubit.ge_pi()` — `dev.qubit` *is*
  the qubit), factory signatures (no element selector), calib schema (`calib.qubit.pi_ge`), and
  two-qubit gates (`dev.gate("CZ")` has **no operand slots** and no per-pair calib schema).
- **The tell:** the design is already internally inconsistent — storage (`swap("M1-S3")`) and readout
  are collections; qubit/manipulate/coupler are singletons.
- **Fix (cheap now):** elements **collection-first from day one** (`dev.q(0).ge_pi()`, `calib.qubits[0]`,
  factories take an element handle). With one qubit it reads `dev.q(0)`; a second qubit becomes *data*.
  Same "design-for-extension, implement-one" principle the spec applies to horizontal modalities (D15),
  never applied to the vertical multi-qubit axis.

### B5 — Config versioning: keep the mechanism, add the schema layer

The spec over-weights this as an open P-decision and **contradicts itself** (§1.2/§7.3 lean git-native;
§14.1 recommends A/C and says pure git "is not safe"). Resolve toward:

- **Keep** the immutable-snapshot + pointer + `BranchManager` mechanism (`config_versioning.py`,
  `branch_manager.py`) — it works, it's race-safe under the shared worker, unlikely to be exceeded. Put
  it behind §14.1's own **store interface** (default file-snapshot, no queue/DB dependency).
- **Do not migrate to git.** Correction: git would *not* explode in size (it content-addresses +
  delta-compresses near-identical text *better* than the current full-file store). Reject git on the
  **race-safety** grounds §14.1 already identifies, not size.
- **The real win is Pydantic (D16):** schema = template + validation-at-load + a home for schema-version
  + migration functions (add a fridge line / new pulse family safely). This is the actual pain ("loosely
  coupled, no template, no validation") and it's orthogonal to the storage backend.
- **Single-source ≠ single file.** Keep sensible splits — scalar config (YAML) vs tabular datasets (the
  swap CSVs) is a *legitimate* boundary. The single source of truth is the typed `Device` object.
- **Collapse 4 IDs → 1 handle** a measurement records; preserve `commit`/`branch`/`checkout` regardless
  of backend. (`git diff` between two calibration states is a nice-to-have you lack today.)

### B6 — Internal inconsistencies to resolve

- Git-native (§1.2/§7.3) vs immutable-recommended (§14.1) — pick one (B5).
- Collection (storage/readout) vs singleton (qubit/manipulate/coupler) — pick one (B4).
- "One QickSweep1D per axis; RAverager machinery disappears" (§8) vs the real Python outer-loop tier
  (`sweep_runner.py`, `qsim_base.py:406-410`) — acknowledge the two-tier model (B1.3).
- "Everything is a declarative Seq" (§6.3) vs feedback / structure-sweeps / frame-loops that aren't
  (B1.1–B1.4).
- Naming: `QickSweep` (spec) vs `QickParam`/`QickSweep1D` (actual v2, `asm_v2.py:21,215`) — R2 recollection.

---

## Part C — What this spec should own vs defer

- **Own (scoped core spec = WS1+WS2):** Pulse IR **with frames (B1.1), a frame-aware `repeat` (B1.2), the
  two-tier sweep model (B1.3), feedback nodes (B1.4), the glass-box guarantee (B1.5), the collection
  model (B4), and the `Session` seam named (B3)**; compiler; program; mock; wiring; the OC lowering
  contract (B2); core fitters (D4).
- **Defer to the program plan (Part A):** WS3 lifecycle detail, WS6 job-server, migration staging.
- **Defer to follow-on specs:** full fitting refactor (WS5); config-versioning *implementation* (WS4 —
  the decision is made in B5, the code is small and separable).
- **Keep as positioning (non-gating):** §15 horizontal modalities, §16 QILC/Intonato.

**Two decisions before any code:** (A) collection-first elements; (B) a frame / phase-accumulator model
in the IR. Both are cheap to decide now and expensive to retrofit after slice 1. The frame model in
particular is not just correctness insurance — it is the single biggest ergonomic win (it retires the
hand-worked phase bookkeeping that dominates experiment-writing today) **and** the enabler of the
hardware-loop lowering (B1.2) that removes a program-memory ceiling the lab currently hits.
