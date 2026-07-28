# Porting existing experiments to tProc V2 — capability summary

*Scope: the **v2 upgrade only** (can V2 run what we already do), decoupled from the
architecture rewrite proposed in the qicklab spec. Grounded in a survey of ~112
experiments + `MM_base`/`MM_rb_base`/`MM_dual_rail_base`.*

## Verdict

**The migration is capability-safe. No experiment in our stack does something tProc V2
fundamentally cannot do.** ~95%+ of experiments port as pure sweeps + host-computed
phases + standard readout. Real work concentrates in two small, bounded buckets.

## Why it's safe: the one finding that settles it

**We never do dynamic runtime register math.** Across every experiment, measurement
results feed *only* `condj` (branching) — never arithmetic. Every `mathi`/`safe_regwi`
is either a fixed-step linear sweep or a host-precomputed value. The thing that would
genuinely not map to V2's model (arbitrary runtime ASM computation) is something we
don't use.

## How each pattern ports

| What we do | V1 mechanism | V2 | Effort |
|---|---|---|---|
| Sweep freq/gain/len/phase/wait (bulk) | `mathi` in `update()` | `QickSweep1D` + `add_loop` | trivial / cleaner |
| Phase tracking (Δf·t, AC-Stark, dual-rail) | host-computed, `safe_regwi(deg2reg)` | pass physical `phase_deg` | trivial / cleaner |
| Ramsey wait+phase lockstep | two coupled sweep regs | two sweeps on the *same* loop | trivial / cleaner |
| Decimated / single-shot / thresholded acquire | `AcquireMixin` | **same `AcquireMixin`** (shared v1/v2) | none |
| Trotter / in-program pulse loops | Python unroll | unroll as-is; HW loop optional* | none |
| `phrst`, register-driven waits | native | native | re-verify timing |
| **Measurement feedback** (active reset etc.) | `read`→`condj`→`label` | `read_input`/`cond_jump` | **re-tune latency** |
| **Long pulses** (3 Stark expts) | `register_long_pulse` chunking | same 16-bit limit → port chunk | **mechanical port** |
| DDR4 / MR buffers / `memri` / `bitwi` | — | — | not used |

\* *V2 hardware loops can fold the unrolled Trotter blocks that hit the program-memory
ceiling (per-step phase advance is affine → expressible as a sweep). Optional upgrade,
not required for the port.*

## The two buckets that need real work

1. **Measurement feedback — 8 routines, structurally identical.** `active_reset`,
   `parity_active_reset`, `slow_ge_pulse_active_reset`, `joint_parity_active_reset`,
   `multi_parity_readout`, + 3 single-shot files. All are the same primitive
   (`read → condj(<thresh) → conditional reset π`). The 66 experiments that *call*
   `active_reset` share one base method — the porting surface is ~8 functions, ~2–3
   truly distinct. **Only real risk: the empirical readout-latency wait
   (`wait_after_readout`) must be re-measured on v2 silicon** — it's a hardware
   constant, not an expressiveness gap, and there is no data-ready handshake in *either*
   tProc version. One measurement, reused across all 8.

2. **Long pulses — 3 Stark experiments.** Hold a tone past 2¹⁶ cycles. V2 has the same
   16-bit length limit, so port the chunking loop (or move the tone to a mux generator,
   whose 32-bit length removes the need).

## Free wins (V1 hacks that disappear in V2)

- The register-register `math` dance to dodge the 31-bit immediate limit for phase
  values → gone (V2 phase is a physical-unit sweep).
- The `gain//2` flat_top compensation → gone (V2 `flat_top` handles it internally).
- Trotter throughput: V1 needs ~13 instr/pulse (`set_pulse_registers`) + `sync_all(10)`
  slack; V2 plays a pre-defined pulse in ~2–4 instr (wmem→port copy), so the
  "generators race ahead" pressure drops sharply.

## Recommendation

Port on a **behavior-preserving basis first** (sweeps → `QickSweep`, unrolls stay
unrolled, feedback → `read_input`/`cond_jump`). The single thing to validate on-board
before trusting the port is **feedback latency** — do one calibration run to fix the v2
`wait_after_readout`, then it covers all feedback experiments. The HW-loop Trotter fold
and the architecture rewrite are independent, optional follow-ons.
