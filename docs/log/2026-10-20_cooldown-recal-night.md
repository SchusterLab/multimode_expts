# 2026-10-19/20 night: first device run of autocal after the cooldown, transduction checks

seb + Claude (Claude alone after 22:00, seb ok: "keep pushing, debug if needed, stop at 5 am"). Branch `autocal` (merged
into `main` step by step), sandbox `C:\experiments\260601_Transduction_sandbox\{autocal,ic_debug}`.

## Calibration (autocal, coupler 0.30 mA)

Final versions (update_main=False; main not changed): `CFG-HW-20261019-00062`, storage `CFG-M1-20261019-00030`
(swap fixed at 22:50, see below). Review pages: `autocal/runs/autocal_261019_*/index.html`.

| quantity | before warm-up (10-12, 0.30 mA) | now |
|---|---|---|
| qubit f_ge / f_ef | 3566.106 / 3422.67 MHz | 3561.52 / 3417.98 MHz |
| T1 / T1_ef | — / — | 454 / 159 us |
| single shot (relax 8000, active reset) | ~0.95 | 0.946-0.956 |
| f0-g1 / f1-g2 (driven frame) | 2003.30 / 2002.75 MHz | 1989.63 / 1989.07 MHz |
| M1 f_ge | 4982.15 MHz | 4986.44 MHz |
| revival time | 1.528 us | 1.315 us |
| gain_to_alpha | 2.490e-4 | 2.641e-4 (repeat 2.605e-4; scale check from pn2: ~2.59e-4) |
| M1-S4 swap (L 4.225 us) | 3754, 878.316 MHz | 2973, 874.283 MHz |

Consistency: qubit -4.6 MHz, M1 +4.3 MHz; f0-g1 = (f_ge + f_ef) - f_M1 moves -13.6 (measured -13.7); M1-S4 = f_S4 -
f_M1 moves -4.3 + 0.24.

## Found (measured)

- **f level split by ~10 kHz**: EF Ramseys beat (two-frequency fit split 11.5 +/- 0.4 and 9.9 +/- 0.2 kHz, F 157 / 319;
  EF coherence ~100 us; GE Ramsey no beat). One-frequency fits bias f_ef by ~5 kHz (negligible for the MHz-wide pi
  pulses). Cause not tested (charge-parity splitting of f plausible; a TLS possible). Repeat check: `autocal/ef_beat.*`.
- **Bare-frame vs driven frame** (seb): ladder Ramsey / amplitude Rabi give bare-frame values (g0-e0 Ramsey 3561.53 vs
  error amplification 3562.09 MHz; the e0-f0 Rabi returned the ge pi gain). Removed from the ladder.
- **Swap calibrated to the wrong angle**: with n_step 2 every gain-scan row is a multiple of 4 pulses, so every swap
  angle k pi/4 passes. After the cooldown the coupling per gain unit rose ~30 %; the old gain sat at ~0.65 pi and the
  scan moved to 3 pi/4 (gain 4428). Seen as: eta 0.35 reference B left P1 0.025 in M1 (A 0.59: env load incomplete).
  Chevron JOB-20261019-00165: full swap at 2.785 us at gain 4428. Fixed: seed 2880 from the chevron, recalibrated
  (2973, 874.283; full swap 4.160 +/- 0.022 us, chevron JOB-20261019-00175), B / A eta 0.327 / 0.337 (symmetric).
  autocal now uses n_step 1 for the storage gain scans (only k pi/2 pass).
- **Gate |2> P3 0.06 is a Wigner-scale artefact**: photon-number Ramsey of |2> (pn2, JOB-20261019-00182) P(n) =
  [0.024, 0.009, 0.963, 0.004]; |1> pure (single component at f0 + chi). The gate Wigners reconstruct physically at
  alpha scale s ~0.98 +/- 0.01 (|2> P3 0.006-0.021); the gate chi^2 alone is flat over s. One gain-to-alpha calibration
  scatters ~1.5 % (2.641 / 2.605e-4). Night channel data: analyse with s = 0.98.
- **Parity revival vs Ramsey chi**: revival -14 % (chi +16 %) but the photon-number Ramsey chi only +4 % (0.281 ->
  0.2915 MHz observed). Not resolved (the two programs count waits differently, cf. the 0.929 cycle factor).
- Sandbox bookkeeping: `losssplit.py` numbered cases with `len(glob('case_??.json'))`, which restarted at 100 and
  overwrote the weekend cases 100 / 101; restored from `final/final_results.json` (job ids certain, times approximate);
  night records are now `night1019_case_*.json` (`--prefix`), numbering from the highest existing case.

## Night measurements (sandbox `ic_debug/losssplit/night1019_case_*.json`)

1 gate, 2 eta reference (wrong swap), 3 chevron (wrong swap), 4 chevron (fixed swap), 5 eta reference, 6 pn2,
7 icpt@0.35, then (chain `night_1019.sh`) f-level check, null23, icpt@0.35. Results: see the end of this entry.
