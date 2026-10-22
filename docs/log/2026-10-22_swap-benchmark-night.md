# 2026-10-22: swap benchmark at five strengths, sector length Rabi, Ic 2.0x vs 1.0x (night 10-21/22)

seb ok (10-21 ~17:00-22:45). Sandbox `C:\experiments\260601_Transduction_sandbox\coherence\` (scripts `run_swap_bench.py`,
`swap_cal.py`, `swap_cal2d.py`, `run_len_rabi.py`, `fit_swap_bench.py`, `plot_swap_bench.py`; chains `night_1021*.sh`;
results `results_bench_*.json`, `fits_swap_bench.json`, `results_len_rabi_s1.00.json`; figures in `figs/`) and
`ic_debug/final/swapAB_analysis.py`. Classes on main: CavityModeRamseyExperiment (fock=2, sweep='phase', mid_pulse),
CavityModeT1Experiment (mid_pulse), commits fcbacb3 / a0b895c (seb reviewed). Config CFG-HW-20261020-00034 throughout;
only the M1-S4 row changes per strength (update_main=False).

## Swap strengths

| strength | M1 version | gain | freq (MHz) | pi (us) | calibration |
|---|---|---|---|---|---|
| 1.0x | CFG-M1-20261020-00010 | 2973 | 874.2829 | 4.225 | 10-19 |
| 1.33x | CFG-M1-20261021-00015 | 3931 | 874.3456 | 3.177 | swap_cal.py OK (converged 1.1 % / 3.8 kHz, accepted < 2 % / 5 kHz) |
| 0.75x | CFG-M1-20261021-00019 | 2239 | 874.2477 | 5.634 | partial snapshot: the +/-5 % fine gain window covers +/-1.4 sigma of the broad peak (width check failed); coarse gain and fine frequency clean |
| 1.67x | CFG-M1-20261021-00025 | 4871 | 874.4256 | 2.530 | swap_cal.py OK (0.3 % / 2.1 kHz) |
| 2.0x | CFG-M1-20261021-00043 | 5837 | 874.5337 | 2.113 | 1D scans do not converge (Stark coupling df/dg = 2 K g: +150 gain = +17 kHz); 2D EA freq x gain: product 0.96 at the optimum after correcting the readout convention |

Seeds: gain 2973 s; frequency from the Stark fit f = F0 + K g^2 of 10-19 (K 9.8e-9 MHz/gain^2), within 3-6 kHz.
Bug found and fixed (sandbox): ErrorAmplification2DExperiment.analyze defaults to start_readout 'g' for non-multiphoton
pulses; for a storage swap the photon back in M1 reads 'e' (pass start_readout='e'). Worth fixing the default in the
class (shared code, not changed; for seb).

## Benchmark (|0>+|n> in M1, N partial-swap round trips theta/-theta, decoder phase sweep; parity of |2>)

Fit A(N)/A(0) = exp(-N/Nd) sqrt(1 - V sin^2(pi N / T)) (B, C), exp(-N/Nd) (D idle, E pump detuned). Rate = (2 pi/T)/t_rt.

| strength | 2-photon rate B / C1 (2pi x kHz) | 3-photon rate C2 (2pi x kHz) | error per partial swap 2-photon / 3-photon (rad) |
|---|---|---|---|
| 0.75x | 8.8 / 11.1 | 32.8 | 0.19 / 0.69 |
| 1.0x | 11.1 / 12.4 | 36.8 | 0.18 / 0.58 |
| 1.33x | 11.9 / 13.0 | 39.7 | 0.14 / 0.47 |
| 1.67x | 11.9 / 11.5 | 43.9 | 0.11 / 0.42 |
| 2.0x | 8.6 (T poorly fixed: 46 +/- 12 > scan) / 11.0 | 36.4 | 0.07 / 0.29 |
| 1.0x repeat 03:00 | 10.2 / - | 36.9 | 0.16 / 0.58 |

- B (|0>+|2>, sectors 0, 2) and the parity of |2> oscillate with the same period (photons go to S4 and come back): a
  coherent error per round trip; |0>+|1> (sector 1) shows none. Sectors with 2 photons (B fock 2, C fock 1) share one
  rate; the 3-photon sector is ~3.3x faster: consistent with a pair (Kerr-like, ~n(n-1)) term. Hypothesis.
- The rate is constant within ~+/-15 % over 2.6x in gain: a fixed nonlinearity per unit swap time; the error per swap
  falls ~ as 1 / strength.
- D and E agree (pump-induced dephasing negligible at these pulse lengths; the 10-21 afternoon E effect needed 68 us
  of pump). D parity follows T1.
- Sector length Rabi (1.0x, one swap, |1,0> and |2,0>, parity): k2/k1 = 0.977 +/- 0.023, zero of P(1,1) at the same
  full-swap length, amplitude 0.92 of ideal: populations nearly linear; the nonlinearity shows mainly as a phase
  (first order), which the round trips accumulate.

## Ic at eta 0.35, 2.0x vs 1.0x (interleaved, 2 sets each, 00:32-02:44; swapAB1021_case_*)

| swap | eta_c | Ic pooled | per set | MODEL thru (T1 215 / 1239 us) | Ic - MODEL |
|---|---|---|---|---|---|
| 1.0x | 0.332 | -0.087 +/- 0.043 | -0.126 +/- 0.034, -0.040 +/- 0.042 (chi2 2.6/1) | +0.104 | -0.191 |
| 2.0x | 0.345 | -0.022 +/- 0.029 | -0.041 +/- 0.062, -0.006 +/- 0.053 (chi2 0.2/1) | +0.090 | -0.112 |

2.0x - 1.0x: Ic +0.065 +/- 0.052, deficit +0.079 +/- 0.052 (~1.3-1.5 sigma): in the predicted direction, not
significant. ~4-6 sets per arm needed for ~3 sigma.

## Readout

Each losssplit process: the active-reset single shot was rejected (angle ~-18 deg), the plain one found +162 deg, the
active-reset one then passed (fid ~0.95): the 10-21 angle check works. The stored phase of CFG-HW-20261020-00034 is
from before the 10-21 10:38 QICK reload; main should get a corrected readout when it is set.

## Next (proposed, not decided)

- More interleaved Ic sets 2.0x vs 1.0x (overnight) for a significant answer.
- Closure test: a channel model with the measured pair nonlinearity during the swap; does it explain the Ic deficit
  and the rho02 vs rho13 asymmetry, at both strengths?
- Sector chevron with |1,1> (HOM) needs a prepulse in CavityModeT1Experiment (small change, for review).
