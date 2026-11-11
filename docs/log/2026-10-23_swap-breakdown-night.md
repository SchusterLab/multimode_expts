# 2026-10-23: swap breakdown night (decoder vs Wigner, |2>-input P1, 1.0x vs 2.0x, env vacancy); strong swaps to 3.5x

seb (10-22 15:00-21:00): calibrate stronger swaps; then (21:00) "a clear answer and breakdown of what happens with this
swap"; drop measurements without value. Sandbox `C:\experiments\260601_Transduction_sandbox\coherence\` (scripts named below,
results `results_*.json`, summary `coh_test_summary_1023.md`, P1 budget `p1_budget_1023.py/.json`). All versions
update_main=False; main not changed. No repo code changed.

## Strong swaps (M1-S4, CFG-HW-20261020-00034 base)

| strength | M1 version | gain | freq (MHz) | pi (us) | note |
|---|---|---|---|---|---|
| 2.0x (round-trip corrected) | CFG-M1-20261022-00047 | 5837 | 874.52373 | 2.113 | 10-21 00043 was 10 kHz too high (`rt_freq_scan.py`) |
| 2.5x | CFG-M1-20261022-00024 | 7258 | 874.72259 | 1.690 | saved from measured values (fine-gain fit R^2 0.59) |
| 3.0x | CFG-M1-20261022-00036 | 8743 | 874.97394 | 1.408 | 2D map gain kept (fine gain: width out of range) |
| 3.5x | CFG-M1-20261022-00044 | 10091 | 875.27347 | 1.207 | check stopped at 27 / 48 points (seb: coherence test first) |

- Above 2x the error-amplification maps are flat in frequency (+/-45 kHz) and the fine-gain fits fail (broad noisy peaks).
  `swap_cal2d.py`: N = 5, 9 above 2x, windows x s/2, relaxed fine-frequency gate (fit and symmetry centre within 2 kHz x s/2),
  fine gain: mean with the map if R^2 >= 0.4 and within 2 %, else the clean map gain (1 % gain = ~3e-4 swap error).
  The 1D coarse gain is biased at 3x (~4 % low). Seeds: `seed_swap.py` (Stark f = F0 + K g^2, K 9.99e-9 MHz/gain^2).
- Checks (`run_swap_bench.py --check`, same elapsed time): B |0>+|2> error rate 11.3 kHz at 2.5x (static-Kerr model 11.1),
  16.6 kHz at 3x; B |0>+|1> (1 photon, no Kerr) shows a coherent oscillation at 2.5x and 3x (a swap detuning the EA maps cannot
  see); pump at 2 MHz detuning vs idle: +7 /ms at 2.5x and 3x (0-5 /ms below). The round trip is first-order sensitive to the
  swap detuning: the 2.0x scan (B |0>+|1>, N 8 and 16, +/-40 kHz) gave clean fringes and -10 kHz.

## Simulations (MODEL; square sidebands, no decay unless stated)

- Static M1 self-Kerr |K| = 14.7 kHz reproduces all four benchmark cases and the strength-independent rate
  (`sim_kerr_roundtrip.py`); its Ic cost at eta 1/3 is only 0.013 (1.0x) / 0.003 (2.0x) (`sim_kerr_signatures.py`).
  Config kerr is -3.0 kHz: unresolved.
- Ic sensitivity at eta 1/3 (exact Ic, ideal +0.215): rho02 x 0.9 -> +0.149; rho02 and rho13 x 0.9 -> +0.079; forbidden P1
  5 % -> +0.097; low-set values (rho02 0.79, rho13 0.90, P1 5.4 %) -> -0.048. Modest errors explain the low Ic.
- Decoder (CavityModeRamsey fock 2) on the one-swap state: with the 0.40 us f1-g2 it takes rho13 at -25 % weight; with a
  ~1.4 us f1-g2 (an f2-g3 turn synchronised) it reads rho02 alone (`sim_decoder_oneswap.py`, `sim_decoder_sync.py`).
- A measurement-free two-branch decoder (D and D shifted by one photon, 8 pulses) reaches the optimal Kraus Fe 0.748 at eta
  1/3 (even branch only 0.600); with today's sideband lengths 0.310; with synchronised lengths (f1-g2 ~1.6, f2-g3 ~1.7,
  f0-g1 ~1.25 us) 0.742 (`sim_full_decoder.py`). Not built.

## Selective f1-g2 and the coherence test

- `sel_f1g2_cal.py` -> CFG-HW-20261022-00029: f0-g1 night values (1989.6315 MHz, 0.560 us, gain 8000), f1-g2 1992.1415 MHz,
  gain 2191, 1.392 us. At gain 2260 the line is 3.05 MHz above the gain-8000 value (sideband Stark shift ~-3.3 MHz at 8000).
- `run_coh_test.py` per cycle: decoder ref (no swap) / one (one partial swap, eta 0.35) / idle (swap at gain 0); populations
  |1>, |2> through D; Wigners of '+' and of the |2> input, with and without the swap (same encoder). 13 cycles: t1/t2 (1.0x,
  19:11-21:03), b10 (1.0x, 5) and b20 (2.0x, 4) alternating 22:48-03:45.

| quantity | 1.0x (b10) | 2.0x (b20) | all runs |
|---|---|---|---|
| decoder rho02 after the swap (one/ref x 3) | 0.83 +/- 0.03 | 0.88 +/- 0.04 | 0.87 +/- 0.02 (chi2 12/12) |
| Wigner rho02 after the swap (one/ref x 3) | 0.91 +/- 0.06 | 1.02 +/- 0.05 | 0.93 +/- 0.03 (chi2 19/12) |
| Wigner prep |rho02| / 0.5 (no swap) | 0.97 | 0.97 | t1/t2 0.92 |
| forbidden P1 (|2> in, swap) | 4.6 +/- 0.7 % | 6.1 +/- 0.6 % | |
| P1 excess after prep + decay + vacancy | +0.8 +/- 0.8 % | +2.5 +/- 0.9 % | |

- Idle arm (same time, no swap): 0.96-1.02 of ref: the time alone costs ~1 %.
- Decoder population check: |1> -> e 8-12 % (MODEL 0 % e, 3.4 % f): selectivity is good enough that the rho13 bias is
  <~2 %. The decoder reads 0.06 +/- 0.05 below the Wigner over all runs (b10/b20 alone -0.12 +/- 0.04, t1/t2 +0.04 +/- 0.10);
  the Wigner varies more between runs. Not resolved; neither readout shows the low-set 0.79 as a readout artefact.
- Forbidden P1 budget (`p1_budget_1023.py`): the |2> preparation carries 1-5 % |0> and 3-8 % |1> (drifting over the night);
  through the ideal swap they give 1-4 % P1; M1 decay adds 1.0 % (1.0x) / 0.7 % (2.0x). At 1.0x this closes; at 2.0x
  ~2.5 % remains after the vacancy term.

## Env vacancy and eta (photon-number Ramsey icref@0.35, decay-aware)

- The simple A / B formula counts decay as vacancy (as found on 10-12). With the MODEL decay (T1 M1 215, S4 1239 us; each
  strength's swap time): 1.0x, 8 references 10-19..10-23: -4.0..+1.1 %, mean ~-1 %, i.e. ~0. 2.0x (10-22 00:32, 10-23 04:35):
  +2.3, +2.5 %. The 2.0x full swap loads the env photon ~3.5 +/- 1 % less well than 1.0x (model-independent difference).
- eta_B tonight 0.326-0.330 (1.0x, 21:29-21:54), 0.321 (04:10); 2.0x 0.330. Stable to ~0.006.
- Correction: the 22:00 statement "~2.3 % vacancy this cooldown" used the simple formula and was wrong.

## Breakdown (1.0x swap, rho02 of the '+' channel output, relative to ideal)

| part | cost | how |
|---|---|---|
| '+' preparation | 3-8 % | Wigner no-swap |rho02| 0.92-0.97 |
| time of the swap (decay, dephasing) | ~1 % | idle arm |
| static Kerr during the swap | ~2 % | MODEL fitted to the benchmark |
| env vacancy | ~0 % at 1.0x (~5 % at 2.0x) | decay-aware icref |
| remainder (two-path) | ~4 % (Wigner) .. ~12 % (decoder) | the 10-08/09 "two-path deficit", seen again |

Forbidden P1 at 1.0x: preparation + decay. At 2.0x the env loading adds ~1.5 % and ~2.5 % is unexplained.

## Incidents

- Job server overloaded 15:05-17:45 (one core busy; 4-18 s per request; 6 connections via sshd, likely a dashboard over a
  tunnel); 75 s per point instead of 1 s; fixed by guan. jobs.db is 2.0 GB.
- My chain scripts: a start before the server was up (all submits refused); a stop pattern that also matched the new
  chain; a first night chain that survived its wrapper and ran a duplicate 2.0x scan (22:00-22:35, both kept their own
  points). The watchdog now checks the server (3 misses), the worker, stalls, and duplicate measurement processes.

## Next (proposed, not decided)

- The two-path remainder: decoder vs Wigner disagree by ~0.06-0.12; a repeat with the full Ic set (4 inputs) read both
  ways would give its Ic effect directly.
- 2.0x: recalibrate the full swap used for the env loading (round trip at full length) and repeat the P1 / vacancy check.
- The |2> preparation (P0 1-5 %, P1 3-8 %, drifting) costs more Ic than the swap at 1.0x: a prep refresh rule.
