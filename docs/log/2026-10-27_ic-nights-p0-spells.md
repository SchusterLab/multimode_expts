# 2026-10-27: encoder inputs, e0-f0 spells, P0 check, two Ic nights (10-26 evening to 10-28 morning)

seb + Claude. Sandbox `C:\experiments\260601_Transduction_sandbox\coherence\`; step-by-step record in
`night1023/NOTES.md` (10-26 / 10-27 / 10-28 bullets). No repo code changed; main config not changed (local snapshots,
`update_main=False`). Coupler 0.30 mA, eta 0.33, arms A (1.0x channel) and B (2.0x channel).

## What was done

- **Night 10-26/27** (`night1026.py`): Ic sets with the real encoder inputs (E0, E1, +, +i), 3 sets per arm, E1 qubit
  checks around each set, E1 photon-number Ramsey sentinel per pair.
- **10-27 day, qubit not in g after the |2> prep** (seb's questions, one step at a time): sideband drift vs P0
  (offline), slow f1-g2 at gains 8000 / 6000 / 4000, qubit checks after each channel step, relax scan and dummy
  pulses (heating), single shots vs g / e / f refs and a 3-state classifier, readout-artifact tests, EA recalibration of
  the four qubit rungs with an A/B test, pulse trains (error amplification) of every rung, T1_ef / T2_ef (Ramsey and
  echo).
- **e0-f0 spells** found in the pulse-train data and monitored (`spell_monitor_1027.py`, `spell_tls_1027.py`: single
  shot + T1_ef + ef Ramsey per cycle, run in seb's batch mode at priority 1 with gaps for other users).
- **External review** (read-only subagent) of the day's work; offline failure-map check.
- **10-27 evening:** Wigner vs Ramsey P0 comparison (`p0_compare_1027.py`), then **night 10-27/28** (`night1027.py`
  = night1026 + a 3-job check nothing / E1 / C around each set, C = g0-e0 + e0-f0 = the spell tag), 4 sets per arm,
  interleaved with Jonginn's jobs (seb), deadline 05:00.
- **10-28 morning:** per-set Ic, input fidelities (`input_fidelity_1028.py`), two-night figure
  (`plot_ic_nights_1028.py` -> `figs/ic_nights_1028.png`), objectives agreed and written into
  `docs/status/transduction.md`.

## What was found (measured unless marked MODEL)

- Encoder: E0 clean; E1 ~ the |2> ladder (P0 / P1 / P2 4.1 / 1.4 / 93.6 % Wigner in the encoder test, 96-98 % P2 by Ramsey
  at night); |+> coherence at its population limit.
- Qubit not in g after the ladder (1.2-3 %, mostly f) is not heating (dummy pulses, relax), not a readout artifact
  (f1 reads at the f0 blob; displaced M1 gives no false reads), not fixed by the rung EA. Dummy pulses: no heating;
  the relax scan is NOT resolved (round scatter 0.5-1 % > any trend). Per-pulse error slopes: g0-e0 ~0, g1-e1 0.04,
  f1-g2 0.48, e1-f1 0.32 +/- 0.03 % per pulse (pulse trains 10-27; e1-f1 errors go to g, ~10 % floor not understood);
  f0-g1 0.39 % per pulse (10-26, f0g1_mult).
- Failure map of the full ladder (reviewer; the second-half qubit rungs are not number-selective): e0-f0 fail -> P0,
  f0-g1 fail -> final e, g1-e1 fail -> P1, e1-f1 fail -> final e, f1-g2 fail -> final f. Check: stage-6 e minus stage-3 f = +0.20 / -0.02 / -0.28 / +0.09 %
  (mean 0.00; 2 of the 4 data sets are one interleaved run).
- ef charge-parity split 13.6 kHz (two Ramsey lines, T2 ~106 us each; echo ~190-200 us); too small to matter (MODEL).
- e0-f0 spells: 3 periods (14:10, 14:57-15:03, 15:25) with 9-13 % e after g0-e0 + e0-f0 (normal 4-5 %); the full ladder
  was not worse during them. None after 15:26 (~4.3 h, ~56 cycles), none in the night. T1_ef 177 us (sd 37), f_ef
  stable (sd 1.5 kHz) outside spells. Cause not tested (TLS near ef is a hypothesis).
- **Wigner vs Ramsey P0: no large systematic.** 9 paired comparisons, mean difference +0.2 %, sd 3 % (~2x the expected
  noise of ~1.6 % per pair: Wigner 1.1-1.7 % per ref job + Ramsey 0.3-0.6 %). The 3 night1026 pairs all have
  Wigner < Ramsey (-2.4 / -0.3 / -2.2 %, mean -1.6 +/- 0.9 %). Pooling all refs of a night changes the night1026 Ic
  by < 0.01.
- **Ic, measured inputs (MODEL A 0.14 / B 0.17):** night 10-26/27 A 0.060 +/- 0.032, B 0.123 +/- 0.026; night 10-27/28
  A 0.154 +/- 0.014, B 0.101 +/- 0.028. Per set (10-27/28, in time order): A 0.192 / 0.133 / 0.129 / 0.120, B 0.053 /
  0.028 / 0.126 / 0.170. Two-night averages A 0.139 +/- 0.013 (nights disagree, chi2 7.2 / 1), B 0.113 +/- 0.019
  (chi2 0.3 / 1). With the 10-25 scan at eta 0.33 (A 0.135 +/- 0.032, 2 sets; B 0.152 +/- 0.023, 3 sets; old parity
  calibration and ladder inputs, so not like-for-like): A was low for the whole night 10-26/27 and 0.12-0.19
  otherwise; B's session values agree (spread 0.05, ~1.4 sigma) but B rose inside night 10-27/28.
- Prep vs Ic, night 10-27/28 (no spell occurred): per-set Ic vs E1 qubit check r = 0.07, vs spell tag r = 0.29 (8
  sets); sentinels flat (P2 96.0-97.8 %, P0 1.4-2.3 %). Not tested for night 10-26/27 (no spell tag; the E1 check
  cannot see e0-f0 failures). The 1-photon eta is flat (icref 0.31-0.32, both arms, both nights). The |2>-branch
  output P1 (+/-1.5 % per point) vs Ic: pooled r = -0.41, inside night 10-27/28 r = -0.67, B alone ~-0.86 (4 sets)
  -> an open hint, not tested.
- (Audit, 10-28) Night 10-27/28 interleaved with Jonginn's jobs at 0.45 mA (104 of 149 config snapshots of 10-27/28);
  ours ran at 0.30 mA, so the worker ramped the coupler between jobs. Untested; it cannot explain night 10-26/27
  (Jonginn stopped).
- Offline f rejection in the Wigner final read (Q separates f): the f excess of E1 over E0 is only 0.4-0.6 % and
  overlaps the g / e tails -> not useful (MODEL gain <= 0.005).
- Input fidelities (per pair): E0 ~100 % (by construction: lam comes from E0), E1 96-98 % (+ 1.2-3 % qubit not in g),
  + / +i 94-97 % per pair (coherence ~97 %, 93-100, of its population limit). Pooling the + / +i refs over pairs gives
  ~70 % because the rho02 phase moves 13-178 deg between pairs (frame drift); + vs +i stays at -91 +/- 2.5 deg. A
  constant phase does not affect the Ic; drift inside an output set was not checked.

## Decisions (who)

- seb: work on the encoder path; batch mode at priority 1 with gaps for the spell / TLS monitor; restart the spell
  monitor if nothing shows; return to the main thread (Ic) in the evening; interleave the night run with Jonginn;
  deadline 05:00.
- seb + Claude: stop list (no more T2_ef, rung EA, relax / dummy tests, sideband gain trades, f1-g2 retuning, channel
  qubit checks); write the objectives into the status file. Priority between the three objectives: open (seb).

## Corrections made during these days (kept for the record)

- Retracted on 10-27 (NOTES and the session): "E1 qubit rise from drift" (MODEL size too small), "readout artifact"
  (a job-to-job systematic of the ge-pi setting), "failures all f", "e0-f0 saturation" (one spell pooled with good
  rounds), "f1 at a shifted blob", "half of P0 follows f1-g2" (P3 rose, P0 + P3 flat), "+7 % P0 per spell" (assumed
  a pi miss).
- 10-28 01:30 "sets agree within a night" was wrong for B in the night of 10-27/28 (it rose ~0.12).
- 10-28 "B is stable at ~0.11 over three sessions" was wrong: the 0.111 was at eta 0.35; at eta 0.33 the 10-25 scan
  gave B 0.152.
- 10-28 audit (independent agent, read-only) of the status file and this log: corrected "both arms move ~0.1 between
  sessions" (only A, one night), "P0 scatter = Wigner noise" (~2x), "prep / spells ruled out" (night 10-27/28 only),
  "P1 hint not supported" (still open), the phase range, the failure-map tolerance; added the coupler ramps, the
  e1-f1 slope, the lost process rules and the missing logs.
