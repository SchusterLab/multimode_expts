# Transduction (Fock-state decoder, lossy-beamsplitter channel)

**Last updated: 2026-10-29** (seb + Claude; the 10-28 version audited by an independent agent; 10-29: night 10-28/29 added). Work happens on `main`
(`C:\python\multimode_expts`). Experiment scripts and running notes live in the sandbox
`C:\experiments\260601_Transduction_sandbox\coherence\` (notes: `night1023/NOTES.md`, one bullet per step since 10-23;
there is NO repo log for 10-23..10-26: the breakdown night, herald test, sector-3 scan, eta scan and parity fix are in
NOTES only). History: 10-01..10-12 at eta 0.35 (logs under Docs), cooldown ~10-19, recalibration 10-20/21.

## Objectives (seb + Claude, 10-28; read before proposing a measurement)

Priority between them: **not yet chosen by seb** -- ask.

1. **A solid main number:** a positive coherent information Ic at eta 0.33 nominal (icref eta_c 0.31-0.32; ideal
   channel 0.200 at eta_c 0.31, 0.215 at 0.33-0.34), with honest errors, that repeats between sessions.
2. **A loss budget:** every part of the gap ideal -> measured gets a measured size (prep, decay, Kerr, the rest).
3. **A higher Ic:** better prep (mostly the |2> branch) and / or a better channel.

Rules: a new measurement must move one of these; write down which one, and the result that would change the next step,
before it runs. No conclusion from a single set; report the chi2 of repeated sets. Scans in RANDOM point order.

## Where things stand (measured-input Ic = the main number; MODEL = decay + Kerr with ideal inputs)

- **Setup.** Arm A: env |1> into S4 (1.0x swap) + channel swap 1.0x (874.2829 MHz, gain 2973, 2.58 us). Arm B: the same
  env load + channel swap 2.0x (874.5237 MHz, gain 5837, 1.29 us). MODEL A 0.14, B 0.17. Inputs from the real encoder
  (E0, E1 = g0-e0 pi + encoder, +, +i) since 10-26. Scripts `night1026.py` / `night1027.py` / `night1028.py`; analysis
  `analyze_night1026.py [--src night1028] [--sets n] [--pool-refs]`.
- **Objective 1 -- main number.** Measured-input Ic at eta 0.33 nominal:

  | Session | A | B | Note |
  | --- | --- | --- | --- |
  | 10-25/26 eta scan | 0.135 +/- 0.032 (2 sets) | 0.152 +/- 0.023 (3 sets) | old parity calibration, ladder inputs: not like-for-like |
  | night 10-26/27 | 0.060 +/- 0.032 (3) | 0.123 +/- 0.026 (3) | encoder inputs, new parity cal., Jonginn stopped |
  | night 10-27/28 | 0.154 +/- 0.014 (4) | 0.101 +/- 0.028 (4) | same, interleaved with Jonginn's jobs |
  | night 10-28/29 | 0.114 +/- 0.026 (3) | 0.124 +/- 0.024 (4) | snapshot CFG-HW-20261028-00170 (35 ns rungs from the 10-28 EA); no other user, coupler 0.30 mA all night |

  - A: the sets scatter more than their errors. All 10 A sets vs one constant: chi2 21.1 / 9 (p ~0.01), sd 0.062 vs
    median error 0.042. Night 10-26/27 low (0.034 / 0.044 / 0.073); night 10-28/29 0.120 / 0.002 / 0.172 (chi2 ~9.8 / 2
    inside one night). Cause not found.
  - B: consistent. All 11 B sets vs one constant chi2 11.9 / 10; the 3 night values agree (chi2 0.5 / 2). The rise
    inside night 10-27/28 (0.053 / 0.028 / 0.126 / 0.170) did not repeat (10-28/29: 0.055 / 0.132 / 0.149 / 0.118).
  - Arm ranking: not established. Three-night averages A 0.134 +/- 0.012 (nights disagree, chi2 8.0 / 2; the error is
    too small because of the extra scatter), B 0.117 +/- 0.015. Figure: sandbox `figs/ic_nights_1029.png`.
  - Per set, the Ic follows the mean output coherence |rho02|, |rho13| (r +0.90 in A, +0.53 in B; partly by
    construction). The output populations do not follow it (r with out P1 -0.24). A's coherence scatters ~2x more
    than B's between sets: 1.44x its noise (+ and +i as two measures), B 0.76x. NOT a phase drift between the jobs of
    a set: the + vs +i output phase equals the refs' within +/-5 deg in both arms (`coh_phase_1029.py`). Left (NOT
    tested): a fast dephasing whose size changes from set to set; A's longer swap may be more sensitive to a swap
    detuning (A's icref eta also scatters more: sd 0.007 vs 0.002).
  - Errors: within the two nights the set scatter is near the bootstrap errors, except B 10-27/28; in the 10-25 scan
    the set scatter (0.05-0.08) was larger than the bootstrap errors (10-08: ML Ic error ~1.8x too small). Treat a
    single-session error as a lower bound.
  - Method: per-pair vs pooled refs changes the Ic by < 0.01 (night1026); ~0.02 was the 10-24 estimate with the older
    ML method at eta 0.35.
- **Objective 2 -- budget.**
  - Prep (ideal-input minus measured-input Ic): 0.06-0.08 in the two nights; 0.12-0.14 in the 10-25 scan (old
    calibration, ladder inputs).
  - Input fidelities: E0 98.6-98.7 % (pooled per night; lam is set from E0, so ~1 by construction); E1 96.0-97.8 %
    (Ramsey P2) or 95.7-97.4 % (Wigner, pooled per night), plus 1.2-3.4 % of shots with the qubit not in g (mostly
    f); + / +i 94-97 % per pair, coherence ~97 % (93-100) of its population limit.
  - |2> branch failure map (the second-half qubit rungs are NOT number-selective): e0-f0 fail -> P0 (not seen by the
    qubit); f0-g1 fail -> final e; g1-e1 fail -> P1; e1-f1 fail -> final e; f1-g2 fail -> final f. Check: stage-6 e
    minus stage-3 f = +0.20 / -0.02 / -0.28 / +0.09 % (mean 0.00; 2 of the 4 data sets are one interleaved run).
  - Per-pulse error (pulse trains, 10-27; f0-g1 from 10-26): g0-e0 ~0, g1-e1 0.04, f0-g1 0.39, f1-g2 0.48,
    e1-f1 0.32 +/- 0.03 %/pulse (e1-f1 errors go to g and its train has a ~10 % floor: not understood, and in conflict
    with the failure map's implied e1-f1 failure ~0).
  - Decoherence alone (MODEL, measured T1 / T2, `decay_budget_1028.py`): ~1.2 % of the 3-4 % |2>-ladder loss.
  - **f1 -> g conversion** (10-28, `hold_f_1028.py`; Guan saw it first at 0.45 mA): at 0.30 mA, |f,1> goes to qubit g
    at 1.6-1.9 %/us (~6x weaker than at 0.45 mA); |f,0> does not. It goes to g0 (~70 %) and g1 (~30 %), not g2: an
    error channel. It explains the e1-f1 train g channel in size. MODEL for the ladder: +0.3-0.4 % P0, +0.1-0.2 % P1
    per |2> prep. Physics not known.
  - Qubit T2 at M1 n = 1 (10-29, `ramsey_n1_1029.py`): ge 148 +/- 6 us (n = 0: 324 +/- 21) = n = 0 + M1 photon loss
    (MODEL 129 +/- 3), nothing extra. ef 30-35 us (n = 0: 104 +/- 25, model-dependent: the charge-parity split moves
    7-14 kHz) vs MODEL (+ M1 loss + f1 -> g) 43 +/- 4 us: at most a small extra (~1.5-2 sigma). Crude MODEL: the
    ladder T2s give ~1.6 % of the ~3 % + / +i coherence loss, the largest part in f1.
  - **Sideband drive speeds up f1 -> g** (10-29, `hold_drive_1029.py`): a sideband-channel tone 15 MHz from f1-g2 adds
    2.8-5.8 %/us at gain 8000 and 29-37 %/us at gain 12000 (idle ~1.6 %/us); |f,0> does not convert. Where P0 is made
    (`stage_p0_1029.py`): E1 cut after f0-g1 P0 0.8 +/- 0.4 %, full E1 1.9 +/- 0.1 % -> the second half adds 1.1 +/- 0.4 %.
    MODEL: the drive part during f1-g2 (gain 8000, ~0.2 us in f1) ~0.6-1.2 % -> explains it, and the 10-24 fast-f1-g2 +2 %
    P0. Qubit rung frequency residuals (pi(0) pi(180)) <= 15 kHz: not a source.
    The driven conversion goes to g0 (96 % at gain 12000; `pn_drive_1029.py`): qubit and photon lost together.
  - **Slower f1-g2** (gain 6000, 0.524 us, `CFG-HW-20261029-00027`, `slow_f1g2_1029.py`): E1 P0 1.69 -> 0.91 %, new - old
    -0.78 +/- 0.36 % (3 interleaved rounds, 2.2 sigma), but NOT repeated in 6 more rounds (P0 -0.01 +/- 0.16; P1 -0.32
    +/- 0.08 instead; gain 5000 the same, `CFG-HW-20261029-00032`). All 9 pairs: P2 +0.43 +/- 0.26 % (1.7 sigma), qubit-
    not-in-g unchanged. The MODEL (hold-tone rate applied to the resonant pulse) is not confirmed. Not adopted.
    E1 P0 at the same table moves ~0.5 % between runs (1.33 / 1.69 / 1.88 %) -- larger than one run's error.
  - Channel: the 1-photon eta is the same in both arms and both nights (icref 0.31-0.32). One sector-3 Ramsey block
    (10-25 03:20-03:59) put the 3-photon sector on eta in both arms (offsets -0.023 / -0.013); it says nothing about
    later nights.
- **Checked, with limits (do not repeat without a new reason):**
  - Wigner vs Ramsey P0: no LARGE systematic. 9 paired comparisons, mean +0.2 %, sd 3 % (~2x the expected noise);
    the 3 night1026 pairs (the Ic-relevant ones) all have Wigner < Ramsey, -1.6 +/- 0.9 %. Pooling refs changes the Ic
    by < 0.01.
  - Prep and Ic, night 10-27/28 only (no spell occurred): per-set Ic vs E1 qubit check r = 0.07, vs spell tag r = 0.29
    (8 sets); E1 Ramsey sentinels flat (P2 96.0-97.8 %, P0 1.4-2.3 %). NOT tested for night 10-26/27 (no spell tag;
    the E1 check cannot see e0-f0 failures; one sentinel per pair). Night 10-28/29: 14 spell tags all normal (C read e
    3.2-4.7 %), 4 sentinels flat (P0 1.8-2.4 %, P2 95.9-97.0 %), icref eta A 0.306 / 0.317, B 0.322 / 0.323 (start / end);
    set 17 (A, Ic 0.002) had the highest E1 qubit check (2.7 / 2.2 %), but set 22 (A, 0.172) had 1.9 / 2.3 %.
  - e0-f0 "spells" (~5-8 % extra e after g0-e0 + e0-f0 for minutes): 3 on 10-27 (14:10, 14:57-15:03, 15:25, the last
    still going when the monitor ended); none in ~3.2 h of monitoring (56 cycles) between 15:26 and 19:47 and none in
    16 tagged checks in night 10-27/28. Cause (TLS near ef?) not tested.
  - B swap frequency: arm C (+23.9 kHz, 10-23, 2 sets at eta 0.35) 0.133 +/- 0.030 vs B 0.111 +/- 0.019 = no clear
    gain, but C had an unexplained ~5 % transfer asymmetry (icref sums 0.944-0.954).
  - Shorter qubit rungs (10-28, `sigma_scan_1028.py`, 1D EA in gain and frequency; 2D maps gave the same optimum):
    e0-f0 / e1-f1 at 20 and 12.5 ns are cleaner per pulse, no leakage, but no E1 gain (P0 -0.35 +/- 0.26, P1 +0.51 +/-
    0.17 %). g0-e0 / g1-e1 at 20 ns: E1 P2 +0.9 +/- 0.7 % (~1 sigma), Wigner P1 3.2 +/- 1.5 %; not adopted. At 12.5 ns
    the ge line moves > +2 MHz (outside the EA window): not calibrated. Strong pulses need ~4-9 % less gain than 1/sigma
    scaling (cause not known). Fast sidebands: 10-24, fewer failures but +2 % P0 from the fast f1-g2, no net gain.
  - No lever found: qubit rung EA recalibration, dummy-pulse heating, sideband gain trades, f1-g2 retuning, T2_ef /
    charge-parity split (13.6 kHz), the herald at its present length (its cost is time: ~3.1 us of M1 decay), channel
    qubit checks. The relax scan is NOT resolved (round scatter 0.5-1 % > any trend).
  - Readout: the third IQ cluster is f; failed shots are real states (f1 reads at the f0 blob). Offline f rejection
    in the Wigner final read is not useful (f excess of E1 over E0 only 0.4-0.6 %, overlaps the g / e tails).
  - Phase: the Ic is not sensitive to a constant phase. The ref + vs +i phase is -91 +/- 2.5 deg; the ref rho02 phase
    moves 13-178 deg between pairs (frame drift; pooling raw refs therefore gives + / +i ~70 %). Drift inside an
    output set: not checked.
- **Open:**
  - Why A's sets scatter more than their errors (see Objective 1). A Wigner job cannot be split in time (shots stored
    per displacement point). Drift between jobs ruled out (10-29).
  - **Coupler ramps:** NOT the cause of A's scatter (night 10-28/29 had no ramps and A still scattered). May still
    have played a part in B's rise in night 10-27/28. Details: in night 10-27/28 Jonginn's jobs ran at 0.45 mA and ours at 0.30 mA (104 of 149
    snapshots of 10-27/28 at 0.45 mA); the worker sets the yoko for each job. On 10-24 ramps came with ~-80 kHz
    sideband shifts and -6.4 deg readout drift. The 10-21 switch test covered only 0.30 <-> 0.25 mA. This cannot
    explain night 10-26/27 (Jonginn stopped).
  - The |2>-branch output P1 vs Ic: pooled r = -0.41, inside night 10-27/28 r = -0.67, B alone r ~ -0.86 (4 sets).
    A hint, not tested.
  - Spells as a cause in night 10-26/27 (not tagged).
  - The prep loss: the |2> branch (sideband rungs; e0-f0). The faster herald (~0.02 of the standard Ic, 10-25
    estimate) needs a repo code change (fixed 2.0 us wait in `MM_base.post_selection_measure`): seb to decide.
  - e1-f1 pulse-train floor (~10 %); its errors into g are the f1 -> g conversion (above).
  - f1 -> g conversion: rate vs sideband gain (a SLOWER f1-g2 may win, MODEL); where the driven conversion goes (g0 / g1);
    rate vs coupler setpoint. Gaps between pulses are only ~0.02 us: not a lever.
  - Is the drive effect a Stark shift of |f,1> toward a lossy state, the same knob as the coupler setpoint? (seb asked
    10-29; planned, not run.) Coupler 0.25 -> 0.45 mA moves M1 -7.7 MHz (qubit < 0.04 MHz), rate ~20x; ln(rate) slope
    ~0.45 / MHz predicts the drive rates from the sideband Stark shift (~3.4 MHz at gain 8000, ~6.5 at 12000) -- size
    matches, SIGN unclear (the line shift cannot say which level moves). Test: with the +/-15 MHz tone on, Ramseys of
    the qubit (ge, ef) and of M1 -> which level shifts and which way; rate and shift at 3-4 tone gains (4000-12000).
  - Frequency error of the 35 ns rungs: a pi(0) pi(180) train check (same-phase odd-m trains hide it in the floor).
  - A's dip at eta 0.30 / 0.36 in the 10-25 scan (0.043 / 0.076 vs MODEL 0.12 / 0.14, repeated).

## Device and process notes (current)

- Coupler 0.30 mA for our jobs (other users use other setpoints; the worker ramps the yoko per job). HW snapshot
  `CFG-HW-20261027-00034` (local, not main): readout after the 10-27 10:16 board restart (phase ~437), local parity
  calibration of 10-26 (bare qubit f_ge 3561.497, revival 1.470 us), EA rung table. Storage `CFG-M1-20261020-00010`
  (1.0x), 2.0x row from `CFG-M1-20261022-00047`. Main config not set by these campaigns (scripts use
  `update_main=False`).
- Start state: sandbox `calstate.py` / `last_cal.json` (seb: no autocal before every campaign; readout only if > 3 h
  old). Night scripts run the sideband N = 1 maps (f0-g1, f1-g2 drift ~10 kHz / h) before each pair and the readout
  single shot + qubit.ge_ramsey_fine every 3 h. Relax 8000 us whenever S4 holds photons. After a frequency event,
  recalibrate qubit f_ge and M1 f_ge, not only the multiphoton table.
- Queue: normal priority for measurements; priority 1 only for short autocal steps (seb also allowed short batches
  with gaps for a monitor). Night runs interleave with Jonginn's jobs unless he stops.
- A QICK board restart rotates the readout (10-27: 582 -> 434) and can break the worker's Pyro link: restart the
  worker and redo the readout single shot.
- Analysis rules: frame-align sets before pooling; reference eta_c from the simulated B / A icref.
- Spell tag for any run: HistogramExperiment of g0-e0 + e0-f0, 3-state read, "read e". On the device normal reads are
  3.3-6.5 %, spells 8.9-13.2 % (with the self-calibrated read of night1027: normal 3.4-5.2 %). Do not flag 5 %.

## Code notes

- `experiments/transduction/`: `channel_model.py` (ideal channel, decoder, Fe, Ic), `sequences.py` (logical prep,
  channel recipe), `process_tomo.py` (4-input sets, metrics, bootstrap), `estimators.py` (joint ML channel fit).
  Tests: `tests/test_transduction_process_tomo.py`, `tests/test_transduction_estimators.py`.
- Wigner HDF5 files keep the single shots (`i0` / `q0`, all read lanes); the parity analysis bins the I quadrature only.
- Known issue (not fixed): `analyze_set(n_thru=...)` uses the NOMINAL eta; use the sandbox analysis.
- Known issue (not fixed): the Ramsey time axis of `experiments/single_qubit/t2_ramsey.py` is ~7 % short.
- Known issue (sandbox): the constrained photon-number Ramsey fit can put the |2> weight into n = 3; read by hand.
- Known issue (status unknown since 10-12): the swap EA product-peak frequency fit fails at a coarse step.

## Docs

- `docs/log/2026-10-27_ic-nights-p0-spells.md`: 10-26..10-28 -- encoder inputs, spells, P0 check, two Ic nights.
- `docs/log/2026-10-22_swap-benchmark-night.md`, `2026-10-21_night-ic-eta.md`, `2026-10-21_coupler-switch-test.md`,
  `2026-10-21_coherence-report.md`, `2026-10-20_cooldown-recal-night.md`, `2026-10-20_sidebands-g2a.md`: after the
  cooldown.
- `docs/log/2026-10-12_transduction-ic-eta-sweep.md`, `2026-10-09_transduction-review-hom-speed.md`,
  `2026-10-09_transduction-eta-scan-night.md`, `2026-10-08_transduction-deficit-located.md`,
  `2026-10-08_transduction-time-series.md`, `2026-10-07_transduction-swap-gain-scan.md`,
  `2026-10-06_transduction-loss-split.md`, `2026-10-04_transduction-tls-event.md`,
  `2026-10-03_transduction-gap-search.md`, `2026-10-02_transduction-night-sweep.md`,
  `2026-10-01_transduction-ic-debug.md`: the 10-01..10-12 history (eta 0.35 era).
- Related (other themes): `2026-10-24_kerr-n123.md` (guan, dressed M1 Kerr).
- Full record: sandbox `coherence/night1023/NOTES.md` (10-23 on) and `ic_debug/LOG.md` (before).
