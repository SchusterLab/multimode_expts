# 2026-10-28/29: shorter qubit pulses, f1 -> g conversion at 0.30 mA, Ic night 10-28/29 (seb + Claude)

Sandbox: `C:\experiments\260601_Transduction_sandbox\coherence\`. Step-by-step notes: `night1023/NOTES.md`.
MODEL = calculation; everything else is measured.

## Why

Item 3 of the status (the |2>-branch prep loss) and objective 1 (does the Ic repeat between nights).

## Done

- `decay_budget_1028.py` (MODEL, measured T1 / T2): decoherence explains ~1.2 % of the 3-4 % |2>-ladder loss.
- `sigma_scan_1028.py`: e0-f0 / e1-f1 and g0-e0 / g1-e1 at sigma 35 / 20 / 12.5 ns, 1D error amplification in gain and
  frequency (2 passes), pulse trains with a 4-class read (leakage = 'far'), E1 Ramsey + Wigner per variant.
  `fit_map2d_1028.py`: the 2D gain x frequency maps, fitted (seb: no single-pixel optimum); same optimum as the 1D EA.
- `hold_f_1028.py` (seb, after Guan's 0.45 mA result, `2026-10-28_fock-prep-f-decay.md`): hold |f,0> and |f,1> 0-20 us at
  0.30 mA, then a photon-number Ramsey of the qubit-g shots.
- Snapshot `CFG-HW-20261028-00170` (seb: "update our local config with the best possible parameter"; 35 ns rungs from the
  10-28 EA; main config not changed). Night `night1028.py` 22:26 - 04:52 from it (seb's overnight plan).

## Found

- Shorter ef pulses: cleaner per pulse, no leakage, no E1 gain. Shorter ge pulses (20 ns): no significant E1 gain
  (P2 +0.9 +/- 0.7 %). At 12.5 ns the ge line moves > +2 MHz: not calibrated. Strong pulses need ~4-9 % less gain than
  1/sigma (cause not known). Not adopted: Claude kept 35 ns for the snapshot (no gain shown). seb: skip the 2D map when
  it agrees with the 1D EA.
- The floor of a same-phase odd-m pi train holds the single-pulse frequency error (it does not accumulate). My earlier
  reading ("detuning accumulates" at 35 ns) was wrong. A pi(0) pi(180) check is postponed.
- f1 -> g at 0.30 mA: 1.6-1.9 %/us (Guan at 0.45 mA: ~6x faster); f0 does not convert. Goes to g0 ~70 %, g1 ~30 %, not g2.
  It explains the e1-f1 train g channel in size. MODEL ladder cost: +0.3-0.4 % P0, +0.1-0.2 % P1. Physics not known.
- Night 10-28/29: Ic A 0.114 +/- 0.026 (3 sets), B 0.124 +/- 0.024 (4 sets). No other user; coupler fixed at 0.30 mA.
  Spell tags, sentinels and icref all normal.
- Three nights: B 0.117 +/- 0.015, nights agree (chi2 0.5 / 2), 11 sets consistent. A 0.134 +/- 0.012 but the sets scatter
  beyond their errors (10 sets chi2 21.1 / 9; set 17 tonight 0.002, set 22 0.172) also without coupler ramps.
  Per set the Ic follows the output coherence (partly by construction); A's coherence scatters 2x more than B's.

## Next (proposed, not decided)

- A's scatter: check the output phase / coherence inside one set (split the shots in time), A vs B.
- f1 -> g: rate vs coupler setpoint; less time in f1.
