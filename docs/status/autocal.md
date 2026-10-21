# Where things stand: automated calibration

**Last updated: 2026-10-20.** Theme: `experiments/autocal/` (automated replacement of the calibration notebooks).
Branch `autocal`, worktree `C:\python\multimode_expts_autocal`; merged into `main` (fast-forward) after each fix.
Design: `docs/autocal.md`. This file is overwritten at the end of each work session on this theme.

## State

- First full device run 2026-10-19 after the cooldown (0.30 mA): `CFG-HW-20261019-00062`, storage
  `CFG-M1-20261019-00030` (record: `docs/log/2026-10-20_cooldown-recal-night.md`).
- 10-20 additions (seb ok), on main, 25 tests pass: N = 1 sideband map, active-reset single shot before gain to alpha,
  gain to alpha B' (41 gains 0-4480, 2000 reps x 1 round). Device results: `docs/log/2026-10-20_sidebands-g2a.md`.
- Latest versions (update_main=False): `CFG-HW-20261020-00034` (B' gain to alpha 2.558e-4, N = 1 sidebands, photon-1
  rungs 15:50), storage `CFG-M1-20261020-00010`. The best |2> preparation uses the 10-19 sideband values
  (f0-g1 1989.6315, f1-g2 1989.0730, lengths 0.5599 / 0.3952), not the N = 1 ones; the 10-20/21 night runs with them
  (losssplit `--sb`). Main is not set (seb decides).

## What is next

1. Replace the sideband frequency rule (N = 1 centre) by a final-state optimisation (P2 of |2>; grid or fit, see the
   known limits in `docs/autocal.md`). Ask seb first.
2. seb: set main (hardware = 20-00034 with the 10-19 sideband values; storage 20-00010).
3. Still open from 10-19: storage gain scans with 20 pulses or a 2 % limit; two-frequency EF Ramsey check; swap
   chevron check when the swap gain moves > 10 %.
4. Shared code, reported to seb, not changed: `ParityGainExperiment` fails with rounds > 1 and active reset
   (reshape error); the `MM_base.custom_pulse` docstring lists channel ids the code does not use (it takes the DAC
   channel).
