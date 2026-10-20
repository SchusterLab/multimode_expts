# Where things stand: automated calibration

**Last updated: 2026-10-20.** Theme: `experiments/autocal/` (automated replacement of the calibration notebooks).
Branch `autocal`, worktree `C:\python\multimode_expts_autocal`; merged into `main` (fast-forward) after each fix.
Design: `docs/autocal.md`. This file is overwritten at the end of each work session on this theme.

## State

- First device run done on 2026-10-19 after the cooldown, at 0.30 mA, phase by phase with fixes during the run
  (record: `docs/log/2026-10-20_cooldown-recal-night.md`). Final versions (update_main=False, main not set):
  `CFG-HW-20261019-00062`, storage `CFG-M1-20261019-00030`. 20 tests pass.
- seb has not reviewed the review pages yet (`C:\experiments°601_Transduction_sandboxutocaluns\`).

## What is next

1. seb: review the night, decide whether to set main to the final versions.
2. Fix the known limits in `docs/autocal.md` (gain scans with 20 pulses or a 2 % limit; averaged gain-to-alpha or a
   |2>-Ramsey scale check; two-frequency EF Ramsey check; a swap chevron check when the swap gain moves > 10 %).
3. A second full run from a cold start to measure the time and the number of stops.
