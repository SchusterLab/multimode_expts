# Where things stand: automated calibration

**Last updated: 2026-10-19.** Theme: `experiments/autocal/` (automated replacement of the calibration notebooks).
Branch `autocal`, worktree `C:\python\multimode_expts_autocal`. Design: `docs/autocal.md`. This file is overwritten at
the end of each work session on this theme.

## State

- First version written and tested on the mock station (11 tests in `tests/test_autocal.py` pass). Not run on the
  device yet. Not merged into `main`.

## What is next

1. seb: review, then OK to merge `autocal` into `main` (needed for a device run: the station takes its config paths
   from the checkout it is imported from, and the worker uses the main checkout).
2. First device run after the 2026-10 cooldown, at the transduction set point (coupler 0.30 mA), seeds:
   `CFG-HW-20261019-00001` (qubit / readout), `seed_sections` from `CFG-HW-20261012-00075` (manipulate, multiphoton),
   storage dataset `CFG-M1-20261012-00012`. Phase by phase, seb reviews `index.html` after each phase.
3. Tune the check thresholds on real data (they are set from synthetic data and the notebook scan sizes).
