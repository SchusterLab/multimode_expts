# AGP ensemble display appearance

Jonginn supplied the earlier AGP scan figure as the visual reference. The agent
had incorrectly identified the newer `fitting.plot_style` appearance as matching
that figure before checking the actual display code. The reference instead comes
from `CentralBosonAGPDisorderScan.display()` with the original global 14-point font.

## Changes

- `experiments/qsim/agp_ensemble.py`: scalar scan display uses a local Matplotlib
  default context with a 14-point base font and the reference marker/line styles.
- `experiments/qsim/agp_sequential.py`: the sequential scan's measured markers,
  theory lines, legend and panel height match that reference.
- `experiments/qsim/agp_sequential_ensemble.py`: the delegated diagonal scan uses
  the reference context; KK/KW/WW metric panels are arranged side by side, one row
  per cutoff, preserving returned axes indexed by `(component, cutoff)`.

The user's supplied reference takes precedence over the shared plotting-style
default for these displays. Calibration and walk style changes attempted earlier
in the session were withdrawn. Existing notebook edits were preserved. No estimator,
saved arrays, theory calculation, measurement setting or remote file was changed.

## Validation

- 28 existing tests pass in `test_agp_ensemble.py`, `test_agp_sequential.py` and
  `test_agp_sequential_ensemble.py`.
- Seven renders using explicitly synthetic arrays cover single/multiple cutoffs,
  diagonal and metric displays, and theory disabled. White backgrounds, visible
  spines, no grid, fonts, blue measurement lines, linear axes and restoration of
  ambient rc settings were checked; scalar and metric PNGs were visually inspected.
- `git diff --check` passes. No staging, commit or deployment was performed.
