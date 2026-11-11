# Floquet crosstalk tests: tools, config branches and a bug in the 2D analysis

guan asked on 2026-11-07 (evening) for a wider repeat of jonginn's `JOB-20261107-00441`
(`FockPopulation2DExperiment`, S4-S7 swapped) and then left the overnight testing of the
crosstalk hypotheses to Claude, with a hard stop before the warm-up on 2026-11-08, 09:00.
The physics (numbers, figures, interpretation) is in guan's vault day files
`Lab/guan/AGP/2026/11/2026-11-07.md` and `2026-11-08.md`; this entry records only code,
tools and decisions.

## What was run and where the code is

- Script: `measurement_notebooks/guan/20261107_agp_2d_wide_detuning.py` (new, guan's
  sandbox; cells 1-13 are the code that went to the queue, cell 14 lists the module gaps).
- Kernels on the Lab server: `guan_2d_wide` (measurement, real station on the 00441 config
  versions) and `guan_2d_analysis` (mock station, only for `log_analysis`/`add_note`).
- Exploratory analysis scripts are in `.cache/crosstalk_20261107/` (git-ignored): damped-sine
  frequency per detuning, symmetric-pair residual, per-term/pair/fine analyses, and the
  one-photon model with an extra level. They are not modules; see "Module gaps".
- `tools/kernel_exec.py run --no-wait` cannot send to a busy kernel: the client waits for a
  kernel_info reply that a busy kernel sends only after the running cell, and the detached
  sender exits with code 3 ("did not send the code"). Workaround used: a background shell
  loop that waits for a marker line in the kernel log and then submits the next file.

## Config branches (guan's `measurement_notebooks/guan/branches.jsonl`)

Decision (guan, 2026-11-07): recalibrations for other gaps go only to new versions on their
own branches; no current version changes.

- `crosstalk_gap1_00441`: the 4 versions of `JOB-20261107-00441` (Floquet `CFG-FL-20261105-00030`).
- `crosstalk_gap10`: Floquet recalibrated at `scramble_sync_cycles=10` (`CFG-FL-20261108-00020`).
- `crosstalk_gap2`: Floquet recalibrated at `scramble_sync_cycles=2` (`CFG-FL-20261108-00040`).
- The station was checked out back to `crosstalk_gap1_00441` after each gap test.

The recalibration uses jonginn's `floquet_calibration.py` settings (Floquet error
amplification for frequency and gain, then `SidebandStarkAmplification` for the 12 pairwise
phases). Its postprocessors were given guards: a fit is accepted only inside its scan and with
a finite, small error. On zero (mock) data the unguarded phase postprocessor accepted a
meaningless value inside the window; the error condition fixes that.

## Code findings

- Bug: `FockPopulation2DExperiment.analyze()` takes the FFT peak over all bins above 0 to start
  the sine fit. The slow decay puts the largest amplitude in the lowest bin, so at large
  detuning (small oscillation) the start value is the decay bin and the sine fit fails; the
  one-hyperbola dispersion fit then fails or is meaningless. Seen in 00438/00439 and most points
  of the wide scans. Not fixed (no code edits without permission); fix proposed in cell 14.
- The same class fits one hyperbola. When an extra level couples in, that fit has no meaning;
  a symmetric-pair residual per |d| is the direct check.
- Changing `scramble_sync_cycles` needs a new Floquet calibration (frequency and pairwise
  phases): the cycle-averaged resonance and the per-slot phases depend on the gap.

## Module gaps (from cell 14)

- Fix the FFT start value in `FockPopulation2DExperiment.analyze()` (remove a slow background
  or search above a minimum frequency; fit a damped sine with a slow background).
- Move the guarded Floquet recalibration (cell 9) next to jonginn's calibration runners.
- If the extra-level model is kept: model into `theory/qsim/central_boson.py`, fits into
  `fitting/qsim/`.
