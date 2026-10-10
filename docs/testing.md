Status: current (2026-09-27). Guidance for what the tests in this repo are for.

# What we test, and why

A measurement result is correct only if every link in this chain is correct. Each
test should serve one link, directly or as a proxy. A test that serves no link does
not belong in the suite.

| # | Link | What "correct" means | Direct test | Good proxy |
|---|---|---|---|---|
| 1 | intent → program | The program plays the pulses we meant: transitions, frequencies, phases, timing, order. | Only the device: the calibration converges, the fringes sit at the expected detuning, and so on. | Property tests on the pulse schedule (phase ledgers, cycle timing) with values found independently of the code. |
| 2 | program → data | The device runs the program, and the data is saved with enough information (config versions, parameters) to analyze it again. | Only the device. | Mock acquire: the program compiles with the qick checks, and the saved file reads back with the right shapes and provenance. |
| 3 | data → numbers | The analysis turns raw data into the right physical numbers: fits, phase corrections, reconstruction. | Synthetic data with a known answer, or exact identities (round trips, invariances). | Reproduction of a result that was checked independently of this code. |
| 4 | numbers → state | Calibration write-back puts the right number in the right place, and never a bad one (NaN, wrong type, large jump). | Give a known fit result to the hook, then check the Station value. | Guards at the write functions (`ds_*` updates, config sets), tested once; a before/after table of the calibration state in the hardware report. |
| 5 | workflow | A person can run the notebook on the device, top to bottom, and switch hooks off by hand. | The hardware suite. | Mock dry run of the notebook as it is, hooks included; static checks (imports resolve, no undefined names). |
| 6 | reproducibility | Saved data gives the same numbers when it is analyzed again offline. | Analyze stored data again and compare with stored numbers. | — |

## Notes

- **Stored numbers and pinned ASM test "no change", not "correct".** They are the
  direct test of link 6, and of no unintended change during a refactor. They are not
  evidence for links 1 or 3. Keep them strict (a change fails the test). On an
  intended change, regenerate them in the same commit and review the diff.
- **Links 1, 2 and 4 are fully checked only on the device.** Everything offline is a
  proxy for them. So the hardware suite must be cheap to run and cheap to read: one
  report with the plots, the calibration state before and after, and a few flags (fit
  converged, change larger than expected). A person reads it; it asserts no numbers.
- **Pre/post-processor hooks stay in the notebook** and stay short. Test them where
  they are (the mock dry run), not as copies in a test file. Put guards where they
  write, not in each hook.
