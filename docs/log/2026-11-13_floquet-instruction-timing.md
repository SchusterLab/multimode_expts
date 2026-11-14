# 2026-11-13: tProc timing check for Floquet trains

(Claude, with the user, branch `floquet-instruction-timing`, worktree
`C:\python\multimode_expts_floquet_timing`.)

## Why

With short Floquet pulses and a small sync, the tProc can need more clocks to
issue the next pulse than the slot lasts. It then falls behind: the timed FIFO
empties, the next `set` arrives after its time and plays at once, so the pulse
starts late. The train then runs longer than `calculate_floquet_cycle_us`
says, and nothing reported it.

## What was done

- New module `experiments/floquet_instruction_timing.py`. After a Floquet
  train is emitted, `warn_if_tproc_falls_behind_floquet_train(program,
  train_name, first_instruction_index)` reads the instructions the train added
  to `program.prog_list`, runs through them in tProc order (it follows each
  `loopnz` through its counter), cuts them into pulse slots (from one `set` to
  the next), and keeps a running balance of how far the tProc is ahead of the
  schedule, limited by the timed FIFO depth. It warns if a pulse starts late,
  with the smallest extra sync that removes the lateness.
- The clocks per instruction and the FIFO depth are read from the qick
  firmware source in the `../qick` checkout (`firmware/ip/axis_tproc64x32_x8_v1/src/`):
  `ctrl.sv` (control state machine: one clock per state, from `DECODE_ST` to
  the next `DECODE_ST`) and `tproc64x32_x8.v` (parameter `N` of
  `fifo_time_i`). The opcodes come from `QickProgram.instructions`. No clock
  numbers are typed into the code.
- Called in two places, after the train:
  `FloquetProgram._play_scramble_with_phase_offsets` and
  `FloquetProgram._play_m1s_frac_train` (`experiments/qsim/floquet_train.py`).
  The check only reads `prog_list`; the compiled programs do not change.
- Tests: `tests/test_floquet_instruction_timing.py` (firmware reader, slots,
  loops, time register, FIFO head start, and a real FockPopulationProgram on the
  mock station with short pulses).

## Decisions (user)

- Check only Floquet trains, not other pulses.
- Read every number from the firmware files; no typed constants.
- Readable code: short functions, names that say what they do, no extra
  checks where Python, numpy or qick already raise.
- A warning only; nothing is saved in the HDF5 files.
- Only in `floquet_train.py`. A call in `ErrorAmplificationProgram.body` was written, then
  taken out by the user: not wanted there.
- Put it in the main worktree the same day, while the AGP scans run (a one-time request, not a
  change of the branch rules).

## Found

- Clocks per instruction (from `ctrl.sv`): `regwi` 2, `synci` 3, `set` 4,
  `mathi`/`math`/`bitwi` 5, `loopnz` 6 when it jumps and 3 when it exits.
- Clocks to issue one Floquet pulse:
  - hardware loop (`_play_floquet_hardware_loop`): 12, plus 6 per cycle for the jump;
  - software loop with the preloaded bank: 9 when the phase register value is
    below 2**30 (phase below 90 deg), else 14 or 19, because qick's
    `safe_regwi` splits larger values into two or three instructions;
  - `_play_m1s_frac_train` hardware loop: 25 (one pulse per loop pass);
  - error amplification: 9.
- `FLOQUET_DEFAULTS` (`experiments/qsim/notebook_helpers/defaults.py`) sets
  `floquet_hardware_loop=True`, so notebooks that take these defaults use the
  hardware loop.
- Programs with their own Floquet loop (for example
  `sideband_scramble.SidebandScrambleProgram`) do not go through
  `floquet_train.py` and are not checked.

## Limits

- A model of the firmware source, not a measurement of the board. It assumes
  that the FPGA image was built from that source.
- The balance starts at 0 (the tProc just on time at the first pulse), so the
  lateness it gives is an upper bound for a train that starts with a head
  start.
- One warning per compiled program: a job that compiles one program per sweep
  point gives one warning per point.
