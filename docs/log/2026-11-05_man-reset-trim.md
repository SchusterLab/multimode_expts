# 2026-11-05: M1 reset: 3 dump pulses at the measured lines (was 6); one copy of man_reset

Who: guan with Claude, kernels `guan_agp_meas` / `guan_agp_plot` (Lab server, port 8765).
Scripts: `measurement_notebooks/guan/20261105_agp_highW.py` cells 5-15,
`analysis_notebooks/guan/agp/20261105_agp_plots.py` cells 7-12. Device numbers and job IDs:
vault `guan/AGP/2026/11/2026-11-05.md`, notes under JOB-20261105-00231 and -00247.

## Change (guan's decision)

- `MM_base.man_reset` with `chi_dressed`: one M-D2 dump pulse per qubit state, offsets from
  `MM_base.man_reset_offsets_mhz(cfg)` = -chi_f, -chi_ge, 0 (f, e, g: the order of the qubit
  decay), chi_f = chi_ge + chi_ef. `cfg.expt.dump_reset_iter_num` repeats each pulse (was only in
  the qsim copy).
- The copies `QsimProgram.man_reset` and `t2_cavity_fluxexcursion.SidebandScrambleDarkProgram.man_reset`
  are deleted; both programs use `MM_base.man_reset`.
- Test `tests/test_man_reset_offsets.py`. The 24 program goldens with an active reset are made
  again; with the channel-1 frequencies masked, each new listing is the old one minus the
  removed pulses.
- Per shot of a FockPopulation job: 1.166 -> 0.94 ms (the first M1 reset 480 -> 240 us).

## Why: what the old list was

- Before: offsets n * chi, n = 0, 1, 2, chi in (chi_ge, chi_ge + chi_ef): six pulses. c5fbacd
  (guan, 2025-06) ported this from a config list, with the comments "what does n=2 do?", "minus
  sign correct?". In 2026-02 (bf5) the sign was "+ n chi" for the M1-D1 dump ("works when M1D1
  freq goes down"); then the default dump became D2, which is above M1, so the sign is wrong for
  D2. The old list worked only because 0 (twice) and +0.93 MHz are about 0.3 MHz from the e and
  f dump lines and remove most of the photon.
- `device.manipulate.chi_ge` and `chi_ef` are read only by the reset. `chi_ef` was 1.231 MHz in
  every config since 2026-01-15 and was never measured.

## Steps and checks (photon left in M1 after the reset, qubit in g / e / f)

Check: `ManActiveResetVerificationProgram`, prepare M1 |1>, qubit in g, e or f, reset, map a
left photon to qubit e. New options: `prep_man_photons` (0 for the floor), `pre_reset_pulse`.

| Pulse list | g / e / f |
|---|---|
| old six pulses | -0.2 / 0.8 / 0.8 % |
| 0, +chi_ge, +chi_f (measured chi) | 0.1 / 7.0 / 6.7 % |
| 0, chi_ge, 2 chi_ge (old config chi_ge) | -0.1 / 5.5 / 5.8 % |
| 0, -chi_ge, -chi_f (guan: D2 above M1) | -0.1 / 0.9 / 2.8 % |
| -chi_f, -chi_ge, 0 (adopted) | -0.3 / 0.6 / 1.9 % |

Errors about 0.3 %; the floor changes by about 0.4 % from run to run. The f case is still about
1 % worse than the old list; the f line may be off -chi_f (the f_n-g_n+1 table gives chi_f about
-0.571 against -0.593 MHz from qubit spectroscopy). Before a normal shot the qubit is in g or e,
so f matters least.

## Rollout

- The offsets need measured chi in the config: `CFG-HW-20261105-00040` (jonginn's 00039 with
  chi_ge -0.3175, chi_ef -0.2759 MHz). guan sends it to jonginn. Configs with the legacy
  chi_ef = 1.231 put the f pulse at -0.93 MHz, at no line (e and g are still right).
- The main hardware-config pointer is not changed.

## Open

- A dump-line spectroscopy (one M1-D2 pulse at a swept offset, qubit in g, e, f) would give the
  f line and the line widths; then the f pulse can move.
- Calibration code that measures chi (`guan/single_qubit_autocalibrate_v2.py` `update_chi`,
  `update_chi_ef`, by Ramsey) should write both fields; its Ramsey sign can flip if the shift is
  larger than the Ramsey detuning.
