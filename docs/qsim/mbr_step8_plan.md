# MBR redesign, step 8 plan: a usable MBR surface

Status: approved by guan 2026-09-25 (order A, B, D, C). 8A in progress. It uses
`mbr_redesign.md` for the rules and patterns. Where this file is silent, that file applies.

## 0. Goal and scope (guan, 2026-09-25)

"Usable" means:
- a few measurement and analysis notebooks with a clear structure and no free globals;
- if you follow the symbols from a notebook, you can see what is done;
- the classes follow the Experiment protocol (`acquire`, `analyze`, `display`);
- the code on the hot path is easy to read.

Out of scope: shared infra that is not qsim-specific (`MM_base`, ...), and the dormant and
deprecated code (cooling, dark mode, flux excursion, ...). Exception: if such code is coupled
into the MBR base, it moves out of the MBR path.

## 1. Chunks and order

| Chunk | What | Net |
|---|---|---|
| 8A | The MBR base: remove the dark-mode chain from the MBR program and experiment | ASM golden, mock acquire |
| 8B | Matrix Pencil and `MBRSpectrumExperiment`: readable steps, SFF out | analysis golden (needs the data mount) |
| 8D | Measurement notebooks: single elements and propagator through class `acquire` | dryrun tool |
| 8C | Analysis notebook: a short canonical path; the rest to `dormant/` or an audit notebook | analysis suite |

8C waits for the answers in section 4.

## 2. 8A: facts (from the survey)

The MRO of `MBRRamseyProgram` today:
`SidebandScrambleProgram -> DarkBaseProgram -> DarkModeEncoding, FloquetTrain,
ManipulateModePulses -> QsimBaseProgram -> MMAveragerProgram`.

What MBR uses from it:
- `DarkBaseProgram.initialize` (MM base init, swap dataset, Floquet pulse setup). It also
  registers the displacement pulse if `perform_wigner` or `init_alpha` is set. MBR rejects
  `perform_wigner` but not `init_alpha`.
- `ManipulateModePulses.man_reset`. It overrides `MM_base.man_reset`, and MBR's active reset
  plays it. **The new base must keep this reset** (`test_pulse_layer_layout.py` records the
  same trap for `DarkBaseRProgram`).
- `FloquetTrain`: `_play_scramble_with_phase_offsets` (about 370 lines),
  `_play_closed_floquet_cycle_pairs`, the phase-offset helpers, `calculate_floquet_cycle_us`.
- `QsimBaseProgram`: `retrieve_swap_parameters`, `_initialize_floquet_pulses`.

What MBR does not use: `SidebandScrambleProgram` (only its `core_pulses`, which MBR never
calls; the `MBRRamseyProgram` docstring is wrong about this), `DarkModeEncoding`, the other
`ManipulateModePulses` methods, and `DarkBaseProgram.body`.

The experiment side: the three job classes inherit `DarkBaseExperiment.acquire`, a generic
1D/2D sweep with parity-check and normalize branches that MBR never sets.

Live options that no canonical job uses:
- all three jobs set `floquet_hardware_loop=False` (StarkCal refuses it), but `mbr_defaults`
  sets it to `True`;
- all three jobs use `spectroscopy_phase_correction_mode="final_analyzer"`. The `decoder` mode
  and `decoder_phase_matrix` are used only by the deprecated `NPhotonHamiltonianSpectroscopyProgram`;
- `palindrome_scramble` is `False` everywhere; `storage_phase_matrix` is "not really being
  used" (its own docstring).

The ASM golden (16 programs, two pinned config sets) covers active reset, man and storage
reset, the gauss and preload envelopes, and the software Floquet loop. It does not cover the
hardware loop or the `decoder` mode.

## 3. 8A: steps

1. **8A1, the base.** `MBRRamseyProgram(FloquetTrain, QsimBaseProgram)` with its own
   `initialize`, and `man_reset = ManipulateModePulses.man_reset` set explicitly, with the
   reason. `init_alpha` is rejected like `perform_wigner`. A new `MBRJobExperiment` base with a
   short `acquire` (outer sweep x `ramsey_phase`), and the three job classes on it.
   `readout_lane_count` moves to `qsim_base` (dark code imports it from there). Net: ASM
   golden byte-identical; a test that each MBR program resolves every method it calls to the
   same function as before.
2. **8A2, the Floquet playback.** Split `_play_scramble_with_phase_offsets` into named steps
   (checks, pulse arguments, software loop, hardware loop). First add golden programs for the
   hardware loop, because the golden does not cover it now. Net: ASM golden byte-identical.
3. **8A3, small fixes.** Correct the docstrings; `mbr_defaults` agrees with the jobs on
   `floquet_hardware_loop`.

8A does not remove the unused options (section 2). That is question 4.1.

Found in 8A (for the `notebook_helpers` cleanup): `notebook_helpers/mbr_campaign.build_campaign`
has its own copy of the MBR defaults. It duplicates `mbr_campaign.mbr_defaults` (whose docstring
says nothing else defines these keys), and it sets `floquet_waveform`, which `mbr_defaults`
leaves out on purpose. The notebooks use this copy. Merge the two.

Found in 8A1: the ASM golden does not see which `man_reset` the MBR programs play (with the
pinned configs, both compile to the same program). `test_pulse_layer_layout.py` holds it now.

## 4. Questions

1. (guan) Remove the options that no canonical job uses (`decoder` mode and
   `decoder_phase_matrix`, `palindrome_scramble`, `storage_phase_matrix`, the hardware loop)
   from the MBR path? The deprecated `NPhotonHamiltonianSpectroscopyProgram` then stops
   working (decision 7-1: moved code is not maintained). Proposal: yes, after 8A.
2. (guan) Remove `analyze_sff` and `display_sff` from `MBRSpectrumExperiment`, to agree with
   decision 7-3? Proposal: yes, in 8B.
3. (guan, jonginn) 8C, the canonical analysis notebook. Proposal: keep load manifest ->
   `analyze` -> `display` for the calibration set, the spectrum and disorder, and the self-Kerr
   fit as a class method. Move out the N=2 raw reprocessing, the FFT / peak-finder / MPM
   comparisons, the report replots, and the Aug 15-17 reproduction (keep that one usable until
   the physics audit ends).
