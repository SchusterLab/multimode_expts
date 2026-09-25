# MBR redesign, step 8 plan: a usable MBR surface

Status: approved by guan 2026-09-25 (order A, B, D, C). 8A, 8B and 8D done; 8C waits for question 3. It uses
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
| 8B | Matrix Pencil and `MBRSpectrumExperiment`: readable steps | analysis golden (needs the data mount) |
| 8D | Measurement notebook `mbr.py`: section 4 through the spectrum class | dryrun tool |
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

4. **8A4, the old phase correction out** (decision 4.1). `MBRRamseyProgram` removes the AC
   Stark phase only on the final half-pi. It refuses `'decoder'` mode, `decoder_phase_matrix`
   and `storage_phase_matrix`. `FloquetTrain` loses `decoder_phase_offsets`.
   `floquet_phase_calibration.py` (the programs that measured the matrix) moves to
   `deprecated/`. Palindrome stays. Net: ASM golden byte-identical; a test that the removed
   options are refused.

Found in 8A (for the `notebook_helpers` cleanup): `notebook_helpers/mbr_campaign.build_campaign`
has its own copy of the MBR defaults. It duplicates `mbr_campaign.mbr_defaults` (whose docstring
says nothing else defines these keys), and it sets `floquet_waveform`, which `mbr_defaults`
leaves out on purpose. The notebooks use this copy. Merge the two.

Found in 8A1: the ASM golden does not see which `man_reset` the MBR programs play (with the
pinned configs, both compile to the same program). `test_pulse_layer_layout.py` holds it now.

## 4. Questions

1. **Decided (guan, 2026-09-25).** The `'decoder'` mode and `storage_phase_matrix` are from
   the per-pulse AC Stark frame tracking. With Kerr, that frame and the final-half-pi frame
   differ by a gauge. We chose the final half-pi because it was easier than fixing the
   per-pulse one. So they go to `deprecated/` (done in 8A4). **Palindrome stays**: it is a
   symmetric (second-order) Trotter step, and MBR's free evolution is a Trotter train. The
   hardware loop stays in `FloquetTrain` (dark-mode code uses it).
2. **Decided (guan, 2026-09-25): keep SFF.** `MBRSpectrumExperiment.analyze_sff` is
   |sum_n A_n(t) / D|^2 over the complete fixed-N basis. That is the direct measurement; the
   spectrum is its FFT. Decision 7-3 was about the other SFF, the disorder-ensemble
   `DisorderSFFExperiment`. 8B makes `analyze_sff` readable.
3. (guan, jonginn) 8C, the canonical analysis notebook. Proposal: keep load manifest ->
   `analyze` -> `display` for the calibration set, the spectrum and disorder, and the self-Kerr
   fit as a class method. Move out the N=2 raw reprocessing, the FFT / peak-finder / MPM
   comparisons, the report replots, and the Aug 15-17 reproduction (keep that one usable until
   the physics audit ends).

## 5. 8B: what was done

- `fitting/qsim/matrix_pencil.py`: the two long functions are now the five steps of the
  module docstring, one function each; one `_FrequencyCircle` for the modulo-sampling
  arithmetic (it was defined twice, inline); one `_row_settings` for the checks both
  entry points share; `_MergeTolerance` for the two cross-row merge modes; one
  `_windowed_spectrum` for the FFT that the global fit and `refit_occupation` share.
  Same results: the two MPM goldens match at 1e-12; a temporary harness compared the
  new code with the old on 1032 synthetic cases over every option (rtol 1e-10, because
  threaded BLAS in `lstsq` is not bit-reproducible run to run).
- `MBRSpectrumExperiment.analyze`: the calibration merge-tolerance block is now
  `_matrix_pencil_merge_options`, with its own test (no golden uses that mode).
- `analyze_sff`: the docstring says what it is (question 4.2).

## 6. 8D: what was done

- `mbr.py` section 4 ("single matrix elements", hand-built TimeTrace jobs) is now
  "selected occupations": an `MBRSpectrumExperiment` over a subset. Only diagonal traces
  are canonical (decision 7-2).
- Section 5 (propagator) stays as it is: `MBRHamTomoExperiment` has no `acquire` on
  purpose (each part takes hours, with decisions between parts; see its docstring). The
  survey called this a gap; it is the design.
- `mbr_tomography.py` and `mbr_disorder.py` still call the `notebook_helpers` planners.
  That is the `notebook_helpers` promotion work, not 8D.
