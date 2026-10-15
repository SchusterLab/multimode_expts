# 2026-10-07: drift check, swap-fit fix, swap-gain scan (transduction)

seb + Claude. Full record: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md` (10-07 sections); scripts and
results in `ic_debug/losssplit/` (`losssplit.py`, `analyze_night1007.py`, `night1007.md`, `night1007.png`).

## Done

- **configs/versions restore** (seb, morning): checked 18 649 / 18 666 OK; 9 missing (`CFG-HW-20261006-00108..00116`);
  8 old checksum mismatches (Jan-May 2026). A new station loads the main configs.
- **Drift check** (seb asked; `losssplit.py --recal-only`). The first try failed: the worker had lost its Pyro link to the
  board; seb restarted the worker. Result vs 10-06 21:37: readout angle jumped ~37 deg (probably a QICK server restart);
  sidebands +18-19 kHz; qubit / M1 f_ge, ladder gains stable; ladder EA frequencies scatter +-20 kHz (fit scatter).
- **Swap-calibration bug** (found at the start of the scan): the EA product-peak fit gave a false 878.5145 MHz (raw data:
  878.318). The peak is ~3 kHz wide and falls between points at a 4-8 kHz step. Fixed in `losssplit.py` (sandbox, not
  common code): mirror-symmetry center as the coarse value, and a fine scan that must agree with it. A second bug (Stark
  seed slope from two points at the same gain) was fixed too. seb stopped the bad runs; nothing bad was measured.
- **Swap-gain scan (plan C of the 10-06/07 night)**: 9 eta 0.35 sets, gain x0.5-x1.5 (length 4.225 us / f), reference
  interleaved, own swap calibration per set, full recalibration (with single shot) every 2 h.

## Found (measured)

- No measurable dependence of the Ic gap on the swap gain: reference (5 sets) mean -0.076, others (4) -0.114, ~1 sigma.
- The reference sets alone scatter -0.006 to -0.167 (std 0.076, error 0.042), with a stable swap calibration (gain
  3750-3761, freq within 1 kHz). The gap follows the output coherence (|rho02|, |rho13|) again.
- Swap at 0.3 mA: Stark slope ~1.2e-9 MHz / gain^2; gain x length constant within ~1 %.
- Correction: at 19:18 I (Claude) reported "the reference is better than any other gain, 3.5 sigma" from 3 reference
  sets. The next two reference sets falsified it.

## Decisions

- seb: full 4-input Wigner sets (not the reduced set); load the 14:10 recalibration instead of a new start recalibration.
- Open for seb: readout into the main config; the 9 missing version files; whether the product-fit issue should be fixed
  in common code.
