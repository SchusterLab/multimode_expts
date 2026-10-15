# 2026-10-08 (day): the channel coherence deficit located (transduction)

seb + Claude. Full record: `C:\experiments\260601_Transduction_sandbox\ic_debug\LOG.md` (10-08 sections). Scripts and
results in `ic_debug/losssplit/`: `losssplit.py` (cases `locate`, `locate2`, `samesess`, `phase01`), `analyze_locate*.py`,
`analyze_samesess.py`, `analyze_swaplen.py`, `analyze_phase01.py`, `sim_kerr_swap.py` (MODEL), result `.md` files and
`figs/` (also the drift figures `tseries_*.png`, `pulse_drift_*.png`, `ladder_vs_sb.png`).

## Done

- Reports for seb on the night data: drift figures (large fonts); the fn-gn+1 sideband drift (-60..-90 kHz overnight) is
  explained by a ~1-2 % rise of the sideband drive amplitude at the chip (measured Stark shift -3.05 MHz; shorter pi
  length); the qubit ladder gains follow at ~0.3 % (r = 0.92-0.96); the swap pump does not. Not the cause of the deficit.
- Stray result files of 10-03 (`budget_measured.md`, `eta_jitter.md`, `eta_shape2b.md`) moved from the repo root to
  `ic_debug/pool/`; four pool scripts now write next to themselves.
- Direct tests (cases 55-58, swap fixed at 3754 / 878.3159 MHz, Wigner with parity-count bootstrap errors).

## Found (measured)

- eta of the partial swap (S4 empty), photon-number Ramsey: 0.328-0.335 all day (nominal 0.35). A swap length scan finds no
  fixed extra time (L0 = -15 +/- 58 ns); the offset is within that scan's precision.
- env photon alone (no swap) 0.98 +/- 0.015; swap alone (S4 empty) 0.99-1.00 +/- 0.03; full channel |rho02| 0.853 +/- 0.028
  (-3.3 sigma vs the swap alone, same session), |rho13| 0.985 +/- 0.025.
- |0>+|1> input through the channel: |rho01| 0.766 +/- 0.051, |rho12| 0.981 +/- 0.017.
- In both channels the coherence whose amplitude has two interfering paths (env and M1 photons stay / exchange) is low;
  the single-path coherence is at the ideal. A random phase growing ~N^2 is disfavoured (~4 sigma); a static Kerr during the
  swap is disfavoured by the swap-alone test (MODEL).
- Readout: after this morning's QICK restart the IQ plane turned ~+158 deg; the run script now bootstraps the readout with
  a plain single shot when the active-reset one fails.

- **Env vacancy (evening, case 59, photon-number Ramsey):** the env photon is missing at the swap in p = 0.022 +/- 0.008
  of the shots (0.040 if the small n = 2, 3 fit floor is kept); the channel timing changes nothing. Because the two-path
  coherences have opposite signs for a Fock env and a vacuum env, this small vacancy removes ~2p (rho02) and ~4p (rho01).
  It explains ~1/3-2/3 of the two-path deficit; ~7-15 % remains (2-4 sigma).
- **Ramsey fit floor (case 60):** vacuum and |1>-alone references show ~0.01-0.015 at photon numbers that cannot be
  there, so the normalised vacancy is right: p = 0.020 +/- 0.008 (ordinary: prep + decay). It explains about a third of
  the two-path loss; ~10-16 % remains (3-3.5 sigma). Distinguishability (HOM visibility >= 0.98) and a random
  photon-number-sector phase (rho13 is ideal) are excluded for the remainder. MODEL: a random cross-Kerr during the swap
  (~20 kHz rms) fits five of six numbers but not |rho01| of the |0>+|1> channel.

## Corrections

- On 10-07 / 10-08 Claude dismissed an env vacancy because the vacuum-env |rho02| has the same SIZE; in a mixture the
  complex values add, and they have opposite signs. Corrected in the evening.

- The test 2 result first looked like a ~6 % deficit with the nominal eta; with the measured eta it is 1.00. The "~65 ns"
  extra swap time I quoted once was a slip: the estimate from the fixed-ramp model is ~160 ns.
- I proposed a HOM visibility test and a waiting-time test; seb pointed out both were done (10-02 to 10-06).

## Open (for seb)

- Reconcile the coherence result with the HOM populations (10-05: P1 .047 +/- .004 ~ P0 - P2 .058 +/- .007, read then as
  "missing photons, little distinguishability"). Then decide the next direct test.
- The main-config readout phase (703.40) is now ~160 deg off after the restart.
