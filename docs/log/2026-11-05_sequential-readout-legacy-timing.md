# Restore the legacy multiparity measurement timing

The user requested preservation of the empirically optimized legacy measurement
timing. A read-only comparison found that the new intermediate second-bit readout
omitted `sync_all(10)` and moved readout-register initialization after the first
Ramsey. These differences are removed in the local sequential source module.

- Initialize the readout and threshold registers before the first Ramsey, in the
  same instruction order as legacy multiparity.
- Restore `sync_all(10)` immediately before every second-bit measurement, including
  intermediate readout modes. Remove its former final-mode-only placement.
- Preserve the existing feedback waits, pulse settings, and one terminal relaxation.
- Strengthen the existing pulse tests so both measurement boundaries are checked
  for single- and multi-mode readout, in normal and fast pulse modes.

All 31 tests across the three new sequential test modules pass. Additional local
mock traces at depths 54, 58 and 60, with one and five readout modes, verify that
the machine-code prefix through the first M1's second measurement matches legacy
through its ADC wait. Multi-mode execution differs afterward to reset the qubit
and continue reading; single-mode terminal measurement remains identical.

No existing base module was edited. No remote file, device setting, measurement
JOB or vault note was changed. The pippin inspection remained read-only; the fix
is in the shared local checkout. Hardware results are not recorded in this log.
