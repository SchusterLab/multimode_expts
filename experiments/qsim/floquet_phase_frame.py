# -*- coding: utf-8 -*-
"""The rotating-frame phase ledger a Floquet pulse train carries along.

What this is
------------
Every calibrated M1-Sx swap pulse shifts the phase of the *other* storage
modes -- an off-resonant Stark shift, measured once per mode pair and stored
in the Floquet swap dataset. A pulse train is therefore played against a
running ledger: emit a pulse at the current phase of the mode being driven,
then advance every mode's entry by that pulse's column of the calibration.
Load, scramble and read all share one ledger, and a dark-mode readout is only
correct if the phase it consumes is the phase the scramble left behind.

Why it is a module of functions
-------------------------------
The ledger arithmetic is pure: a list of degrees, a calibration source and
which mode was pulsed. It needs no tProc, no station and no config. Pulled
out of the program base it becomes checkable on its own -- which matters more
here than anywhere else in the pulse layer, because a wrong phase is exactly
the failure that still compiles, still acquires, and quietly produces the
wrong spectrum.

Three separate ledgers exist, and they are not interchangeable:

* the **Floquet** ledger, advanced from the dataset's pairwise
  ``phase_from_*`` columns. It skips the mode just pulsed -- that mode's own
  phase is carried by the pulse itself, not by the ledger.
* the **ds_storage** ledger, advanced from an in-memory matrix that *may*
  have a calibrated diagonal, so the pulsed mode is included.
* the **decoder** ledger, one entry per decoder control axis (M1 photon
  lowering first, then the storage axes), advanced from the directional
  decoder matrix.

Mixing them is a physics error, not a style one: the decoder matrix is
directional and must never be applied to the Floquet ledger.
"""


def mod360(phase_deg):
    """Wrap a phase into [0, 360). One definition, so every site agrees."""
    return phase_deg % 360.0


def advance_floquet_offsets(phase_offsets, swap_stors, pulsed_stor, swap_ds):
    """Advance the Floquet ledger in place after pulsing ``pulsed_stor``.

    Uses only the Floquet-to-Floquet pairwise calibration:

        phase[stor_B] += swap_ds.get_phase_from("M1-S{stor_B}", "M1-S{pulsed}")

    The mode just pulsed is skipped. The directional decoder matrix must
    never be used here; see ``advance_matrix_offsets``.
    """
    pulsed_name = f"M1-S{pulsed_stor}"

    for j_stor, stor_B in enumerate(swap_stors):
        if stor_B == pulsed_stor:
            continue

        phase_shift = swap_ds.get_phase_from(f"M1-S{stor_B}", pulsed_name)
        phase_offsets[j_stor] = mod360(phase_offsets[j_stor] + phase_shift)


def advance_matrix_offsets(offsets, matrix, pulsed_column):
    """Advance a ledger in place by one column of a phase matrix.

    ``matrix[i, j]`` is the phase added to entry ``i`` when the pulse in
    column ``j`` plays. Every entry is advanced, the pulsed one included:
    both matrix ledgers (ds_storage, decoder) may carry a calibrated
    diagonal, unlike the Floquet ledger above.
    """
    for affected_index in range(len(offsets)):
        offsets[affected_index] = mod360(
            offsets[affected_index] + matrix[affected_index, pulsed_column])


def detuning_phase_deg(detuning_MHz, elapsed_us):
    """Rotating-frame phase a detuned mode accumulates over an interval.

    MHz * us is cycles, so degrees is 360 * that. This is how synthetic
    disorder enters the ledger: the detuning is applied to the pulse
    frequency during the scramble, and the phase it accrued is what the
    analyzer pulses afterwards have to be referred to.
    """
    return 360.0 * detuning_MHz * elapsed_us
