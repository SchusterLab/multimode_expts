"""
Fock-state decoder building blocks (gate-based pulse sequences).

These are the qubit<->cavity transduction primitives for the lossy-beamsplitter
decoder. They emit gate-based pulse lists (the ``[channel, transition, pulse,
phase]`` form consumed by ``get_prepulse_creator``), so they compose with
``StateTomography1QExperiment`` and (later) the dedicated decoder experiment.

Conventions
-----------
A gate entry is ``[channel_name, transition, pulse_name, phase_deg]``.
The transduction encoder E maps

    (a|g> + b|e>) (x) |0>_cav   -->   |g> (x) (a|0> + b|2>)_cav

and the decoder D = E^dagger inverts it. E is derived from the lab's validated
``prep_fock_state([0, 2])`` shelving ladder with its leading input rotation
removed, so it is *state-independent* (acts the same on any input qubit state).
"""

import numpy as np


def revert_pulse_seq(seq):
    """Inverse of a gate-based pulse sequence.

    Reverses the gate order and adds 180 deg to each phase. For rotation/swap
    gates this is the exact inverse, since R(theta, phi)^-1 = R(theta, phi+180)
    and (R1 R2 ... Rn)^-1 = Rn^-1 ... R2^-1 R1^-1.

    Parameters
    ----------
    seq : list of [channel, transition, pulse, phase] gate entries.

    Returns
    -------
    list
        New list (inputs not mutated) with reversed order and phases + 180.
    """
    out = []
    for gate in reversed(seq):
        gate = list(gate)
        gate[3] = (gate[3] + 180) % 360
        out.append(gate)
    return out


def build_encoder_seq(mm, man_no=1):
    """State-independent transduction encoder E (qubit -> {|0>, |2>} cavity).

    Built from ``prep_fock_state(man_no, [0, 2])`` with its leading g0-e0 input
    rotation stripped, leaving the pure shelving ladder
    (e0-f0, g0-e0, f0-g1, g0-e0, e1-f1, f1-g2 -- all pi pulses).

    Parameters
    ----------
    mm : MM_dual_rail_base (or any object exposing ``prep_fock_state``).
    man_no : manipulate mode number.

    Returns
    -------
    list of gate entries.
    """
    full = mm.prep_fock_state(man_no, [0, 2], broadband=False)
    assert full and full[0][2] == 'hpi', (
        "Expected prep_fock_state([0,2]) to start with the g0-e0 hpi input "
        f"rotation; got {full[0] if full else None}. Encoder derivation needs "
        "review.")
    return [list(g) for g in full[1:]]


def build_decoder_seq(mm, man_no=1):
    """Decoder D = E^dagger ({|0>, |2>} cavity -> qubit). Inverse of the encoder."""
    return revert_pulse_seq(build_encoder_seq(mm, man_no))


# ----------------------------------------------------------------------------
# Channel: environment prep + partial beamsplitter
# ----------------------------------------------------------------------------

def eta_to_swap_ratio(eta):
    """Partial-beamsplitter swap-length ratio for transmissivity ``eta``.

    A full SWAP (pi pulse) corresponds to swap angle theta = pi. A partial swap
    of angle theta uses length L_pi * (theta / pi), with
    ``theta = 2 * arccos(sqrt(eta))`` so that ``eta = cos^2(theta/2)`` is the
    single-photon transmissivity: eta = 1 -> ratio 0 (no swap, identity),
    eta = 0 -> ratio 1 (full SWAP).

    NOTE: this linear length<->angle mapping (as used in the Transduction
    notebook) is approximate at high eta, where the fixed flat-top *ramps*
    (~6 sigma) contribute a swap angle comparable to the (short) scaled flat
    region. For accurate high-eta operation, calibrate the swap angle vs flat
    length directly (a length-Rabi on the M-S beamsplitter) and invert.
    """
    eta = float(np.clip(eta, 0.0, 1.0))
    theta = 2.0 * np.arccos(np.sqrt(eta))
    return theta / np.pi


def build_env_prep_seq(env_stor=3, man_no=1):
    """Gate sequence to load |1> into storage mode S{env_stor} (the environment).

    Puts |1> in the manipulate mode (g0-e0, e0-f0, f0-g1) then full-SWAPs it to
    the storage mode. Assumes man_no = 1 (the multiphoton transitions address
    manipulate mode 1).
    """
    assert man_no == 1, "build_env_prep_seq currently assumes man_no == 1"
    return [
        ['multiphoton', 'g0-e0', 'pi', 0],
        ['multiphoton', 'e0-f0', 'pi', 0],
        ['multiphoton', 'f0-g1', 'pi', 0],
        ['storage', f'M{man_no}-S{env_stor}', 'pi', 0],
    ]


def channel_swap_gate(env_stor=3, man_no=1):
    """Gate-based full SWAP between the manipulate mode and S{env_stor}.

    Compile this, then scale the flat-top length (row 2 of the compiled pulse)
    by ``eta_to_swap_ratio(eta)`` to realize the partial beamsplitter.
    """
    return [['storage', f'M{man_no}-S{env_stor}', 'pi', 0]]


def scale_swap_length(compiled_pulse, ratio):
    """Return a copy of a compiled pulse array with its length row (index 2)
    scaled by ``ratio`` (for the partial beamsplitter). ``compiled_pulse`` is the
    ``get_prepulse_creator(...).pulse.tolist()`` form: 7 rows, one column/pulse.
    """
    out = [list(row) for row in compiled_pulse]
    out[2] = [L * ratio for L in out[2]]
    return out
