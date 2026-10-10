"""
Pulse sequences for the transduction channel experiment.

One place for the logical-state preparation and the channel recipe
(environment |1> in S_env, encode, partial M-S swap), so that the notebook,
the overnight studies and the offline analysis use the same sequence.

Compiled sequences are the ``get_prepulse_creator(...).pulse.tolist()`` form
(7 rows, one column per pulse) that ``WignerTomography1ModeExperiment`` takes
as ``pre_sweep_pulse``.
"""

from experiments.transduction.decoder import (
    build_env_prep_seq, channel_swap_gate, eta_to_swap_ratio, scale_swap_length,
)

# Informationally complete inputs of the process tomography (span I, X, Y, Z).
PROC_INPUTS = ('0', '1', '+', '+i')

# logical label -> photon list for prep_fock_state. Logical |1> = Fock 2.
# '+i' and '-i' are swapped: the hardware flips the sign of the imaginary part.
LOGICAL_PHOTONS = {'0': [0], '1': [2], '+': [0, 2], '-': [0, -2],
                   '+i': [0, -2j], '-i': [0, 2j]}


def build_logical_prep(mm, logical, enc_phase_corr_deg=0.0, man_no=1):
    """Gate list that prepares the logical state ``logical`` in the manipulate mode.

    The encoder phase correction is added to the phase of the leading g0-e0 hpi
    pulse (superposition inputs only).
    """
    photons = LOGICAL_PHOTONS[logical]
    seq = [list(g) for g in mm.prep_fock_state(man_no, photons, broadband=False)]
    if len(photons) == 2 and seq and seq[0][2] == 'hpi':
        seq[0][3] = (seq[0][3] + enc_phase_corr_deg) % 360
    return seq


def compile_seq(mm, seq):
    """Gate list -> compiled 7-row pulse list."""
    return mm.get_prepulse_creator(seq).pulse.tolist()


def reference_prep(mm, logical='+', enc_phase_corr_deg=0.0, man_no=1):
    """Compiled preparation of ``logical`` with no environment and no channel."""
    return compile_seq(mm, build_logical_prep(mm, logical, enc_phase_corr_deg, man_no))


def channel_prep(mm, eta, logical, env_stor, enc_phase_corr_deg=0.0, man_no=1):
    """Compiled channel sequence: env |1> in S{env_stor} + encode(logical) +
    partial M-S{env_stor} swap with transmissivity ``eta``."""
    base = compile_seq(mm, build_env_prep_seq(env_stor=env_stor, man_no=man_no)
                       + build_logical_prep(mm, logical, enc_phase_corr_deg, man_no))
    swap = scale_swap_length(compile_seq(mm, channel_swap_gate(env_stor=env_stor, man_no=man_no)),
                             eta_to_swap_ratio(eta))
    return [base[r] + swap[r] for r in range(len(base))]
