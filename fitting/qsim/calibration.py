"""Numerics of the qsim calibration notebooks (multiphoton, Floquet).

Pure functions: arrays in, arrays out. Callers:
`measurement_notebooks/202609_qsim_migration/multiphoton_calibration.py`.
Moved out of `experiments/qsim/notebook_helpers/multiphoton_calibration.py` in
MBR redesign step 9C (`docs/qsim/mbr_step9_plan.md`).
"""
import numpy as np


def population_transfer(z, z_g, z_e):
    """-> the projection of IQ points onto one readout axis: 0 at ``z_g``, 1 at ``z_e``.

    A population-transfer coordinate from that photon number's own g/e
    references, not a leakage-resolved gate fidelity.
    """
    readout_axis = z_e - z_g
    return np.real((np.asarray(z) - z_g) * np.conj(readout_axis)) / abs(readout_axis) ** 2


def rabi_transfer(iq_from_g, iq_from_e):
    """Amplitude-Rabi transfer per photon number, on its own g/e axis.

    ``iq_from_g[n]`` and ``iq_from_e[n]`` are the complex IQ of the Rabi scans
    from |g,n> and |e,n>; the first gain point (0) of each is its reference.
    -> (g_to_e, e_to_g), indexed [photon number, gain point].
    """
    g_to_e, e_to_g = [], []
    for z_from_g, z_from_e in zip(iq_from_g, iq_from_e):
        z_from_g, z_from_e = np.asarray(z_from_g), np.asarray(z_from_e)
        z_g, z_e = z_from_g[0], z_from_e[0]
        g_to_e.append(population_transfer(z_from_g, z_g, z_e))
        e_to_g.append(1 - population_transfer(z_from_e, z_g, z_e))
    return np.asarray(g_to_e), np.asarray(e_to_g)


def validation_transfer(iq, n_photon_numbers):
    """The exact-path validation: cases |g,n>, |e,n> (references), then the two transfers.

    ``iq`` is the complex IQ, shaped (photon number, 4) after reshaping.
    -> array (photon number, 2): [g->e, e->g].
    """
    iq = np.asarray(iq).reshape(n_photon_numbers, 4)
    out = np.zeros((n_photon_numbers, 2))
    for row, (z_g, z_e, z_g_to_e, z_e_to_g) in enumerate(iq):
        out[row, 0] = population_transfer(z_g_to_e, z_g, z_e)
        out[row, 1] = 1 - population_transfer(z_e_to_g, z_g, z_e)
    return out


def return_error(iq_by_depth):
    """Even-swap error amplification: mean |z_depth - z_0|^2 over the deeper rows.

    ``iq_by_depth`` is (depth, sweep point); row 0 is the zero-swap
    preparation/readout reference. Every depth uses an even number of swaps,
    so a perfect swap returns every row to row 0. Lower is better.
    -> the score per sweep point.
    """
    z = np.asarray(iq_by_depth)
    return np.mean(np.abs(z[1:] - z[0]) ** 2, axis=0)
