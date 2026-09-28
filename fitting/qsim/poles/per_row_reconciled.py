"""Fitter A: the current Matrix Pencil, unchanged (spec 4.4).

A thin wrapper of :func:`fitting.qsim.matrix_pencil.analyze_matrix_pencil`: per-row
pencils with rank sweeps, reconciled across rows. Its known defects are in
``docs/log/2026-09-27_matrix-pencil.md`` and ``tests/test_matrix_pencil_synthetic.py``.
"""
from math import comb

import numpy as np

from fitting.qsim.matrix_pencil import MatrixPencilSettings, analyze_matrix_pencil
from fitting.qsim.poles.pole_fit import PoleFit, check_shape, sample_time
from slab import AttrDict

#: The settings of the 7-1 disorder campaign
#: (``experiments.qsim.mbr_disorder_ensemble.DEFAULT_MATRIX_PENCIL``), with the
#: pole cap at D = 35 as its analysis set it; the results so far rest on these.
CAMPAIGN_SETTINGS = MatrixPencilSettings(
    match_decay=False,
    track_frequency_tolerance_bins=1.0,
    merge_frequency_tolerance_bins=0.10,
    dedup_frequency_tolerance_bins=0.10,
    minimum_supporting_rows=1,
    requested_max_modes=comb(3 + 5 - 1, 3),
)


def fit(A, time_us, settings=CAMPAIGN_SETTINGS, row_groups=None):
    """-> the PoleFit of analyze_matrix_pencil; every row diagonal. ``row_groups`` unused
    (the current code uses them only with calibration errors, which A is not given)."""
    A = check_shape(A, time_us)
    sample_time(time_us)
    occupations = [(row,) for row in range(len(A))]
    result = analyze_matrix_pencil(AttrDict(dict(A=A, occupations=occupations)),
                                   np.asarray(time_us, dtype=float), settings)
    modes = result.modes
    return PoleFit(np.asarray(modes.frequencies_MHz), np.asarray(modes.decay_per_us),
                   np.asarray(modes.local_complex_amplitudes), len(modes.frequencies_MHz),
                   np.asarray(modes.frequency_standard_errors_MHz))
