# -*- coding: utf-8 -*-
"""The per-row calibration errors the spectrum gives Matrix Pencil.

``analyze(spectrum_method="matrix_pencil", mpm_merge_frequency_tolerance_bins="calibration")``
merges rows by the phase-calibration standard errors. No golden uses that
mode, so it is checked here, on made-up numbers.
"""
import numpy as np
import pytest
from slab import AttrDict

from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment

errors_MHz = MBRSpectrumExperiment._calibration_frequency_errors_MHz

CALIBRATION = AttrDict(dict(
    occupations=[(1, 0, 2), (0, 3, 0)],
    phase_error=[3.6, -7.2],            # deg per cycle
    hardware=AttrDict(dict(floquet_cycle_us=0.5)),
))
RECONSTRUCTION = AttrDict(dict(final_occupations=[(0, 3, 0), (1, 0, 2), (0, 3, 0)]))


def test_each_row_gets_its_final_occupations_error():
    # |slope error| / (360 deg * cycle time): 3.6 -> 0.02 MHz, 7.2 -> 0.04 MHz.
    np.testing.assert_allclose(errors_MHz(CALIBRATION, RECONSTRUCTION), [0.04, 0.02, 0.04])


@pytest.mark.parametrize("calibration, reconstruction, match", [
    (None, RECONSTRUCTION, "requires the phase calibration"),
    (CALIBRATION, AttrDict(dict(final_occupations=[(9, 9, 9)])), "missing phase standard errors"),
])
def test_refuses_what_it_cannot_resolve(calibration, reconstruction, match):
    with pytest.raises(ValueError, match=match):
        errors_MHz(calibration, reconstruction)
