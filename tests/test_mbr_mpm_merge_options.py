# -*- coding: utf-8 -*-
"""The Matrix Pencil cross-row merge tolerance that the spectrum resolves.

``analyze(spectrum_method="matrix_pencil", mpm_merge_frequency_tolerance_bins="calibration")``
merges rows by the phase-calibration standard errors. No golden uses that
mode, so it is checked here, on made-up numbers.
"""
import numpy as np
import pytest
from slab import AttrDict

from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment

merge_options = MBRSpectrumExperiment._matrix_pencil_merge_options

CALIBRATION = AttrDict(dict(
    occupations=[(1, 0, 2), (0, 3, 0)],
    phase_error=[3.6, -7.2],            # deg per cycle
    hardware=AttrDict(dict(floquet_cycle_us=0.5)),
))
RECONSTRUCTION = AttrDict(dict(final_occupations=[(0, 3, 0), (1, 0, 2), (0, 3, 0)]))


def test_calibration_mode_gives_each_row_its_final_occupations_error():
    options = merge_options({"merge_frequency_tolerance_bins": "calibration"},
                            CALIBRATION, RECONSTRUCTION, sigma=2.5, floor_kHz=0.3)
    # |slope error| / (360 deg * cycle time): 3.6 -> 0.02 MHz, 7.2 -> 0.04 MHz.
    np.testing.assert_allclose(options["row_frequency_standard_errors_MHz"], [0.04, 0.02, 0.04])
    assert options["merge_frequency_tolerance_bins"] is None
    assert options["merge_frequency_tolerance_sigma"] == 2.5
    assert options["merge_frequency_tolerance_floor_MHz"] == pytest.approx(3e-4)


def test_bins_mode_leaves_the_bins_and_adds_no_errors():
    options = merge_options({"merge_frequency_tolerance_bins": 0.8}, None, RECONSTRUCTION,
                            sigma=3.0, floor_kHz=0.1)
    assert options["merge_frequency_tolerance_bins"] == 0.8
    assert "row_frequency_standard_errors_MHz" not in options


def test_explicit_options_win_over_the_defaults():
    options = merge_options({"merge_frequency_tolerance_sigma": 5.0}, None, RECONSTRUCTION,
                            sigma=3.0, floor_kHz=0.1)
    assert options["merge_frequency_tolerance_sigma"] == 5.0


@pytest.mark.parametrize("calibration, reconstruction, match", [
    (None, RECONSTRUCTION, "requires the phase calibration"),
    (CALIBRATION, AttrDict(dict(final_occupations=[(9, 9, 9)])), "missing phase standard errors"),
])
def test_calibration_mode_refuses_what_it_cannot_resolve(calibration, reconstruction, match):
    with pytest.raises(ValueError, match=match):
        merge_options({"merge_frequency_tolerance_bins": "calibration"}, calibration, reconstruction,
                      sigma=3.0, floor_kHz=0.1)


def test_an_unknown_mode_string_is_refused():
    with pytest.raises(ValueError, match="numeric, None, or 'calibration'"):
        merge_options({"merge_frequency_tolerance_bins": "calib"}, CALIBRATION, RECONSTRUCTION,
                      sigma=3.0, floor_kHz=0.1)
