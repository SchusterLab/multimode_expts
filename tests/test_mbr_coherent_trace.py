# -*- coding: utf-8 -*-
"""The coherent normalized-trace FFT, sum_n A_n(t)/A_n(0), on made-up levels."""
import numpy as np
import pytest
from slab import AttrDict

from fitting.qsim.mbr_spectrum import coherent_trace_spectrum


def _data(scale_rows=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(7)
    energies_MHz = np.array([-0.3, 0.0, 0.2, 0.45])
    weights = rng.random((3, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    time_us = np.arange(64) * 0.25
    A = (weights @ np.exp(-2j * np.pi * np.outer(energies_MHz, time_us))) * np.asarray(scale_rows)[:, None]
    spectrum = AttrDict(dict(time_us=time_us, fft_window="hann",
                             energy_MHz=np.fft.fftshift(np.fft.fftfreq(len(time_us), d=0.25)),
                             energies_MHz=energies_MHz, eigenstate_weights=weights))
    return AttrDict(dict(A=A)), spectrum


def test_theory_returns_give_the_theory_spectrum():
    """A_n(t) equal to the theory returns (any row scale) -> the same spectrum, scale 1."""
    reconstruction, spectrum = _data(scale_rows=(2.0, 0.5 + 1j, 3.0))
    result = coherent_trace_spectrum(reconstruction, spectrum)
    np.testing.assert_allclose(result.measured, result.theory_unscaled, rtol=1e-10, atol=1e-12)
    assert result.theory_scale == pytest.approx(1.0)
    # The strongest peak sits on a level.
    peak_MHz = result.energy_MHz[np.argmax(result.measured)]
    assert np.min(np.abs(peak_MHz - spectrum.energies_MHz)) <= 0.5 / (64 * 0.25)


def test_a_zero_return_at_t0_is_refused():
    reconstruction, spectrum = _data()
    reconstruction.A[1, 0] = 0.
    with pytest.raises(ValueError, match="zero return amplitude"):
        coherent_trace_spectrum(reconstruction, spectrum)
