"""Fitter E: peaks of the windowed FFT (spec 4.5), the resolution before Matrix Pencil.

Pure numerics.
"""
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import find_peaks

from fitting.qsim.mbr_spectrum import windowed_fft
from fitting.qsim.poles.pole_fit import (check_shape, normalize_to_initial_return, pole_fit,
                                         poles_from, sample_time)


class FFTPeakSettings(BaseModel):
    """The free parameters of the FFT peak finder."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: A window lowers the side lobes (fewer false peaks) and widens the peaks
    #: (more merging).
    window: Literal["raw", "hann", "hamming", "blackman"] = "hann"
    #: Zero padding factor: finer peak positions, not finer resolution.
    zero_padding: Annotated[int, Field(ge=1)] = 8
    #: A peak must stand out by this fraction of the highest peak (prominence).
    #: Lower finds weak levels and more side lobes.
    peak_threshold: Annotated[float, Field(gt=0., lt=1.)] = 0.05


def fit(A, time_us, settings=FFTPeakSettings(), row_groups=None):
    """-> the PoleFit of the FFT peaks of sum_b A_b(t) / A_b(0). Decays are 0."""
    dt_us = sample_time(time_us)
    a = normalize_to_initial_return(check_shape(A, time_us))
    frequency_MHz, spectrum = trace_spectrum(a, dt_us, settings)
    peaks, _ = find_peaks(spectrum, prominence=settings.peak_threshold * np.max(spectrum))
    poles = poles_from(frequency_MHz[peaks], np.zeros(len(peaks)), dt_us)
    return pole_fit(poles, a, dt_us, len(peaks))


def trace_spectrum(a, dt_us, settings):
    """-> (E, |FFT| of sum_b a_b), on the fftshifted grid of ``windowed_fft``."""
    n_fft = settings.zero_padding * a.shape[1]
    spectrum = windowed_fft(np.sum(a, axis=0), settings.window, settings.zero_padding)
    return np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt_us)), spectrum
