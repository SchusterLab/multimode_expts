"""Split fft helper functions for better readability
"""

import numpy as np


def choose_fft_windows(fft_window):
    _windows = {"raw": np.ones,
                "hann": np.hanning,
                "hamming": np.hamming, 
                "blackman": np.blackman}
    if fft_window not in _windows:
        raise ValueError(f"fft windows must be {" ".join(_windows.keys())}")
    return _windows[fft_window]


def do_fft(f,
           n,
           axis):
    _transfomred = np.fft.ifft(f, n = n, axis = axis)
    _shift = np.fft.fftshift(_transfomred, axes = axis)
    return np.abs(_shift)
    
    
    
def calculate_ldos_from_propagator(propagator,
                                   cycles,
                                   dt,
                                   fft_window = 'raw',
                                   zero_padding = 1,
                                   axis = 1):
    window = choose_fft_windows(fft_window)
    n_fft = int(zero_padding * len(cycles))
    fft_scale = n_fft / np.sum(window)
    
    energy_MHz = np.fft.fftshift(np.fft.fftfreq(n_fft, d = dt))
    fft_ldos = fft_scale * do_fft(propagator * window, 1)
    return fft_ldos, energy_MHz

