import numpy as np
from scipy.fft import rfft, rfftfreq


def guess_freq(x, y):
    # note: could also guess phase but need zero-padding
    # just guessing freq seems good enough to escape from local minima in most cases
    yf = rfft(y - np.mean(y))
    xf = rfftfreq(len(x), x[1] - x[0])
    peak_idx = np.argmax(np.abs(yf[1:])) + 1
    return np.abs(xf[peak_idx]), np.angle(yf[peak_idx])


def guess_sinusoidal_params(x, y):
    """
    Guess parameters for a sinusoidal function: A*sin(2*pi*f*x + phi) + C

    Parameters
    ----------
    x : array_like
        Independent variable (e.g., time)
    y : array_like
        Dependent variable (response)

    Returns
    -------
    freq : float
        Estimated frequency
    amp : float
        Estimated amplitude
    offset : float
        Estimated offset (DC component)
    """
    # Estimate offset as the mean
    offset = np.mean(y)

    # Estimate amplitude as half the peak-to-peak range
    amp = np.ptp(y) / 2

    # Estimate frequency using FFT
    freq, _ = guess_freq(x, y)

    return freq, amp, offset
