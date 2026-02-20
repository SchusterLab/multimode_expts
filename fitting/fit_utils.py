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


def guess_decaysin_params(x, y):
    """
    Estimate parameters for a decaying sinusoid:
        A * sin(2*pi*f*x + phi) * exp(-(x-x0)/T) + y0

    Uses zero-padded FFT for frequency/phase and Hilbert envelope for
    amplitude/decay estimation. Falls back to simpler heuristics if the
    envelope analysis fails.

    Returns
    -------
    amp, freq, phase_deg, decay, offset
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.mean(y)
    y_centered = y - offset

    # --- Frequency and phase via zero-padded FFT ---
    n = len(y_centered)
    npad = max(4 * n, 1024)
    yf = rfft(y_centered, n=npad)
    xf = rfftfreq(npad, x[1] - x[0])
    magnitudes = np.abs(yf)
    magnitudes[0] = 0
    peak_idx = np.argmax(magnitudes)
    freq = xf[peak_idx]
    # rfft uses cos convention; convert to sin: add 90 deg
    phase_deg = np.angle(yf[peak_idx]) * 180 / np.pi + 90

    # --- Amplitude and decay from Hilbert envelope ---
    xrange = x[-1] - x[0]
    try:
        from scipy.signal import hilbert
        analytic = hilbert(y_centered)
        envelope = np.abs(analytic)
        mask = envelope > 0.15 * np.max(envelope)
        if np.sum(mask) > 5:
            log_env = np.log(np.clip(envelope[mask], 1e-20, None))
            coeffs = np.polyfit(x[mask], log_env, 1)
            decay = -1.0 / coeffs[0] if coeffs[0] < -1e-10 else xrange
            amp = np.exp(coeffs[1])
        else:
            amp = np.ptp(y) / 2
            decay = xrange
    except Exception:
        amp = np.ptp(y) / 2
        decay = xrange

    decay = np.clip(decay, 0.1 * xrange, 50 * xrange)
    amp = max(amp, 1e-10)

    return amp, freq, phase_deg, decay, offset
