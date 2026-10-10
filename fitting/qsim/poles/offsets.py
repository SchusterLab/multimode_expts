"""The row-to-row frequency offset sigma: two of the three estimates of spec section 6.

- Floor (:func:`calibration_sigma`): the standard error of each final occupation's
  Stark-calibration phase slope, as a frequency.
- Upper bound (:func:`model_offset_sigma`): per row, the one frequency offset (with a decay
  and a complex scale) that best matches the row to the model's return; the spread of those
  offsets. Model errors enter it too, so it bounds sigma from above.

The third, the ceiling, is where benchmark 2 loses the statistic. Pure numerics.
"""
import numpy as np


def calibration_sigma(phase_error_deg_per_cycle, cycle_us):
    """-> sigma_cal = |phase slope error| / (360 deg x T_cycle), in MHz, per occupation."""
    return np.abs(np.asarray(phase_error_deg_per_cycle, dtype=float)) / (360. * cycle_us)


def model_returns(eigenstate_weights, energies_MHz, time_us):
    """-> m_b(t) = sum_lambda |<b|lambda>|^2 exp(-2 pi i E_lambda t), one row per measured occupation."""
    return eigenstate_weights @ np.exp(-2j * np.pi * np.outer(energies_MHz, time_us))


def row_offsets(a, model, time_us, offsets_MHz, decays_per_us):
    """-> per row, the offset delta (on the grid) maximizing the explained power

        |<r, a_b>|^2 / <r, r>,   r(t) = exp(-gamma t) exp(-2 pi i delta t) m_b(t),

    over the grid of delta and gamma (the best complex scale is then <r, a_b> / <r, r>).
    """
    roll = np.exp(np.outer(-2j * np.pi * np.asarray(offsets_MHz), time_us))            # delta x t
    decay = np.exp(-np.outer(decays_per_us, time_us))                                    # gamma x t
    r = decay[None, :, None, :] * roll[None, None, :, :] * model[:, None, None, :]       # b x gamma x delta x t
    explained = np.abs(np.sum(r.conj() * a[:, None, None, :], axis=-1)) ** 2 / np.sum(np.abs(r) ** 2, axis=-1)
    best = explained.reshape(len(a), -1).argmax(axis=1) % len(offsets_MHz)
    return np.asarray(offsets_MHz)[best]


def model_offset_sigma(a, model, time_us, max_offset_MHz=5e-3, step_MHz=2e-5,
                       decays_per_us=np.linspace(0., 0.03, 13)):
    """-> (the standard deviation of the per-row offsets, the offsets), in MHz.

    The offsets are searched in +-max_offset_MHz at step_MHz; a row whose best offset is at
    the edge of the range is not matched by any roll, and its offset is NaN.
    """
    grid = np.arange(-max_offset_MHz, max_offset_MHz + step_MHz / 2, step_MHz)
    offsets = row_offsets(a, model, time_us, grid, decays_per_us)
    offsets = np.where(np.abs(offsets) >= max_offset_MHz - step_MHz / 2, np.nan, offsets)
    return float(np.nanstd(offsets)), offsets
