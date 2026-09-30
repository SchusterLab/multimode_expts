"""Physical synthetic checks for occupation return-amplitude envelope fits."""

import numpy as np
import pytest

from measurement_notebooks.jonginn.decay_fitting import fit_complex_decay


def coherent_trace(t):
    return 0.55 * np.exp(-2j * np.pi * 0.02 * t) + 0.45 * np.exp(2j * np.pi * 0.037 * t)


@pytest.mark.parametrize("beta", [1, 2])
def test_recovers_amplitude_tau_with_coherent_beating_and_floor(beta):
    t = np.linspace(0, 240, 241)
    coherent = coherent_trace(t)
    tau = 110.0
    power = 0.004 + 0.8 * abs(coherent) ** 2 * np.exp(-2 * (t / tau) ** beta)
    measured = np.sqrt(power) * np.exp(1j * np.angle(coherent))
    result = fit_complex_decay(t, measured, coherent, beta=beta)
    assert result["accepted"]
    assert result["tau_us"] == pytest.approx(tau, rel=0.001)
    assert result["floor_power"] == pytest.approx(0.004, abs=1e-6)


def test_undamped_beats_have_no_accepted_finite_tau():
    t = np.linspace(0, 240, 241)
    coherent = coherent_trace(t)
    result = fit_complex_decay(t, coherent, coherent)
    assert not result["accepted"]
    assert np.isnan(result["tau_us"])
    assert result["status"] == "no_decay_evidence"


def test_phase_rotations_leave_tau_unchanged():
    t = np.linspace(0, 240, 241)
    coherent = coherent_trace(t)
    measured = coherent * np.exp(-t / 90)
    reference = fit_complex_decay(t, measured, coherent)
    rotated = fit_complex_decay(t, measured * np.exp(1j * t**2), coherent * 1j)
    assert reference["accepted"] and rotated["accepted"]
    assert rotated["tau_us"] == pytest.approx(reference["tau_us"], rel=1e-7)


def test_window_keeps_physical_time_origin_for_gaussian():
    t = np.linspace(40, 200, 161)
    coherent = coherent_trace(t)
    measured = coherent * np.exp(-(t / 120) ** 2)
    result = fit_complex_decay(t, measured, coherent, beta=2)
    assert result["accepted"]
    assert result["tau_us"] == pytest.approx(120, rel=0.001)


def test_noisy_coherent_decay_is_recovered():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 240, 241)
    coherent = coherent_trace(t)
    measured = coherent * np.exp(-t / 100) + 0.006 * (
        rng.normal(size=t.size) + 1j * rng.normal(size=t.size)
    )
    result = fit_complex_decay(t, measured, coherent)
    assert result["accepted"]
    assert result["tau_us"] == pytest.approx(100, rel=0.1)


def test_apparent_fit_is_labeled_separately():
    t = np.linspace(0, 240, 241)
    result = fit_complex_decay(t, np.exp(-t / 80))
    assert result["accepted"]
    assert result["metric"] == "apparent_power_envelope"
    assert result["tau_us"] == pytest.approx(80, rel=0.001)


@pytest.mark.parametrize("case", ["truncated", "nonfinite", "unsorted", "negative", "short"])
def test_invalid_input_is_explicit(case):
    t = np.arange(20, dtype=float)
    measured = np.exp(-t / 20).astype(complex)
    if case == "truncated":
        measured = measured[:-1]
    elif case == "nonfinite":
        measured[3] = np.nan
    elif case == "unsorted":
        t[5] = t[4]
    elif case == "negative":
        t -= 1
    elif case == "short":
        t, measured = t[:6], measured[:6]
    with pytest.raises(ValueError):
        fit_complex_decay(t, measured)


def test_nonfinite_or_misaligned_theory_is_explicit():
    t = np.arange(20, dtype=float)
    measured = np.exp(-t / 20).astype(complex)
    with pytest.raises(ValueError, match="same one-dimensional shape"):
        fit_complex_decay(t, measured, measured[:-1])
    theory = measured.copy()
    theory[4] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_complex_decay(t, measured, theory)
