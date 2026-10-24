"""ErrorAmplificationExperiment.analyze: the final readout state from the pulse type, the
config override, periodic=False by default, and no fit (but no error) for custom sweep pulses.
Synthetic data only: no station, no hardware."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from slab import AttrDict

from experiments.single_qubit import error_amplification
from experiments.single_qubit.error_amplification import (
    ErrorAmplificationExperiment,
    expected_final_state,
)

IG, IE = 100.0, -80.0


@pytest.mark.parametrize("pulse_type, state", [
    (["qubit", "ge", "pi", 0], "g"),
    (["qubit", "ge_broadband", "pi", 0], "g"),
    (["qubit", "ef", "pi", 0], "g"),
    (["man", "M1", "pi", 0], "e"),
    (["storage", "M1-S4", "pi", 0], "e"),
    (["floquet", "M1-S4", "pi/40", 0], "e"),
    (["multiphoton", "g1-e1", "pi", 0], "g"),
    (["multiphoton", "e2-f2", "pi", 0], "g"),
    (["multiphoton", "f0-g1", "pi", 0], "e"),
    (["multiphoton", "f2-g3", "pi", 0], "e"),
])
def test_expected_final_state(pulse_type, state):
    assert expected_final_state(AttrDict(dict(pulse_type=pulse_type))) == state


@pytest.mark.parametrize("empty", [None, [], [[], [], []]])
def test_empty_sweep_pulses_keep_the_rule(empty):
    cfg = AttrDict(dict(pulse_type=["storage", "M1-S4", "pi", 0], pre_sweep_pulse=empty, post_sweep_pulse=empty))
    assert expected_final_state(cfg) == "e"


@pytest.mark.parametrize("key", ["pre_sweep_pulse", "post_sweep_pulse"])
def test_custom_sweep_pulse_is_unknown(key):
    cfg = AttrDict({"pulse_type": ["qubit", "ge", "pi", 0], key: [["qubit", "ge", "pi", 0]]})
    assert expected_final_state(cfg) is None


def _experiment(state_right, center, **expt):
    """An experiment with synthetic data: the readout is state_right on resonance and leaks to
    the other state off resonance, more for more pulses."""
    x = np.linspace(1985.7, 1986.3, 51)
    n_pts = np.arange(1, 8)
    p_right = np.exp(-n_pts[:, None] * ((x[None, :] - center) / 0.15) ** 2)
    i_right, i_other = (IE, IG) if state_right == "e" else (IG, IE)
    avgi = i_right * p_right + i_other * (1 - p_right)
    experiment = ErrorAmplificationExperiment.__new__(ErrorAmplificationExperiment)
    experiment.cfg = AttrDict(dict(device=dict(readout=dict(Ig=[IG], Ie=[IE])),
                                   expt=dict(parameter_to_test="frequency", **expt)))
    experiment.data = dict(x_pts=x, N_pts=n_pts, avgi=avgi, avgq=np.zeros_like(avgi))
    return experiment


def test_analyze_uses_the_derived_state():
    # f2-g3 reads e when right (the 2026-10-24 case: the old default 'g' fit the wrong edge).
    expt = _experiment("e", 1985.85, pulse_type=["multiphoton", "f2-g3", "pi", 0])
    expt.analyze()
    assert expt.data["fit_avgi"][2] == pytest.approx(1985.85, abs=0.012)
    expt.analyze(state_fin="g")
    assert abs(expt.data["fit_avgi"][2] - 1985.85) > 0.05


def test_config_state_fin_wins():
    expt = _experiment("g", 1986.1, pulse_type=["storage", "M1-S4", "pi", 0], state_fin="g")
    expt.analyze()
    assert expt.data["fit_avgi"][2] == pytest.approx(1986.1, abs=0.012)


@pytest.mark.parametrize("expt_periodic, call_periodic, used", [
    (None, None, False),   # default: gain and frequency scans do not wrap
    (True, None, True),    # cfg.expt.periodic
    (True, False, False),  # the argument wins
])
def test_periodic_choice(monkeypatch, expt_periodic, call_periodic, used):
    calls = []
    real = error_amplification.fitter.fitgaussian
    monkeypatch.setattr(error_amplification.fitter, "fitgaussian",
                        lambda *a, periodic, **k: calls.append(periodic) or real(*a, periodic=periodic, **k))
    extra = {} if expt_periodic is None else dict(periodic=expt_periodic)
    expt = _experiment("g", 1986.0, pulse_type=["qubit", "ge", "pi", 0], **extra)
    expt.analyze(periodic=call_periodic)
    assert calls == [used]


def test_custom_sweep_pulse_warns_and_skips_the_fit():
    expt = _experiment("e", 1986.0, pulse_type=["qubit", "ge", "pi", 0],
                       pre_sweep_pulse=[["qubit", "ge", "pi", 0]])
    with pytest.warns(UserWarning, match="final state is unknown"):
        data = expt.analyze()
    assert data is expt.data
    assert "prod_avgi" not in data and "fit_avgi" not in data
    expt.display()  # the 2D maps only, no error
    plt.close("all")
    expt.analyze(state_fin="e")
    assert expt.data["fit_avgi"][2] == pytest.approx(1986.0, abs=0.012)
