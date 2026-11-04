# -*- coding: utf-8 -*-
"""What HistogramExperiment.analyze() hands the single-shot postprocessor.

`expt.readout_cfg` is copied into `device.readout` by every notebook's
postprocessor, so these tests pin it: the phase is the one the shots were
taken with plus the fitted residual (so a second analyze does not rotate
twice), the values are g-e only even with f shots, and the g/e/f analysis
reads g vs e/f on I and e vs f on Q.
"""
import numpy as np
import pytest
from slab import AttrDict

from experiments.single_qubit.single_shot import HistogramExperiment


def make_expt(centers, phase=10.0, n=4000, sigma=1.0, seed=1):
    rng = np.random.default_rng(seed)
    data = {}
    for s, c in centers.items():
        data["I" + s] = c.real + sigma * rng.standard_normal(n)
        data["Q" + s] = c.imag + sigma * rng.standard_normal(n)
    expt = HistogramExperiment.__new__(HistogramExperiment)
    expt.cfg = AttrDict(dict(expt=dict(active_reset=False, check_f="f" in centers),
                             device=dict(readout=dict(phase=[phase], threshold=[0.0]))))
    expt.data = data
    return expt


# g at the origin, e along 30 deg, f off the g-e line near e (a right angle at e)
GE = {"g": 0 + 0j, "e": 6 * np.exp(1j * np.pi / 6)}
GEF = dict(GE, f=GE["e"] + 4 * np.exp(1j * (np.pi / 6 + np.pi / 2)))


def test_readout_cfg_holds_the_ge_settings():
    expt = make_expt(GE)
    expt.analyze(verbose=False)
    ro = expt.readout_cfg
    assert set(ro) == {"phase", "threshold", "Ig", "Ie", "confusion_matrix_without_reset"}
    assert ro.phase == [pytest.approx(10.0 - 30.0, abs=1.0)]
    assert ro.Ig[0] < ro.threshold[0] < ro.Ie[0]
    assert ro.Ie[0] - ro.Ig[0] == pytest.approx(6.0, abs=0.2)


def test_a_second_analyze_does_not_rotate_twice():
    expt = make_expt(GE)
    expt.analyze(verbose=False)
    first = expt.readout_cfg.phase
    expt.analyze(verbose=False)
    assert expt.readout_cfg.phase == first


def test_f_shots_keep_the_ge_axis_and_give_a_3x3_assignment():
    ge, gef = make_expt(GE), make_expt(GEF)
    ge.analyze(verbose=False)
    gef.analyze(verbose=False)
    assert gef.data["angle"] == pytest.approx(ge.data["angle"], abs=1.0)
    assert gef.readout_cfg.threshold == [pytest.approx(ge.readout_cfg.threshold[0], abs=0.3)]
    A = gef.data["assignment_matrix"]
    assert A.shape == (3, 3)
    np.testing.assert_allclose(A.sum(axis=1), 1.0)
    assert np.all(np.diag(A) > 0.9)
    assert gef.data["f_high_q"]


def test_f_on_the_other_side_of_the_ge_line():
    mirrored = dict(GE, f=GE["e"] + 4 * np.exp(1j * (np.pi / 6 - np.pi / 2)))
    expt = make_expt(mirrored)
    expt.analyze(verbose=False)
    assert not expt.data["f_high_q"]
    assert np.all(np.diag(expt.data["assignment_matrix"]) > 0.9)


def test_an_unusable_histogram_gives_no_readout_cfg():
    expt = make_expt({"g": 0j, "e": 0j}, sigma=0.0)  # no pi pulse: the blobs coincide
    with pytest.raises(ValueError, match="invalid readout"):
        expt.analyze(verbose=False)
    assert not hasattr(expt, "readout_cfg")
