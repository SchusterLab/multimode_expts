"""Joint ML channel estimate (Ic) on simulated data with a known channel."""
import glob
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.transduction.channel_model import ideal_channel_output, logical_ket
from experiments.transduction.estimators import (
    WignerData, charge_mask, choi_to_outputs, estimate_ic, ic_from_choi, outputs_to_choi,
)
from experiments.transduction.process_tomo import DIM, rho_linear, theory_metrics, wigner_analysis
from experiments.transduction.sequences import PROC_INPUTS


def _ideal(eta):
    return {L: ideal_channel_output(logical_ket(L, DIM), eta, DIM).full() for L in PROC_INPUTS}


def _fake_set(rhos, n_shots, seed=0, n_points=120, offset=-0.15):
    """Fake Wigner jobs: random grid, ideal readout, a common parity offset like the real data."""
    rng = np.random.default_rng(seed)
    alpha = 1.8 * np.sqrt(rng.uniform(0, 1, n_points)) * np.exp(2j * np.pi * rng.uniform(0, 1, n_points))
    expts = {}
    for L in PROC_INPUTS:
        nt = np.full(n_points, float(n_shots))
        pc = dict(confusion_matrix=[1.0, 0.0, 0.0, 1.0], alpha_scale=1.0, pulse_correction=True,
                  plus=dict(n_total=nt, n_excited=nt * 0), minus=dict(n_total=nt, n_excited=nt * 0))
        wd = WignerData(dict(alpha=alpha, parity_counts=pc), DIM)
        wd.offset = np.full(n_points, offset)
        ne = wd.simulate(rhos[L], rng)
        pc['plus']['n_excited'], pc['minus']['n_excited'] = ne[0], ne[1]
        expts[L] = SimpleNamespace(data=dict(alpha=alpha, parity_counts=pc))
    return expts


def test_choi_round_trip_and_ideal_ic():
    r = _ideal(0.35)
    J = outputs_to_choi(r)
    back = choi_to_outputs(J)
    assert all(np.allclose(back[L], r[L]) for L in PROC_INPUTS)
    assert ic_from_choi(J) == pytest.approx(theory_metrics(0.35)['Ic'], abs=1e-9)
    assert np.allclose(J[~charge_mask(DIM)], 0)          # the ideal channel is photon-number covariant


@pytest.mark.parametrize('eta', [0.35, 0.87])
def test_ml_recovers_ideal_channel_at_high_shots(eta):
    expts = _fake_set(_ideal(eta), n_shots=2_000_000)
    lin = {L: WignerData(expts[L].data, DIM).linear() for L in PROC_INPUTS}
    out = estimate_ic(expts, lin, DIM, n_boot=0)
    assert out['estimator'] == 'ml_covariant'
    assert out['Ic'] == pytest.approx(theory_metrics(eta)['Ic'], abs=0.01)
    assert out['chi2'] == pytest.approx(out['n_points'], rel=0.3)


def test_covariance_check_rejects_a_non_covariant_channel():
    """A coherence the photon-number structure forbids (rho_03 in '+') must fail the check."""
    r = _ideal(0.5)
    for L, s in (('+', 1), ('+i', 1j)):
        r[L] = r[L].copy()
        r[L][0, 3] += 0.08 * s
        r[L][3, 0] += 0.08 * np.conj(s)
    expts = _fake_set(r, n_shots=200_000)
    lin = {L: WignerData(expts[L].data, DIM).linear() for L in PROC_INPUTS}
    out = estimate_ic(expts, lin, DIM, n_boot=0)
    assert out['cov_check_p'] < 0.02 and out['estimator'] == 'ml_cptp'


S3_JOBS = {L: f'JOB-20260930-{10 + k:05d}' for k, L in enumerate(PROC_INPUTS)}
_HAVE_DATA = bool(glob.glob(r'C:\experiments\*\data\JOB-20260930-00010_WignerTomography1ModeExperiment.h5'))


@pytest.mark.skipif(not _HAVE_DATA, reason='9/30 S3 set only on the measurement PC')
def test_parity_and_linear_match_lab_pipeline():
    from experiments.transduction.process_tomo import load_wigner
    for L in ('0', '+'):
        e = load_wigner(S3_JOBS[L])
        wd = WignerData(e.data, DIM)
        assert np.allclose(wd.parity(), e.data['parity'], atol=1e-12)
        lab = rho_linear(wigner_analysis(e, DIM), e.data['parity'], DIM).full()
        assert np.allclose(wd.linear(), lab, atol=1e-8)
