"""Transduction process tomography: one pipeline for data and theory."""
import glob
import tempfile

import numpy as np
import pytest
import qutip as qt

from experiments.transduction.channel_model import (
    apply_logical_z, channel_output, coherent_information, entanglement_fidelity,
)
from experiments.transduction.decoder import eta_to_swap_ratio
from experiments.transduction.process_tomo import (
    DIM, ChannelSet, analyze_set, fast_alpha_list, fast_pops, fast_report, ideal_outputs,
    linearity, load_result, load_wigner, measure_set, metrics, save_result, theory_metrics,
)
from experiments.transduction.sequences import PROC_INPUTS, build_logical_prep, channel_prep

ETAS = [0.1, 0.35, 0.6, 0.87]


@pytest.mark.parametrize('eta', ETAS)
def test_theory_matches_analytic_channel(eta):
    N00, N02, N20, N22, _, _ = channel_output(eta, DIM)
    Fe = entanglement_fidelity(N00, N20, N02, N22, eta, DIM, decode=True, physical=True)
    Ic = coherent_information(N00, N20, N02, N22, eta, DIM, decode=False)
    th = theory_metrics(eta)
    assert th['Fe'] == pytest.approx(Fe, abs=1e-9)
    assert th['Ic'] == pytest.approx(Ic, abs=1e-9)
    assert th['phi_ch'] == pytest.approx(0.0, abs=1e-9)


def test_theory_values_at_035():
    th = theory_metrics(0.35)
    assert th['Ic'] == pytest.approx(0.211, abs=1e-3)
    assert th['Fe'] == pytest.approx(0.752, abs=1e-3)


@pytest.mark.parametrize('rung', [1, 2])
@pytest.mark.parametrize('eta', ETAS)
def test_hardware_phase_is_removed(eta, rung):
    """A logical-Z phase on all outputs must not change Fe or Ic (in Rung 2 too:
    the old Rung-2 code removed the ideal pi as well)."""
    rot = {L: apply_logical_z(r, 1.234) for L, r in ideal_outputs(eta, rung=rung).items()}
    got, th = metrics(rot, eta, rung=rung), theory_metrics(eta, rung=rung)
    for k in ('Fe', 'Fe_phys', 'Ic'):
        assert got[k] == pytest.approx(th[k], abs=1e-9)
    assert got['phi_ch'] == pytest.approx(np.degrees(1.234), abs=1e-6)


@pytest.mark.parametrize('eta', ETAS)
def test_linearity_zero_for_ideal_channel(eta):
    for v in linearity(ideal_outputs(eta)).values():
        assert np.allclose(v, 0, atol=1e-12)


def test_fast_alpha_list():
    pts = fast_alpha_list(4)
    assert len(pts) == 1 + 4 * 11 and pts[0] == [0.0, 0.0]


class _FakeMM:
    """Stand-in for MM_dual_rail_base: compiles each gate to one column, length 1."""
    def prep_fock_state(self, man_no, photons, broadband=False):
        if len(photons) == 2:
            return [['multiphoton', 'g0-e0', 'hpi', 0], ['multiphoton', 'f1-g2', 'pi', 0]]
        return [['multiphoton', f'f{k}-g{k + 1}', 'pi', 0] for k in range(photons[0])]

    def get_prepulse_creator(self, seq):
        class _P:
            pulse = np.array([[hash(tuple(g)) % 997 for g in seq], [0] * len(seq), [1.0] * len(seq),
                              [g[3] for g in seq], [0] * len(seq), [0] * len(seq), [0] * len(seq)], dtype=float)
        return _P()


def test_logical_prep_phase_correction():
    mm = _FakeMM()
    assert build_logical_prep(mm, '+', 30.0)[0][3] == 30.0
    assert build_logical_prep(mm, '1', 30.0) == mm.prep_fock_state(1, [2])


@pytest.mark.parametrize('eta', [0.2, 0.35, 0.9])
def test_channel_prep_layout(eta):
    """env prep (4 gates) + encode + one swap column scaled to the eta ratio."""
    mm = _FakeMM()
    seq = channel_prep(mm, eta, '+', env_stor=4)
    assert len(seq) == 7
    assert len(seq[2]) == 4 + 2 + 1
    assert seq[2][-1] == pytest.approx(eta_to_swap_ratio(eta))
    assert seq[2][:-1] == [1.0] * 6


S3_SET = {L: f'JOB-20260930-{10 + k:05d}' for k, L in enumerate(PROC_INPUTS)}
_HAVE_DATA = bool(glob.glob(r'C:\experiments\*\data\JOB-20260930-00010_WignerTomography1ModeExperiment.h5'))


@pytest.mark.skipif(not _HAVE_DATA, reason='9/30 S3 set only on the measurement PC')
def test_real_set_reproduces_known_numbers():
    s = ChannelSet(eta=0.35, env_stor=3, jobs=dict(S3_SET))
    res = analyze_set(s, n_boot=0, n_boot_ml=0)
    assert res['Ic_linear'] == pytest.approx(0.005, abs=2e-3)
    assert res['Fe'] == pytest.approx(0.491, abs=2e-3)
    # joint ML (estimator study, 2026-09-30: E3ml +0.013, chi2 401 / 404 points)
    assert res['estimator'] == dict(Fe='linear', Ic='ml_covariant')
    assert res['Ic'] == pytest.approx(0.013, abs=2e-3)
    assert res['chi2'] == pytest.approx(401, abs=2) and res['n_points'] == 404
    with tempfile.TemporaryDirectory() as d:
        back = load_result(save_result(d, s, res))
    assert back['Ic'] == pytest.approx(res['Ic'])
    assert back['jobs'] == S3_SET
    assert np.allclose(back['rhos']['+'].full(), res['rhos']['+'].full())


class _ReplayRunner:
    """Stands in for the Wigner runner: 'measures' by loading stored jobs, in order."""
    def __init__(self, jobs):
        self.jobs, self.calls, self.last_job_ids = list(jobs), [], []

    def execute(self, overrides=None, batch_size=10, **kwargs):
        self.calls.append(dict(overrides=overrides, batch_size=batch_size, **kwargs))
        self.last_job_ids = self.jobs[:len(overrides)]
        return [load_wigner(j) for j in self.last_job_ids]


@pytest.mark.skipif(not _HAVE_DATA, reason='9/30 S3 set only on the measurement PC')
def test_measure_set_notebook_path():
    """Section 4 path: measure_set (all inputs in one batch) -> analyze_set."""
    runner = _ReplayRunner([S3_SET[L] for L in PROC_INPUTS])
    s = measure_set(runner, _FakeMM(), 0.35, 3, reps=1000, enc_phase_corr_deg=12.0, relax_delay=8000)
    call = runner.calls[0]
    assert call['batch_size'] == 4 and call['measure_parity'] is False and call['prepulse'] is True
    # env (4) + encode ('0': 0, '1': 2, '+'/'+i': 2 fake gates) + swap (1)
    assert [len(o['pre_sweep_pulse'][2]) for o in call['overrides']] == [5, 7, 7, 7]
    assert s.jobs == S3_SET and s.meta['enc_phase_corr_deg'] == 12.0
    assert analyze_set(s, n_boot=0, ic_estimator='linear')['Ic'] == pytest.approx(0.005, abs=2e-3)
    pops = {L: fast_pops(e) for L, e in s.expts.items()}
    assert all(abs(p.sum() - 1) < 1e-9 for p in pops.values())
    assert set(fast_report(pops)['linearity']) == {'+', '+i'}
