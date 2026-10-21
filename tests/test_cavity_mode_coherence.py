"""Cavity coherence (10-21): MM_base.prep_man_fock_state(n=...), the new CavityModeT1Experiment
(t1_cavity_mode: fit models and analysis) and the fock=2 / phase_cycle options of
CavityModeRamseyExperiment, with end-to-end builds on the mock station (the QICK program build and
ASM compile are real in mock mode; the data are zeros)."""
import json
from pathlib import Path

import numpy as np
import pytest
from slab import AttrDict

from experiments.MM_base import MM_base
from experiments.MM_dual_rail_base import MM_dual_rail_base
from experiments.single_qubit.t1_cavity_mode import CavityModeT1Experiment, exp_decay, parity_decay
from experiments.transduction.decoder import build_encoder_seq


class _Bare:
    """prep_man_fock_state / prep_fock_state do not use the program state."""
    prep_fock_state = MM_dual_rail_base.prep_fock_state


def prep(state, **kw):
    return MM_base.prep_man_fock_state(_Bare(), 1, state, **kw)


def test_prep_man_fock_state_default_unchanged():
    assert prep('0') == []
    assert prep('1') == [['multiphoton', 'g0-e0', 'pi', 0], ['multiphoton', 'e0-f0', 'pi', 0],
                         ['multiphoton', 'f0-g1', 'pi', 0]]
    assert prep('+', broadband=True) == [['multiphoton', 'g0-e0', 'hpi', 0], ['multiphoton', 'e0-f0', 'pi', 0],
                                         ['multiphoton', 'f0-g1', 'pi', 0]]
    assert prep('-i')[0] == ['multiphoton', 'g0-e0', 'hpi', -90]
    with pytest.raises(ValueError):
        prep('x')


def test_prep_man_fock_state_fock_2():
    assert [p[1] for p in prep('2')] == ['g0-e0', 'e0-f0', 'f0-g1', 'g1-e1', 'e1-f1', 'f1-g2']
    # |0> + |2>: the transduction encoder after the g0-e0 hpi input rotation
    zero_two = prep('+', broadband=True, n=2)
    assert zero_two[0] == ['multiphoton', 'g0-e0', 'hpi', 0]
    assert zero_two[1:] == build_encoder_seq(_Bare(), 1)
    assert zero_two == MM_dual_rail_base.prep_fock_state(_Bare(), 1, [0, 2], broadband=True)
    assert zero_two[-1] == ['multiphoton', 'f1-g2', 'pi', 0]


def test_parity_decay_shape():
    t1 = 300.0
    assert parity_decay(0, 0.1, 1.0, t1, fock=1) == pytest.approx(-0.9)           # odd parity
    assert parity_decay(0, 0.1, 1.0, t1, fock=2) == pytest.approx(1.1)            # even parity
    assert parity_decay(t1 * np.log(2), 0.1, 1.0, t1, fock=2) == pytest.approx(0.1)   # P0 - P1 + P2 = 0
    assert parity_decay(1e6, 0.1, 1.0, t1, fock=2) == pytest.approx(1.1)          # vacuum


def _t1_expt(fock, readout, t, y):
    e = CavityModeT1Experiment.__new__(CavityModeT1Experiment)
    e.cfg = AttrDict(dict(expt=dict(mode='manipulate', fock=fock, readout=readout)))
    e.data = dict(xpts=t, avgi=y)
    return e


@pytest.mark.parametrize('fock, readout', [(1, 'map'), (1, 'parity'), (2, 'parity')])
def test_t1_analysis_recovers_t1(fock, readout):
    rng = np.random.default_rng(fock)
    t = np.linspace(0, 1500, 101)
    y = (exp_decay(t, -20.0, 35.0, 340.0) if readout == 'map' else parity_decay(t, -5.0, 22.0, 340.0, fock))
    d = _t1_expt(fock, readout, t, y + rng.normal(0, 0.6, t.size)).analyze()
    assert d['T1'] == pytest.approx(340.0, rel=0.05)
    assert 0 < d['T1_err'] < 20


CONFIGS = Path(__file__).parent / "data" / "cavity_coherence"


@pytest.fixture(scope="module")
def mock_station():
    """Mock station from the committed 10-20 config snapshot (file paths: no database, no archive)."""
    pytest.importorskip("qick")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from experiments import MultimodeStation
    from job_server import JobClient
    configs = {kind: str(CONFIGS / name) for kind, name in json.loads((CONFIGS / "configs.json").read_text()).items()
               if not kind.startswith("_")}
    try:
        station = MultimodeStation(user="pytest", experiment_name="261021_cavity_coherence", project="Coherence",
                                   mock=True, **configs)
    except Exception as exc:  # no soccfg snapshot, ...
        pytest.skip(f"cannot build a mock station here: {type(exc).__name__}: {exc}")
    assert station.is_mock
    return station, JobClient()


def _run(station, client, cls, **cfg):
    from experiments import CharacterizationRunner
    base = dict(start=0.01, step=2.0, expts=6, reps=10, rounds=1, qubits=[0], active_reset=False, relax_delay=200,
                man_mode_idx=1, man_mode_no=1)
    runner = CharacterizationRunner(station=station, ExptClass=cls, default_expt_cfg=AttrDict(dict(base, **cfg)),
                                    postprocessor=None, job_client=client, show=False)
    return runner.execute(go_kwargs=dict(analyze=False, progress=False, display=False))


@pytest.mark.parametrize('cfg', [dict(mode='manipulate', fock=1, readout='map'),
                                 dict(mode='manipulate', fock=1, readout='parity'),
                                 dict(mode='manipulate', fock=2, readout='parity'),
                                 dict(mode='storage', storage_mode_idx=4, fock=1, readout='map')])
def test_t1_cavity_mode_builds_and_runs(mock_station, cfg):
    station, client = mock_station
    expt = _run(station, client, CavityModeT1Experiment, **cfg)
    assert np.asarray(expt.data['avgi']).shape == (6,)
    assert np.allclose(expt.data['xpts'], 0.01 + 2.0 * np.arange(6))


def test_t1_cavity_mode_refuses_map_for_fock_2(mock_station):
    station, client = mock_station
    with pytest.raises(Exception, match="parity"):
        _run(station, client, CavityModeT1Experiment, mode='manipulate', fock=2, readout='map')


@pytest.mark.parametrize('echoes', [0, 1])
def test_cavity_ramsey_fock_2_phase_cycle(mock_station, echoes):
    from experiments.single_qubit.t2_cavity_mode import CavityModeRamseyExperiment
    station, client = mock_station
    sb = station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['frequency']
    f0g1, f1g2 = float(sb[0]), float(sb[1])
    expt = _run(station, client, CavityModeRamseyExperiment, mode='manipulate', fock=2, ramsey_freq=0.05, echoes=echoes,
                parity_meas=False, phase_cycle=True, f0g1_freq=f0g1 + 0.123)
    d = expt.data
    assert np.allclose(d['avgi'], (d['avgi_0'] - d['avgi_180']) / 2) and d['avgi'].shape == (6,)
    prog = expt.prog                                   # the last program built: phase_offset 180
    assert prog.creator.pulse[0][0] == pytest.approx(f1g2)                       # the Ramsey pulse is f1-g2
    prep_freqs = [float(f) for f in prog.auto_prep_pulse[0]]
    assert any(f == pytest.approx(f0g1 + 0.123) for f in prep_freqs)             # override reaches the prep
    assert float(sb[0]) == pytest.approx(f0g1)                                   # and is undone afterwards
    assert prog.cfg.expt.phase_offset == pytest.approx(180.0)


def test_cavity_ramsey_storage_rejects_fock_2(mock_station):
    from experiments.single_qubit.t2_cavity_mode import CavityModeRamseyExperiment
    station, client = mock_station
    with pytest.raises(Exception, match="fock"):
        _run(station, client, CavityModeRamseyExperiment, mode='storage', storage_mode_idx=4, fock=2, ramsey_freq=0.05)


# --- sweep='phase' and mid_pulse (10-21: swap round trips on |0>+|2>) -------------------------------------

def test_fit_fringe_recovers_amplitude_and_phase():
    from experiments.single_qubit.t2_cavity_mode import fit_fringe, fringe
    rng = np.random.default_rng(3)
    phase = np.arange(12) * 30.0
    y = fringe(phase, -4.0, 9.0, 72.0) + rng.normal(0, 0.5, phase.size)
    d = fit_fringe(phase, y)
    assert d['fringe_amp'] == pytest.approx(9.0, abs=4 * d['fringe_amp_err'])
    assert d['fringe_phase_deg'] == pytest.approx(72.0, abs=4 * d['fringe_phase_err_deg'])
    assert 0 < d['fringe_amp_err'] < 1 and 0 < d['fringe_phase_err_deg'] < 10


ROUND_TRIP = [['storage', 'M1-S4', 'pi', 0], ['storage', 'M1-S4', 'pi', 180]]


def test_cavity_ramsey_phase_sweep_with_swap_round_trips(mock_station):
    from experiments.single_qubit.t2_cavity_mode import CavityModeRamseyExperiment
    station, client = mock_station
    expt = _run(station, client, CavityModeRamseyExperiment, mode='manipulate', fock=2, ramsey_freq=0.0, parity_meas=False,
                sweep='phase', expts=12, start=0.05, mid_pulse=ROUND_TRIP * 2)
    assert np.allclose(expt.data['xpts'], 30.0 * np.arange(12))
    prog = expt.prog
    assert len(prog.mid_pulse[0]) == 4                                           # two round trips
    assert prog.creator.pulse[4][0] not in prog.mid_pulse[4]                     # not on the Ramsey channel
    d = expt.analyze()
    assert {'fringe_amp', 'fringe_amp_err', 'fringe_phase_deg'} <= set(d)


def test_cavity_ramsey_mid_pulse_compiled_array(mock_station):
    from experiments.MM_dual_rail_base import MM_dual_rail_base as MMD
    from experiments.single_qubit.t2_cavity_mode import CavityModeRamseyExperiment
    station, client = mock_station
    compiled = MMD(station.hardware_cfg, station.soccfg).get_prepulse_creator(ROUND_TRIP).pulse.tolist()
    expt = _run(station, client, CavityModeRamseyExperiment, mode='manipulate', fock=2, ramsey_freq=0.0, parity_meas=False,
                sweep='phase', expts=12, start=0.05, mid_pulse=compiled, mid_gate_based=False)
    assert [list(map(str, row)) for row in expt.prog.mid_pulse] == [list(map(str, row)) for row in compiled]


def test_cavity_ramsey_refuses_mid_pulse_on_the_ramsey_channel(mock_station):
    from experiments.single_qubit.t2_cavity_mode import CavityModeRamseyExperiment
    station, client = mock_station
    with pytest.raises(Exception, match="Ramsey channel"):
        _run(station, client, CavityModeRamseyExperiment, mode='manipulate', fock=2, ramsey_freq=0.0, parity_meas=False,
             sweep='phase', expts=12, mid_pulse=[['multiphoton', 'f1-g2', 'pi', 0]])


def test_cavity_ramsey_refuses_phase_sweep_with_phase_cycle(mock_station):
    from experiments.single_qubit.t2_cavity_mode import CavityModeRamseyExperiment
    station, client = mock_station
    with pytest.raises(Exception, match="phase_cycle"):
        _run(station, client, CavityModeRamseyExperiment, mode='manipulate', fock=2, ramsey_freq=0.0, sweep='phase',
             expts=12, phase_cycle=True)


def test_t1_cavity_mode_with_mid_pulse(mock_station):
    station, client = mock_station
    expt = _run(station, client, CavityModeT1Experiment, mode='manipulate', fock=2, readout='parity', expts=3,
                mid_pulse=ROUND_TRIP * 3)
    assert len(expt.prog.mid_pulse[0]) == 6
