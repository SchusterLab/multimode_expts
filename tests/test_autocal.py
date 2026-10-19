"""experiments.autocal: fit checks on synthetic data, the run loop (apply only on a passed check, stop at the first
failure, snapshots never move main), and the default plan building on a mock station."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fitting import fitting as fitter
from experiments.autocal import AutocalRun, AutocalStop, Outcome, Phase, Step, _checks as chk
from experiments.autocal._core import Change

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------------------- checks
def test_lorentzian_accepts_a_clean_line_and_rejects_noise_and_edges():
    x = np.linspace(-10, 10, 400)
    y = fitter.lorfunc(x, 0.0, 1.0, 1.5, 0.3) + 0.03 * RNG.standard_normal(x.size)
    p, _ = fitter.fitlor(x, y)
    assert chk.lorentzian(x, y, p)[0]
    noise = 0.03 * RNG.standard_normal(x.size)
    assert not chk.lorentzian(x, noise, fitter.fitlor(x, noise)[0])[0]
    y_edge = fitter.lorfunc(x, 0.0, 1.0, 9.8, 0.3) + 0.03 * RNG.standard_normal(x.size)
    ok, info = chk.lorentzian(x, y_edge, fitter.fitlor(x, y_edge)[0])
    assert not ok and 'edge' in info['reason']


def test_gaussian_product_peak():
    x = np.linspace(878.1, 878.4, 75)
    y = fitter.gaussianfunc(x, 0.05, 0.95, 878.27, 0.02) + 0.02 * RNG.standard_normal(x.size)
    p, _ = fitter.fitgaussian(x, y, periodic=False)
    assert chk.gaussian(x, y, p)[0]
    assert not chk.gaussian(x, y, [0.05, 0.1, 878.27, 0.02])[0]          # low contrast


def test_ramsey_band():
    t = np.linspace(0.01, 10, 101)
    p = [0.4, 1.05, 0.0, 30.0, 0.5, 0.01]
    y = fitter.decaysin(t, *p) + 0.01 * RNG.standard_normal(t.size)
    assert chk.ramsey(t, y, p, ramsey_freq=1.0)[0]
    ok, info = chk.ramsey(t, y, [0.4, 1.9, 0.0, 30.0, 0.5, 0.01], ramsey_freq=1.0)
    assert not ok and 'band' in info['reason']


def test_flat_data_fails_every_check():
    """A mock or dead readout gives flat data; a fit to it must never pass (10-19: T1 passed on zeros)."""
    x, y = np.linspace(0, 100, 50), np.zeros(50)
    assert not chk.lorentzian(x, y, [0, 1e-9, 50, 5])[0]
    assert not chk.gaussian(x, y, [0, 1.0, 50, 5])[0]
    assert not chk.ramsey(x, y, [1e-9, 1.0, 0, 30, 0, 0], ramsey_freq=1.0)[0]
    assert not chk.rabi(x, y, [1e-9, 0.01, 0, 30, 0, 0], 50)[0]
    assert not chk.decay(x, y, [0, 1e-9, 0, 30])[0]


def test_decay_and_jump():
    t = np.linspace(0, 2000, 100)
    p = [0.1, 1.0, 0.0, 300.0]
    y = fitter.expfunc(t, *p) + 0.01 * RNG.standard_normal(t.size)
    assert chk.decay(t, y, p)[0]
    assert chk.jump(3561.6, 3561.5, max_abs=1)[0] and not chk.jump(3570, 3561.5, max_abs=1)[0]
    assert not chk.jump(130, 100, max_rel=0.15)[0]


# ----------------------------------------------------------------------------------------- run loop
class FakeStation:
    user = 'pytest'

    def __init__(self):
        self.hardware_cfg = SimpleNamespace(hw=SimpleNamespace(yoko_coupler=SimpleNamespace(current=0.0)))
        self.value = {'a': 1.0, 'b': 2.0}
        self.snapshots = []

    def snapshot_hardware_config(self, update_main=False):
        self.snapshots.append(('HW', update_main)); return 'CFG-HW-TEST'

    def snapshot_man1_storage_swap(self, update_main=False):
        self.snapshots.append(('M1', update_main)); return 'CFG-M1-TEST'


def _config_step(name, key, new, ok):
    def _eval(station, expt, kw):
        old = station.value[key]
        return Outcome(ok=ok, info={} if ok else {'reason': 'synthetic failure'},
                       changes=[Change(key, old, new, lambda: station.value.__setitem__(key, new))])
    return Step(name, None, {}, lambda s: {}, _eval)


def test_run_applies_passed_steps_stops_at_failure_and_never_sets_main(tmp_path):
    st = FakeStation()
    phases = [Phase('p1', [_config_step('p1.a', 'a', 5.0, True)], snapshot_storage=True),
              Phase('p2', [_config_step('p2.b', 'b', 7.0, False), _config_step('p2.a', 'a', 9.0, True)])]
    run = AutocalRun(st, None, phases, out_dir=tmp_path, coupler_A=0.0003)
    with pytest.raises(AutocalStop):
        run.run()
    assert st.value == {'a': 5.0, 'b': 2.0}                 # p1.a applied, p2.b not (failed), p2.a not reached
    assert st.hardware_cfg.hw.yoko_coupler.current == 0.0003
    assert st.snapshots and all(main is False for _, main in st.snapshots)
    rec = json.loads((run.dir / 'record.json').read_text())
    assert [s['step'] for s in rec['steps']] == ['p1.a', 'p2.b']
    assert rec['steps'][1]['ok'] is False and rec['steps'][1]['changes'][0]['new'] == 7.0
    page = run.report.read_text(encoding='utf-8')
    assert 'p2.b' in page and 'FAILED' in page and 'synthetic failure' in page and 'not applied' in page


def test_start_at_and_phase_filter(tmp_path):
    st = FakeStation()
    phases = [Phase('p1', [_config_step('p1.a', 'a', 5.0, True), _config_step('p1.b', 'b', 6.0, True)]),
              Phase('p2', [_config_step('p2.a', 'a', 9.0, True)])]
    run = AutocalRun(st, None, phases, out_dir=tmp_path)
    assert [s.name for _, s in run.steps(start_at='p1.b')] == ['p1.b', 'p2.a']
    assert [s.name for _, s in run.steps(phases=['p2'])] == ['p2.a']
    run.run(phases=['p1'], start_at='p1.b')
    assert st.value == {'a': 1.0, 'b': 6.0}


# ----------------------------------------------------------------------------------------- default plan on a mock station
VERSIONS = Path(r'C:\python\multimode_expts\configs\versions')


@pytest.fixture(scope='module')
def mock_station():
    pytest.importorskip('qick')
    import matplotlib
    matplotlib.use('Agg')
    from experiments import MultimodeStation
    hw = sorted((VERSIONS / 'hardware_config').glob('CFG-HW-*.yml'))
    m1 = sorted((VERSIONS / 'man1_storage_swap').glob('CFG-M1-*.csv'))
    mp = sorted((VERSIONS / 'multiphoton_config').glob('CFG-MP-*.yml'))
    fl = sorted((VERSIONS / 'floquet_storage_swap').glob('CFG-FL-*.csv'))
    if not (hw and m1 and mp and fl):
        pytest.skip('no config version archive on this machine')
    try:
        return MultimodeStation(user='pytest', experiment_name='261019_autocal_test', mock=True, hardware_config=str(hw[-1]),
                                multiphoton_config=str(mp[-1]), storage_man_file=str(m1[-1]), floquet_file=str(fl[-1]))
    except Exception as exc:
        pytest.skip(f'cannot build a mock station here: {type(exc).__name__}: {exc}')


def test_default_plan_builds_every_step_on_a_mock_station(mock_station, tmp_path):
    from experiments.autocal import default_phases
    run = AutocalRun(mock_station, None, default_phases(mock_station), out_dir=tmp_path)
    rows = run.budget(verbose=False)
    names = [r[1] for r in rows]
    assert names[0] == 'readout.res_spec' and names[-1].startswith('storage.M1-S4')
    assert {'ladder.f0-g1.chevron', 'ladder.f1-g2.ea_freq', 'manipulate.gain_to_alpha'} <= set(names)
    assert 0.5 < sum(r[2] for r in rows) / 3600 < 4
    for ph, st in run.steps():                               # every acquisition step keeps the coupler yoko ramp on
        assert st.expt is None or st.defaults.get('avoid_yoko') is False, st.name


@pytest.mark.parametrize('name', ['qubit.t1_ge', 'ladder.f0-g1.lenrabi', 'storage.M1-S4.ea_freq_coarse'])
def test_mock_acquisition_is_rejected_and_nothing_applied(mock_station, tmp_path, name):
    """Mock data is flat: the step runs end to end (compile, acquire, evaluate) and its check must reject it."""
    from experiments.autocal import default_phases

    def state(s):
        dev = s.hardware_cfg.device; ds = s.ds_storage
        return (json.dumps({k: dev[k] for k in ('readout', 'qubit', 'multiphoton', 'manipulate')}, sort_keys=True, default=str),
                [(r, ds.get_freq(r), ds.get_gain(r), ds.get_pi(r)) for r in ('M1', 'M1-S4')])
    before = state(mock_station)
    run = AutocalRun(mock_station, None, default_phases(mock_station), out_dir=tmp_path, log=False)
    with pytest.raises(AutocalStop):
        run.run(start_at=name, stop_at=name)
    rec = run.record['steps'][-1]
    assert rec['step'] == name and rec['ok'] is False
    assert state(mock_station) == before
