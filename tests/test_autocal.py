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


def test_stop_inside_a_phase_saves_the_passed_steps(tmp_path):
    """10-19: the run stopped at the EF Ramsey and the passed GE values of the phase were lost."""
    st = FakeStation()
    phases = [Phase('p1', [_config_step('p1.a', 'a', 5.0, True), _config_step('p1.b', 'b', 7.0, False)])]
    run = AutocalRun(st, None, phases, out_dir=tmp_path)
    with pytest.raises(AutocalStop):
        run.run()
    assert st.value == {'a': 5.0, 'b': 2.0}
    assert run.record['snapshots'] == [{'phase': 'p1', 'hardware': 'CFG-HW-TEST', 'partial, stopped at': 'p1.b'}]
    assert all(main is False for _, main in st.snapshots)


def test_gaussian_rejects_a_pattern_the_fit_does_not_describe():
    """10-19 e0-f0 at a wrong gain: broad, with a second rise at the edge; product and fit agree poorly."""
    x = np.linspace(3416, 3420, 50)
    y = fitter.gaussianfunc(x, 0.0, 0.8, 3418.9, 0.7) + np.where(x > 3419.6, 0.8, 0.0) + 0.12 * RNG.standard_normal(x.size)
    assert not chk.gaussian(x, y, [0.0, 0.8, 3418.9, 0.7])[0]
    xr = np.linspace(-1, 1, 50); yr = fitter.gaussianfunc(xr, 0, 1, 0, 0.1) + 0.3 * np.sin(9 * xr)   # SNR fine, shape wrong
    ok, info = chk.gaussian(xr, yr, [0, 1, 0, 0.1], min_snr=0)
    assert not ok and 'R^2' in info['reason']


def test_stop_file_stops_between_steps_and_keeps_the_passed_values(tmp_path):
    st = FakeStation()

    def _eval_and_touch(station, expt, kw):                  # step 1 passes and asks for a stop
        (run.dir / 'STOP').touch()
        return Outcome(ok=True, changes=[Change('a', 1.0, 5.0, lambda: station.value.__setitem__('a', 5.0))])
    phases = [Phase('p1', [Step('p1.a', None, {}, lambda s: {}, _eval_and_touch), _config_step('p1.b', 'b', 7.0, True)])]
    run = AutocalRun(st, None, phases, out_dir=tmp_path)
    with pytest.raises(AutocalStop, match='STOP'):
        run.run()
    assert st.value == {'a': 5.0, 'b': 2.0}
    assert run.record['snapshots'][0]['partial, stopped at'] == 'p1.b'


def _edge_then(second_ok):
    """A step whose first scan has its clean peak at the edge (recentre) and whose repeat passes or fails."""
    calls = []

    def _eval(station, expt, kw):
        calls.append(station.value['a'])
        if len(calls) == 1:
            return Outcome(ok=False, recentre=True, info={'reason': 'center at the edge'},
                           changes=[Change('a', 1.0, 3.0, lambda: station.value.__setitem__('a', 3.0),
                                           lambda: station.value.__setitem__('a', 1.0))])
        new = 3.1
        return Outcome(ok=second_ok, info={} if second_ok else {'reason': 'still bad'},
                       changes=[Change('a', station.value['a'], new, lambda: station.value.__setitem__('a', new),
                                       lambda: None)])
    return _eval, calls


@pytest.mark.parametrize('second_ok, final', [(True, 3.1), (False, 1.0)])
def test_edge_peak_moves_the_centre_and_repeats_once(tmp_path, monkeypatch, second_ok, final):
    st = FakeStation()
    ev, calls = _edge_then(second_ok)
    step = Step('p1.ea', 'FakeExperiment', {}, lambda s: {}, ev)
    run = AutocalRun(st, None, [Phase('p1', [step])], out_dir=tmp_path)
    monkeypatch.setattr(run, '_runner', lambda st_: SimpleNamespace(execute=lambda **kw: None, last_job_ids=['JOB-X']))
    if second_ok:
        run.run()
    else:
        with pytest.raises(AutocalStop):
            run.run()
    assert calls == [1.0, 3.0]                                # the repeat scanned around the moved centre
    assert st.value['a'] == final                            # failed repeat: back to the value before the first scan
    assert [s.get('recentred_retry') for s in run.record['steps']] == ['next', None]


def test_sharp_peak_finds_the_resonance_not_the_off_resonant_plateau():
    """10-19 M1-S4 coarse scan: off-resonant plateau ~0.6 with spikes up to 1.0; the resonance a narrow 0.9 peak
    between dark bands."""
    from experiments.autocal._storage import sharp_peak
    y = np.r_[0.6 + 0.15 * RNG.standard_normal(35), 0.05 * np.ones(15), [0.4, 0.9, 0.8, 0.4], 0.05 * np.ones(21)]
    y[20] = 1.0
    c, i = sharp_peak(y)
    assert i == 51 and c > 0.7


def test_ramsey_frequency_error_limit():
    t = np.linspace(0.01, 80, 101)
    p = [20.0, 0.0887, 0.0, 44.0, -10.0, 0.0]
    y = fitter.decaysin(t, *p) + 6.0 * RNG.standard_normal(t.size)      # 10-19 EF Ramsey: start amplitude / residual ~3.4
    cov = np.diag([1, 1e-6, 1, 1, 1, 1])
    assert chk.ramsey(t, y, p, 0.1, cov=cov)[0]
    ok, info = chk.ramsey(t, y, p, 0.1, cov=np.diag([1, 0.03 ** 2, 1, 1, 1, 1]))
    assert not ok and 'fit error' in info['reason']


def test_photon0_seed_keeps_the_driven_frame_value_unless_far_from_the_bare_qubit():
    from slab import AttrDict
    from experiments.autocal import _ladder

    def station(f_table, g_table):
        rung = dict(frequency=[f_table, 3561.0], gain=[g_table, 9000], length=[0, 0], sigma=[0.035, 0.035])
        cfg = AttrDict(dict(device=dict(qubit=dict(f_ge=[3561.52], f_ef=[3417.98], pulses=dict(pi_ge=dict(gain=[9077]))),
                                        multiphoton=dict(pi={'gn-en': rung}, hpi={'gn-en': deepcopy_rung(rung)}))))
        return SimpleNamespace(hardware_cfg=cfg)

    def deepcopy_rung(r):
        return {k: list(v) for k, v in r.items()}

    warm = station(3562.09, 9204)                            # stored driven-frame values, 0.57 MHz / 1.4 % from bare
    out = _ladder._seed_rung('g0-e0', {'pi': warm.hardware_cfg.device.multiphoton.pi,
                                       'hpi': warm.hardware_cfg.device.multiphoton.hpi})(warm, None, {})
    assert [c.new for c in out.changes][:2] == [3562.09, 9204]
    cold = station(3566.68, 9449)                            # before the cooldown: 5.2 MHz away
    out = _ladder._seed_rung('g0-e0', {'pi': cold.hardware_cfg.device.multiphoton.pi,
                                       'hpi': cold.hardware_cfg.device.multiphoton.hpi})(cold, None, {})
    assert [c.new for c in out.changes][:2] == [3561.52, 9449]      # frequency from the bare qubit; gain within 10 %: kept


def test_ladder_rungs_skip_the_bare_frame_steps_by_default():
    from experiments.autocal import _ladder
    names = [s.name for s in _ladder.rung_steps('g1-e1', {})]
    assert names == ['ladder.g1-e1.seed', 'ladder.g1-e1.ea_pi_freq', 'ladder.g1-e1.ea_pi_gain']
    assert 'ladder.g0-e0.ramsey' in [s.name for s in _ladder.rung_steps('g0-e0', {}, bare_frame=True)]


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


@pytest.mark.parametrize('name', ['qubit.t1_ge', 'ladder.f0-g1.lenrabi', 'ladder.f1-g2.n1_map', 'manipulate.single_shot_ar',
                                  'storage.M1-S4.ea_freq_coarse'])
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


def test_sideband_ends_with_the_n1_map_and_gain_to_alpha_follows_an_active_reset_shot(mock_station, tmp_path):
    """seb 10-20: the ladder needs the single-pulse sideband optimum (10-02 drive memory); the gain-to-alpha scan the
    active-reset confusion matrix the Wigner uses."""
    from experiments.autocal import default_phases
    names = [st.name for ph in default_phases(mock_station) for st in ph.steps]
    for sb in ('f0-g1', 'f1-g2'):
        i = names.index(f'ladder.{sb}.n1_map')
        assert names[i - 1] == f'ladder.{sb}.ea_freq' and names[i + 1] == f'ladder.{sb}.lenrabi_final'
    i = names.index('manipulate.gain_to_alpha')
    assert names[i - 1] == 'manipulate.single_shot_ar'
    run = AutocalRun(mock_station, None, default_phases(mock_station), out_dir=tmp_path)
    kw, cfg, _ = run.build(next(st for ph in run.phases for st in ph.steps if st.name == 'manipulate.single_shot_ar'))
    assert cfg.active_reset is True and cfg.pre_selection_reset is True and cfg.pre_selection_parity is True


def test_vacuum_width_fit_recovers_gain_to_alpha():
    """B' (10-20): offset fixed at 0, centre at 0; the fit error matches the scatter of the synthetic runs."""
    from experiments.autocal._manipulate import G2A_PTS, G2A_START, G2A_STEP, vacuum_width_fit
    x = G2A_START + G2A_STEP * np.arange(G2A_PTS)
    vals = []
    for k in range(30):
        y = 0.94 * np.exp(-2 * (2.58e-4 * x) ** 2) + 0.016 * RNG.standard_normal(x.size)       # 0.032 at 500 reps -> 0.016 at 2000
        g, sg, c, sc, o, rms = vacuum_width_fit(x, y, 2.6e-4)
        vals.append(g)
        assert abs(c - 0.94) < 0.03 and o == 0.0
    vals = np.array(vals)
    assert abs(vals.mean() / 2.58e-4 - 1) < 0.004 and 0.5 * sg < vals.std() < 2 * sg and vals.std() / 2.58e-4 < 0.01


def test_gain_to_alpha_step_uses_the_flank_grid(mock_station, tmp_path):
    from experiments.autocal import default_phases
    run = AutocalRun(mock_station, None, default_phases(mock_station), out_dir=tmp_path)
    st = next(st for ph in run.phases for st in ph.steps if st.name == 'manipulate.gain_to_alpha')
    kw, cfg, _ = run.build(st)
    assert (cfg.start, cfg.step, cfg.expts, cfg.reps, cfg.rounds) == (0, 112, 41, 500, 4)
    assert 6 < st.seconds(cfg, kw) / 60 < 8
