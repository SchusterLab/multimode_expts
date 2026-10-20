"""Manipulate (M1) properties, from measurement_notebooks/QEC/multiphoton_calibration_v2.ipynb (cells 81-108):
parity revival time -> M1 frequency (parity spectroscopy, then cavity Ramsey) -> displacement scale (gain to alpha).

Differences, on purpose:
- the parity spectroscopy window keeps the notebook's width (60 MHz) but is centred on the current M1 frequency;
- gain_to_alpha / alpha_scale are applied only within 15 % of the old values (the notebook applies them by hand
  after a look at the plot); outside that the step fails and shows the plot.
Not included: the chi / Kerr gain sweep (cells 111-115) needs hand-picked peak indices.
"""
import numpy as np

from fitting import fitting as fitter
from fitting.fit_display_classes import LengthRabiFitting
from experiments.autocal import _checks as chk
from experiments.autocal._core import Outcome, Phase, Step, default_pre
from experiments.autocal._qubit import SINGLESHOT, _ss_eval, ramsey_delta
from experiments.autocal._util import fit_line, item, plot_fit, prep_photon, pulses

AVOID_YOKO = False
MAN = 1
# cell 1 (multiphoton notebook): active reset on
AR = dict(active_reset=True, ef_reset=False, storage_reset=False, coupler_reset=False, pre_selection_reset=True, man_reset=False,
          use_qubit_man_reset=False, pre_selection_parity=True, ps_threshold_sigma=-0)

PARITY_DELAY = dict(start=0.01, step=0.05, expts=50, reps=250, rounds=1, qubits=[0], parity_fast=True, normalize=False,
                    active_reset=False, man_reset=True, gate_based=False, storage_reset=True, man_mode_no=MAN, relax_delay=2500,
                    param_to_calib='length', shift_sign=1, avoid_yoko=AVOID_YOKO)
PARITY_FREQ = dict(start=4960, stop=5020, step=0.60, reps=500, rounds=1, qubits=[0], normalize=False, single_shot=False,
                   singleshot_reps=10000, span=20, man_mode_no=MAN, displace=(True, 0.1, 1000), const_pulse=(False, 1),
                   f0g1_cavity=0, prepulse=False, pre_sweep_pulse=None, relax_delay=2500, avoid_yoko=AVOID_YOKO)
CAV_RAMSEY = dict(start=0.01, step=0.05, expts=200, ramsey_freq=3.7, reps=100, rounds=1, qubits=[0], checkEF=False, f0g1_cavity=0,
                  init_gf=False, active_reset=False, man_reset=True, storage_reset=True, user_defined_pulse=None, parity_meas=True,
                  man_mode_no=MAN, storage_ramsey=[False, 2, True], man_ramsey=None, coupler_ramsey=False, custom_coupler_pulse=None,
                  echoes=[False, 0], prepulse=False, postpulse=False, gate_based=False, pre_sweep_pulse=None, post_sweep_pulse=None,
                  prep_e_first=True, relax_delay=2500, avoid_yoko=AVOID_YOKO)
PARITY_GAIN = dict(start=0, step=20, expts=40, reps=250, rounds=1, qubit=0, qubits=[0], normalize=False, single_shot=False,
                   singleshot_reps=1000, singleshot_active_reset=False, singleshot_man_reset=True, singleshot_storage_reset=True,
                   active_reset=False, man_reset=True, storage_reset=True, span=1000, prep_e=False, manipulate=MAN,
                   displace=None, const_pulse=(False, 1), f0g1_cavity=0, prepulse=False, pre_sweep_pulse=None, relax_delay=2500,
                   pulse_correction=False, parity_fast=True, scaling_amplitude=True, plot_IQ=False, man_mode_no=MAN,
                   avoid_yoko=AVOID_YOKO)


def man(station):
    return station.hardware_cfg.device.manipulate


def parity_delay_pre(station, defaults, **kw):
    cfg = default_pre(station, defaults, **kw)
    if cfg.param_to_calib == 'length':
        cfg.prepulse = True
        cfg.pre_sweep_pulse = pulses(station, prep_photon(station, 1, cfg.man_mode_no))
    else:
        cfg.prepulse, cfg.pre_sweep_pulse = False, []
    return cfg


def cav_ramsey_pre(station, defaults, **kw):
    cfg = default_pre(station, defaults, **kw); m = man(station); i = cfg.man_mode_no - 1
    cfg.man_idx = i
    if cfg.user_defined_pulse is None:
        cfg.user_defined_pulse = [True, m.f_ge[i], 10, m.displace_sigma[i], 0, 4]
    if cfg.man_ramsey is None:
        cfg.man_ramsey = [False, i]
    if cfg.custom_coupler_pulse is None:
        cfg.custom_coupler_pulse = [[944.25], [1000], [0.316677658], [0], [1], ['flat_top'], [0.005]]
    cfg.pre_sweep_pulse = cfg.pre_sweep_pulse or []
    cfg.post_sweep_pulse = cfg.post_sweep_pulse or []
    return cfg


def parity_gain_pre(station, defaults, **kw):
    cfg = default_pre(station, defaults, **kw)
    if cfg.displace is None:
        cfg.displace = (True, man(station).displace_sigma[MAN - 1])
    return cfg


def _revival_eval(station, expt, kw):
    d = expt.data
    an = LengthRabiFitting(d, config=expt.cfg); an.analyze(fitparams=[100, 0.1, 0, None, None, None])
    p = an.results.get('fit_avgi')
    t = float(an._calculate_pi_lengths(list(p))[0])        # as LengthRabiFitting.display (analyze does not set pi_length)
    key = 'revival_time_fast' if expt.cfg.expt.parity_fast else 'revival_time'
    ok, info = chk.both(chk.rabi(d['xpts'], d['avgi'], p, t), chk.jump(t, man(station)[key][MAN - 1], max_rel=0.3))
    pp = list(p) + ([d['xpts'][0]] if p is not None and len(p) == 5 else [])
    return Outcome(ok=ok, info=info, changes=[item(man(station)[key], MAN - 1, t, f'manipulate {key}')],
                   plot=plot_fit(d['xpts'], d['avgi'], fitter.decaysin, pp, vline=t, xlabel='parity delay (us)'))


def _parity_freq_eval(station, expt, kw):
    d = expt.data
    x, y = np.asarray(d['xpts'], float), np.asarray(d['avgi'], float); p = fit_line(x, y)
    ok, info = chk.both(chk.lorentzian(x, y, p), chk.jump(p[2], man(station).f_ge[MAN - 1], max_abs=10))
    return Outcome(ok=ok, info=info, changes=[item(man(station).f_ge, MAN - 1, p[2], 'manipulate f_ge')],
                   plot=plot_fit(x, y, fitter.lorfunc, p, vline=p[2], xlabel='M1 drive frequency (MHz)'))


def _cav_ramsey_eval(station, expt, kw):
    dlt, p, rf, cov = ramsey_delta(expt); d = expt.data
    ok, info = chk.ramsey(d['xpts'], d['avgi'], p, rf, cov=cov); info['delta'] = dlt
    return Outcome(ok=ok, info=info, changes=[item(man(station).f_ge, MAN - 1, float(man(station).f_ge[MAN - 1]) + dlt, 'manipulate f_ge')],
                   plot=plot_fit(d['xpts'], d['avgi'], fitter.decaysin, p, xlabel='time (us)'))


def _parity_gain_eval(station, expt, kw):
    expt.analyze(scale=True, plot=False); d = expt.data; m = man(station)
    g2a, sc = float(d['gain_to_alpha']), float(d['scale'])
    ok, info = chk.both(chk.jump(g2a, m.gain_to_alpha[MAN - 1], max_rel=0.15),
                        chk.jump(sc, m.alpha_scale[MAN - 1], max_rel=0.15))
    info.update(gain_to_alpha=g2a, alpha_scale=sc)
    plot = plot_fit(d['xpts'], d['avgi'], xlabel='displacement gain') if 'xpts' in d and 'avgi' in d else None
    return Outcome(ok=ok, info=info, plot=plot,
                   changes=[item(m.gain_to_alpha, MAN - 1, g2a, 'manipulate gain_to_alpha'),
                            item(m.alpha_scale, MAN - 1, sc, 'manipulate alpha_scale')])


def phases():
    src = 'multiphoton_calibration_v2'
    return [Phase('manipulate', [
        Step('manipulate.revival_time', 'ParityDelayExperiment', PARITY_DELAY,
             lambda s: dict(param_to_calib='length', step=0.02, expts=200, reps=200, parity_fast=False),
             _revival_eval, preprocessor=parity_delay_pre, source=f'{src} cells 81, 88'),
        Step('manipulate.parity_freq', 'ParityFreqExperiment', PARITY_FREQ,
             lambda s: dict(start=man(s).f_ge[MAN - 1] - 30, stop=man(s).f_ge[MAN - 1] + 30),
             _parity_freq_eval, source=f'{src} cells 96-97 (window centred on the M1 frequency)',
             seconds=lambda cfg, kw: (cfg.stop - cfg.start) / cfg.step * cfg.reps * (cfg.relax_delay + 30) * 1e-6),
        Step('manipulate.cavity_ramsey', 'single_qubit.t2_cavity.CavityRamseyExperiment', CAV_RAMSEY,
             lambda s: dict(ramsey_freq=3., step=0.01, expts=100, reps=500, prep_e_first=False,
                            user_defined_pulse=[True, man(s).f_ge[MAN - 1], 1500, man(s).displace_sigma[MAN - 1], 0, 4]),
             _cav_ramsey_eval, preprocessor=cav_ramsey_pre, source=f'{src} cells 99-101'),
        # the parity correction of the Wigner and of the gain-to-alpha scan uses confusion_matrix_with_active_reset; the
        # readout / qubit single shots run without active reset (readout being recalibrated) and never update it (10-19:
        # the gain-to-alpha scan used a 10-17 matrix from another set point, e-e 0.9785 vs 0.964 measured at 21:45)
        Step('manipulate.single_shot_ar', 'HistogramExperiment', SINGLESHOT,
             lambda s: dict(check_f=False, relax_delay=2500, reps=5000, **AR),
             lambda s, e, kw: _ss_eval(s, e, kw, min_fid=0.85), source='losssplit single_shot (Wigner active-reset settings; seb ok 10-20)'),
        Step('manipulate.gain_to_alpha', 'single_qubit.parity_gain.ParityGainExperiment', PARITY_GAIN,
             lambda s: dict(pulse_correction=True, reps=500, step=150, expts=100, parity_fast=False, **AR),
             _parity_gain_eval, preprocessor=parity_gain_pre, source=f'{src} cells 104-108'),
    ])]
