"""Readout and qubit steps, from measurement_notebooks/QEC/single_qubit_autocalibrate_v2.ipynb (cells 14-56).

Settings are the notebook's (defaults + the execute(...) overrides of each cell). Differences, on purpose:
- every fitted value passes a check (``_checks``) before it is applied;
- the EF spectroscopy window is centred on the current f_ef (the notebook starts at a fixed 3415 MHz);
- the Ramsey preprocessor sets prepulse / postpulse to booleans (the notebook's trailing commas made 1-tuples).
"""
import numpy as np

from fitting import fitting as fitter
from fitting.fit_display_classes import RamseyFitting
from experiments.autocal import _checks as chk
from experiments.autocal._core import Outcome, Phase, Step, default_pre
from experiments.autocal._util import item, plot_fit

AVOID_YOKO = False

# --------------------------------------------------------------------------------------------- readout
RESSPEC = dict(expts=250, reps=500, relax_delay=50, pulse_e=False, pulse_f=False, pulse_cavity=False,
               cavity_pulse=[4984.373226159381, 800, 2, 0], qubit=0, avoid_yoko=AVOID_YOKO)


def _resspec_kw(station):
    span, center = 1.5, float(station.hardware_cfg.device.readout.frequency[0])
    return dict(start=center - span / 2, step=span / RESSPEC['expts'])


def _resspec_eval(station, expt, kw):
    """Report only: the notebook does not apply the readout frequency (hanger fit, postprocessor off)."""
    d = expt.data; old = float(station.hardware_cfg.device.readout.frequency[0])
    f = float(d['fit'][0]) if 'fit' in d and d['fit'] is not None else float('nan')
    return Outcome(ok=True, info=dict(fit_freq=f, config_freq=old, shift=f - old, note='not applied (as the notebook)'),
                   plot=plot_fit(d['xpts'], d['amps'], vline=f, xlabel='readout frequency (MHz)', ylabel='amplitude'))


SINGLESHOT = dict(reps=5000, relax_delay=500, check_f=False, active_reset=False, man_reset=False, storage_reset=False, qubit=0,
                  pulse_manipulate=False, cavity_freq=4984.373226159381, cavity_gain=400, cavity_length=2, prepulse=False,
                  pre_sweep_pulse=None, gate_based=True, qubits=[0], avoid_yoko=AVOID_YOKO)
# cell 1: active reset off while the readout is recalibrated
SS_RESET = dict(active_reset=False, ef_reset=True, storage_reset=False, coupler_reset=False, pre_selection_reset=True,
                man_reset=False, use_qubit_man_reset=True, pre_selection_parity=False)


def _ss_eval(station, expt, kw, min_fid=None, min_sep=3.0, min_diag=0.7):
    """min_fid None (the first shot, pi pulse not yet calibrated): the check is the blob separation in widths and the
    confusion diagonal; the rotation and threshold are what this shot is for. min_fid set (the final shot, after the
    qubit phase): the fidelity check of the transduction runs (0.85)."""
    expt.analyze(plot=False, station=station, subdir=station.autocalib_path)
    d = expt.data; ro = station.hardware_cfg.device.readout
    fid = float(np.asarray(d['fids']).ravel()[0]); cm = np.asarray(d['confusion_matrix'], float)
    ig, ie = np.asarray(d['Ig_rot'], float).ravel(), np.asarray(d['Ie_rot'], float).ravel()
    sep = float(abs(np.median(ie) - np.median(ig)) / np.mean([1.4826 * np.median(abs(v - np.median(v))) for v in (ig, ie)]))
    sane = bool(fid < 0.9995 and np.all(np.isfinite(cm)) and 0 < cm.ravel()[1] < 0.3 and 0 < cm.ravel()[2] < 0.3)
    ok = sane and (fid > min_fid if min_fid is not None else (sep > min_sep and cm.ravel()[0] > min_diag and cm.ravel()[3] > min_diag))
    info = dict(fid=fid, separation_widths=sep, angle=float(d['angle']), confusion=cm.tolist())
    if not ok:
        info['reason'] = (f'fidelity {fid:.4f} < {min_fid}' if min_fid is not None and sane else
                          f'blobs not separated ({sep:.1f} widths) or confusion matrix not plausible')
    thr = d['thresholds']
    ch = [item(ro.phase, 0, float(ro.phase[0]) + float(d['angle']), 'readout phase'),
          item(ro, 'threshold', thr, 'readout threshold', cast=lambda v: v),
          item(ro, 'threshold_list', [thr], 'readout threshold_list', cast=lambda v: v),
          item(ro, 'Ie', [float(np.median(d['Ie_rot']))], 'readout Ie', cast=lambda v: v),
          item(ro, 'Ig', [float(np.median(d['Ig_rot']))], 'readout Ig', cast=lambda v: v),
          item(ro, 'confusion_matrix_with_active_reset' if expt.cfg.expt.active_reset else 'confusion_matrix_without_reset',
               d['confusion_matrix'], 'readout confusion matrix', cast=lambda v: np.asarray(v).tolist())]

    def _plot(ax):
        ax.hist(np.asarray(d['Ig_rot']).ravel(), 80, alpha=0.6, label='g'); ax.hist(np.asarray(d['Ie_rot']).ravel(), 80, alpha=0.6, label='e')
        ax.set_xlabel('I rotated'); ax.legend(); ax.text(0.02, 0.92, f'fid {fid:.4f}', transform=ax.transAxes)
    return Outcome(ok=ok, info=info, changes=ch, plot=_plot)


# --------------------------------------------------------------------------------------------- qubit ge / ef
GESPEC = dict(expts=200, reps=100, rounds=1, length=1, gain=100, sigma=0.1, qubit=[0], qubits=[0], prepulse=False,
              pre_sweep_pulse=[], gate_based=False, relax_delay=250, avoid_yoko=AVOID_YOKO)
EFSPEC = dict(start=3415, step=0.05, expts=500, reps=200, rounds=1, length=1, gain=100, qubit_f=False, qubit=0,
              cavity_drive=False, wait_qubit=False, relax_delay=500, avoid_yoko=AVOID_YOKO)


def _spec_eval(key, max_jump):
    def _eval(station, expt, kw):
        d = expt.data; q = station.hardware_cfg.device.qubit
        x, y = np.asarray(d['xpts'], float)[1:-1], np.asarray(d['avgi'], float)[1:-1]
        p, _ = fitter.fitlor(x, y)
        ok, info = chk.both(chk.lorentzian(x, y, p), chk.jump(p[2], q[key][0], max_abs=max_jump))
        return Outcome(ok=ok, info=info, changes=[item(q[key], 0, p[2], f'qubit {key}')],
                       plot=plot_fit(x, y, fitter.lorfunc, p, vline=p[2], xlabel='frequency (MHz)'))
    return _eval


def ramsey_pre(station, defaults, **kw):
    cfg = default_pre(station, defaults, **kw)
    ef = bool(cfg.if_ef)
    cfg.checkEF = ef
    cfg.qubit_ge_init = cfg.qubit_ge_after = bool(ef and cfg.ef_init)
    cfg.prepulse = False if cfg.pre_sweep_pulse is None else (True if cfg.prepulse is None else bool(cfg.prepulse))
    cfg.postpulse = False if cfg.post_sweep_pulse is None else (True if cfg.postpulse is None else bool(cfg.postpulse))
    return cfg


GERAMSEY = dict(ramsey_freq=0.2, start=0.01, step=0.5, expts=101, reps=200, rounds=1, pre_sweep_pulse=None, post_sweep_pulse=None,
                if_ef=False, ef_init=True, qubits=[0], user_defined_freq=[False, 3568.2038290468167, 5304, 0.035], f0g1_cavity=0,
                normalize=False, active_reset=False, man_reset=False, storage_reset=False, prepulse=None, postpulse=None,
                pre_active_reset_pulse=False, gate_based=False, advance_phase=0, echoes=[False, 0], relax_delay=200,
                avoid_yoko=AVOID_YOKO)
EFRAMSEY = dict(GERAMSEY, if_ef=True, ef_init=True, ramsey_freq=3)


def ramsey_delta(expt):
    """(delta f, fit params, ramsey_freq) of a Ramsey job (RamseyFitting; delta = the smallest |f_adjust|)."""
    an = RamseyFitting(expt.data, config=expt.cfg); an.analyze()
    adj = np.asarray(an.data['f_adjust_ramsey_avgi'], float)
    return float(adj[np.argmin(np.abs(adj))]), an.data['fit_avgi'], float(expt.cfg.expt.ramsey_freq)


def _ramsey_eval(key):
    def _eval(station, expt, kw):
        q = station.hardware_cfg.device.qubit
        dlt, p, rf = ramsey_delta(expt); d = expt.data
        ok, info = chk.ramsey(d['xpts'], d['avgi'], p, rf)
        info['delta'] = dlt
        return Outcome(ok=ok, info=info, changes=[item(q[key], 0, float(q[key][0]) + dlt, f'qubit {key}')],
                       plot=plot_fit(d['xpts'], d['avgi'], fitter.decaysin, p, xlabel='time (us)'))
    return _eval


def amprabi_pre(station, defaults, **kw):
    cfg = default_pre(station, defaults, **kw)
    if cfg.sigma_test is None:
        cfg.sigma_test = station.hardware_cfg.device.qubit.pulses.pi_ge.sigma[0]
    cfg.checkEF = cfg.pulse_ge_init = cfg.pulse_ge_after = bool(cfg.if_ef)
    return cfg


AMPRABI = dict(start=0, step=60, expts=151, reps=200, rounds=1, sigma_test=None, qubit=0, pulse_type='gauss', drag_beta=0.0,
               pulse_ge_init=False, pulse_ge_after=False, checkZZ=False, checkEF=False, qubits=[0], flat_length=0,
               normalize=False, single_shot=False, singleshot_reps=10000, span=50, user_defined_freq=[False, 3568.203829046816],
               prepulse=False, postpulse=False, if_ef=False, avoid_yoko=AVOID_YOKO)
EFAMPRABI = dict(AMPRABI, if_ef=True, step=120, expts=150)


def _amprabi_eval(leg):
    def _eval(station, expt, kw):
        d = expt.data; pu = station.hardware_cfg.device.qubit.pulses
        if 'pi_gain_avgi' not in d:
            expt.analyze(data=d)
        pi, hpi, p = int(d['pi_gain_avgi']), int(d['hpi_gain_avgi']), d['fit_avgi']
        ok, info = chk.rabi(d['xpts'], d['avgi'], p, pi)
        info['hpi'] = hpi
        if ok and not (0.3 < hpi / pi < 0.7 and pi < 32000):
            ok, info['reason'] = False, f'hpi / pi = {hpi / pi:.2f} or pi gain {pi} not plausible'
        return Outcome(ok=ok, info=info,
                       changes=[item(pu[f'pi_{leg}'].gain, 0, pi, f'pi_{leg} gain', cast=int),
                                item(pu[f'hpi_{leg}'].gain, 0, hpi, f'hpi_{leg} gain', cast=int)],
                       plot=plot_fit(d['xpts'], d['avgi'], fitter.decaysin, list(p) + ([d['xpts'][0]] if len(p) == 5 else []),
                                     vline=pi, xlabel='gain'))
    return _eval


T1GE = dict(start=0, step=20, expts=100, reps=100, rounds=1, qubit=0, qubit_ef=False, normalize=False, relax_delay=2500,
            avoid_yoko=AVOID_YOKO)
T1EF = dict(T1GE, step=5, reps=50, qubit_ef=True)


def _t1_eval(key, ch):
    def _eval(station, expt, kw):
        d = expt.data; q = station.hardware_cfg.device.qubit
        x, y = np.asarray(d['xpts'], float)[:-1], np.asarray(d[ch], float)[:-1]
        p, _ = fitter.fitexp(x, y)
        ok, info = chk.decay(x, y, p)
        old = q[key][0] if key in q else float('nan')
        c = item(q[key], 0, abs(p[3]), f'qubit {key}') if key in q else None
        return Outcome(ok=ok, info=dict(info, old=old), changes=[c] if c else [],
                       plot=plot_fit(x, y, fitter.expfunc, p, xlabel='time (us)', ylabel=ch))
    return _eval


def phases():
    q = lambda s: s.hardware_cfg.device.qubit
    src = 'single_qubit_autocalibrate_v2'
    readout = Phase('readout', [
        Step('readout.res_spec', 'ResonatorSpectroscopyExperiment', RESSPEC, _resspec_kw, _resspec_eval, source=f'{src} cells 14-15'),
        Step('readout.single_shot', 'HistogramExperiment', SINGLESHOT, lambda s: dict(check_f=False, relax_delay=2500, reps=5000, **SS_RESET),
             _ss_eval, source=f'{src} cells 18-20'),
    ])
    qubit = Phase('qubit', [
        Step('qubit.ge_spec', 'PulseProbeSpectroscopyExperiment', GESPEC,
             lambda s: dict(start=q(s).f_ge[0] - 50, step=100 / 1000, expts=1000, reps=200, span=100, center=q(s).f_ge[0]),
             _spec_eval('f_ge', max_jump=40), source=f'{src} cells 23-24'),
        Step('qubit.ge_ramsey', 'RamseyExperiment', GERAMSEY,
             lambda s: dict(ramsey_freq=1.0, step=0.1, active_reset=False, relax_delay=2500, if_ef=False, echoes=[False, 10]),
             _ramsey_eval('f_ge'), preprocessor=ramsey_pre, source=f'{src} cells 27-28'),
        Step('qubit.ge_amprabi', 'AmplitudeRabiExperiment', AMPRABI, lambda s: dict(relax_delay=2500, step=120),
             _amprabi_eval('ge'), preprocessor=amprabi_pre, source=f'{src} cells 32-33'),
        Step('qubit.ge_ramsey_fine', 'RamseyExperiment', GERAMSEY,
             lambda s: dict(ramsey_freq=0.1, step=1, active_reset=True, relax_delay=250),
             _ramsey_eval('f_ge'), preprocessor=ramsey_pre, source=f'{src} cell 33 (after the amplitude Rabi)'),
        Step('qubit.t1_ge', 'T1Experiment', T1GE, lambda s: {}, _t1_eval('T1', 'avgi'), source=f'{src} cells 36-37'),
        Step('qubit.ef_spec', 'PulseProbeEFSpectroscopyExperiment', EFSPEC,
             lambda s: dict(start=q(s).f_ef[0] - 0.5 * EFSPEC['step'] * EFSPEC['expts']),
             _spec_eval('f_ef', max_jump=10), source=f'{src} cells 41-42 (window centred on f_ef)'),
        Step('qubit.ef_ramsey', 'RamseyExperiment', EFRAMSEY,
             lambda s: dict(ramsey_freq=0.1, step=0.8, relax_delay=2500, active_reset=True),
             _ramsey_eval('f_ef'), preprocessor=ramsey_pre, source=f'{src} cells 44-45'),
        Step('qubit.ef_amprabi', 'AmplitudeRabiExperiment', EFAMPRABI, lambda s: dict(relax_delay=2500),
             _amprabi_eval('ef'), preprocessor=amprabi_pre, source=f'{src} cells 49-50'),
        Step('qubit.ef_ramsey_fine', 'RamseyExperiment', EFRAMSEY,
             lambda s: dict(ramsey_freq=0.1, step=1, relax_delay=2500),
             _ramsey_eval('f_ef'), preprocessor=ramsey_pre, source=f'{src} cell 50 (after the EF amplitude Rabi)'),
        Step('qubit.t1_ef', 'T1Experiment', T1EF, lambda s: {}, _t1_eval('T1_ef', 'avgq'), source=f'{src} cells 53-54'),
        Step('qubit.single_shot_final', 'HistogramExperiment', SINGLESHOT,
             lambda s: dict(check_f=False, relax_delay=2500, reps=5000, **SS_RESET),
             lambda s, e, kw: _ss_eval(s, e, kw, min_fid=0.85),
             source=f'{src} cells 18-20, repeated after the pi pulse calibration (autocal)'),
    ])
    return [readout, qubit]
