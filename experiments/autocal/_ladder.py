"""Manipulate photon ladder, from measurement_notebooks/QEC/multiphoton_calibration_v2.ipynb (cells 7-65).

Per qubit rung (g_n-e_n, e_n-f_n): error amplification of the pi pulse (frequency, gain) -> of the pi/2 pulse
(frequency, gain). The notebook's Ramsey and amplitude Rabi before it measure the bare-frame frequency and gain; the
error amplification gives the driven-frame values, which the pulses use (they differ at larger drive: 10-19, g0-e0
Ramsey 3561.53 vs error amplification 3562.09 MHz). seb 10-19: skip them; bare_frame=True puts them back. Per sideband (f_n-g_n+1): spectroscopy -> chevron ->
length Rabi -> error amplification (frequency) -> length Rabi. Settings are the notebook's.

Differences, on purpose:
- a seed step before each rung sets its starting values (offsets from `reference`, see phases()). Photon 0: the stored (driven-frame) pi value if it is within
  SEED_KEEP of the new bare qubit value, else the bare value (a cold start; the driven-frame offset, ~0.6 MHz for
  g0-e0, is inside the +/-2 MHz error-amplification window). Photon n > 0: the new rung n-1 plus the offset
  (rung n - rung n-1) of the table at the start of the run, which keeps chi. The notebook seeds the first time from
  the rung below without the offset;
- pi/2 pulses only on the rungs in HPI_RUNGS (default g0-e0: the only multiphoton pi/2 of the logical preparation,
  MM_dual_rail_base.prep_fock_state; the other pi/2 entries of the table are not calibrated). The pi/2 pulse has its
  own shape (sigma 0.02 vs 0.035 for g0-e0, so its gain is close to the pi gain): its seed keeps its gain and follows
  the new pi frequency with the start-of-run pi/2 - pi offset; the amplitude Rabi (pi shape) sets the pi gain only.
  The notebook sets both gains from the Rabi and then centres the pi/2 scans on the pi values, which lands near the
  right pi/2 gain for the same reason;
- the error amplification is centred on the current value (after the Rabi);
- only a sideband (f0-g1) updates the M1 row of the storage dataset; the qubit rungs never do (the notebook's error
  amplification postprocessor wrote any photon-0 frequency, also g0-e0, into that row).
"""
from copy import deepcopy

import numpy as np

from fitting import fitting as fitter
from fitting.fit_display_classes import ChevronFitting, LengthRabiFitting
from experiments.autocal import _checks as chk
from experiments.autocal._core import Outcome, Phase, Step, default_pre
from experiments.autocal._qubit import ramsey_delta
from experiments.autocal._util import ds_item, fit_line, item, plot_fit, plot_image, prep_photon, pulses

AVOID_YOKO = False
MAN = 1
HPI_RUNGS = ('g0-e0',)
SEED_KEEP = dict(frequency=1.5, gain=0.10)        # MHz; relative

RAMSEY = dict(start=0.01, step_size=0.05, expts=75, ramsey_freq=0.2, reps=250, rounds=1, qubits=[0],
              user_defined_freq=[False, 3568.2038290468167, 5304, 0.035], f0g1_cavity=0, normalize=False, active_reset=False,
              man_reset=True, storage_reset=False, prepulse=None, postpulse=None, pre_sweep_pulse=None, post_sweep_pulse=None,
              pre_active_reset_pulse=False, gate_based=False, advance_phase=0, echoes=[False, 0], relax_delay=2500, if_ef=False,
              ef_init=True, transition=None, man_no=MAN, avoid_yoko=AVOID_YOKO)
AMPRABI = dict(start=50, step=450, expts=75, reps=100, rounds=1, sigma_test=None, qubit=0, pulse_type='gauss', drag_beta=0.0,
               pulse_ge_init=False, pulse_ge_after=False, checkZZ=False, checkEF=False, qubits=[0], flat_length=0, normalize=False,
               single_shot=False, singleshot_reps=10000, span=50, user_defined_freq=[False, 3568.203829046816], prepulse=False,
               pre_sweep_pulse=None, post_sweep_pulse=None, postpulse=False, gate_based=False, relax_delay=2500, if_ef=False,
               transition=None, man_no=MAN, avoid_yoko=AVOID_YOKO)
ERRAMP = dict(reps=100, rounds=1, qubit=0, n_pulses=10, active_reset=False, man_reset=True, storage_reset=True, relax_delay=2500,
              start=0, expts=10, step=100, parameter_to_test='gain', pulse_type=['qubit', 'ge', 'pi', 0], man_no=MAN,
              pulse_name='pi', avoid_yoko=AVOID_YOKO)
QSPEC = dict(man_mode_no=MAN, start=1999, step=0.01, expts=500, reps=500, rounds=1, length=1, gain=1000, pulse_type='gaussian',
             qubit_f=False, qubits=[0], transition=None, relax_delay=200, avoid_yoko=AVOID_YOKO)
CHEVRON = dict(start=2, step=0.1, expts=25, reps=100, rounds=1, qubits=[0], gain=8000, ramp_sigma=0.005, use_arb_waveform=False,
               pi_ge_before=False, pi_ef_before=False, pi_ge_after=False, normalize=False, active_reset=False,
               check_man_reset=[False, 0], check_man_reset_pi=[], transition=None, err_amp_reps=0, swap_lossy=False,
               prepulse=False, postpulse=False, pre_sweep_pulse=None, post_sweep_pulse=None, man_mode_no=MAN, avoid_yoko=AVOID_YOKO)
LENRABI = dict(man_mode_no=MAN, start=None, step=0.045, qubits=[0], expts=50, reps=100, rounds=1, gain=8000, freq=None,
               use_arb_waveform=False, pi_ge_before=False, pi_ef_before=False, pi_ge_after=False, normalize=False,
               active_reset=False, man_reset=True, stor_reset=True, check_man_reset=[False, 0], swap_lossy=False,
               check_man_reset_pi=[], prepulse=False, postpulse=False, post_sweep_pulse=None, pre_sweep_pulse=None, err_amp_reps=0,
               relax_delay=2500, avoid_yoko=AVOID_YOKO)


def key_of(tr):
    """'g1-e1' -> 'gn-en'; 'f0-g1' -> 'fn-gn+1' (keys of device.multiphoton)."""
    return tr[0] + 'n-' + tr[3] + ('n+1' if tr[0] == 'f' else 'n')


def mp(station, pulse='pi'):
    return station.hardware_cfg.device.multiphoton[pulse]


# --------------------------------------------------------------------------------------------- preprocessors
def _qubit_rung_pulses(station, tr):
    """Pre / post pulses of a qubit rung (cells 7, 9): |n> in M1, plus g_n-e_n around an e_n-f_n sweep."""
    n = int(tr[1]); pre = prep_photon(station, n, MAN); post = []
    if tr[0] == 'e':
        pre.append(['multiphoton', f'g{n}-e{n}', 'pi', 0]); post = [['multiphoton', f'g{n}-e{n}', 'pi', 0]]
    return pulses(station, pre), pulses(station, post)


def ramsey_pre(station, defaults, **kw):
    cfg = default_pre(station, defaults, **kw)
    cfg.checkEF = False; cfg.qubit_ge_init = cfg.qubit_ge_after = False
    pre, post = _qubit_rung_pulses(station, cfg.transition)
    cfg.pre_sweep_pulse, cfg.post_sweep_pulse = pre, post
    cfg.prepulse, cfg.postpulse = len(pre) > 0, len(post) > 0
    cfg.step = cfg.step_size
    return cfg


def amprabi_pre(station, defaults, **kw):
    cfg = default_pre(station, defaults, **kw)
    if cfg.sigma_test is None:
        cfg.sigma_test = station.hardware_cfg.device.qubit.pulses.pi_ge.sigma[0]
    cfg.checkEF = cfg.pulse_ge_init = cfg.pulse_ge_after = False
    pre, post = _qubit_rung_pulses(station, cfg.transition)
    cfg.pre_sweep_pulse, cfg.post_sweep_pulse = pre, post
    cfg.prepulse, cfg.postpulse = len(pre) > 0, len(post) > 0
    return cfg


def erramp_pre(station, defaults, **kw):
    cfg = default_pre(station, defaults, **kw)
    cfg.qubits = [cfg.qubit]
    return cfg


def sideband_pre(station, defaults, **kw):
    """Cells 44, 49, 53: |n> in M1, then g_n-e_n and e_n-f_n pi pulses, then the swept f_n-g_n+1 sideband."""
    cfg = default_pre(station, defaults, **kw)
    n = int(cfg.transition[1])
    seq = prep_photon(station, n, MAN) + [['multiphoton', f'g{n}-e{n}', 'pi', 0], ['multiphoton', f'e{n}-f{n}', 'pi', 0]]
    cfg.pre_sweep_pulse = pulses(station, seq); cfg.prepulse = len(cfg.pre_sweep_pulse) > 0
    if 'freq' in cfg and cfg.freq is None:
        cfg.freq = mp(station)['fn-gn+1']['frequency'][n]
    if 'start' in cfg and cfg.start is None:
        cfg.start = station.soccfg.cycles2us(3)
    return cfg


# --------------------------------------------------------------------------------------------- evaluation
def _ea_eval(tr, pulse, param, state_fin='g'):
    k, n = key_of(tr), int(tr[1])

    def _eval(station, expt, kw):
        d = expt.data
        expt.analyze(data=d, state_fin=state_fin, periodic=False)
        x, y, p = np.asarray(d['x_pts'], float), np.asarray(d['prod_avgi'], float), d['fit_avgi']
        ok, info = chk.gaussian(x, y, p)
        val = float(p[2]) if param == 'frequency' else int(round(p[2]))
        ch = [item(mp(station, pulse)[k][param], n, val, f'{pulse} {tr} {param}', cast=float if param == 'frequency' else int)]
        if tr[0] == 'f' and n == 0 and param == 'frequency':
            ch.append(ds_item(station, f'M{MAN}', 'freq', val))
        return Outcome(ok=ok, info=info, changes=ch, recentre=bool(info.get('edge_only')),
                       plot=plot_fit(x, y, fitter.gaussianfunc, p, vline=val, xlabel=param, ylabel='product (normalised)'))
    return _eval


def _ramsey_eval(tr):
    k, n = key_of(tr), int(tr[1])

    def _eval(station, expt, kw):
        dlt, p, rf, cov = ramsey_delta(expt); d = expt.data
        ok, info = chk.ramsey(d['xpts'], d['avgi'], p, rf, cov=cov); info['delta'] = dlt
        ch = [item(mp(station, pu)[k]['frequency'], n, float(mp(station, pu)[k]['frequency'][n]) + dlt, f'{pu} {tr} frequency')
              for pu in (('pi', 'hpi') if tr in HPI_RUNGS else ('pi',))]
        return Outcome(ok=ok, info=info, changes=ch, plot=plot_fit(d['xpts'], d['avgi'], fitter.decaysin, p, xlabel='time (us)'))
    return _eval


def _amprabi_eval(tr):
    k, n = key_of(tr), int(tr[1])

    def _eval(station, expt, kw):
        d = expt.data
        if 'pi_gain_avgi' not in d:
            expt.analyze(data=d)
        pi, hpi, p = int(d['pi_gain_avgi']), int(d['hpi_gain_avgi']), d['fit_avgi']
        ok, info = chk.rabi(d['xpts'], d['avgi'], p, pi); info['hpi'] = hpi
        if ok and not (0.3 < hpi / pi < 0.7 and pi < 32000):
            ok, info['reason'] = False, f'hpi / pi = {hpi / pi:.2f} or pi gain {pi} not plausible'
        ch = [item(mp(station, 'pi')[k]['gain'], n, pi, f'pi {tr} gain', cast=int)]       # pi/2: own shape, see docstring
        pp = list(p) + ([d['xpts'][0]] if len(p) == 5 else [])
        return Outcome(ok=ok, info=info, changes=ch, plot=plot_fit(d['xpts'], d['avgi'], fitter.decaysin, pp, vline=pi, xlabel='gain'))
    return _eval


def _qspec_eval(sb):
    n = int(sb[1])

    def _eval(station, expt, kw):
        d = expt.data
        x, y = np.asarray(d['xpts'], float), np.asarray(d['avgi'], float); p = fit_line(x, y)
        # n >= 1: the preparation has more steps, less contrast, and leftover |f_n-1> adds the lower sideband line next
        # to it (10-19 f1-g2: dip 7 deep vs scatter 1.9, signal-to-noise 4.1, centre pulled ~0.3 MHz toward f0-g1).
        # The spectroscopy only places the chevron window; the chevron and the error amplification set the value.
        ok, info = chk.lorentzian(x, y, p, max_width_frac=0.05, min_snr=5.0 if n == 0 else 3.5)
        return Outcome(ok=ok, info=info, changes=[item(mp(station)['fn-gn+1']['frequency'], n, p[2], f'pi {sb} frequency')],
                       plot=plot_fit(x, y, fitter.lorfunc, p, vline=p[2], xlabel='sideband frequency (MHz)'))
    return _eval


def _chevron_eval(sb):
    n = int(sb[1])

    def _eval(station, expt, kw):
        d = expt.data; f = np.asarray(d['freq_sweep'], float)
        an = ChevronFitting(frequencies=f, time=d['xpts'][0], response_matrix=d['avgi'], config=station.hardware_cfg, station=station)
        an.analyze(); best = an.results.get('best_frequency_contrast')
        ok = best is not None and np.isfinite(best) and f.min() < best < f.max()
        info = dict(best=best, lo=float(f.min()), hi=float(f.max()))
        if not ok:
            info['reason'] = 'no chevron centre inside the sweep (at an end or no fit)'
        ch = [item(mp(station)['fn-gn+1']['frequency'], n, best if best is not None else np.nan, f'pi {sb} frequency')]
        if n == 0:
            ch.append(ds_item(station, f'M{MAN}', 'freq', best if best is not None else np.nan))
        return Outcome(ok=bool(ok), info=info, changes=ch,
                       plot=plot_image(d['xpts'][0], f, d['avgi'], xlabel='length (us)', ylabel='frequency (MHz)'))
    return _eval


def _lenrabi_eval(sb):
    n = int(sb[1])

    def _eval(station, expt, kw):
        d = expt.data
        an = getattr(expt, '_length_rabi_analysis', None)
        if an is None:
            an = LengthRabiFitting(d, config=expt.cfg); an.analyze()
        p = an.results.get('fit_avgi', d.get('fit_avgi'))
        pi = float(an.results['pi_length'] if 'pi_length' in an.results else an._calculate_pi_lengths(list(p))[0])
        ok, info = chk.rabi(d['xpts'], d['avgi'], p, pi)
        ch = [item(mp(station)['fn-gn+1']['length'], n, pi, f'pi {sb} length')]
        if n == 0:
            ch.append(ds_item(station, f'M{MAN}', 'pi', pi))
        pp = list(p) + ([d['xpts'][0]] if p is not None and len(p) == 5 else [])
        return Outcome(ok=ok, info=info, changes=ch, plot=plot_fit(d['xpts'], d['avgi'], fitter.decaysin, pp, vline=pi, xlabel='length (us)'))
    return _eval


# --------------------------------------------------------------------------------------------- seeds
def _seed_rung(tr, table0):
    """Config step: starting values of a qubit rung (see the module docstring)."""
    k, n = key_of(tr), int(tr[1])
    leg = 'ge' if tr[0] == 'g' else 'ef'

    def _eval(station, expt, kw):
        q = station.hardware_cfg.device.qubit; ch = []; new_pi = {}; rule = {}
        for par in ('frequency', 'gain'):
            cur = mp(station, 'pi')[k][par]
            if n == 0:
                bare = q[f'f_{leg}'][0] if par == 'frequency' else q.pulses[f'pi_{leg}'].gain[0]
                d = abs(cur[n] - bare) if par == 'frequency' else abs(cur[n] - bare) / bare
                new = cur[n] if d < SEED_KEEP[par] else bare
                rule[par] = 'stored (driven frame)' if d < SEED_KEEP[par] else 'bare qubit (stored value too far)'
            else:
                new = cur[n - 1] + (table0['pi'][k][par][n] - table0['pi'][k][par][n - 1])
                rule[par] = 'rung below + start-of-run offset'
            new_pi[par] = new
            ch.append(item(cur, n, new, f'seed pi {tr} {par}', cast=float if par == 'frequency' else int))
        if tr in HPI_RUNGS:
            off = table0['hpi'][k]['frequency'][n] - table0['pi'][k]['frequency'][n]
            ch.append(item(mp(station, 'hpi')[k]['frequency'], n, new_pi['frequency'] + off, f'seed hpi {tr} frequency'))
            ch.append(item(mp(station, 'hpi')[k]['gain'], n, table0['hpi'][k]['gain'][n], f'seed hpi {tr} gain', cast=int))
        return Outcome(ok=True, info=rule, changes=ch)
    return _eval


def _seed_sideband(sb, table0):
    n = int(sb[1])

    def _eval(station, expt, kw):
        cur = mp(station)['fn-gn+1']['frequency']; f0 = table0['pi']['fn-gn+1']['frequency']
        new = cur[n] if n == 0 else cur[n - 1] + (f0[n] - f0[n - 1])
        return Outcome(ok=True, info=dict(rule='kept' if n == 0 else 'sideband below + start-of-run offset'),
                       changes=[item(cur, n, new, f'seed pi {sb} frequency')])
    return _eval


# --------------------------------------------------------------------------------------------- steps
def rung_steps(tr, table0, bare_frame=False):
    k, n = key_of(tr), int(tr[1]); src = 'multiphoton_calibration_v2'

    def ea_kw(pulse, param, band, expts, reps=None):
        def _kw(station):
            c = mp(station, pulse)[k][param][n]
            kw = dict(transition=tr, parameter_to_test=param, step=2 * band / expts, start=c - band, expts=expts,
                      pulse_type=['multiphoton', tr, pulse, 0])
            if reps:
                kw['reps'] = reps
            return kw
        return _kw
    steps = [
        Step(f'ladder.{tr}.seed', None, {}, lambda s: {}, _seed_rung(tr, table0), source='autocal (see _ladder docstring)'),
    ]
    if bare_frame:
        steps += [
            Step(f'ladder.{tr}.ramsey', 'RamseyExperiment', RAMSEY, lambda s: dict(transition=tr), _ramsey_eval(tr),
                 preprocessor=ramsey_pre, source=f'{src} cells 7, 15-16'),
            Step(f'ladder.{tr}.amprabi', 'AmplitudeRabiExperiment', AMPRABI, lambda s: dict(transition=tr), _amprabi_eval(tr),
                 preprocessor=amprabi_pre, source=f'{src} cells 9, 19-20'),
        ]
    steps += [
        Step(f'ladder.{tr}.ea_pi_freq', 'ErrorAmplificationExperiment', ERRAMP, ea_kw('pi', 'frequency', 2, 50, reps=50),
             _ea_eval(tr, 'pi', 'frequency'), preprocessor=erramp_pre, source=f'{src} cells 24-26'),
        Step(f'ladder.{tr}.ea_pi_gain', 'ErrorAmplificationExperiment', ERRAMP, ea_kw('pi', 'gain', 500, 25),
             _ea_eval(tr, 'pi', 'gain'), preprocessor=erramp_pre, source=f'{src} cell 30'),
    ]
    if tr in HPI_RUNGS:
        steps += [
            Step(f'ladder.{tr}.ea_hpi_freq', 'ErrorAmplificationExperiment', ERRAMP, ea_kw('hpi', 'frequency', 1, 25),
                 _ea_eval(tr, 'hpi', 'frequency'), preprocessor=erramp_pre, source=f'{src} cell 36 (centred on the pi/2 value)'),
            Step(f'ladder.{tr}.ea_hpi_gain', 'ErrorAmplificationExperiment', ERRAMP, ea_kw('hpi', 'gain', 2000, 25),
                 _ea_eval(tr, 'hpi', 'gain'), preprocessor=erramp_pre, source=f'{src} cell 39 (centred on the pi/2 value)'),
        ]
    return steps


def sideband_steps(sb, table0):
    n = int(sb[1]); src = 'multiphoton_calibration_v2'
    f = lambda s: float(mp(s)['fn-gn+1']['frequency'][n])
    g = lambda s: int(mp(s)['fn-gn+1']['gain'][n])
    L = lambda s: float(mp(s)['fn-gn+1']['length'][n])
    return [
        Step(f'ladder.{sb}.seed', None, {}, lambda s: {}, _seed_sideband(sb, table0), source='autocal (see _ladder docstring)'),
        Step(f'ladder.{sb}.qspec', 'PulseProbeF0g1SpectroscopyExperiment', QSPEC,
             lambda s: dict(start=1965, step=(2010 - 1965) / 1499, expts=1500, gain=1000, length=20., transition=sb),
             _qspec_eval(sb), preprocessor=sideband_pre, source=f'{src} cells 44-46'),
        Step(f'ladder.{sb}.chevron', 'LengthRabiGeneralF0g1Experiment', CHEVRON,
             lambda s: dict(transition=sb, sweep_start=f(s) - 5, sweep_stop=f(s), sweep_npts=11, gain=g(s), start=2, batch=True),
             _chevron_eval(sb), preprocessor=sideband_pre, sweep='freq', source=f'{src} cells 49-51'),
        Step(f'ladder.{sb}.lenrabi', 'LengthRabiGeneralF0g1Experiment', LENRABI, lambda s: dict(transition=sb, gain=g(s), freq=f(s)),
             _lenrabi_eval(sb), preprocessor=sideband_pre, source=f'{src} cells 53-55'),
        Step(f'ladder.{sb}.ea_freq', 'ErrorAmplificationExperiment', ERRAMP,
             lambda s: dict(transition=sb, parameter_to_test='frequency', step=0.4 / 50, start=f(s) - 0.2, expts=50, n_pulses=20,
                            n_step=3, pulse_type=['multiphoton', sb, 'pi', 0]),
             _ea_eval(sb, 'pi', 'frequency', state_fin='e'), preprocessor=erramp_pre, source=f'{src} cells 57-58'),
        Step(f'ladder.{sb}.lenrabi_final', 'LengthRabiGeneralF0g1Experiment', LENRABI,
             lambda s: dict(transition=sb, gain=g(s), freq=f(s), expts=100, step=1.5 * L(s) / 100),
             _lenrabi_eval(sb), preprocessor=sideband_pre, source=f'{src} cells 63-64'),
    ]


def phases(station, top=2, bare_frame=False, reference=None):
    """The ladder up to |top> (2 for the transduction logical {0, 2}): g0-e0, e0-f0, f0-g1, g1-e1, e1-f1, f1-g2, ...
    bare_frame: also the notebook's Ramsey and amplitude Rabi on each qubit rung (see the module docstring).
    reference: a device.multiphoton table whose rung-to-rung offsets seed the rungs n > 0 (one consistent earlier
    calibration); None = the table at the start of this run. Pass it when a run resumes inside the ladder: the
    start-of-run table then mixes new lower rungs with old upper ones, and its offsets would carry the old shift."""
    src = reference if reference is not None else station.hardware_cfg.device.multiphoton
    table0 = {pu: deepcopy(src[pu]) for pu in ('pi', 'hpi')}
    steps = []
    for n in range(top):
        steps += rung_steps(f'g{n}-e{n}', table0, bare_frame) + rung_steps(f'e{n}-f{n}', table0, bare_frame) + sideband_steps(f'f{n}-g{n + 1}', table0)
    return [Phase('ladder', steps, snapshot_storage=True)]
