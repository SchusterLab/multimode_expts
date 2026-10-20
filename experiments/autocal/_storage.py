"""M1 - storage sideband (default S4, the transduction environment mode).

1. spectroscopy of the M1-Sx sideband (single_qubit_autocalibrate_v2 cell 110 settings), to find the line again;
2. the full-swap calibration at the fixed pi length of the dataset, by error amplification (the transduction
   procedure of 2026-10, ic_debug/losssplit calibrate_swap_conv, seb's settings: 11 pulses, n_step 2, 75 points,
   50 reps): frequency coarse (0.3 MHz) -> gain (+/-15 %) -> frequency fine (0.1 MHz) -> gain fine (+/-5 %) ->
   frequency fine, then a convergence check (gain < 0.5 %, frequency < 1.5 kHz between the two fine rounds).

Frequency guard (10-07): the ~3 kHz product peak can fall between points at a coarse step and the product fit can
then give a false centre. The mirror-symmetry centre of the pattern (the chevron is symmetric about the resonance) is
computed too; the coarse scan uses it when the product fit disagrees, the fine scans require both to agree.
The notebook's sideband chevron (cells 123-125) is not used for S4: it changes the pi length, and the transduction
swap keeps the pi length fixed and calibrates gain and frequency.
"""
import numpy as np

from fitting import fitting as fitter
from experiments.autocal import _checks as chk
from experiments.autocal._core import Outcome, Phase, Step, default_pre
from experiments.autocal._util import ds_item, fit_line, plot_fit, prep_photon, pulses

AVOID_YOKO = False
MAN = 1

STOR_SPEC = dict(step=0.08, expts=250, reps=200, qubit=[0], flux_drive=[0, 1, 3000, 5], freq=None, gain=None, length=None,
                 prepulse=True, postpulse=True, active_reset=False, relax_delay=500, avoid_yoko=AVOID_YOKO)
EA_STOR = dict(reps=50, rounds=1, qubits=[0], man_mode_no=MAN, stor_is_dump=False, active_reset=False, man_reset=True,
               storage_reset=True, relax_delay=2500, expts=75, n_start=0, n_step=2, n_pulses=11, avoid_yoko=AVOID_YOKO)


def stor_spec_pre(station, defaults, **kw):
    """Cell 110: sweep centred on the dataset frequency; M1 |1> before, the f0-g1 and e-f pi pulses after."""
    cfg = default_pre(station, defaults, **kw)
    name = f'M{MAN}-S{cfg.stor_mode_no}'; ds = station.ds_storage
    freq = ds.get_freq(name)
    cfg.freq = freq if cfg.freq is None else cfg.freq
    cfg.gain = ds.get_gain(name) if cfg.gain is None else cfg.gain
    cfg.length = ds.get_pi(name) if cfg.length is None else cfg.length
    bw = cfg.step * cfg.expts
    cfg.start = cfg.freq - bw / 2
    ch = 'low' if freq < 1800 else 'high'
    cfg.flux_drive = [ch, cfg.flux_drive[1], cfg.gain, cfg.length]
    prep = prep_photon(station, 1, MAN)
    cfg.pre_sweep_pulse = pulses(station, prep)
    cfg.post_sweep_pulse = pulses(station, prep[-1:-3:-1])
    return cfg


def sym_center(x, y, min_half=0.25):
    """Mirror-symmetry centre over all rows: the x0 minimising the mean squared difference of y(x0 + d) and y(x0 - d),
    at least min_half of the span on each side."""
    y = (y - y.mean()) / y.std(); n = len(x); best = (np.inf, 0)
    for i in range(int(min_half * n), int((1 - min_half) * n)):
        m = min(i, n - 1 - i); r = np.mean((y[:, i - m:i][:, ::-1] - y[:, i + 1:i + 1 + m]) ** 2)
        if r < best[0]:
            best = (r, i)
    return float(x[best[1]])


def _spec_eval(row):
    def _eval(station, expt, kw):
        d = expt.data
        x, y = np.asarray(d['xpts'], float), np.asarray(d['avgi'], float); p = fit_line(x, y)
        ok, info = chk.both(chk.lorentzian(x, y, p), chk.jump(p[2], station.ds_storage.get_freq(row), max_abs=2.0))
        return Outcome(ok=ok, info=info, changes=[ds_item(station, row, 'freq', p[2])],
                       plot=plot_fit(x, y, fitter.lorfunc, p, vline=p[2], xlabel=f'{row} sideband frequency (MHz)'))
    return _eval


def _ea_kw(row, mode, param, span_of):
    def _kw(station):
        ds = station.ds_storage; n = EA_STOR['expts']
        span = span_of(station)
        if param == 'frequency':
            c = ds.get_freq(row); start, step = c - span / 2, span / (n - 1)
        else:
            c = ds.get_gain(row); start, step = int(c - span / 2), max(1, int(span / (n - 1)))
        return dict(stor_mode_no=mode, parameter_to_test=param, pulse_type=['storage', row, 'pi', 0], start=start, step=step, span=span)
    return _kw


def _ea_eval(row, param, hist, fine):
    """Product-peak fit; for the frequency also the symmetry centre (coarse: fallback, fine: must agree in 3 steps)."""
    def _eval(station, expt, kw):
        d = expt.data
        expt.analyze(data=d, state_fin='e', periodic=False)
        x, prod, p = np.asarray(d['x_pts'], float), np.asarray(d['prod_avgi'], float), d['fit_avgi']
        ok, info = chk.gaussian(x, prod, p)
        val = float(p[2]); info['how'] = 'product'
        if param == 'frequency':
            xs = sym_center(x, np.asarray(d['avgi'], float)); info['symmetry'] = xs; step = float(kw['step'])
            agree = ok and abs(val - xs) <= 3 * step
            if not agree:
                if fine:
                    ok, info['reason'] = False, info.get('reason') or f'product fit {val:.5f} and symmetry centre {xs:.5f} disagree'
                elif abs(xs - (x[0] + x[-1]) / 2) <= 0.22 * (x[-1] - x[0]):
                    ok, val, info['how'] = True, xs, 'symmetry'
                    info.pop('reason', None)
                else:
                    ok, info['reason'] = False, 'symmetry centre at the edge: resonance not inside the scan'
        if ok:                                               # convergence history: accepted values only
            hist.setdefault(param, []).append(val)
        new = val if param == 'frequency' else int(round(val))
        return Outcome(ok=ok, info=info, recentre=bool(info.get('edge_only')) and info.get('how') == 'product',
                       changes=[ds_item(station, row, 'freq' if param == 'frequency' else 'gain', new,
                                        cast=float if param == 'frequency' else int)],
                       plot=plot_fit(x, prod, fitter.gaussianfunc, p, vline=val, xlabel=f'{row} {param}', ylabel='product (normalised)'))
    return _eval


def _converged(row, hist, tol_gain=0.005, tol_freq=0.0015):
    def _eval(station, expt, kw):
        g, f = hist.get('gain', []), hist.get('frequency', [])
        if len(g) < 2 or len(f) < 3:
            return Outcome(ok=False, info=dict(reason='not enough calibration rounds'))
        dg, df = abs(g[-1] - g[-2]) / g[-2], abs(f[-1] - f[-2])
        ok = dg < tol_gain and df < tol_freq
        info = dict(gain_change_rel=dg, freq_change_MHz=df, gains=g, freqs=f)
        if not ok:
            info['reason'] = f'not converged: gain {100 * dg:.2f} % (tol {100 * tol_gain} %), freq {1e3 * df:.2f} kHz (tol {1e3 * tol_freq} kHz)'
        return Outcome(ok=ok, info=info)
    return _eval


def phases(mode=4):
    row = f'M{MAN}-S{mode}'; hist = {}
    gspan = lambda frac: (lambda s: int(frac * s.ds_storage.get_gain(row)))
    fspan = lambda span: (lambda s: span)
    src = 'transduction swap calibration (losssplit calibrate_swap_conv)'
    return [Phase('storage', [
        Step(f'storage.{row}.spec', 'FluxSpectroscopyF0g1Experiment', STOR_SPEC,
             lambda s: dict(stor_mode_no=mode, gain=1000, length=5, step=0.08, expts=400, reps=250),
             _spec_eval(row), preprocessor=stor_spec_pre, source='single_qubit_autocalibrate_v2 cell 110'),
        Step(f'storage.{row}.ea_freq_coarse', 'ErrorAmplificationExperiment', EA_STOR, _ea_kw(row, mode, 'frequency', fspan(0.3)),
             _ea_eval(row, 'frequency', hist, fine=False), source=src),
        Step(f'storage.{row}.ea_gain', 'ErrorAmplificationExperiment', EA_STOR, _ea_kw(row, mode, 'gain', gspan(0.3)),
             _ea_eval(row, 'gain', hist, fine=False), source=src),
        Step(f'storage.{row}.ea_freq_fine', 'ErrorAmplificationExperiment', EA_STOR, _ea_kw(row, mode, 'frequency', fspan(0.1)),
             _ea_eval(row, 'frequency', hist, fine=True), source=src),
        Step(f'storage.{row}.ea_gain_fine', 'ErrorAmplificationExperiment', EA_STOR, _ea_kw(row, mode, 'gain', gspan(0.1)),
             _ea_eval(row, 'gain', hist, fine=False), source=src),
        Step(f'storage.{row}.ea_freq_fine2', 'ErrorAmplificationExperiment', EA_STOR, _ea_kw(row, mode, 'frequency', fspan(0.1)),
             _ea_eval(row, 'frequency', hist, fine=True), source=src),
        Step(f'storage.{row}.converged', None, {}, lambda s: {}, _converged(row, hist), source=src),
    ], snapshot_storage=True)]
