"""Small helpers shared by the step modules: config changes, pulse sequences, plots."""
import numpy as np

from fitting import fitting as fitter
from experiments.autocal._checks import r_squared
from experiments.autocal._core import Change


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def item(container, key, new, label, cast=float):
    """Change of container[key] (a list element, a dict / AttrDict entry) to cast(new)."""
    old = container[key]
    new = cast(new)

    def _set():
        container[key] = new

    def _undo():
        container[key] = old
    return Change(label=label, old=as_float(old), new=new, set=_set, undo=_undo)


def ds_item(station, row, column, new, cast=float):
    """Change of one value of the man1-storage-swap dataset (station.ds_storage)."""
    ds = station.ds_storage
    get = dict(freq=ds.get_freq, gain=ds.get_gain, pi=ds.get_pi, h_pi=ds.get_h_pi)[column]
    upd = dict(freq=ds.update_freq, gain=ds.update_gain, pi=ds.update_pi, h_pi=ds.update_h_pi)[column]
    new, old = cast(new), get(row)
    return Change(label=f'ds {row} {column}', old=as_float(old), new=new, set=lambda: upd(row, new), undo=lambda: upd(row, old))


def fit_line(x, y):
    """Lorentzian fit of a spectroscopy line, robust to a narrow line on a wide scan: fitlor's default start guess
    and a start at the strongest point (3-point median smoothed; width a few steps); keeps the better fit (R^2).
    10-19: the f0-g1 line at 1992.9 MHz, ~0.5 MHz wide on a 45 MHz scan, was missed by the default guess."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    med = float(np.median(y))
    ys = np.median(np.vstack([np.roll(y, -1), y, np.roll(y, 1)]), axis=0); ys[0], ys[-1] = y[0], y[-1]
    i = int(np.argmax(np.abs(ys - med)))
    tries = []
    for start in (None, [med, float(ys[i] - med), float(x[i]), 3 * float(abs(x[1] - x[0]))]):
        try:
            p, _ = fitter.fitlor(x, y, fitparams=None if start is None else list(start))
            if np.all(np.isfinite(p)):
                tries.append((r_squared(y, fitter.lorfunc(x, *p)), list(p)))
        except Exception:
            pass
    return max(tries)[1] if tries else [float('nan')] * 4     # nan: the check reports 'no fit'


def mm_base(station):
    from experiments.MM_dual_rail_base import MM_dual_rail_base
    return MM_dual_rail_base(station.hardware_cfg, station.soccfg)


def pulses(station, seq):
    """Compiled pulse list (the format of pre_sweep_pulse) of a gate sequence, with the current config."""
    if not seq:
        return []
    return mm_base(station).get_prepulse_creator(seq, station.hardware_cfg).pulse.tolist()


def prep_photon(station, n, man_no=1):
    """Gate sequence that prepares |n> in manipulate mode man_no (as the notebooks: prep_man_photon)."""
    return list(mm_base(station).prep_man_photon(man_no=man_no, photon_no=n)) if n > 0 else []


def plot_fit(x, y, model=None, p=None, xlabel='', ylabel='I (a.u.)', vline=None, sl=slice(None)):
    """Plot function for Outcome.plot: data, the fit, and the applied value."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def _plot(ax):
        ax.plot(x[sl], y[sl], 'o', ms=3, color='#2a78d6', label='data')
        if model is not None and p is not None and np.all(np.isfinite(p)):
            xx = np.linspace(x[sl].min(), x[sl].max(), 600)
            ax.plot(xx, model(xx, *p), color='#c2410c', lw=1.8, label='fit')
        if vline is not None and np.isfinite(vline):
            ax.axvline(vline, color='#1baf7a', ls='--', lw=1.5, label=f'{vline:.6g}')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(color='#e3e3df', lw=0.6)
    return _plot


def plot_image(x, y, z, xlabel='', ylabel='', vline=None):
    def _plot(ax):
        ax.pcolormesh(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float), shading='auto', cmap='viridis')
        if vline is not None and np.isfinite(vline):
            ax.axvline(vline, color='w', ls='--', lw=1.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return _plot
