"""Optional offline IQ correction of saved MBR jobs, without changing raw HDF5.

This preserves the two-cloud refit from the legacy data_postprocess notebook.
It assumes the saved g/e contrast sign is correct and cannot repair active reset.
"""
from copy import copy, deepcopy

import h5py
import numpy as np
from slab import AttrDict

from fitting.qsim.readout import fit_readout_clouds


def refit_readout(expt, max_samples=50000):
    """Return an analysis-only job copy and its IQ-fit provenance.

    Fit all final-read shots with deterministic sampling; allow the constant
    ADC offset between raw-shot and averaged-buffer paths only after checking
    that it explains their difference. Reject postselected acquisitions.
    """
    cfg = expt.cfg.expt
    if cfg.get('pre_selection_reset', False) or cfg.get('post_select_pre_pulse', False):
        raise ValueError('readout refit does not support postselected acquisitions')
    means_i, means_q = np.asarray(expt.data['avgi']), np.asarray(expt.data['avgq'])
    if means_i.shape != means_q.shape or means_i.ndim != 2:
        raise ValueError('expected matching 2D spectroscopy averages')
    reps = int(cfg.reps)
    rng = np.random.default_rng(0)
    with h5py.File(expt.fname, 'r') as handle:
        idata, qdata = handle['idata'], handle['qdata']
        if (idata.ndim != 2 or idata.shape != qdata.shape
                or idata.shape[0] != means_i.size or idata.shape[1] % reps):
            raise ValueError('saved shot axes do not match sweep points and reps')
        npoints, nreads = idata.shape[0], idata.shape[1] // reps
        if nreads < 1:
            raise ValueError('no final readout in saved shots')
        per_point = min(reps, max(1, max_samples // npoints))
        average, samples = np.empty((npoints, 2)), []
        for start in range(0, npoints, 64):
            stop = min(start + 64, npoints)
            ii = np.asarray(idata[start:stop, nreads-1::nreads], dtype=float)
            qq = np.asarray(qdata[start:stop, nreads-1::nreads], dtype=float)
            if not np.isfinite(ii).all() or not np.isfinite(qq).all():
                raise ValueError('non-finite final-read shots')
            average[start:stop] = np.column_stack((ii.mean(axis=1), qq.mean(axis=1)))
            pick = rng.choice(reps, per_point, replace=False)
            samples.append(np.column_stack((ii[:, pick].ravel(), qq[:, pick].ravel())))
    saved = np.column_stack((means_i.ravel(), means_q.ravel()))
    offset = np.median(saved - average, axis=0)
    if not np.allclose(saved, average + offset, rtol=1e-5, atol=1e-4):
        raise ValueError('saved means differ from final-shot means beyond a constant ADC offset')
    qubit = cfg.qubits[0]
    ig, ie = expt.cfg.device.readout.Ig[qubit], expt.cfg.device.readout.Ie[qubit]
    fit = fit_readout_clouds(np.concatenate(samples) + offset, ie - ig,
                            max_samples=max_samples)
    fit.update(source_file=str(expt.fname), adc_offset=offset,
               old_Ig=float(ig), old_Ie=float(ie))
    theta = np.deg2rad(fit['angle_deg'])
    corrected = copy(expt)
    corrected.cfg, corrected.data = deepcopy(expt.cfg), AttrDict(dict(expt.data))
    corrected.data['avgi'] = means_i * np.cos(theta) - means_q * np.sin(theta)
    corrected.data['avgq'] = means_i * np.sin(theta) + means_q * np.cos(theta)
    corrected.data['amps'] = np.abs(corrected.data['avgi'] + 1j*corrected.data['avgq'])
    corrected.data['phases'] = np.angle(corrected.data['avgi'] + 1j*corrected.data['avgq'])
    corrected.cfg.device.readout.Ig[qubit] = fit['Ig']
    corrected.cfg.device.readout.Ie[qubit] = fit['Ie']
    for key in ('Pe', 'complex_return', 'return_quadrature'):
        corrected.data.pop(key, None)
    corrected.analyze()
    return corrected, fit
