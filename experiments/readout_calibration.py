# -*- coding: utf-8 -*-
"""Apply a single-shot histogram result to the station's readout config.

What it does
------------
`HistogramExperiment` measures the g/e blobs; this turns that measurement
into the readout settings every later experiment reads: the rotation angle
that puts the blobs on the I axis, the discrimination threshold, the blob
centres, and the confusion matrix.

Why it is a module
------------------
It was a copy-pasted cell in at least seven notebooks across four users --
byte-identical in all of them when this module was written, which is luck
rather than design. It *writes config*: `device.readout.phase`, `threshold`,
`threshold_list`, `Ie`, `Ig`, and one of two confusion matrices. A drift
between copies would mean two users' readouts calibrated differently, with
nothing failing and nothing recorded to say so.

Notebook use is unchanged in shape -- it is still a `postprocessor`:

    CharacterizationRunner(station=station, ExptClass=meas.HistogramExperiment,
                           default_expt_cfg=singleshot_defaults,
                           postprocessor=apply_singleshot_calibration,
                           job_client=client)

The angle accumulates
---------------------
`phase` is set to the *existing* phase plus the fitted angle, because the
measurement was taken through the current rotation and reports the residual.
Running this twice on one histogram therefore double-rotates; that is
inherent to the quantity, not a defect here, and is why the function takes a
fresh `expt` rather than re-reading the station.
"""
import numpy as np


def apply_singleshot_calibration(station, expt):
    """Analyse a single-shot histogram and write its result into the station.

    Args:
        station: the live `MultimodeStation`; its `hardware_cfg` is mutated
            in place, and nothing is snapshotted -- the caller decides
            whether to persist a config version.
        expt: an acquired `HistogramExperiment`. Analysed here (with plotting
            off) so that the fields written are the ones just fitted.

    Writes, under `hardware_cfg.device.readout`: `phase` (existing plus the
    fitted angle), `threshold`, `threshold_list`, `Ie`, `Ig`, and either
    `confusion_matrix_with_active_reset` or
    `confusion_matrix_without_reset` according to `expt.cfg.expt.active_reset`.
    """
    expt.analyze(plot=False, station=station, subdir=station.autocalib_path)
    fids = expt.data['fids']
    confusion_matrix = expt.data['confusion_matrix']
    thresholds_new = expt.data['thresholds']
    angle = expt.data['angle']
    print(fids)

    hardware_cfg = station.hardware_cfg
    readout = hardware_cfg.device.readout
    readout.phase = [readout.phase[0] + angle]
    readout.threshold = thresholds_new
    readout.threshold_list = [thresholds_new]
    readout.Ie = [np.median(expt.data['Ie_rot'])]
    readout.Ig = [np.median(expt.data['Ig_rot'])]
    if expt.cfg.expt.active_reset:
        readout.confusion_matrix_with_active_reset = confusion_matrix
    else:
        readout.confusion_matrix_without_reset = confusion_matrix
    print('Updated readout!')
