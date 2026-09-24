# -*- coding: utf-8 -*-
"""EntireFloquetCyclePhaseCalibrationProgram: the pulse program of the old jobs.

The aggregate class ``MBRPhaseCorrectionExperiment`` that used to live here moved without
changes to ``experiments/qsim/deprecated/legacy_mbr.py`` (``docs/qsim/mbr_redesign.md``,
section 2). The pulses now live in ``MBRStarkCalProgram``; this subclass keeps
the old one-analyzer-phase-per-job config working until its callers move.
"""
from experiments.qsim.mbr_stark_cal import MBRStarkCalProgram


class EntireFloquetCyclePhaseCalibrationProgram(MBRStarkCalProgram):
    """The old calibration program: one analyzer phase per job.

    Same pulses as :class:`MBRStarkCalProgram`, which now owns them. Here the
    analyzer phase is a scalar in the config (``spectroscopy_analyzer_phase``)
    and only the preparation phase is swept, so each occupation needs two
    jobs.

    ``n_physical_cycle`` also permits odd guide points (``phase_unwrap_mode:
    odd_guide``). An odd point plays all complete forward/inverse pairs
    followed by one forward cycle.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        if ecfg.get("phase_unwrap_mode", "pair") == "odd_guide":
            n_physical_cycle = int(ecfg.n_physical_cycle)
        else:
            n_physical_cycle = 2 * int(ecfg.n_cycle_pair)
        ecfg.spectroscopy_prep_phase = float(
            ecfg.get("spectroscopy_prep_phase", 0.0))
        ecfg.spectroscopy_analyzer_phase = float(
            ecfg.get("spectroscopy_analyzer_phase", 0.0))
        self._initialize_closed_cycles(n_physical_cycle)
