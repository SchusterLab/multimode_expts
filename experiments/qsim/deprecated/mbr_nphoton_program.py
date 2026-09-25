# -*- coding: utf-8 -*-
"""NPhotonHamiltonianSpectroscopyProgram, the old diagonal program -- DEPRECATED.

Moved from `experiments/qsim/mbr_ramsey.py` on 2026-09-24 (MBR redesign step
7e), without changes. It is an empty subclass of `MBRRamseyProgram`; new jobs
use `MBRTimeTraceProgram`. See `docs/qsim/mbr_step7_plan.md`.

Not maintained; may break when live code changes. If it breaks, add a note here
and do not fix it.
"""
from experiments.qsim.mbr_ramsey import MBRRamseyProgram


class NPhotonHamiltonianSpectroscopyProgram(MBRRamseyProgram):
    """The old diagonal spectroscopy program: one analyzer phase per job.

    Same sequence as :class:`MBRRamseyProgram`, with
    ``spectroscopy_prep_phase`` swept and ``spectroscopy_analyzer_phase`` a
    scalar in the config.
    """
