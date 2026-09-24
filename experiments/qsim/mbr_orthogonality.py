# -*- coding: utf-8 -*-
"""EncodingOrthogonalityProgram: the pulse program of the old jobs.

The aggregate class ``MBROrthogonalityExperiment`` that used to live here moved without
changes to ``experiments/qsim/deprecated/legacy_mbr.py`` (``docs/qsim/mbr_redesign.md``,
section 2).
"""
from experiments.qsim.mbr_spectroscopy_program import (
    NPhotonHamiltonianSpectroscopyProgram,
)


class EncodingOrthogonalityProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure coherent cross-return amplitudes between encoder paths.

    One job fixes ``spectroscopy_occupations`` (the encoded column). Its outer
    software sweep packs decoder occupation and analyzer phase into
    ``decoder_analyzer_row``; the inner sweep is the usual preparation phase
    ``0/180``. Only zero Floquet cycles are accepted, so this probes the access
    paths rather than Floquet time evolution.
    """
    def initialize(self):
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        decoder_occupations = [list(occupation) for occupation in ecfg.orthogonality_decoder_occupations]
        analyzer_phases = ecfg.orthogonality_analyzer_phases
        decoder_analyzer_row = int(ecfg.decoder_analyzer_row)
        decoder_index = decoder_analyzer_row // 2
        analyzer_phase_index = decoder_analyzer_row % 2
        decoder_occupation = decoder_occupations[decoder_index]
        ecfg.spectroscopy_analyzer_phase = float(
            analyzer_phases[analyzer_phase_index])
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.final_analyzer_phase_per_cycle_deg = 0.
        ecfg.floquet_cycle = 0
        ecfg.spectroscopy_final_occupations = decoder_occupation

        super().initialize()

    def _get_inverse_pulses(self, _):
        """
        Overrides ``_get_inverse_pulses`` in the parent ``NPhotonHamiltonianSpectroscopyProgram``
        for a given decoder_occupation

        """

        # Parent body asks for inverse(encoder); use the selected decoder here.
        return super()._get_inverse_pulses(self.decoder_encoder_pulses)
