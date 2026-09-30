# -*- coding: utf-8 -*-
"""EncodingPropagatorProgram: the pulse program of the old jobs -- DEPRECATED.

Moved from `experiments/qsim/mbr_propagator.py` on 2026-09-24 (MBR redesign
step 7e), without changes except imports. New jobs use `MBROrthoColumnProgram`.
Not maintained; may break when live code changes. If it breaks, add a note here
and do not fix it.

The aggregate class ``MBRPropagatorExperiment`` that used to live here moved without
changes to ``experiments/qsim/deprecated/legacy_mbr.py`` (``docs/qsim/mbr_redesign.md``,
section 2).
"""
from experiments.qsim.deprecated.mbr_nphoton_program import NPhotonHamiltonianSpectroscopyProgram


class EncodingPropagatorProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure one raw column of the short-time propagator."""

    def initialize(self):
        ecfg = self.cfg.expt
        cycle_decoder_analyzer = list(ecfg.cycle_decoder_analyzer)
        decoder_occupation = list(cycle_decoder_analyzer[1:-1])

        ecfg.floquet_cycle = int(cycle_decoder_analyzer[0])
        ecfg.spectroscopy_analyzer_phase = float(
            cycle_decoder_analyzer[-1])
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.final_analyzer_phase_per_cycle_deg = 0.
        if ("propagator_occupations" in ecfg
                and ecfg.get("phase_correction_location", "analysis") == "pulse"):
            decoder = ecfg.propagator_occupations.index(decoder_occupation)
            ecfg.final_analyzer_phase_per_cycle_deg = (
                ecfg.propagator_decoder_phase_correction_deg[decoder]
            )
        ecfg.spectroscopy_final_occupations = decoder_occupation

        super().initialize()

    def _get_inverse_pulses(self, _):
        # The parent body requests inverse(encoder); decode the selected row.
        return super()._get_inverse_pulses(self.decoder_encoder_pulses)
