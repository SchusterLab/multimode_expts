# -*- coding: utf-8 -*-
"""One column <decoder|U(q)|initial> over decoders: the MBR ortho column job.

One job (docs/qsim/mbr_redesign.md, sections 1-3): one initial occupation at
one number of Floquet cycles ``q`` (default 0), swept over the decoded
occupation. All four Ramsey phase combinations are inside the job, so after
``analyze`` it holds one complex 1D array, ``complex_return = Q_0 - i Q_90``
per decoder, in the phase frame it was acquired in.

``MBROrthogonalityExperiment`` assembles the columns of one ``q`` into the
matrix ``M_q``; at ``q = 0`` that is the encoder/decoder overlap matrix.

The AC Stark shift of each decoder is removed on the pulse: the final
analyzer half-pi is shifted by ``-q * decoder_phase_per_cycle_deg[decoder]``.
The values come from an ``MBRCalibrationSetExperiment``, whose manifest path
the job records in ``cfg.expt.calibration_manifest``.

Converted old jobs whose correction was left to analysis carry it as
``cfg.expt.analysis_phase_per_cycle_deg`` instead; new jobs never set it.
``complex_return`` is always as acquired; the assembled class applies it.
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.mbr_ramsey import MBRJobExperiment, MBRRamseyProgram
from experiments.qsim.mbr_stark_cal import RAMSEY_PHASES


class MBROrthoColumnProgram(MBRRamseyProgram):
    """The Ramsey sequence for one ``decoder_occupation`` and one ``ramsey_phase``.

    ``floquet_cycle`` is the fixed ``q`` of the job. The analyzer correction
    is the entry of ``decoder_phase_per_cycle_deg`` for the current decoder.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        prep_phase, analyzer_phase = ecfg.ramsey_phase
        decoder = [int(n) for n in ecfg.decoder_occupation]
        decoders = [[int(n) for n in occupation] for occupation in ecfg.decoder_occupations]
        ecfg.spectroscopy_prep_phase = float(prep_phase)
        ecfg.spectroscopy_analyzer_phase = float(analyzer_phase)
        ecfg.spectroscopy_final_occupations = decoder
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.final_analyzer_phase_per_cycle_deg = float(
            ecfg.decoder_phase_per_cycle_deg[decoders.index(decoder)])
        super().initialize()


class MBROrthoColumnExperiment(MBRJobExperiment):
    """Job: the return of one initial occupation into every decoder, at one ``q``.

    ``acquire`` sweeps ``decoder_occupation`` (outer) and ``ramsey_phase``
    (inner), so ``avgi`` has shape ``(n_decoders, 4)``.
    """

    default_program = MBROrthoColumnProgram

    @staticmethod
    def job_config(initial, decoders, swap_stors, cycle=0, decoder_phases_deg=None,
                   calibration_manifest=None, detunings=None, sync_cycles=10, reps=300):
        """-> the expt-config overrides for the column of one initial occupation.

        ``decoder_phases_deg`` is the analyzer correction per decoder, in
        ``decoders`` order (``MBRCalibrationSetExperiment.phase_for``), and
        ``calibration_manifest`` the path of the set it came from.
        """
        swap_stors = [int(stor) for stor in swap_stors]
        decoders = [[int(n) for n in occupation] for occupation in decoders]
        detunings = [0.] * len(swap_stors) if detunings is None else [float(d) for d in detunings]
        decoder_phases_deg = ([0.] * len(decoders) if decoder_phases_deg is None
                              else [float(phase) for phase in decoder_phases_deg])
        if len(decoder_phases_deg) != len(decoders):
            raise ValueError("need one decoder phase per decoder")
        return dict(
            reps=int(reps),
            storage_reset=swap_stors,
            swap_stors=swap_stors,
            detunings=detunings,
            spectroscopy_occupations=[int(n) for n in initial],
            decoder_occupations=decoders,
            decoder_phase_per_cycle_deg=decoder_phases_deg,
            floquet_cycle=int(cycle),
            ramsey_phases=deepcopy(RAMSEY_PHASES),
            swept_params=["decoder_occupation", "ramsey_phase"],
            scramble_sync_cycles=int(sync_cycles),
            floquet_hardware_loop=False,
            update_phases=True,
            palindrome_scramble=False,
            spectroscopy_phase_correction_mode="final_analyzer",
            calibration_manifest=None if calibration_manifest is None else str(calibration_manifest),
        )

    @property
    def initial_occupation(self):
        return tuple(int(n) for n in self.cfg.expt.spectroscopy_occupations)

    @property
    def decoder_occupations(self):
        return [tuple(int(n) for n in occupation)
                for occupation in self.cfg.expt.decoder_occupations]

    @property
    def cycle(self):
        return int(self.cfg.expt.floquet_cycle)

    @property
    def analysis_phase_per_cycle_deg(self):
        """-> the correction still owed in analysis, per decoder (zeros for new jobs)."""
        phases = self.cfg.expt.get("analysis_phase_per_cycle_deg", None)
        if phases is None:
            return np.zeros(len(self.decoder_occupations))
        return np.asarray(phases, dtype=float)

    def analyze(self, data=None, **kwargs):
        """Complex return per decoder, as acquired.

        ``Pe`` is the excited-state probability per point; each quadrature is
        the difference between preparation phases 0 and 180 at one analyzer
        phase, and ``complex_return = Q_0 - i Q_90``.
        """
        if data is not None:
            self.data = data
        data = self.data
        ramsey_phases = [tuple(float(p) for p in pair)
                         for pair in np.asarray(data["xpts"]).tolist()]
        column = {pair: index for index, pair in enumerate(ramsey_phases)}
        decoders = np.asarray(data["ypts"])
        signal = np.asarray(data["avgi"]).reshape(len(decoders), len(ramsey_phases))

        q = self.cfg.expt.qubits[0]
        Ig = self.cfg.device.readout.Ig[q]
        Ie = self.cfg.device.readout.Ie[q]
        if np.isclose(Ig, Ie):
            raise ValueError("Ig and Ie are identical; recalibrate readout")
        Pe = (signal - Ig) / (Ie - Ig)
        q0 = Pe[:, column[(0., 0.)]] - Pe[:, column[(180., 0.)]]
        q90 = Pe[:, column[(0., 90.)]] - Pe[:, column[(180., 90.)]]
        data["Pe"] = Pe
        data["decoder_occupations"] = decoders
        data["complex_return"] = q0 - 1j * q90
        return data

    def display(self, data=None, **kwargs):
        """|A| and phase per decoder, and the column in the IQ plane."""
        if data is not None:
            self.data = data
        if "complex_return" not in self.data:
            self.analyze()
        complex_return = np.asarray(self.data["complex_return"])
        labels = [str(occupation) for occupation in self.decoder_occupations]
        rows = np.arange(len(labels))

        fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, 0.25 * len(labels) + 2)),
                                 constrained_layout=True)
        axes[0].barh(rows, np.abs(complex_return))
        axes[0].set_yticks(rows)
        axes[0].set_yticklabels(labels)
        axes[0].invert_yaxis()
        axes[0].set(xlabel="|A|", ylabel="decoder occupation")
        axes[1].scatter(complex_return.real, complex_return.imag, s=20)
        axes[1].axhline(0., color="0.85")
        axes[1].axvline(0., color="0.85")
        axes[1].set(xlabel=r"$Q_0$", ylabel=r"$-Q_{90}$", title="IQ plane")
        axes[1].set_aspect("equal", adjustable="datalim")

        title = f"{self.initial_occupation} at q = {self.cycle}"
        fname = getattr(self, "fname", None)
        if fname:
            title += "\n" + str(fname).replace("\\", "/").split("/")[-1]
        fig.suptitle(title)
        return fig
