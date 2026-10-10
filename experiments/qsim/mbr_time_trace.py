# -*- coding: utf-8 -*-
"""The return amplitude <final|U(t)|initial> over time: the MBR time trace job.

One job (docs/qsim/mbr_redesign.md, sections 1-3): one (initial, final)
occupation pair, swept over the number of Floquet cycles. ``initial ==
final`` is the diagonal return a spectrum is built from; ``initial != final``
is an off-diagonal element. All four Ramsey phase combinations are inside
the job, so after ``analyze`` it holds one complex 1D array,
``complex_return = Q_0 - i Q_90``, in the phase frame it was acquired in.

The AC Stark shift of the final occupation is removed on the pulse: the
final analyzer half-pi is shifted by ``-floquet_cycle *
final_analyzer_phase_per_cycle_deg``. Its value comes from an
``MBRCalibrationSetExperiment``, whose manifest path the job records in
``cfg.expt.calibration_manifest``.
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from experiments.qsim.mbr_ramsey import MBRJobExperiment, MBRRamseyProgram
from experiments.qsim.mbr_stark_cal import RAMSEY_PHASES


class MBRTimeTraceProgram(MBRRamseyProgram):
    """The Ramsey sequence at one ``floquet_cycle`` and one ``ramsey_phase``.

    ``ramsey_phase`` is the swept [preparation, analyzer] pair; the rest is
    :class:`MBRRamseyProgram` with ``spectroscopy_final_occupations`` as the
    decoded occupation.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        prep_phase, analyzer_phase = ecfg.ramsey_phase
        ecfg.spectroscopy_prep_phase = float(prep_phase)
        ecfg.spectroscopy_analyzer_phase = float(analyzer_phase)
        super().initialize()


class MBRTimeTraceExperiment(MBRJobExperiment):
    """Job: the return of one (initial, final) pair vs Floquet cycles.

    ``acquire`` sweeps ``floquet_cycle`` (outer) and ``ramsey_phase``
    (inner), so ``avgi`` has shape ``(n_cycles, 4)``.
    """

    default_program = MBRTimeTraceProgram

    @staticmethod
    def job_config(initial, final, cycles, swap_stors, phase_per_cycle_deg=0.,
                   calibration_manifest=None, detunings=None, sync_cycles=10,
                   reps=300):
        """-> the expt-config overrides for the job of one occupation pair.

        ``phase_per_cycle_deg`` is the analyzer correction of the *final*
        occupation (``MBRCalibrationSetExperiment.phase_for(final)``) and
        ``calibration_manifest`` the path of the set it came from.
        """
        swap_stors = [int(stor) for stor in swap_stors]
        detunings = [0.] * len(swap_stors) if detunings is None else [float(d) for d in detunings]
        return dict(
            reps=int(reps),
            storage_reset=swap_stors,
            swap_stors=swap_stors,
            detunings=detunings,
            spectroscopy_occupations=[int(n) for n in initial],
            spectroscopy_final_occupations=[int(n) for n in final],
            floquet_cycles=[int(n) for n in cycles],
            ramsey_phases=deepcopy(RAMSEY_PHASES),
            swept_params=["floquet_cycle", "ramsey_phase"],
            scramble_sync_cycles=int(sync_cycles),
            floquet_hardware_loop=False,
            update_phases=True,
            palindrome_scramble=False,
            spectroscopy_phase_correction_mode="final_analyzer",
            final_analyzer_phase_per_cycle_deg=float(phase_per_cycle_deg),
            calibration_manifest=None if calibration_manifest is None else str(calibration_manifest),
        )

    @property
    def initial_occupation(self):
        return tuple(int(n) for n in self.cfg.expt.spectroscopy_occupations)

    @property
    def final_occupation(self):
        ecfg = self.cfg.expt
        return tuple(int(n) for n in ecfg.get("spectroscopy_final_occupations",
                                               ecfg.spectroscopy_occupations))

    def analyze(self, data=None, **kwargs):
        """Complex return per Floquet cycle, as acquired.

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
        cycles = np.asarray(data["ypts"])
        signal = np.asarray(data["avgi"]).reshape(len(cycles), len(ramsey_phases))

        q = self.cfg.expt.qubits[0]
        Ig = self.cfg.device.readout.Ig[q]
        Ie = self.cfg.device.readout.Ie[q]
        if np.isclose(Ig, Ie):
            raise ValueError("Ig and Ie are identical; recalibrate readout")
        Pe = (signal - Ig) / (Ie - Ig)
        q0 = Pe[:, column[(0., 0.)]] - Pe[:, column[(180., 0.)]]
        q90 = Pe[:, column[(0., 90.)]] - Pe[:, column[(180., 90.)]]
        data["Pe"] = Pe
        data["cycles"] = cycles
        data["complex_return"] = q0 - 1j * q90
        return data

    def display(self, data=None, **kwargs):
        """Re and Im of the return vs cycles, and its path in the IQ plane."""
        if data is not None:
            self.data = data
        if "complex_return" not in self.data:
            self.analyze()
        complex_return = np.asarray(self.data["complex_return"])
        cycles = np.asarray(self.data["cycles"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        axes[0].plot(cycles, complex_return.real, "o-", label="Re A")
        axes[0].plot(cycles, complex_return.imag, "o-", label="Im A")
        axes[0].plot(cycles, np.abs(complex_return), "--", color="0.5", label="|A|")
        axes[0].set(xlabel="Floquet cycles", ylabel="return amplitude")
        axes[0].legend()
        points = axes[1].scatter(complex_return.real, complex_return.imag, c=cycles,
                                 cmap="viridis", s=20)
        axes[1].set(xlabel=r"$Q_0$", ylabel=r"$-Q_{90}$", title="IQ path")
        axes[1].set_aspect("equal", adjustable="datalim")
        fig.colorbar(points, ax=axes[1], label="Floquet cycles")

        initial, final = self.initial_occupation, self.final_occupation
        title = (str(initial) if initial == final
                 else rf"$\langle {final}|U(t)|{initial}\rangle$")
        fname = getattr(self, "fname", None)
        if fname:
            title += "\n" + str(fname).replace("\\", "/").split("/")[-1]
        fig.suptitle(title)
        return fig
