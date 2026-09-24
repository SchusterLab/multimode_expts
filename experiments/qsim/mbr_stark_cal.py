# -*- coding: utf-8 -*-
"""Closed-cycle phase of one occupation: the MBR Stark-shift calibration job.

One job (docs/qsim/mbr_redesign.md, sections 1-3): one occupation, swept over
the number of closed Floquet cycle pairs. All four Ramsey phase combinations
-- preparation half-pi at 0/180 deg, analyzer half-pi at 0/90 deg -- are
inside the same job, so after ``analyze`` it holds one complex 1D array, the
return ``A = Q_0 - i Q_90`` per cycle-pair count.

What it measures. A closed pair is an ordered Floquet cycle followed by its
reverse-order inverse, which cancels the intended exchange motion while
repeatable diagonal phase (the AC Stark shift) accumulates. The phase slope of
``A`` against the number of physical cycles (two per pair) is the phase per
entire Floquet cycle, determined only modulo 180 deg/cycle because the pairs
sit two physical cycles apart.

``MBRCalibrationSetExperiment`` assembles these jobs over occupations and
turns them into the analyzer phase correction.
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.dark_base import DarkBaseExperiment
from experiments.qsim.mbr_ramsey import MBRRamseyProgram
from fitting.qsim import mbr_phase

# [preparation, analyzer] half-pi phases in degrees, the inner sweep of a job.
RAMSEY_PHASES = [[0., 0.], [180., 0.], [0., 90.], [180., 90.]]


class MBRStarkCalProgram(MBRRamseyProgram):
    """Prepare one occupation, play closed cycle pairs, analyze.

    One closed pair is

        ordered Floquet cycle -> reverse-order inverse Floquet cycle.

    The inverse cycle uses the same tracked logical axes as spectroscopy and
    adds 180 degrees to every beam-splitter pulse. ``n_cycle_pair`` pairs
    contain ``2 * n_cycle_pair`` physical entire cycles.

    ``ramsey_phase`` is the swept [preparation, analyzer] pair.
    ``final_analyzer_phase_per_cycle_deg`` is used only for an end-to-end sign
    check: it is multiplied by the number of physical cycles and subtracted
    from the final qubit half-pi.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        prep_phase, analyzer_phase = ecfg.ramsey_phase
        ecfg.spectroscopy_prep_phase = float(prep_phase)
        ecfg.spectroscopy_analyzer_phase = float(analyzer_phase)
        self._initialize_closed_cycles(2 * int(ecfg.n_cycle_pair))

    def _initialize_closed_cycles(self, n_physical_cycle):
        """Checks and settings shared with the old calibration program."""
        ecfg = self.cfg.expt
        name = type(self).__name__
        if n_physical_cycle < 0:
            raise ValueError("n_physical_cycle must be non-negative")
        if "spectroscopy_occupations" not in ecfg:
            raise ValueError(f"{name} requires spectroscopy_occupations")
        if ecfg.get("floquet_hardware_loop", False):
            raise ValueError(
                "The exact multi-mode forward/inverse calibration currently "
                "uses the software-emitted pulse sequence; set "
                "floquet_hardware_loop=False"
            )

        detunings = ecfg.get("detunings", None)
        if detunings is not None and detunings is not False \
                and np.asarray(detunings).size > 0 \
                and not np.allclose(detunings, 0.0):
            raise ValueError(
                "Entire-cycle phase calibration uses zero detuning.  "
                "Disorder is part of the target Hamiltonian and must not be "
                "calibrated out."
            )

        ecfg.final_analyzer_phase_per_cycle_deg = float(ecfg.get(
            "final_analyzer_phase_per_cycle_deg", 0.0
        ))
        ecfg.zero_floquet_gain = bool(ecfg.get(
            "zero_floquet_gain", False
        ))
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.n_physical_cycle = n_physical_cycle
        ecfg.floquet_cycle = 0
        ecfg.palindrome_scramble = False
        ecfg.ro_stor = 0
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        n_physical_cycle = int(ecfg.n_physical_cycle)
        n_cycle_pair = n_physical_cycle // 2
        phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        prepulse_cfg = [[
            "qubit", "ge", "hpi",
            float(ecfg.spectroscopy_prep_phase),
        ]] + deepcopy(self.encoder_pulses)
        prepulse_cfg = self._add_wait_after_storage_pulses(
            prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="entire_cycle_phase_pre_")
        self.sync_all()

        self._play_closed_floquet_cycle_pairs(
            n_cycle_pair=n_cycle_pair,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            extra_forward=n_physical_cycle % 2,
        )

        postpulse_cfg = self._get_inverse_pulses(
            self.encoder_pulses)
        analyzer_phase = (
            float(ecfg.spectroscopy_analyzer_phase)
            - n_physical_cycle
            * float(ecfg.final_analyzer_phase_per_cycle_deg)
        )
        postpulse_cfg.append([
            "qubit", "ge", "hpi", self._mod360(analyzer_phase),
        ])
        postpulse_cfg = self._add_wait_after_storage_pulses(
            postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="entire_cycle_phase_post_")
        self.sync_all()
        self.measure_wrapper()


class MBRStarkCalExperiment(DarkBaseExperiment):
    """Job: the closed-cycle return of one occupation vs cycle pairs.

    ``acquire`` sweeps ``n_cycle_pair`` (outer) and ``ramsey_phase`` (inner),
    so ``avgi`` has shape ``(n_cycle_pairs, 4)``. ``analyze`` turns it into
    ``complex_return`` and fits the phase per physical cycle.
    """

    def __init__(self, soccfg=None, path='', prefix=None, config_file=None,
                 expt_params=None, program=None, progress=None, **kwargs):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, expt_params=expt_params,
                         program=program or MBRStarkCalProgram,
                         progress=progress, **kwargs)

    @staticmethod
    def job_config(occupation, cycle_pairs, swap_stors, sync_cycles=10, reps=1500):
        """-> the expt-config overrides for the job of one occupation.

        Merged over the runner's default config, which carries the settings
        every MBR job shares (reset, readout, Floquet waveform).
        """
        swap_stors = [int(stor) for stor in swap_stors]
        return dict(
            reps=int(reps),
            storage_reset=swap_stors,
            swap_stors=swap_stors,
            spectroscopy_occupations=[int(n) for n in occupation],
            n_cycle_pairs=[int(n) for n in cycle_pairs],
            ramsey_phases=deepcopy(RAMSEY_PHASES),
            swept_params=["n_cycle_pair", "ramsey_phase"],
            scramble_sync_cycles=int(sync_cycles),
            floquet_hardware_loop=False,
            detunings=[0.] * len(swap_stors),  # disorder is not calibrated out
            update_phases=True,
            palindrome_scramble=False,
            final_analyzer_phase_per_cycle_deg=0.,
        )

    @property
    def occupation(self):
        return tuple(int(n) for n in self.cfg.expt.spectroscopy_occupations)

    def analyze(self, data=None, **kwargs):
        """Complex return per cycle-pair count, and its phase per cycle.

        ``Pe`` is the excited-state probability per point; each quadrature is
        the difference between preparation phases 0 and 180 at one analyzer
        phase, and ``complex_return = Q_0 - i Q_90``. The phase fit is stored
        beside it (see :func:`fitting.qsim.mbr_phase.fit_closed_cycle_phase`).
        If the data cannot be fitted, the fit fields are NaN and a message is
        printed; this never raises, so a job whose fit fails still saves.
        """
        if data is not None:
            self.data = data
        data = self.data
        ramsey_phases = [tuple(float(p) for p in pair)
                         for pair in np.asarray(data["xpts"]).tolist()]
        column = {pair: index for index, pair in enumerate(ramsey_phases)}
        cycle_pairs = np.asarray(data["ypts"])
        signal = np.asarray(data["avgi"]).reshape(len(cycle_pairs), len(ramsey_phases))

        q = self.cfg.expt.qubits[0]
        Ig = self.cfg.device.readout.Ig[q]
        Ie = self.cfg.device.readout.Ie[q]
        if np.isclose(Ig, Ie):
            raise ValueError("Ig and Ie are identical; recalibrate readout")
        Pe = (signal - Ig) / (Ie - Ig)
        q0 = Pe[:, column[(0., 0.)]] - Pe[:, column[(180., 0.)]]
        q90 = Pe[:, column[(0., 90.)]] - Pe[:, column[(180., 90.)]]

        data["Pe"] = Pe
        data["physical_cycles"] = 2 * cycle_pairs
        data["complex_return"] = q0 - 1j * q90
        try:
            data.update(mbr_phase.fit_closed_cycle_phase(
                data["complex_return"], data["physical_cycles"]))
        except ValueError as error:
            print(f"[{type(self).__name__}] {self.occupation}: no phase fit ({error})")
            nan = np.full(len(cycle_pairs), np.nan)
            data.update(return_phase=nan, phase_fit=nan.copy(),
                        relative_return=np.abs(data["complex_return"]),
                        valid_mask=np.zeros(len(cycle_pairs), dtype=bool),
                        phase_per_cycle=np.nan, phase_error=np.nan)
        return data

    def display(self, data=None, fig=None, **kwargs):
        """Raw complex return, relative magnitude and the phase fit."""
        if data is not None:
            self.data = data
        data = self.data
        if "complex_return" not in data:
            self.analyze()
        complex_return = np.asarray(data["complex_return"])
        cycles = np.asarray(data["physical_cycles"])

        if fig is None:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[4., 1.])
        iq_axis = fig.add_subplot(grid[0, 0])
        relative_axis = fig.add_subplot(grid[1, 0])
        phase_axis = fig.add_subplot(grid[:, 1])

        iq_axis.plot(complex_return.real, complex_return.imag, "o--", color="black",
                     linewidth=1.2, markersize=4, label="raw return")
        points = iq_axis.scatter(complex_return.real, complex_return.imag, c=cycles,
                                 cmap="viridis", s=28, zorder=3)
        iq_limit = 1.15 * np.nanmax(np.abs(complex_return))
        if not np.isfinite(iq_limit) or iq_limit <= 0.:
            iq_limit = 1.
        iq_axis.axhline(0., color="0.85")
        iq_axis.axvline(0., color="0.85")
        iq_axis.set(xlim=(-iq_limit, iq_limit), ylim=(-iq_limit, iq_limit),
                    xlabel=r"$Q_0$", ylabel=r"$-Q_{90}$", title="raw complex return")
        iq_axis.set_aspect("equal", adjustable="box")
        iq_axis.legend()
        fig.colorbar(points, ax=iq_axis, label="number of physical entire Floquet cycles")

        relative_axis.plot(cycles, data["relative_return"], "o-")
        relative_axis.axhline(1., color="0.7")
        relative_axis.set(xlabel="number of physical entire Floquet cycles",
                          ylabel=r"$|A|/|A(0)|$", title="relative return")

        phase_axis.plot(cycles, data["return_phase"], "o", label="measured")
        phase_axis.plot(cycles, data["phase_fit"], label="fit")
        phase_axis.set(xlabel="number of physical entire Floquet cycles",
                       ylabel="return phase (deg)")
        phase_axis.legend()

        title = (f"{self.occupation}: {float(data['phase_per_cycle']):.4f} +/- "
                 f"{float(data['phase_error']):.4f} deg / cycle")
        fname = getattr(self, "fname", None)
        if fname:
            title += "\n" + str(fname).replace("\\", "/").split("/")[-1]
        fig.suptitle(title)
        return fig
