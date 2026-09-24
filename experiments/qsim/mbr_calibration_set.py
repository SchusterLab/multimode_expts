# -*- coding: utf-8 -*-
"""A set of Stark-shift calibration jobs over occupations, and its phase correction.

An assembled class (docs/qsim/mbr_redesign.md, sections 1, 2 and 5): it holds
one :class:`MBRStarkCalExperiment` per occupation and never goes to the
worker itself. It gives the measured phase per entire Floquet cycle of every
occupation, and turns it into the analyzer phase correction that the time
trace programs subtract.

    cal = MBRCalibrationSetExperiment(occupations, cycle_pairs=range(65),
                                      swap_stors=[1, 2, 3, 4])
    cal.acquire(runner, batch_size=10)     # runner.ExptClass is MBRStarkCalExperiment
    cal.analyze(); cal.display()
    cal.save()                             # manifest YAML + assembled HDF5
    cal.phase_for((3, 0, 0, 0, 0))         # deg / cycle, for one occupation

    cal = MBRCalibrationSetExperiment.from_manifest(path)

The runner carries the settings every MBR job shares (reset, readout, reps
defaults); this class adds what to sweep.
"""
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import AssembledExperiment
from experiments.qsim.mbr_saved import saved_parameters
from experiments.qsim.mbr_stark_cal import MBRStarkCalExperiment
from fitting.qsim import mbr_phase


class MBRCalibrationSetExperiment(AssembledExperiment):
    """StarkCal jobs over occupations; gives ``phase_for(occupation)``."""

    child_class = MBRStarkCalExperiment

    def __init__(self, occupations, cycle_pairs, swap_stors, sync_cycles=10,
                 reps=1500, notes=""):
        super().__init__(notes=notes)
        self.occupations = [tuple(int(n) for n in o) for o in occupations]
        self.cycle_pairs = [int(n) for n in cycle_pairs]
        self.swap_stors = [int(stor) for stor in swap_stors]
        self.sync_cycles = int(sync_cycles)
        self.reps = int(reps)

    # -- acquisition ------------------------------------------------------

    def job_overrides(self):
        """-> one expt-config override dict per occupation."""
        return [self.child_class.job_config(occupation, self.cycle_pairs,
                                            self.swap_stors, self.sync_cycles,
                                            self.reps)
                for occupation in self.occupations]

    @classmethod
    def from_children(cls, children, job_ids=(), notes=""):
        """Assemble already acquired or loaded StarkCal jobs.

        The occupations, cycle pairs and swap modes are read from the jobs'
        cfg.expt, in job order.
        """
        children = list(children)
        if not children:
            raise ValueError("a calibration set needs at least one StarkCal job")
        first = children[0].cfg.expt
        cycle_pairs = [int(n) for n in first.n_cycle_pairs]
        swap_stors = [int(stor) for stor in first.swap_stors]
        for child in children[1:]:
            ecfg = child.cfg.expt
            if [int(n) for n in ecfg.n_cycle_pairs] != cycle_pairs:
                raise ValueError(f"{child.occupation}: different cycle pairs")
            if [int(stor) for stor in ecfg.swap_stors] != swap_stors:
                raise ValueError(f"{child.occupation}: different swap modes")
        occupations = [child.occupation for child in children]
        if len(set(occupations)) != len(occupations):
            raise ValueError("each occupation must appear once")

        calibration = cls(occupations, cycle_pairs, swap_stors,
                          sync_cycles=int(first.get("scramble_sync_cycles", 10)),
                          reps=int(first.reps), notes=notes)
        calibration.children = children
        calibration.job_ids = list(job_ids)
        calibration._check_children()
        return calibration

    # -- analysis ---------------------------------------------------------

    def analyze(self):
        """Phase per entire Floquet cycle of every occupation, mod 180.

        Runs each job's own ``analyze`` if it has not run yet. Also records
        the jobs' hardware parameters (Floquet cycle time, couplings, Kerr),
        which the phase correction needs.
        """
        self._check_children()
        for child in self.children:
            if "complex_return" not in child.data:
                child.analyze()
            if not np.isfinite(float(child.data["phase_per_cycle"])):
                raise ValueError(f"{child.occupation} has no phase fit; "
                                 f"see its display()")
        saved = saved_parameters(self.children)
        self.data = AttrDict(dict(
            occupations=list(self.occupations),
            physical_cycles=np.asarray(self.children[0].data["physical_cycles"]),
            complex_returns=np.asarray([c.data["complex_return"] for c in self.children]),
            phase_mod180=np.asarray([float(c.data["phase_per_cycle"]) for c in self.children]),
            phase_error=np.asarray([float(c.data["phase_error"]) for c in self.children]),
            hardware=saved.hardware,
            mode_labels=saved.mode_labels,
        ))
        return self.data

    def phase_correction(self, cycle_branches=0, correction_sign=1.):
        """-> the analyzer correction per occupation, in deg / physical cycle.

        ``cycle_branches`` picks the 180 deg/cycle branch: one integer for
        every occupation, a list in occupation order, or a dict
        ``{occupation: branch}`` (unlisted occupations get 0). The M1
        self-Kerr phase is removed; see
        :func:`fitting.qsim.mbr_phase.build_phase_correction`.
        ``calibration_manifest`` records where it came from, if saved.
        """
        if "phase_mod180" not in self.data:
            self.analyze()
        branches = mbr_phase.cycle_branches(self.occupations, cycle_branches)
        correction = mbr_phase.build_phase_correction(
            self.occupations, self.data.phase_mod180, branches,
            self.data.hardware.physical_kerr_MHz,
            self.data.hardware.floquet_cycle_us,
            correction_sign=correction_sign)
        correction.calibration_manifest = (
            str(self.manifest_path) if self.manifest_path else None)
        return correction

    def phase_for(self, occupation, cycle_branch=0):
        """-> the analyzer correction of one occupation, in deg / physical cycle."""
        occupation = tuple(int(n) for n in occupation)
        if occupation not in self.occupations:
            raise KeyError(f"{occupation} is not in this calibration set")
        correction = self.phase_correction({occupation: cycle_branch})
        return correction.phase_by_occupation[occupation]

    def display(self, ncols=None):
        """Each job's fit in a grid, then the phase-per-cycle summary."""
        if "phase_mod180" not in self.data:
            self.analyze()
        count = len(self.children)
        if ncols is None:
            nrows = max(1, int(np.floor(np.sqrt(count))))
            ncols = int(np.ceil(count / nrows))
        else:
            ncols = min(int(ncols), count)
            nrows = int(np.ceil(count / ncols))
        fig = plt.figure(figsize=(7 * ncols, 4 * nrows), constrained_layout=True)
        subfigures = fig.subfigures(nrows, ncols, squeeze=False)
        for child, subfigure in zip(self.children, subfigures.flat):
            child.display(fig=subfigure)
        for subfigure in subfigures.flat[count:]:
            subfigure.set_visible(False)

        rows = np.arange(count)
        summary, ax = plt.subplots(figsize=(9, max(4, 0.35 * count + 2)),
                                   constrained_layout=True)
        ax.errorbar(self.data.phase_mod180, rows, xerr=self.data.phase_error, fmt="o")
        ax.axvline(0., color="0.7")
        ax.set(xlabel="measured phase mod 180 (deg / entire cycle)",
               ylabel="occupation", title="entire-cycle calibration summary")
        ax.set_yticks(rows)
        ax.set_yticklabels([str(o) for o in self.occupations])
        ax.invert_yaxis()
        return summary

    # -- persistence ------------------------------------------------------

    def manifest_parameters(self):
        return dict(occupations=[list(o) for o in self.occupations],
                    cycle_pairs=self.cycle_pairs,
                    swap_stors=self.swap_stors)

    def assembled_arrays(self):
        return dict(
            occupations=np.asarray(self.occupations, dtype=int),
            physical_cycles=self.data.physical_cycles,
            complex_returns=self.data.complex_returns,
            phase_mod180=self.data.phase_mod180,
            phase_error=self.data.phase_error,
            couplings_MHz=self.data.hardware.couplings_MHz,
        )

    def assembled_attrs(self):
        hardware = self.data.hardware
        return dict(
            floquet_cycle_us=float(hardware.floquet_cycle_us),
            physical_kerr_MHz=float(hardware.physical_kerr_MHz),
            hardware_source=str(hardware.source),
            mode_labels=list(self.data.mode_labels),
        )
