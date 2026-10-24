"""Central-boson AGP data in three layers, each file pointing to the layer below.

- ``FockPopulationExperiment`` (``fock_population.py``): one job, one prepared
  state |i>, one read mode r, a cycle sweep. Shots, and raw and corrected
  ``<i|n_r(t)|i>`` and ``<i|n_r^2(t)|i>``.
- ``CentralBosonAGPExperiment``: the jobs of one disorder strength W, one per
  |i> with the direction mode d occupied. Adds the sector-averaged two-time
  correlations C_1(t) and C_2(t). No shots.
- ``CentralBosonAGPDisorderScan``: one ``CentralBosonAGPExperiment`` per W.
  Only the C_k(W, t) maps and their axes.

Each assembled layer writes a manifest YAML (the files of the layer below) and
a small HDF5 with only its own arrays (``AssembledExperiment.save``). Load an
HDF5 alone with ``from_h5file`` to look at it; ``from_manifest`` re-assembles
from the layer below. analyze() integrates C1 with the AGP kernel, without
decay correction or fitting. analyze(mu=...) takes kHz; default is g L/D.
"""
from itertools import product
from math import comb
import warnings

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict
from fitting.qsim.agp import finite_time_agp

from experiments.assembled_data import AssembledExperiment
from experiments.qsim.fock_population import (
    FockPopulationExperiment,
    FockPopulationSetExperiment,
    assembled_from_h5file,
    mode_name,
)


def direction_sector(photon_sector_number, swap_stors, direction):
    """-> fixed-N occupations [n_M1, *swap_stors] with mode ``direction`` occupied.

    ``direction`` is 0 for M1 or a physical storage ID in ``swap_stors``.
    The states are in ascending lexicographic order.
    """
    modes = [0, *swap_stors]
    if direction not in modes:
        raise ValueError(f"direction {direction} is not M1 (0) or one of {list(swap_stors)}")
    index = modes.index(direction)
    return [list(state) for state in product(range(photon_sector_number + 1), repeat=len(modes))
            if sum(state) == photon_sector_number and state[index] > 0]


def sector_dimension(photon_sector_number, mode_count):
    """-> the number of fixed-N Fock states of ``mode_count`` modes."""
    return comb(photon_sector_number + mode_count - 1, mode_count - 1)


def rms_khz(detunings):
    """-> the RMS of detunings given in MHz, in kHz (0 for no storages)."""
    detunings = np.asarray(detunings, dtype=float)
    return float(1e3 * np.sqrt(np.mean(detunings**2))) if detunings.size else 0.


class CentralBosonAGPExperiment(FockPopulationSetExperiment):
    """The population jobs of one disorder strength, and their two-time correlations.

    One job per fixed-N state |i> with the direction mode d occupied, all read
    in mode r = ``ro_stor`` (default d). With D the full sector dimension,

        C_k(t) = Tr_N[n_r^k(t) n_d^k(0)] / D = sum_i n_d,i^k <i|n_r^k(t)|i> / D,

    for k = 1, 2. States with n_d = 0 add nothing and are not measured. The
    moments are photon-number moments only while n <= 3 (multiparity reads n mod 4).

        agp = CentralBosonAGPExperiment(3, direction=7, swap_stors=[4, 5, 6, 7],
                                        cycles=range(0, 400, 4), detunings=detunings,
                                        calibration=calibration_h5)
        agp.acquire(runner)   # 15 short jobs; runner.ExptClass is FockPopulationExperiment
        agp.analyze(); agp.save(); agp.display()

    ``data`` adds to the population set's arrays: ``C1``, ``C2`` (cycles,),
    ``C1_corrected``, ``C2_corrected`` with a correction, ``C1_err``, ``C2_err``
    (shot-only, of the corrected values when there are some), ``disorder_khz``,
    ``direction``, ``photon_sector_number`` and ``sector_dimension``.
    """

    def __init__(self, photon_sector_number, direction, swap_stors, cycles, detunings,
                 ro_stor=None, disorder_khz=None, calibration=None, notes=""):
        self.photon_sector_number = int(photon_sector_number)
        self.direction = int(direction)
        self.ro_stor = self.direction if ro_stor is None else int(ro_stor)
        occupations = direction_sector(self.photon_sector_number, list(swap_stors), self.direction)
        super().__init__([(occupation, self.ro_stor) for occupation in occupations], cycles,
                         swap_stors, detunings, calibration=calibration, notes=notes)
        # The label of the disorder strength; the detuning RMS when none is given.
        self.disorder_khz = rms_khz(self.detunings) if disorder_khz is None else float(disorder_khz)

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", direction=None, disorder_khz=None):
        """Assemble loaded jobs of one W; they must be the whole direction sector.

        ``direction`` is required: the jobs do not record it.
        """
        if direction is None:
            raise ValueError("give the direction mode (0 for M1, else a storage ID)")
        entries, cycles, swap_stors, detunings, calibration = cls.shared_settings(children)
        modes = {mode for _, mode in entries}
        numbers = {sum(occupation) for occupation, _ in entries}
        if len(modes) != 1 or len(numbers) != 1:
            raise ValueError("the jobs must share one read mode and one photon number")
        agp = cls(numbers.pop(), direction, swap_stors, cycles, detunings, ro_stor=modes.pop(),
                  disorder_khz=disorder_khz, calibration=calibration, notes=notes)
        agp.children, agp.job_ids = agp._ordered(children, job_ids)
        agp._check_children()
        return agp

    @classmethod
    def _from_manifest_kwargs(cls, manifest, path):
        parameters = manifest["parameters"]
        return dict(direction=parameters["direction"], disorder_khz=parameters["disorder_khz"])

    def analyze(self, mu=None, **kwargs):
        """Build correlations and integrate C1; mu is in kHz (None: g L/D)."""
        super().analyze(**kwargs)
        modes = [0, *self.swap_stors]
        initial = self.data.occupations[:, modes.index(self.direction)].astype(float)
        dimension = sector_dimension(self.photon_sector_number, len(modes))
        for k, moment in [(1, "n_mean"), (2, "n2_mean")]:
            weights = initial**k / dimension
            self.data[f"C{k}"] = weights @ self.data[moment]
            if f"{moment}_corrected" in self.data:
                self.data[f"C{k}_corrected"] = weights @ self.data[f"{moment}_corrected"]
            error = self.data["n_err" if k == 1 else "n2_err"]
            self.data[f"C{k}_err"] = np.sqrt(weights**2 @ error**2)
        self.data.update(disorder_khz=np.asarray(self.disorder_khz),
                         direction=np.asarray(self.direction),
                         photon_sector_number=np.asarray(self.photon_sector_number),
                         sector_dimension=np.asarray(dimension))

        # Use recorded coupling and the direction's existing autocorrelation.
        job = self.children[0]
        params = job.recorded_derived_params() or {}
        if self.ro_stor != self.direction or "time_us" not in self.data or "couplings_MHz" not in params:
            warnings.warn("AGP skipped: needs an autocorrelation and recorded timing/coupling.",
                          stacklevel=2)
            return self.data
        couplings = np.asarray(params["couplings_MHz"])[np.asarray(self.swap_stors, dtype=int) - 1]
        if not couplings.size or not np.all(np.isfinite(couplings)):
            raise ValueError("AGP requires finite recorded couplings")
        g_khz = 1000 * float(np.mean(np.abs(couplings)))
        if mu is None:
            mu = g_khz * len(modes) / dimension
        correlation = self.data.get("C1_corrected", self.data["C1"])
        result = finite_time_agp(self.data["time_us"], correlation, mu / 1000)
        self.data["agp"] = float(result["values"] * (2 * np.pi * g_khz / 1000)**2)
        self.data["agp_mu"] = float(mu)  # kHz; agp stores dimensionless g^2 ||A_mu||^2.
        return self.data

    def display(self, raw=False, **kwargs):
        """Top: <i|n_r(t)|i> for every |i>. Bottom: C_1(t) and C_2(t) with shot errors."""
        if "C1" not in self.data:
            self.analyze()
        corrected = "C1_corrected" in self.data and not raw
        suffix, kind = ("_corrected", "corrected") if corrected else ("", "raw")
        cycles = np.asarray(self.data["cycles"])
        labels = [str(list(occupation)) for occupation in self.data["occupations"]]
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                          constrained_layout=True)
        mesh = top.pcolormesh(cycles, np.arange(len(labels)), self.data[f"n_mean{suffix}"],
                              shading="auto")
        top.set_yticks(range(len(labels)), labels, fontsize=7)
        fig.colorbar(mesh, ax=top, label=f"<i|n_{mode_name(self.data['ro_stor'][0])}|i> ({kind})")
        # The errors belong to the corrected values when there are some.
        with_error = corrected == ("C1_corrected" in self.data)
        for k in (1, 2):
            values = np.asarray(self.data[f"C{k}{suffix}"])
            bottom.plot(cycles, values, lw=1, label=f"C_{k}")
            if with_error:
                error = np.asarray(self.data[f"C{k}_err"])
                bottom.fill_between(cycles, values - error, values + error, alpha=.3, lw=0)
        bottom.set(xlabel="Floquet cycles", ylabel=f"Correlation ({kind})")
        bottom.legend()
        top.set_title(f"{type(self).__name__}: N = {int(self.data['photon_sector_number'])}, "
                      f"direction {mode_name(self.data['direction'])}, "
                      f"W = {float(self.data['disorder_khz']):g} kHz")
        return fig

    def manifest_parameters(self):
        return dict(super().manifest_parameters(), photon_sector_number=self.photon_sector_number,
                    direction=self.direction, ro_stor=self.ro_stor, disorder_khz=self.disorder_khz)


class CentralBosonAGPDisorderScan(AssembledExperiment):
    """One ``CentralBosonAGPExperiment`` per disorder strength W, as C_k(W, t) maps.

    Acquire all of it with ``from_pattern(...).acquire(runner)``: every job of
    every W goes through one ``runner.execute``, at most ``batch_size`` waiting
    in the queue, so other users' jobs run between them. Or acquire and save
    one ``CentralBosonAGPExperiment`` per W, and combine them with
    ``from_children``. ``save()`` saves the parts that are not saved yet first.

    ``data`` and the assembled HDF5, rows sorted by W: ``disorder_khz`` (W,),
    ``cycles``, ``time_us`` (when every part has it), ``detunings`` (W, storages),
    and ``C1``, ``C2``, ``C1_err``, ``C2_err``, with ``C1_corrected`` and
    ``C2_corrected`` when every part is corrected, each (W, cycles).
    """

    child_class = CentralBosonAGPExperiment
    from_h5file = classmethod(assembled_from_h5file)
    SHARED = ("photon_sector_number", "direction", "ro_stor", "swap_stors", "cycles")

    def __init__(self, parts, notes=""):
        super().__init__(notes=notes)
        parts = list(parts)
        if not parts:
            raise ValueError("a disorder scan needs at least one CentralBosonAGPExperiment")
        for part in parts:
            if not isinstance(part, self.child_class):
                raise TypeError(f"{part!r} is not a {self.child_class.__name__}")
            for name in self.SHARED:
                if getattr(part, name) != getattr(parts[0], name):
                    raise ValueError(f"the parts have different {name}")
        self.children = sorted(parts, key=lambda part: part.disorder_khz)

    @classmethod
    def from_pattern(cls, disorder_khz, pattern, photon_sector_number, direction, swap_stors,
                     cycles, ro_stor=None, calibration=None, notes=""):
        """Plan a scan: detunings(W) = 1e-3 * W * pattern (MHz), pattern scaled to unit RMS."""
        pattern = np.asarray(pattern, dtype=float)
        rms = np.sqrt(np.mean(pattern**2)) if pattern.size else 0.
        if rms > 0:
            pattern = pattern / rms
        elif any(strength != 0 for strength in disorder_khz):
            raise ValueError("nonzero disorder needs a nonzero pattern")
        parts = [CentralBosonAGPExperiment(photon_sector_number, direction, swap_stors, cycles,
                                           (1e-3 * strength * pattern).tolist(), ro_stor=ro_stor,
                                           disorder_khz=strength, calibration=calibration,
                                           notes=notes)
                 for strength in disorder_khz]
        return cls(parts, notes=notes)

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", **kwargs):
        """Combine saved or loaded CentralBosonAGPExperiments, one per W."""
        scan = cls(children, notes=notes)
        scan.job_ids = list(job_ids)
        return scan

    def job_overrides(self):
        return [override for part in self.children for override in part.job_overrides()]

    def acquire(self, runner, batch_size=10, **execute_kwargs):
        """Run every part's jobs through one ``runner.execute``. -> the parts."""
        if runner.ExptClass is not FockPopulationExperiment:
            raise TypeError(f"runner.ExptClass is {runner.ExptClass.__name__}; "
                            f"{type(self).__name__} needs FockPopulationExperiment")
        jobs = list(runner.execute(overrides=self.job_overrides(), batch_size=batch_size,
                                   **execute_kwargs))
        job_ids = list(runner.last_job_ids)
        start = 0
        for part in self.children:
            stop = start + len(part.entries)
            part.children, part.job_ids = jobs[start:stop], job_ids[start:stop]
            part.data = {}
            start = stop
        self.data = {}
        return self.children

    def analyze(self, mu=None, **kwargs):
        """Analyze every W with mu in kHz (None: g L/D) and collect results."""
        for part in self.children:
            part.analyze(mu=mu, **kwargs)
        parts = [part.data for part in self.children]
        data = dict(disorder_khz=np.asarray([part.disorder_khz for part in self.children]),
                    cycles=np.asarray(parts[0]["cycles"]),
                    detunings=np.asarray([part["detunings"] for part in parts]))
        if all("time_us" in part for part in parts):
            data["time_us"] = np.asarray(parts[0]["time_us"])
        keys = ["C1", "C2", "C1_err", "C2_err"]
        if all("C1_corrected" in part for part in parts):
            keys += ["C1_corrected", "C2_corrected"]
        for key in keys:
            data[key] = np.asarray([part[key] for part in parts])
        if all("agp" in part for part in parts):
            data["agp_list"] = [part["agp"] for part in parts]
            data["agp_mu_list"] = [part["agp_mu"] for part in parts]
        self.data = AttrDict(data)
        return self.data

    def display(self):
        """Plot the measured kernel integral against disorder on linear axes."""
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(self.data["disorder_khz"], self.data["agp_list"], "o-")
        ax.set(xlabel="Disorder RMS (kHz)", ylabel=r"$g^2\|A_\mu\|^2$")
        return fig, ax

    def display_correlation(self, moment=1, raw=False, **kwargs):
        """One correlation as a map: x = Floquet cycles, y = disorder W (kHz).

        Corrected when every part is corrected; ``raw=True`` selects raw.
        ``kwargs`` go to ``pcolormesh``.
        """
        if moment not in (1, 2):
            raise ValueError("moment must be 1 or 2")
        if "C1" not in self.data:
            self.analyze()
        if np.any(np.diff(self.data["disorder_khz"]) == 0):
            raise ValueError("a disorder map needs one part per W")
        key, kind = f"C{moment}", "raw"
        if not raw and f"{key}_corrected" in self.data:
            key, kind = f"{key}_corrected", "corrected"
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        mesh = ax.pcolormesh(self.data["cycles"], self.data["disorder_khz"], self.data[key],
                             shading="auto", **kwargs)
        ax.set(xlabel="Floquet cycles", ylabel="Disorder W (kHz)",
               title=f"{type(self).__name__}: C_{moment} ({kind})")
        fig.colorbar(mesh, ax=ax, label=f"C_{moment} ({kind})")
        return fig, ax

    # -- persistence ------------------------------------------------------

    def save(self, directory=None, notes=None):
        """Save the parts that are not saved yet, then this scan. -> the manifest path."""
        for part in self.children:
            if part.manifest_path is None:
                if not part.data:
                    part.analyze()
                part.save(directory=directory)
        if not self.data:
            self.analyze()
        return super().save(directory=directory, notes=notes)

    def _child_files(self):
        """-> the parts' manifests."""
        return [part.manifest_path for part in self.children]

    @classmethod
    def _load_children(cls, manifest):
        return [cls.child_class.from_manifest(path) for path in manifest["raw_files"]]

    def manifest_parameters(self):
        first = self.children[0]
        return dict(disorder_khz=[part.disorder_khz for part in self.children],
                    **{name: getattr(first, name) for name in self.SHARED})

    def assembled_arrays(self):
        return dict(self.data)
