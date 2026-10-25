"""Central-boson AGP data in three layers, each file pointing to the layer below.

- ``FockPopulationExperiment`` (``fock_population.py``): one job, one prepared
  state |i>, one read mode r, a cycle sweep. Shots, and raw and corrected
  ``<i|n_r(t)|i>`` and ``<i|n_r^2(t)|i>``.
- ``CentralBosonAGPExperiment``: the jobs of one disorder strength W, one per
  |i> with the direction mode d occupied. Adds the sector-averaged two-time
  correlations C_1(t) and C_2(t), the AGP kernel integral of C_1 and the
  closed-system theory of both. No shots.
- ``CentralBosonAGPDisorderScan``: one ``CentralBosonAGPExperiment`` per W.
  Only the C_k(W, t) maps, the AGP (W, mu) maps and their axes.

Each assembled layer writes a manifest YAML (the files of the layer below) and
a small HDF5 with only its own arrays (``AssembledExperiment.save``). Load an
HDF5 alone with ``from_h5file`` to look at it; ``from_manifest`` re-assembles
from the layer below, which ``analyze()`` and the theory need.

analyze(mu=...) takes cutoffs in kHz (one or many; default g L/D). The AGP is
the kernel integral of the measured C_1 over its window (``fitting.qsim.agp``),
with shot and readout-calibration errors, and optionally with a local T1
compensation; no fit. The theory (``theory.qsim``) is the closed system at the
recorded couplings, detunings and Kerr: C_1(t), the AGP of the same window, and
the infinite-time AGP. It is for comparison only and never changes a measured value.
"""
from itertools import product
from math import comb
import json
from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import AssembledExperiment
from experiments.job_paths import JobPathError, _reroot, data_root
from experiments.qsim.fock_population import (
    FockPopulationExperiment,
    FockPopulationSetExperiment,
    assembled_from_h5file,
    mode_name,
    multiparity_moment_error,
)
from fitting.qsim.agp import agp_kernel_weights, calibration_error_factors
from theory.qsim.agp import probe_values, spectral_agp, spectral_weights, two_time_correlation
from theory.qsim.central_boson import fixed_n_hamiltonian

# The photon numbers multiparity readout resolves (n mod 4, n <= 3).
NUMBERS = np.arange(4.)[:, None]


def direction_sector(photon_sector_number, swap_stors, direction):
    """-> fixed-N occupations contributing to the selected AGP direction.

    ``direction`` is 0 for M1 Kerr (n_M1 >= 2), else a storage ID (n_s >= 1).
    The states are in ascending lexicographic order.
    """
    modes = [0, *swap_stors]
    if direction not in modes:
        raise ValueError(f"direction {direction} is not M1 (0) or one of {list(swap_stors)}")
    index = modes.index(direction)
    minimum = 2 if direction == 0 else 1
    return [list(state) for state in product(range(photon_sector_number + 1), repeat=len(modes))
            if sum(state) == photon_sector_number and state[index] >= minimum]


def sector_dimension(photon_sector_number, mode_count):
    """-> the number of fixed-N Fock states of ``mode_count`` modes."""
    return comb(photon_sector_number + mode_count - 1, mode_count - 1)


def rms_khz(detunings):
    """-> the RMS of detunings given in MHz, in kHz (0 for no storages)."""
    detunings = np.asarray(detunings, dtype=float)
    return float(1e3 * np.sqrt(np.mean(detunings**2))) if detunings.size else 0.


def calibration_shots(calibration):
    """-> the shots per prepared state of a calibration record's file, or None.

    The recorded path is re-rooted onto the local data root when it does not
    exist here (``experiments.job_paths``). None when the record names no
    file, or the file is not found; the AGP errors are then shot-only.
    """
    fname = (calibration or {}).get("fname")
    if not fname:
        return None
    path = Path(fname)
    if not path.is_file():
        try:
            path = _reroot(str(fname), data_root()) or path
        except JobPathError:
            return None
    if not path.is_file():
        return None
    with h5py.File(path, "r") as handle:
        if "idata" not in handle or "config" not in handle.attrs:
            return None
        read_num = int(json.loads(handle.attrs["config"]).get("read_num", 0))
        shape = handle["idata"].shape
    if read_num < 1 or len(shape) != 2 or shape[1] % read_num:
        return None
    return shape[1] // read_num


class CentralBosonAGPExperiment(FockPopulationSetExperiment):
    """The population jobs of one disorder strength, and their two-time correlations.

    One job per fixed-N state |i> with nonzero initial probe, all read in mode
    r = ``ro_stor`` (default d). Direction 0 probes P = n_M1(n_M1-1)/2;
    storage directions probe n_s. With O the selected probe and D the full sector,

        C_k(t) = Tr_N[O(t)^k O(0)^k] / D,

    for k = 1 (Kerr) or k = 1, 2 (storage). States with zero initial probe are
    not measured. The moments are photon-number moments only while n <= 3
    (multiparity reads n mod 4).

        agp = CentralBosonAGPExperiment(3, direction=7, swap_stors=[4, 5, 6, 7],
                                        cycles=range(0, 400, 4), detunings=detunings,
                                        calibration=calibration_h5)
        agp.acquire(runner)   # 15 short jobs; runner.ExptClass is FockPopulationExperiment
        agp.analyze(mu=[2, 5, 10], t1_us=635); agp.save(); agp.display()

    ``data`` adds to the population set's arrays: ``C1``, ``C2`` (cycles,),
    ``C1_corrected``, ``C2_corrected`` with a correction, ``C1_err``, ``C2_err``
    (shot-only, of the corrected values when there are some),
    ``C1_calibration_factors`` (cycles, 4) when the calibration file is
    readable, ``disorder_khz``, ``direction``, ``photon_sector_number`` and
    ``sector_dimension``. With timing and couplings recorded and r = d, per
    cutoff in ``agp_mu`` (kHz): ``agp`` and ``agp_err`` (dimensionless
    g^2 ||A_mu||^2), ``agp_t1`` and ``agp_t1_err`` with a T1, and
    ``coupling_mhz``. With a Kerr: ``C1_theory`` (cycles,), ``agp_theory``
    (infinite time), ``agp_theory_window`` (the same kernel and window as the
    data) and ``theory_kerr_mhz``.
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

    @property
    def probes(self):
        """-> [(k, probe)] of the C_k of this direction (``theory.qsim.agp.probe_values``)."""
        return [(1, "kerr")] if self.direction == 0 else [(1, "number"), (2, "number_squared")]

    @property
    def mode_index(self):
        """-> the index of the direction mode in (M1, *swap_stors)."""
        return [0, *self.swap_stors].index(self.direction)

    def recorded_couplings_mhz(self):
        """-> the recorded M1-storage couplings of ``swap_stors`` (MHz), or None."""
        params = self.children[0].recorded_derived_params() or {}
        if "couplings_MHz" not in params:
            return None
        couplings = np.asarray(params["couplings_MHz"], dtype=float)
        couplings = couplings[np.asarray(self.swap_stors, dtype=int) - 1]
        if not couplings.size or not np.all(np.isfinite(couplings)):
            raise ValueError("AGP requires finite recorded couplings")
        return couplings

    def recorded_kerr_mhz(self):
        """-> the M1 Kerr of the jobs' device config (MHz), or None."""
        try:
            return float(np.asarray(self.children[0].cfg.device.manipulate.kerr).reshape(-1)[0])
        except (AttributeError, KeyError, IndexError, TypeError):
            return None

    def theory_spectrum(self, detunings=None, kerr_mhz=None):
        """-> (differences_MHz, weights) of the C_1 probe, closed system.

        At the recorded couplings, ``detunings`` (MHz; default this W's) and
        ``kerr_mhz`` (default the recorded Kerr). Needs the jobs.
        """
        couplings = self.recorded_couplings_mhz()
        kerr = self.recorded_kerr_mhz() if kerr_mhz is None else float(kerr_mhz)
        if couplings is None or kerr is None:
            raise ValueError("the theory needs recorded couplings and a Kerr (give kerr_mhz)")
        model = fixed_n_hamiltonian(self.photon_sector_number, 1 + len(self.swap_stors),
                                    self.detunings if detunings is None else detunings,
                                    couplings, kerr)
        values = probe_values(model.fock_basis, self.mode_index, self.probes[0][1])
        return spectral_weights(model, values)

    def analyze(self, mu=None, t1_us=None, t1_err_us=None, kerr_mhz=None, **kwargs):
        """Build C_k, integrate C_1 with the AGP kernel, and add the closed-system theory.

        ``mu``: cutoff or cutoffs in kHz (None: g L/D). ``t1_us``, ``t1_err_us``:
        T1 of the direction mode, for ``agp_t1``. ``kerr_mhz``: the Kerr of the
        theory (None: the recorded device Kerr).
        """
        super().analyze(**kwargs)
        dimension = sector_dimension(self.photon_sector_number, 1 + len(self.swap_stors))
        corrected = "p_mod_corrected" in self.data
        confusion = self.data.get("confusion_matrix")
        shots = calibration_shots(self.calibration) if corrected else None
        if corrected and shots is None:
            warnings.warn("Calibration file not found: the AGP errors are shot-only.",
                          stacklevel=2)
        for k, probe in self.probes:
            weights = probe_values(self.data["occupations"], self.mode_index, probe) / dimension
            values = probe_values(NUMBERS, 0, probe)
            for suffix in ("", "_corrected") if corrected else ("",):
                self.data[f"C{k}{suffix}"] = weights @ (self.data[f"p_mod{suffix}"] @ values)
            # The errors belong to the corrected values when there are some.
            raw = np.asarray(self.data["p_mod"])
            row_errors = multiparity_moment_error(
                raw.reshape(-1, 4), np.ravel(self.data["shot_counts"]), values, confusion)
            self.data[f"C{k}_err"] = np.sqrt(weights**2 @ row_errors.reshape(raw.shape[:2])**2)
            if k == 1 and shots is not None:
                factors = calibration_error_factors(self.data["p_mod_corrected"], confusion,
                                                    shots, values)
                self.data["C1_calibration_factors"] = np.einsum("r,rta->ta", weights, factors)
        self.data.update(disorder_khz=np.asarray(self.disorder_khz),
                         direction=np.asarray(self.direction),
                         photon_sector_number=np.asarray(self.photon_sector_number),
                         sector_dimension=np.asarray(dimension))

        couplings = self.recorded_couplings_mhz()
        if self.ro_stor != self.direction or "time_us" not in self.data or couplings is None:
            warnings.warn("AGP skipped: needs an autocorrelation and recorded timing/coupling.",
                          stacklevel=2)
            return self.data
        g_mhz = float(np.mean(np.abs(couplings)))
        mu_khz = np.atleast_1d(np.asarray(
            1000 * g_mhz * (1 + len(self.swap_stors)) / dimension if mu is None else mu,
            dtype=float))
        correlation = self.data["C1_corrected" if corrected else "C1"]
        factors = self.data.get("C1_calibration_factors")
        scale = (2 * np.pi * g_mhz)**2  # agp stores dimensionless g^2 ||A_mu||^2

        def estimate(lifetime_us):
            weights = scale * np.array([agp_kernel_weights(self.data["time_us"], cutoff / 1000,
                                                           lifetime_us) for cutoff in mu_khz])
            variance = weights**2 @ self.data["C1_err"]**2
            if factors is not None:
                variance = variance + ((weights @ factors)**2).sum(axis=-1)
            return weights, weights @ correlation, variance

        window, agp, variance = estimate(None)
        self.data.update(agp_mu=mu_khz, agp=agp, agp_err=np.sqrt(variance),
                         coupling_mhz=np.asarray(g_mhz))
        if t1_us is not None:
            _, value, variance = estimate(t1_us)
            if t1_err_us:
                step = 1e-4 * t1_us
                slope = (estimate(t1_us + step)[1] - estimate(t1_us - step)[1]) / (2 * step)
                variance = variance + (slope * t1_err_us)**2
            self.data.update(agp_t1=value, agp_t1_err=np.sqrt(variance),
                             t1_us=np.asarray(float(t1_us)),
                             t1_err_us=np.asarray(float(t1_err_us or 0)))

        kerr = self.recorded_kerr_mhz() if kerr_mhz is None else float(kerr_mhz)
        if kerr is None:
            warnings.warn("Theory skipped: no Kerr recorded; give kerr_mhz.", stacklevel=2)
            return self.data
        differences, weights = self.theory_spectrum(kerr_mhz=kerr)
        theory = two_time_correlation(differences, weights, self.data["time_us"])
        self.data.update(C1_theory=theory, agp_theory_window=window @ theory,
                         agp_theory=spectral_agp(differences, weights, mu_khz / 1000, g_mhz),
                         theory_kerr_mhz=np.asarray(kerr))
        return self.data

    def display(self, raw=False, **kwargs):
        """Top: <i|n_r(t)|i> for every |i>. Bottom: C_1(t), C_2(t) with shot errors, and theory."""
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
            if f"C{k}" not in self.data:
                continue
            values = np.asarray(self.data[f"C{k}{suffix}"])
            bottom.plot(cycles, values, lw=1, label=f"C_{k}")
            if with_error and f"C{k}_err" in self.data:
                error = np.asarray(self.data[f"C{k}_err"])
                bottom.fill_between(cycles, values - error, values + error, alpha=.3, lw=0)
        if "C1_theory" in self.data:
            bottom.plot(cycles, self.data["C1_theory"], "k--", lw=1, label="C_1, closed system")
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
    and ``C1``, ``C2``, ``C1_err``, ``C2_err``, ``C1_theory``, with ``C1_corrected``
    and ``C2_corrected`` when every part is corrected, each (W, cycles). With
    an AGP in every part: ``agp_mu`` (mu,) in kHz and ``agp``, ``agp_err``,
    ``agp_t1``, ``agp_t1_err``, ``agp_theory``, ``agp_theory_window``, each (W, mu).
    """

    child_class = CentralBosonAGPExperiment
    from_h5file = classmethod(assembled_from_h5file)
    SHARED = ("photon_sector_number", "direction", "ro_stor", "swap_stors", "cycles")
    AGP_KEYS = ("agp", "agp_err", "agp_t1", "agp_t1_err", "agp_theory", "agp_theory_window")

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

    def analyze(self, mu=None, t1_us=None, t1_err_us=None, kerr_mhz=None, **kwargs):
        """Analyze every W (see ``CentralBosonAGPExperiment.analyze``) and collect the maps."""
        for part in self.children:
            part.analyze(mu=mu, t1_us=t1_us, t1_err_us=t1_err_us, kerr_mhz=kerr_mhz, **kwargs)
        parts = [part.data for part in self.children]
        data = dict(disorder_khz=np.asarray([part.disorder_khz for part in self.children]),
                    cycles=np.asarray(parts[0]["cycles"]),
                    detunings=np.asarray([part["detunings"] for part in parts]))
        if all("time_us" in part for part in parts):
            data["time_us"] = np.asarray(parts[0]["time_us"])
        keys = ["C1", "C2", "C1_err", "C2_err", "C1_theory"]
        if all("C1_corrected" in part for part in parts):
            keys += ["C1_corrected", "C2_corrected"]
        if all("agp" in part for part in parts):
            cutoffs = [np.asarray(part["agp_mu"]) for part in parts]
            if any(c.shape != cutoffs[0].shape or not np.allclose(c, cutoffs[0]) for c in cutoffs):
                raise ValueError("the parts have different default cutoffs; give mu")
            data["agp_mu"] = cutoffs[0]
            keys += self.AGP_KEYS
        for key in keys:
            if all(key in part for part in parts):
                data[key] = np.asarray([part[key] for part in parts])
        self.data = AttrDict(data)
        return self.data

    def disorder_pattern(self):
        """-> the unit-RMS detuning pattern of the scan (MHz per kHz of W), or None at W = 0 only."""
        nonzero = [part for part in self.children if part.disorder_khz]
        if not nonzero:
            return None
        patterns = [np.asarray(part.detunings) / (1e-3 * part.disorder_khz) for part in nonzero]
        if any(not np.allclose(pattern, patterns[0], atol=1e-9) for pattern in patterns):
            raise ValueError("the parts do not share one disorder pattern")
        return patterns[0]

    def theory_agp(self, disorder_khz, mu=None, kerr_mhz=None):
        """-> the infinite-time closed-system AGP (W, mu) at any ``disorder_khz``.

        On the scan's disorder pattern and recorded couplings. ``mu`` and
        ``kerr_mhz`` default to those of the last ``analyze()``.
        """
        cutoffs = np.atleast_1d(self.data["agp_mu"] if mu is None else mu).astype(float)
        part = self.children[0]
        if kerr_mhz is None and "theory_kerr_mhz" in part.data:
            kerr_mhz = float(part.data["theory_kerr_mhz"])
        pattern = self.disorder_pattern()
        g_mhz = float(np.mean(np.abs(part.recorded_couplings_mhz())))
        values = []
        for strength in np.atleast_1d(disorder_khz):
            if pattern is None and strength:
                raise ValueError("the scan has only W = 0; no pattern to scale")
            detunings = np.zeros(len(part.swap_stors)) if pattern is None else 1e-3 * strength * pattern
            differences, weights = part.theory_spectrum(detunings, kerr_mhz)
            values.append(spectral_agp(differences, weights, cutoffs / 1000, g_mhz))
        return np.asarray(values)

    def display(self, mu=None, theory=True):
        """AGP against W on linear axes, one panel per cutoff. -> (fig, axes).

        Measured (with shot and calibration errors), T1-compensated when
        analyzed with a T1, the closed system over the same window (crosses)
        and, with the parts loaded, the infinite-time closed system (dashed).
        """
        if "agp" not in self.data and "agp_list" in self.data:  # files saved before 2026-10-24
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            ax.plot(self.data["disorder_khz"], self.data["agp_list"], "o-")
            ax.set(xlabel="Disorder RMS (kHz)", ylabel=r"$g^2\|A_\mu\|^2$")
            return fig, np.array([ax])
        cutoffs = np.asarray(self.data["agp_mu"])
        columns = range(len(cutoffs)) if mu is None else [
            int(np.argmin(abs(cutoffs - value))) for value in np.atleast_1d(mu)]
        strengths = np.asarray(self.data["disorder_khz"])
        dense = np.linspace(0, strengths.max(), 121)
        fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.6),
                                 squeeze=False, sharex=True, constrained_layout=True)
        for ax, column in zip(axes[0], columns):
            ax.errorbar(strengths, self.data["agp"][:, column], self.data["agp_err"][:, column],
                        fmt="o-", ms=4, capsize=2, label="measured")
            if "agp_t1" in self.data:
                ax.errorbar(strengths, self.data["agp_t1"][:, column],
                            self.data["agp_t1_err"][:, column], fmt="s-", ms=4, capsize=2,
                            label="local T1 compensated")
            if "agp_theory_window" in self.data:
                ax.plot(strengths, self.data["agp_theory_window"][:, column], "x:", color="gray",
                        label="closed system, same window")
            if theory and self.children and "agp_theory" in self.data:
                ax.plot(dense, self.theory_agp(dense, mu=cutoffs[column])[:, 0], "k--", lw=1,
                        label="closed system, infinite time")
            ax.set(title=rf"$\mu$ = {cutoffs[column]:g} kHz", xlabel="Disorder RMS (kHz)")
        axes[0, 0].set_ylabel(r"$g^2\|A_\mu\|^2$")
        axes[0, 0].legend(fontsize=7)
        return fig, axes[0]

    def display_traces(self, raw=False):
        """C_1(t) for every W with its shot error and the closed-system C_1. -> (fig, axes)."""
        if "C1" not in self.data:
            self.analyze()
        key = "C1" if raw or "C1_corrected" not in self.data else "C1_corrected"
        cycles = np.asarray(self.data["cycles"])
        count = len(self.data["disorder_khz"])
        fig, axes = plt.subplots(count, 1, figsize=(9, 1.8 * count), sharex=True, squeeze=False,
                                 constrained_layout=True)
        for row, ax in enumerate(axes[:, 0]):
            values, error = self.data[key][row], self.data["C1_err"][row]
            ax.plot(cycles, values, lw=1, label=key.replace("_", " "))
            ax.fill_between(cycles, values - error, values + error, alpha=.3, lw=0)
            if "C1_theory" in self.data:
                ax.plot(cycles, self.data["C1_theory"][row], "k--", lw=1, label="closed system")
            ax.set_ylabel(f"W = {self.data['disorder_khz'][row]:g} kHz", fontsize=8)
        axes[0, 0].legend(fontsize=7, ncol=2)
        axes[-1, 0].set_xlabel("Floquet cycles")
        return fig, axes[:, 0]

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
