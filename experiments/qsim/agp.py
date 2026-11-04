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

import yaml

from experiments.assembled_data import (
    ASSEMBLED_DIR,
    AssembledExperiment,
    experiment_root,
    named_stem,
    read_manifest,
)
from experiments.job_paths import JOB_ID_RE, JobPathError, _reroot, data_root
from experiments.qsim.fock_population import (
    FockPopulationExperiment,
    FockPopulationSetExperiment,
    assembled_from_h5file,
    mode_name,
    multiparity_moment_error,
)
from fitting.qsim.agp import (
    agp_kernel_weights,
    calibration_error_factors,
    spectral_function,
    spectrum_frequencies_khz,
    subtract_background,
)
from theory.qsim.agp import (
    probe_values,
    spectral_agp,
    spectral_weights,
    split_dc,
    two_time_correlation,
)
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
        agp.analyze(mu=[2, 5, 10], t1_us=635); agp.save("s7_w30"); agp.display()

    ``data`` adds to the population set's arrays: ``C1``, ``C2`` (cycles,),
    ``C1_corrected``, ``C2_corrected`` with a correction, ``C1_err``, ``C2_err``
    (shot-only, of the corrected values when there are some),
    ``C1_calibration_factors`` (cycles, 4) when the calibration file is
    readable, ``disorder_khz``, ``direction``, ``photon_sector_number`` and
    ``sector_dimension``. With timing recorded and r = d, the spectrum of C_1
    on ``frequency_khz`` (0 to Nyquist by 1/(4T)): ``spectrum``, ``spectrum_err``
    and, background-subtracted, ``spectrum_with_background``. With couplings
    too, per cutoff in ``agp_mu`` (kHz, default ``DEFAULT_MU_KHZ``, the same
    for every W): ``agp`` and ``agp_err`` (dimensionless g^2 ||A_mu||^2),
    ``agp_t1`` and ``agp_t1_err`` with a T1, ``agp_connected`` with a
    background start, ``coupling_mhz`` and ``mu_gld_khz`` (g L/D). With a
    Kerr: ``C1_theory`` (cycles,), ``agp_theory`` (infinite time),
    ``agp_theory_window`` (the same kernel and window as the data),
    ``spectrum_theory`` and ``theory_kerr_mhz``. The HDF5 attribute
    ``analysis`` holds the settings.
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

    # The default cutoff axis (kHz), the same for every W: 41 log steps from 1 to 100, and 2, 5, 20, 50.
    DEFAULT_MU_KHZ = np.unique(np.round(np.concatenate([np.geomspace(1., 100., 41),
                                                        [2., 5., 20., 50.]]), 9))

    def analyze(self, mu=None, t1_us=None, t1_err_us=None, kerr_mhz=None,
                background_start_us=None, window="hann", **kwargs):
        """Build C_k, its spectrum S(f) and AGP(mu) curves, and the closed-system theory.

        All processed curves of this W, on axes that do not depend on it:
        ``agp_mu`` (``mu`` in kHz, default ``DEFAULT_MU_KHZ``) and
        ``frequency_khz`` (0 to Nyquist by 1/(4T), from the time grid). g L/D is
        kept as ``mu_gld_khz``. S is the cosine transform of C_1 (``window``),
        background-subtracted when ``background_start_us`` is given, with
        ``spectrum_with_background`` and ``spectrum_theory``. ``t1_us``, ``t1_err_us``:
        T1 of the direction mode, for ``agp_t1``. ``kerr_mhz``: the Kerr of the
        theory (None: the recorded device Kerr). ``background_start_us``: also
        fit A e^{-rate t} to C_1 for t >= this time and integrate C_1 minus it
        (``agp_connected``, lab book Q5). The decaying DC part of C_1 is what
        the kernel picks up most at small mu; the closed-system theory has no
        such part in ``agp_theory``. ``agp_connected_err`` is shot and
        calibration noise only: not the error of the fitted background.
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

        self.analysis = dict(mu_khz=None if mu is None else np.atleast_1d(mu).tolist(),
                             t1_us=t1_us, t1_err_us=t1_err_us, kerr_mhz=kerr_mhz,
                             background_start_us=background_start_us, window=window)
        if self.ro_stor != self.direction or "time_us" not in self.data:
            warnings.warn("AGP and spectrum skipped: need an autocorrelation and recorded timing.",
                          stacklevel=2)
            return self.data
        correlation = self.data["C1_corrected" if corrected else "C1"]
        if background_start_us is not None:
            connected, fit = subtract_background(self.data["time_us"], correlation,
                                                 self.data["C1_err"], background_start_us)
            self.data.update(background_amplitude=np.asarray(fit["amplitude"]),
                             background_rate_per_us=np.asarray(fit["rate_per_us"]),
                             background_start_us=np.asarray(float(background_start_us)))
        frequency_khz = spectrum_frequencies_khz(self.data["time_us"])
        spectrum, spectrum_err = spectral_function(
            self.data["time_us"], correlation if background_start_us is None else connected,
            self.data["C1_err"], frequency_khz, window)
        self.data.update(frequency_khz=frequency_khz, spectrum=spectrum, spectrum_err=spectrum_err)
        if background_start_us is not None:
            self.data["spectrum_with_background"] = spectral_function(
                self.data["time_us"], correlation, self.data["C1_err"], frequency_khz, window)[0]

        couplings = self.recorded_couplings_mhz()
        if couplings is None:
            warnings.warn("AGP skipped: needs recorded couplings.", stacklevel=2)
            return self.data
        g_mhz = float(np.mean(np.abs(couplings)))
        mu_khz = np.atleast_1d(np.asarray(self.DEFAULT_MU_KHZ if mu is None else mu, dtype=float))
        factors = self.data.get("C1_calibration_factors")
        scale = (2 * np.pi * g_mhz)**2  # agp stores dimensionless g^2 ||A_mu||^2

        def estimate(lifetime_us):
            weights = scale * np.array([agp_kernel_weights(self.data["time_us"], cutoff / 1000,
                                                           lifetime_us) for cutoff in mu_khz])
            variance = weights**2 @ self.data["C1_err"]**2
            if factors is not None:
                variance = variance + ((weights @ factors)**2).sum(axis=-1)
            return weights, weights @ correlation, variance

        kernel, agp, variance = estimate(None)
        self.data.update(agp_mu=mu_khz, agp=agp, agp_err=np.sqrt(variance),
                         coupling_mhz=np.asarray(g_mhz),
                         mu_gld_khz=np.asarray(1000 * g_mhz * (1 + len(self.swap_stors)) / dimension))
        if background_start_us is not None:
            self.data.update(agp_connected=kernel @ connected, agp_connected_err=np.sqrt(variance))
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
        # Against a background-subtracted S, the theory without its DC part (the connected C_1).
        dc = split_dc(differences, weights)[0] if background_start_us is not None else 0.
        self.data.update(C1_theory=theory, agp_theory_window=kernel @ theory,
                         agp_theory=spectral_agp(differences, weights, mu_khz / 1000, g_mhz),
                         theory_kerr_mhz=np.asarray(kerr),
                         spectrum_theory=spectral_function(
                             self.data["time_us"], theory - dc, np.zeros(len(theory)),
                             frequency_khz, window)[0])
        return self.data

    def assembled_attrs(self):
        """The analysis settings of the saved curves (cutoffs, window, background, T1, Kerr)."""
        return dict(analysis=getattr(self, "analysis", {}))

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

    Acquire all of it with ``from_pattern(...).acquire(runner, name)``: W by W,
    at most ``batch_size`` jobs waiting in the queue, so other users' jobs run
    between them; each W is analyzed and saved as soon as its jobs are in, and
    the scan at the end. A stopped run keeps its finished W:
    ``from_saved_parts(name, directory)``. Or combine saved or loaded
    ``CentralBosonAGPExperiment`` parts with ``from_children``; ``save(name)``
    saves the parts that are not saved yet first.

    ``data`` and the assembled HDF5, rows sorted by W: ``disorder_khz`` (W,),
    ``cycles``, ``time_us`` (when every part has it), ``detunings`` (W, storages),
    and ``C1``, ``C2``, ``C1_err``, ``C2_err``, ``C1_theory``, with ``C1_corrected``
    and ``C2_corrected`` when every part is corrected, each (W, cycles). The
    parts' curves on their shared axes: ``agp_mu`` (mu,) in kHz with ``agp``,
    ``agp_err``, ``agp_t1``, ``agp_t1_err``, ``agp_connected``, ``agp_theory``,
    ``agp_theory_window`` (W, mu) and ``mu_gld_khz`` (W,); ``frequency_khz`` (f,)
    with ``spectrum``, ``spectrum_err``, ``spectrum_with_background``,
    ``spectrum_theory`` (W, f). The HDF5 attribute ``analysis`` holds the settings.
    """

    child_class = CentralBosonAGPExperiment
    from_h5file = classmethod(assembled_from_h5file)
    SHARED = ("photon_sector_number", "direction", "ro_stor", "swap_stors", "cycles")
    AGP_KEYS = ("agp", "agp_err", "agp_t1", "agp_t1_err", "agp_theory", "agp_theory_window")
    BACKGROUND_KEYS = ("agp_connected", "agp_connected_err", "background_amplitude",
                       "background_rate_per_us")
    SPECTRUM_KEYS = ("spectrum", "spectrum_err", "spectrum_with_background", "spectrum_theory")

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

    @classmethod
    def from_job_files(cls, paths, direction, complete_only=True, notes=""):
        """Assemble population job files of one direction, for a scan still running or never saved.

        Only autocorrelation jobs (read mode = ``direction``) are used. They are
        grouped into parts by photon number, storages, cycles and detunings. With
        ``complete_only``, a W whose direction sector is not complete yet is left
        out with a warning.
        """
        groups = {}
        for path in paths:
            job = FockPopulationExperiment.from_h5file(path, load_shots=False)
            expt = job.cfg.expt
            if int(expt.ro_stor) != int(direction):
                continue
            key = (sum(expt.occupations), tuple(expt.swap_stors), tuple(expt.floquet_cycles),
                   tuple(np.round(np.asarray(expt.detunings, dtype=float), 12)))
            match = JOB_ID_RE.search(Path(path).name)
            groups.setdefault(key, []).append((job, match.group(0) if match else Path(path).stem))
        parts = []
        for key, members in groups.items():
            jobs, job_ids = zip(*members)
            try:
                parts.append(CentralBosonAGPExperiment.from_children(
                    jobs, job_ids=job_ids, direction=direction,
                    disorder_khz=round(rms_khz(key[-1]), 6)))  # the detuning RMS, without float noise
            except ValueError as error:
                if not complete_only:
                    raise
                warnings.warn(f"left out W = {rms_khz(key[-1]):g} kHz ({len(jobs)} jobs): {error}",
                              stacklevel=2)
        return cls(parts, notes=notes)

    def job_overrides(self):
        return [override for part in self.children for override in part.job_overrides()]

    def acquire(self, runner, name, analyze_kwargs=None, directory=None, batch_size=10,
                **execute_kwargs):
        """Acquire W by W; analyze and save each W when its jobs are in, then the scan.

        One ``runner.execute`` per W, at most ``batch_size`` jobs waiting in the
        queue. ``name`` names the scan and its parts (see ``save``); all names
        are checked before the first job when the runner's station gives the
        experiment folder, else before the first save. ``analyze_kwargs`` go
        to every ``analyze`` (``mu``, ``t1_us``, ``background_start_us``, ...).
        If the run stops, the finished W are saved: ``from_saved_parts(name)``.
        -> the scan manifest path.
        """
        if runner.ExptClass is not FockPopulationExperiment:
            raise TypeError(f"runner.ExptClass is {runner.ExptClass.__name__}; "
                            f"{type(self).__name__} needs FockPopulationExperiment")
        analyze_kwargs = dict(analyze_kwargs or {})
        if directory is None:
            experiment_path = getattr(getattr(runner, "station", None), "experiment_path", None)
            directory = None if experiment_path is None else Path(experiment_path) / ASSEMBLED_DIR
        if directory is not None:
            self._check_names(name, directory)
        for part in self.children:
            part.children = list(runner.execute(overrides=part.job_overrides(),
                                                batch_size=batch_size, **execute_kwargs))
            part.job_ids = list(runner.last_job_ids)
            if directory is None:
                directory = experiment_root(part.children[0].fname) / ASSEMBLED_DIR
                self._check_names(name, directory)
            part.analyze(**analyze_kwargs)
            part.save(part_name(name, part.disorder_khz), directory=directory)
        self.analyze(**analyze_kwargs)
        return self.save(name, directory=directory)

    def _check_names(self, name, directory):
        """Raise before anything is written if the scan or an unsaved part name is taken."""
        named_stem(type(self).__name__, name, directory)
        for part in self.children:
            if part.manifest_path is None:
                named_stem(type(part).__name__, part_name(name, part.disorder_khz), directory)

    @classmethod
    def from_saved_parts(cls, name, directory, notes=""):
        """-> the scan of the parts saved under ``name`` in ``directory`` (after a stopped run)."""
        paths = sorted(Path(directory).glob(f"{cls.child_class.__name__}_{name}_W*kHz.yaml"))
        if not paths:
            raise FileNotFoundError(f"no saved parts of {name!r} in {directory}")
        parts = [cls.child_class.from_manifest(path) for path in paths]
        return cls.from_children(parts, job_ids=[i for part in parts for i in part.job_ids],
                                 notes=notes)

    def analyze(self, mu=None, t1_us=None, t1_err_us=None, kerr_mhz=None,
                background_start_us=None, window="hann", **kwargs):
        """Analyze every W (see ``CentralBosonAGPExperiment.analyze``) and stack its curves.

        The parts share their axes (``agp_mu``, ``frequency_khz``), so the scan
        only stacks: (W, mu) and (W, f) maps.
        """
        for part in self.children:
            part.analyze(mu=mu, t1_us=t1_us, t1_err_us=t1_err_us, kerr_mhz=kerr_mhz,
                         background_start_us=background_start_us, window=window, **kwargs)
        self.analysis = self.children[0].analysis
        parts = [part.data for part in self.children]
        data = dict(disorder_khz=np.asarray([part.disorder_khz for part in self.children]),
                    cycles=np.asarray(parts[0]["cycles"]),
                    detunings=np.asarray([part["detunings"] for part in parts]))
        if all("time_us" in part for part in parts):
            data["time_us"] = np.asarray(parts[0]["time_us"])
        keys = ["C1", "C2", "C1_err", "C2_err", "C1_theory"]
        if all("C1_corrected" in part for part in parts):
            keys += ["C1_corrected", "C2_corrected"]
        for axis, curves in (("frequency_khz", self.SPECTRUM_KEYS),
                             ("agp_mu", self.AGP_KEYS + self.BACKGROUND_KEYS + ("mu_gld_khz",))):
            if all(axis in part for part in parts):
                values = [np.asarray(part[axis]) for part in parts]
                if any(v.shape != values[0].shape or not np.allclose(v, values[0]) for v in values):
                    raise ValueError(f"the parts have different {axis}")
                data[axis] = values[0]
                keys += list(curves)
        for key in keys:
            if all(key in part for part in parts):
                data[key] = np.asarray([part[key] for part in parts])
        self.data = AttrDict(data)
        return self.data

    def summary(self):
        """-> a short text: the settings the parts share, then one line per W."""
        first = self.children[0]
        if "time_us" not in first.data:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                first.analyze()
        source = scan_stem(self.manifest_path) if self.manifest_path else "(not saved)"
        cycles = np.asarray(first.cycles)
        steps = np.unique(np.diff(cycles))
        step = f"step {steps[0]:g}" if len(steps) == 1 else "uneven steps"
        couplings, kerr = first.recorded_couplings_mhz(), first.recorded_kerr_mhz()
        lines = [f"{type(self).__name__} {source}" + (f": {self.notes}" if self.notes else ""),
                 f"N = {first.photon_sector_number}, direction {mode_name(first.direction)}, "
                 f"storages {list(first.swap_stors)}, cycles {cycles[0]}-{cycles[-1]} {step} "
                 f"({len(cycles)}), T = {first.data['time_us'][-1]:.1f} us",
                 "g = " + ("not recorded" if couplings is None
                           else f"{1e3 * np.mean(np.abs(couplings)):.2f} kHz")
                 + ", K = " + ("not recorded" if kerr is None else f"{1e3 * kerr:.2f} kHz")]
        try:
            pattern = self.disorder_pattern()
            lines.append("pattern (unit RMS): " + ("W = 0 only" if pattern is None
                                                   else np.array2string(pattern, precision=3)))
        except ValueError:
            lines.append("pattern: the parts do not share one")
        lines.append(f"{'W (kHz)':>8}  {'jobs':<46} detunings (kHz)")
        for part in self.children:
            ids = sorted(part.job_ids)
            jobs = f"{len(ids)}: {ids[0]} .. {ids[-1]}" if ids else "none recorded"
            lines.append(f"{part.disorder_khz:8g}  {jobs:<46} "
                         + " ".join(f"{1e3 * value:7.1f}" for value in part.detunings))
        return "\n".join(lines)

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
        columns = [int(np.argmin(abs(cutoffs - value))) for value in self.shown_cutoffs(mu)]
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
            if "agp_connected" in self.data:
                ax.errorbar(strengths, self.data["agp_connected"][:, column],
                            self.data["agp_connected_err"][:, column], fmt="D-", ms=4, capsize=2,
                            label="background subtracted")
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

    def display_mu(self, estimator="agp", theory=True):
        """AGP(mu) for every W on a log mu axis, with its error band. -> (fig, ax).

        ``estimator``: "agp", "agp_connected" or "agp_t1". Dashed: the
        infinite-time closed system; the dots mark g L/D of each W.
        """
        if estimator not in self.data:
            raise ValueError(f"no {estimator}: analyze with the settings it needs")
        cutoffs = np.asarray(self.data["agp_mu"])
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for row, strength in enumerate(self.data["disorder_khz"]):
            values = np.asarray(self.data[estimator][row])
            error = np.asarray(self.data[f"{estimator}_err"][row])
            line, = ax.plot(cutoffs, values, lw=1.2, label=f"W = {strength:g} kHz")
            ax.fill_between(cutoffs, values - error, values + error, color=line.get_color(),
                            alpha=.25, lw=0)
            if theory and "agp_theory" in self.data:
                ax.plot(cutoffs, self.data["agp_theory"][row], "--", lw=.8, color=line.get_color())
            if "mu_gld_khz" in self.data:
                gld = float(self.data["mu_gld_khz"][row])
                ax.plot(gld, np.interp(gld, cutoffs, values), "o", ms=4, color=line.get_color())
        ax.set(xscale="log", xlabel=r"$\mu$ (kHz)", ylabel=r"$g^2\|A_\mu\|^2$",
               title=f"{estimator}: dashed closed system (infinite time), dots g L/D")
        ax.legend(fontsize=7, ncol=2)
        return fig, ax

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

    def display_spectrum(self, mu=None, window=None, background_start_us=None, max_khz=None,
                         raw=False, theory=True, yscale="linear"):
        """The one-sided spectral function S(f) of C_1 for every W. -> (fig, axes).

        By default the analyzed curves (``spectrum``, ``spectrum_err``,
        ``spectrum_with_background`` in grey, ``spectrum_theory`` dashed): S is
        the cosine transform of C_1 over its window, per kHz, with the shot
        error, background-subtracted when analyzed so. A ``window``,
        ``background_start_us`` other than the analyzed ones, or ``raw``,
        computes S from C_1 here instead (``fitting.qsim.agp.spectral_function``).
        With the parts loaded: the closed-system weights in bins of 1/T
        (steps). The AGP filters f^2 / (f^2 + mu^2)^2, scaled to a peak of 1,
        are on the right axis (``mu`` in kHz; default ``shown_cutoffs``).
        ``max_khz`` defaults to ``SPECTRUM_MARGIN`` times the highest
        closed-system frequency over all W, at most the Nyquist frequency (the
        Nyquist frequency without the parts). A W whose closed-system weight is
        partly above the Nyquist frequency (it folds back into the measured S)
        is marked with that fraction.
        """
        if "C1" not in self.data:
            self.analyze()
        if "time_us" not in self.data:
            raise ValueError("a spectrum needs the times of the cycles (time_us)")
        analysis = getattr(self, "analysis", {}) or {}
        analyzed_window = analysis.get("window", "hann")
        analyzed_background = analysis.get("background_start_us")
        stored = ("spectrum" in self.data and not raw
                  and window in (None, analyzed_window)
                  and background_start_us in (None, analyzed_background))
        window = window or analyzed_window
        if stored:
            background_start_us = analyzed_background
        subtract = background_start_us is not None
        key = "C1" if raw or "C1_corrected" not in self.data else "C1_corrected"
        times = np.asarray(self.data["time_us"])
        resolution_khz = 1000 / times[-1]
        nyquist_khz = 1000 / (2 * np.median(np.diff(times)))
        frequency_khz = (np.asarray(self.data["frequency_khz"]) if stored
                         else spectrum_frequencies_khz(times))
        model = self.model_spectra_khz() if self.children else []
        if max_khz is None:
            top = max((f[w > 1e-3 * w.max()].max() for f, w in model if len(f)), default=None)
            max_khz = nyquist_khz if top is None else min(nyquist_khz, self.SPECTRUM_MARGIN * top)
        cutoffs = self.shown_cutoffs(mu)
        with_parts = theory and bool(self.children) and "theory_kerr_mhz" in self.children[0].data
        count = len(self.data["disorder_khz"])
        fig, axes = plt.subplots(count, 1, figsize=(9, 2 * count), sharex=True, squeeze=False,
                                 constrained_layout=True)
        for row, ax in enumerate(axes[:, 0]):
            theory_curve = None
            if stored:
                spectrum = np.asarray(self.data["spectrum"][row])
                spread = np.asarray(self.data["spectrum_err"][row])
                unsubtracted = (np.asarray(self.data["spectrum_with_background"][row])
                                if subtract else None)
                if theory and "spectrum_theory" in self.data:
                    theory_curve = np.asarray(self.data["spectrum_theory"][row])
            else:
                values, error = np.asarray(self.data[key][row]), np.asarray(self.data["C1_err"][row])
                unsubtracted = spectral_function(times, values, error, frequency_khz, window)[0]
                if subtract:
                    values = subtract_background(times, values, error, background_start_us)[0]
                spectrum, spread = spectral_function(times, values, error, frequency_khz, window)
            ax.plot(frequency_khz, spectrum, lw=1,
                    label=key.replace("_", " ") + (", background subtracted" if subtract else ""))
            ax.fill_between(frequency_khz, spectrum - spread, spectrum + spread, alpha=.3, lw=0)
            if with_parts:
                part = self.children[row]
                differences, weights = part.theory_spectrum(
                    kerr_mhz=float(part.data["theory_kerr_mhz"]))
                dc, mask = split_dc(differences, weights)
                kept = ~mask if subtract else np.ones(mask.shape, dtype=bool)
                edges = np.arange(0, max_khz + resolution_khz, resolution_khz)
                binned, _ = np.histogram(1000 * np.abs(differences[kept]), edges,
                                         weights=weights[kept])
                ax.stairs(binned / resolution_khz, edges, color="0.5", lw=.8,
                          label="closed system, weights in bins of 1/T")
                if theory_curve is None and theory and "C1_theory" in self.data:
                    theory_curve = spectral_function(
                        times, self.data["C1_theory"][row] - (dc if subtract else 0.),
                        np.zeros(len(times)), frequency_khz, window)[0]
            elif theory_curve is None and theory and "C1_theory" in self.data and not subtract:
                theory_curve = spectral_function(times, self.data["C1_theory"][row],
                                                 np.zeros(len(times)), frequency_khz, window)[0]
            if theory_curve is not None:
                ax.plot(frequency_khz, theory_curve, "k--", lw=1, label="closed system, same window")
            ax.set_yscale(yscale)
            if subtract and unsubtracted is not None:  # grey, and the y range stays on the subtracted S
                limits = ax.get_ylim()
                ax.plot(frequency_khz, unsubtracted, color="0.7", lw=1,
                        label=f"{key.replace('_', ' ')}, with background")
                ax.set_ylim(limits)
            ax.set_ylabel(f"W = {self.data['disorder_khz'][row]:g} kHz\nS (1/kHz)", fontsize=8)
            if model:
                f, w = model[row]
                folded = w[f > nyquist_khz].sum() / w.sum() if w.sum() > 0 else 0.
                if folded >= 1e-3:
                    ax.text(.99, .95, f"{100 * folded:.1f} % of the closed-system weight is above "
                            f"Nyquist ({nyquist_khz:.0f} kHz) and folds back", transform=ax.transAxes,
                            ha="right", va="top", fontsize=7, color="C3")
            if len(cutoffs):
                filters = ax.twinx()
                for cutoff in cutoffs:
                    shape = 4 * cutoff**2 * frequency_khz**2 / (frequency_khz**2 + cutoff**2)**2
                    filters.plot(frequency_khz, shape, ":", lw=1, label=rf"filter, $\mu$ = {cutoff:g} kHz")
                low, high = ax.get_ylim()  # the zero of the filters on the zero of S
                filters.set(ylim=(1.05 * min(low, 0) / max(high, 1e-12), 1.05), yticks=[])
                if row == 0:
                    handles = ax.get_legend_handles_labels()
                    others = filters.get_legend_handles_labels()
                    ax.legend(handles[0] + others[0], handles[1] + others[1], fontsize=7, ncol=2)
            elif row == 0:
                ax.legend(fontsize=7, ncol=2)
        fig.suptitle(f"{type(self).__name__}: spectral function of C_1, {window} window, "
                     f"1/T = {resolution_khz:.2g} kHz", fontsize=10)
        axes[-1, 0].set(xlabel="f (kHz)", xlim=(0, max_khz))
        return fig, axes[:, 0]

    SHOWN_CUTOFFS_KHZ = (5., 10., 20.)

    def shown_cutoffs(self, mu=None):
        """-> the cutoffs (kHz) a plot shows: ``mu``; else the analyzed ones, when 4 or fewer;
        else the analyzed ones nearest to ``SHOWN_CUTOFFS_KHZ``."""
        if mu is not None:
            return np.atleast_1d(mu).astype(float)
        analyzed = np.asarray(self.data.get("agp_mu", []), dtype=float)
        if len(analyzed) <= 4:
            return analyzed
        return np.unique([analyzed[np.argmin(abs(analyzed - value))]
                          for value in self.SHOWN_CUTOFFS_KHZ])

    SPECTRUM_MARGIN = 1.3

    def model_spectra_khz(self):
        """-> per W, (f_kHz, |weights|) of the closed-system C_1 without its DC terms.

        At the Kerr of the last analysis (else the recorded one). ``[]`` when a
        part has no recorded couplings or Kerr.
        """
        spectra = []
        for part in self.children:
            kerr = part.data.get("theory_kerr_mhz")
            try:
                differences, weights = part.theory_spectrum(
                    kerr_mhz=None if kerr is None else float(kerr))
            except ValueError:
                return []
            _, mask = split_dc(differences, weights)
            spectra.append((1000 * np.abs(np.asarray(differences)[~mask]),
                            np.abs(np.asarray(weights)[~mask])))
        return spectra

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

    def save(self, name, directory=None, notes=None):
        """Save the parts that are not saved yet, then this scan. -> the manifest path.

        The scan is ``CentralBosonAGPDisorderScan_<name>``, a part
        ``CentralBosonAGPExperiment_<name>_W<W>kHz`` (``p`` for a decimal
        point). All names are checked before anything is written.
        """
        if directory is None:
            directory = experiment_root(self.children[0].children[0].fname) / ASSEMBLED_DIR
        self._check_names(name, directory)
        for part in [part for part in self.children if part.manifest_path is None]:
            if not part.data:
                part.analyze()
            part.save(part_name(name, part.disorder_khz), directory=directory)
        if not self.data:
            self.analyze()
        return super().save(name, directory=directory, notes=notes)

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

    def assembled_attrs(self):
        """The analysis settings of the saved curves (cutoffs, window, background, T1, Kerr)."""
        return dict(analysis=getattr(self, "analysis", {}))


########################################################################
# Data sets: find, load and compare disorder scans.
########################################################################

SCAN_REGISTRY = Path(__file__).resolve().parents[2] / "analysis_notebooks" / "datasets" / "agp_scans.yaml"
SCAN_CLASS = "CentralBosonAGPDisorderScan"


def part_name(name, disorder_khz):
    """-> the name of one W of the scan ``name``: ``<name>_W120kHz``, ``<name>_W0p5kHz``."""
    return f"{name}_W{float(disorder_khz):g}kHz".replace(".", "p").replace("-", "m")


def scan_stem(path):
    """-> the short name of a scan manifest: its name, ``CentralBosonAGPDisorderScan_<name>``.

    Files saved before names were required (``<time stamp>_CentralBosonAGPDisorderScan[_2]``)
    give their time stamp.
    """
    stem = Path(path).stem
    if stem.startswith(SCAN_CLASS + "_"):
        return stem[len(SCAN_CLASS) + 1:]
    return stem.replace("_" + SCAN_CLASS, "")


def local_path(path):
    """-> ``path`` on this machine: as given when it exists, else re-rooted onto the data root.

    So a path copied on the measurement PC (``C:\\experiments\\...``) works off it.
    """
    given = Path(path)
    if given.exists():
        return given
    rerooted = _reroot(str(path), data_root())
    if rerooted is None or not rerooted.exists():
        raise FileNotFoundError(f"{path} (data root {data_root()})")
    return rerooted


def registered_scans():
    """-> {key: entry} of ``analysis_notebooks/datasets/agp_scans.yaml``."""
    with open(SCAN_REGISTRY, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["scans"]


def _job_range(ids):
    ids = sorted(ids)
    return f"{ids[0]} .. {ids[-1]}" if ids else ""


def list_scans(folder):
    """-> a table (pandas DataFrame) of the disorder-scan manifests in an experiment folder.

    Reads only the manifests, so it is fast. ``folder`` is the experiment folder
    or its ``assembled_data``. ``part_of`` names a scan in the same folder
    whose jobs include all of this one's: then this one is a subset.
    """
    import pandas as pd

    folder = local_path(folder)
    if (folder / "assembled_data").is_dir():
        folder = folder / "assembled_data"
    rows, jobs = [], {}
    for path in sorted(folder.glob(f"*{SCAN_CLASS}*.yaml")):
        manifest = read_manifest(path)
        stem = scan_stem(path)
        jobs[stem] = set()
        for part in manifest["raw_files"]:
            with open(part, encoding="utf-8") as handle:
                jobs[stem] |= set(yaml.safe_load(handle).get("job_ids", []))
        p = manifest["parameters"]
        cycles = np.asarray(p["cycles"])
        rows.append(dict(scan=stem, direction=mode_name(p["direction"]),
                         W_khz=p["disorder_khz"], cycles=f"{cycles[0]}-{cycles[-1]} ({len(cycles)})",
                         jobs=len(jobs[stem]), job_range=_job_range(jobs[stem]),
                         notes=manifest.get("notes", ""), manifest=str(path)))
    for row in rows:
        supersets = [other for other, ids in jobs.items()
                     if other != row["scan"] and jobs[row["scan"]] < ids]
        row["part_of"] = ", ".join(supersets)
    return pd.DataFrame(rows)


def load_scan(source):
    """-> a ``CentralBosonAGPDisorderScan`` from a registry key or a manifest path (loaded, not analyzed).

    A path may be one from the measurement PC (``local_path``).
    """
    entries = registered_scans()
    if isinstance(source, str) and source in entries:
        path = data_root() / entries[source]["manifest"]
    else:
        path = local_path(source)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Calibration file not found")
        return CentralBosonAGPDisorderScan.from_manifest(path)


def load_scans(*keys):
    """-> {key: scan} for registry keys (all of them when none is given)."""
    return {key: load_scan(key) for key in (keys or registered_scans())}


def display_scans(scans, mu, estimator="agp_connected", theory=True, yscale="linear"):
    """AGP against W of several analyzed scans, one panel per cutoff. -> (fig, axes).

    ``scans``: {label: scan}, each analyzed at every ``mu`` (kHz). ``estimator``:
    ``agp_connected`` (needs ``analyze(background_start_us=...)``), ``agp`` or
    ``agp_t1``. Points: measured, with their errors. Dashed, same color: the
    infinite-time closed system on that scan's pattern, couplings and Kerr.
    """
    cutoffs = np.atleast_1d(np.asarray(mu, dtype=float))
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(4.5 * len(cutoffs), 3.8), squeeze=False,
                             sharex=True, constrained_layout=True)
    for index, (label, scan) in enumerate(scans.items()):
        if estimator not in scan.data:
            raise ValueError(f"{label} has no {estimator}; analyze it first "
                             "(agp_connected needs background_start_us)")
        analyzed = np.asarray(scan.data["agp_mu"])
        strengths = np.asarray(scan.data["disorder_khz"])
        pattern = scan.disorder_pattern() if theory and scan.children else None
        for ax, cutoff in zip(axes[0], cutoffs):
            match = np.flatnonzero(np.isclose(analyzed, cutoff))
            if not match.size:
                raise ValueError(f"{label} was not analyzed at mu = {cutoff:g} kHz "
                                 f"(it has {analyzed.tolist()})")
            column = match[0]
            ax.errorbar(strengths, scan.data[estimator][:, column],
                        scan.data[f"{estimator}_err"][:, column], fmt="o", ms=4, capsize=2,
                        color=f"C{index}", label=label)
            if not theory or "agp_theory" not in scan.data:
                continue
            if pattern is None:  # W = 0 only, or no parts: the analyzed theory points
                ax.plot(strengths, scan.data["agp_theory"][:, column], "_", ms=12,
                        color=f"C{index}")
            else:
                dense = np.linspace(0, strengths.max(), 81)
                ax.plot(dense, scan.theory_agp(dense, mu=cutoff)[:, 0], "--", lw=1,
                        color=f"C{index}")
    for ax, cutoff in zip(axes[0], cutoffs):
        ax.set(title=rf"$\mu$ = {cutoff:g} kHz", xlabel="Disorder RMS W (kHz)", yscale=yscale)
    axes[0, 0].set_ylabel(r"$g^2\|A_\mu\|^2$" + f" ({estimator})")
    axes[0, 0].legend(fontsize=7, title="dashed: closed system", title_fontsize=7)
    return fig, axes[0]
