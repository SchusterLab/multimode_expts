"""Definite Fock preparation and population readout (quantum walks, AGP measurement).

- ``FockPopulationProgram``: prepare one Fock occupation, run Floquet cycles, read
  one mode by multiparity readout.
- ``FockPopulationExperiment``: one job of that Program, one occupation and one read
  mode over a cycle sweep. A few minutes on the device. Its file holds the shots and
  the raw and readout-corrected number moments.
- ``FockPopulationSetExperiment``: an assembled set of such jobs on one cycle grid
  (a quantum walk, or one disorder strength of an AGP scan, see ``agp.py``). Its
  files hold only the stacked moments, and a manifest that points to the jobs.

The Program moved on 2026-10-15 from the end of ``floquet_dark_mode_readout.py``,
where jonginn added it as a "temporary amendment for adiabatic gauge potential
measurement". The rest of that file was a compatibility shim that guan deleted
on 2026-10-06; this code is the only part kept.
"""
import json
from copy import deepcopy
from numbers import Integral
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments.assembled_data import AssembledExperiment
# from experiments.dataset import StorageManSwapDataset  # For the paused builder below.
from experiments.qsim.floquet_train import FloquetProgram
from experiments.qsim.multiparity import (
    MultiparityQsimExperiment,
    multiparity_calibration_config,
    multiparity_calibration_from_file,
)
from experiments.qsim.qsim_base import QsimExperiment

# Definite Fock preparation and population readout.


def validate_occupations(occupations, swap_stors):
    """Check nonnegative Fock occupations in M1 plus storage order."""
    if len(occupations) != len(swap_stors) + 1:
        raise ValueError("occupations must be [n_M1] followed by swap_stors")
    if len(set(swap_stors)) != len(swap_stors):
        raise ValueError("swap_stors must contain distinct storage modes")
    if any(isinstance(stor, bool) or not isinstance(stor, Integral)
           or stor not in range(1, 8) for stor in swap_stors):
        raise ValueError("storage modes must be in 1..7")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0
           for n in occupations):
        raise ValueError("occupations must contain non-negative integers")


def set_population_defaults(expt):
    """Fill missing population settings, leaving readout selection to Qsim."""
    defaults = dict(prepulse=False, # Fock ladder is climbed in `corepulse`, so should be false
                    postpulse=True,
                    init_stor=0,
                    ro_stor=0,
                    parity_fast=False,
                    cond_sec_phase=-90) # For multi parity
    for name, value in defaults.items():
        expt.setdefault(name, value)


def validate_population_config(expt):
    """Check preparation and storage routing independently of readout mode."""
    validate_occupations(expt.occupations, expt.swap_stors)
    if expt.prepulse:
        raise ValueError("prepulse must be False: occupations already supplies Fock preparation")
    if expt.ro_stor > 0 and not expt.postpulse:
        raise ValueError("storage readout uses the existing postpulse=True path")
    if expt.get("man_mode_no", 1) != 1:
        raise ValueError("the calibrated preparation path uses M1")


# Retained with the paused gate-list builder below.
# def with_storage_waits(pulses, wait_us):
#     """Copy a gate list, adding a wait after each storage swap."""
#     result = []
#     for pulse in pulses:
#         result.append(list(pulse))
#         if pulse[0] == "storage" and wait_us > 0:
#             result.append(["wait", wait_us])
#     return result


class FockPopulationProgram(FloquetProgram):
    """The ordinary Qsim reset/readout template with a population experiment core.

    ``occupations`` is [n_M1, n_S...] in ``swap_stors`` order. ``ro_stor`` is
    0 for M1. With ``postpulse=True``, the existing Qsim body swaps
    ``ro_stor`` into M1 before the configured readout.
    Preparation uses MM_base.man_stor_swap(), including its fixed swap delay.
    The optional multiphoton gate-list builder is commented out for now;
    use_multiphoton_swap and storage_pulse_wait_us are not used by this path.
    """

    def initialize(self):
        expt = self.cfg.expt
        set_population_defaults(expt)
        validate_population_config(expt)
        super().initialize()

    def _play_gate_list(self, pulses, prefix):
        if not pulses:
            return
        creator = self.get_prepulse_creator(deepcopy(pulses))
        self.sync_all()
        self.custom_pulse(self.cfg, creator.pulse, prefix=prefix)
        self.sync_all() 

    def _prepare_occupation(self):
        expt = self.cfg.expt
        for storage, number in zip(expt.swap_stors, expt.occupations[1:]):
            if number == 0:
                continue
            self._play_gate_list(self.prep_man_fock_state(1, str(number)),
                                 prefix=f"population_prep_S{storage}_")
            self.man_stor_swap(man_idx=1, stor_idx=storage)
        
        self._play_gate_list(self.prep_man_fock_state(1, str(expt.occupations[0])),
                             prefix="population_prep_M1_")
        
    
    def core_pulses(self):
        expt = self.cfg.expt
        self._prepare_occupation()

        storages = list(expt.swap_stors)
        self.phase_offsets = [0.0] * len(storages)
        self.disorder_phase_offsets = [0.0] * len(storages)
        self._play_scramble_with_phase_offsets(
            self.phase_offsets, storages, self.disorder_phase_offsets)
        # QsimProgram.body handles postpulse (ro_stor -> man) and readout.


    # Paused: use MM_base.man_stor_swap() for the current preparation path.
    # def build_fock_preparation(self,
    #                            occupations,
    #                            swap_stors,
    #                            use_multiphoton_swap=False,
    #                            storage_pulse_wait_us=0.2):
    #     """Load initially empty storages once, then load M1 last."""
    #     validate_occupations(occupations, swap_stors)
    #     pulses = []
    #     for storage, number in zip(swap_stors, occupations[1:]):
    #         if number == 0:
    #             continue
    #         pulses.extend(self.prep_man_fock_state(1, str(number)))
    #         row = f"M1-S{storage}"
    #         if use_multiphoton_swap: #TODO: THERE IS NO THIS CONFIG FOR NOW. TO IMPEMENT
    #             row = StorageManSwapDataset.multiphoton_swap_name(row, number)
    #         pulses.append(["storage", row, "pi", 0.0])
    #     pulses.extend(self.prep_man_fock_state(1, str(occupations[0])))
    #     return with_storage_waits(pulses, storage_pulse_wait_us)


# --------------------------------------------------------------------------
# One population job, and an assembled set of them
# --------------------------------------------------------------------------

NUMBERS = np.arange(4)


def mode_name(mode):
    """-> ``M1`` for mode 0, else ``S<mode>``."""
    return "M1" if int(mode) == 0 else f"S{int(mode)}"


def multiparity_moment_error(probabilities, shot_counts, values, confusion_matrix=None):
    """Shot-only standard error of ``sum_n values[n] p(n)``, the matrix fixed.

    The estimator of ``multiparity.multiparity_mean_error`` for any function
    of the label, for example ``values = n**2``. ``probabilities`` is
    (points, 4) raw; with a matrix, the corrected moment is the raw mean of
    the weights ``solve(C, values)``. Fewer than two shots gives NaN.
    """
    weights = np.asarray(values, dtype=float)
    if confusion_matrix is not None:
        weights = np.linalg.solve(np.asarray(confusion_matrix, dtype=float), weights)
    probabilities = np.asarray(probabilities, dtype=float)
    shots = np.asarray(shot_counts, dtype=float)
    mean = probabilities @ weights
    spread = np.maximum(probabilities @ weights**2 - mean**2, 0)
    variance = np.full(len(probabilities), np.nan)
    many = shots > 1
    variance[many] = spread[many] / (shots[many] - 1)
    return np.sqrt(variance)


def calibration_record(calibration):
    """-> the readout correction as a JSON-plain dict ``{matrix, fname?, job_id?}``, or None.

    Accepts what ``cfg.expt.multiparity_confusion_matrix`` accepts: a
    calibration HDF5 filename, a 4x4 matrix C[prepared, measured], a dict
    from ``multiparity_calibration_config``, or an analyzed calibration Experiment.
    """
    if calibration is None:
        return None
    if isinstance(calibration, (str, Path)):
        calibration = multiparity_calibration_from_file(calibration)
    elif isinstance(calibration, QsimExperiment):
        calibration = multiparity_calibration_config(calibration)
    elif not isinstance(calibration, dict):
        calibration = dict(matrix=calibration)
    record = {key: str(value) for key, value in calibration.items() if key != "matrix"}
    record["matrix"] = np.asarray(calibration["matrix"], dtype=float).tolist()
    return record


def recorded_cycle_us(job):
    """-> the compiled Floquet cycle time a job file recorded (derived_params), or None."""
    attrs = job.data.get("attrs") if isinstance(job.data, dict) else None
    if attrs is None and getattr(job, "fname", None) and Path(job.fname).exists():
        with h5py.File(job.fname, "r") as handle:
            attrs = dict(handle.attrs)
    derived = (attrs or {}).get("derived_params")
    if derived is None:
        return None
    if isinstance(derived, (str, bytes)):
        derived = json.loads(derived)
    value = derived.get("floquet_cycle_us")
    return None if value is None else float(value)


class FockPopulationExperiment(MultiparityQsimExperiment):
    """One population job: one prepared occupation, one read mode, a cycle sweep.

    The job a quantum walk or an AGP scan is made of. ``cfg.expt.occupations``
    is [n_M1, *swap_stors], ``ro_stor`` the read mode (0 = M1), and
    ``floquet_cycles`` the sweep. Give the readout correction at submission
    (``cfg.expt.multiparity_confusion_matrix``, best as the dict of
    ``multiparity_calibration_config``): the worker analyzes before it saves,
    so the file then holds raw and corrected results side by side.

    analyze() adds, each of shape (cycles,): ``nmod4_second_moment``,
    ``nmod4_second_moment_corrected`` (with a matrix), ``n2_err`` and
    ``shot_counts``. ``n_err`` and ``n2_err`` are shot-only errors of the
    corrected moments when a matrix is set, else of the raw ones. They are
    photon-number moments only while n <= 3.
    """

    default_program = FockPopulationProgram
    SHOT_KEYS = ("idata", "qdata")

    def sweep_axes(self):
        """One axis: ``floquet_cycle`` over ``cfg.expt.floquet_cycles``."""
        self.cfg.expt.swept_params = ["floquet_cycle"]
        return [("floquet_cycle", self.cfg.expt.floquet_cycles)]

    def analyze(self, data=None, **kwargs):
        super().analyze(data=data, **kwargs)
        shape = np.shape(self.data["nmod4_mean"])
        shots = np.asarray(self.data["idata"]).shape[-1] // int(self.cfg.read_num)
        shot_counts = np.full(shape, shots)
        raw = np.stack([self.data[f"p_mod{n}"] for n in range(4)], axis=-1)
        confusion = self.data.get("confusion_matrix")
        self.data["nmod4_second_moment"] = raw @ NUMBERS**2
        self.data.pop("nmod4_second_moment_corrected", None)
        if confusion is not None:
            corrected = np.stack([self.data[f"p_mod{n}_corrected"] for n in range(4)], axis=-1)
            self.data["nmod4_second_moment_corrected"] = corrected @ NUMBERS**2
        self.data["n2_err"] = multiparity_moment_error(
            raw.reshape(-1, 4), shot_counts.reshape(-1), NUMBERS**2, confusion).reshape(shape)
        self.data["shot_counts"] = shot_counts
        return self.data

    def display(self, data=None, raw=False, **kwargs):
        """Mean n mod 4 vs Floquet cycle with its shot error; corrected when available."""
        if data is None:
            data = self.data
        corrected = "nmod4_mean_corrected" in data
        key, kind = (("nmod4_mean", "raw") if raw or not corrected
                     else ("nmod4_mean_corrected", "corrected"))
        cycles = np.asarray(data["xpts"]).reshape(-1)
        mean = np.asarray(data[key]).reshape(-1)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(cycles, mean, ".-", lw=1)
        # n_err belongs to the corrected mean when there is one.
        if "n_err" in data and (kind == "corrected") == corrected:
            error = np.asarray(data["n_err"]).reshape(-1)
            ax.fill_between(cycles, mean - error, mean + error, alpha=.3, lw=0)
        expt = self.cfg.expt
        name = Path(getattr(self, "fname", None) or type(self).__name__).name
        ax.set(xlabel="Floquet cycles", ylabel=f"Mean n mod 4 ({kind})")
        ax.set_title(f"{list(expt.occupations)}, read {mode_name(expt.ro_stor)}\n{name}", fontsize=9)
        fig.tight_layout()
        return fig, ax

    @classmethod
    def from_h5file(cls, fname, load_shots=True):
        """slab's ``from_h5file``; ``load_shots=False`` skips ``idata``/``qdata``.

        The shots are almost all of a job file, and the assembled classes
        need only the analyzed results, so ``from_manifest`` loads without them.
        """
        if load_shots:
            return super().from_h5file(fname)
        self = cls.__new__(cls)
        self.fname = str(fname)
        with h5py.File(fname, "r") as handle:
            self.data = {key: handle[key][()] for key in handle if key not in cls.SHOT_KEYS}
            self.data["attrs"] = dict(handle.attrs)
        self.cfg = AttrDict(json.loads(self.data["attrs"]["config"]))
        return self


def assembled_from_h5file(cls, path):
    """Arrays-only view of a saved assembled HDF5, without its children.

    ``display()`` works on it; ``analyze()`` and ``save()`` need
    ``from_manifest()``, which loads the layer below.
    """
    with h5py.File(path, "r") as handle:
        arrays = {key: handle[key][()] for key in handle}
        attrs = dict(handle.attrs)
    self = cls.__new__(cls)
    AssembledExperiment.__init__(self, notes="")
    self.data = AttrDict(arrays)
    self.fname = str(path)
    self.manifest_path = Path(path).with_name(attrs["manifest"]) if "manifest" in attrs else None
    return self


class FockPopulationSetExperiment(AssembledExperiment):
    """Population jobs on one cycle grid and one set of detunings, kept together.

    One FockPopulationExperiment per (occupation, read mode). A quantum walk
    keeps the occupation and changes the read mode; one disorder strength of
    an AGP scan (``CentralBosonAGPExperiment``) changes the occupation. The
    jobs carry their own readout correction; this class only stacks their
    results and never needs their shots.

        walk = FockPopulationSetExperiment(
            [([1, 0, 0, 0, 0], mode) for mode in (0, 4, 5, 6, 7)], cycles,
            swap_stors=[4, 5, 6, 7], detunings=[0.] * 4, calibration=calibration_h5)
        walk.acquire(runner)      # runner.ExptClass is FockPopulationExperiment
        walk.analyze(); walk.save(); walk.display()
        walk = FockPopulationSetExperiment.from_manifest(manifest_path)
        walk = FockPopulationSetExperiment.from_h5file(h5_path)   # arrays only, to look

    ``data`` and the assembled HDF5, rows in ``entries`` order:
    ``occupations`` (job, mode), ``ro_stor`` (job,), ``cycles``, ``time_us``
    (when the jobs recorded their cycle time), ``swap_stors``, ``detunings``
    (MHz), and per (job, cycle): ``shot_counts``, ``n_mean``, ``n2_mean``,
    ``n_err``, ``n2_err``, with ``p_mod`` (job, cycle, 4). With a correction
    also ``n_mean_corrected``, ``n2_mean_corrected``, ``p_mod_corrected`` and
    ``confusion_matrix``; the errors then belong to the corrected moments.
    """

    child_class = FockPopulationExperiment
    from_h5file = classmethod(assembled_from_h5file)

    def __init__(self, entries, cycles, swap_stors, detunings, calibration=None, notes=""):
        super().__init__(notes=notes)
        self.entries = [(tuple(int(n) for n in occupation), int(mode))
                        for occupation, mode in entries]
        if not self.entries:
            raise ValueError("a population set needs at least one job")
        if len(set(self.entries)) != len(self.entries):
            raise ValueError("each (occupation, read mode) must appear once")
        self.cycles = [int(cycle) for cycle in cycles]
        self.swap_stors = [int(storage) for storage in swap_stors]
        self.detunings = [float(detuning) for detuning in detunings]
        if len(self.detunings) != len(self.swap_stors):
            raise ValueError("one detuning per storage in swap_stors")
        for occupation, mode in self.entries:
            validate_occupations(list(occupation), self.swap_stors)
            if mode not in (0, *self.swap_stors):
                raise ValueError(f"read mode {mode} is not M1 or one of {self.swap_stors}")
        self.calibration = calibration_record(calibration)

    def job_overrides(self):
        """One runner override per entry; each job carries the readout correction."""
        return [dict(occupations=list(occupation), ro_stor=mode,
                     floquet_cycles=list(self.cycles), swap_stors=list(self.swap_stors),
                     detunings=list(self.detunings),
                     multiparity_confusion_matrix=deepcopy(self.calibration))
                for occupation, mode in self.entries]

    @staticmethod
    def shared_settings(children):
        """-> (entries, cycles, swap_stors, detunings, calibration) the jobs share.

        Occupation and read mode may change from job to job; the cycle grid,
        the storages, the detunings and the readout correction may not.
        """
        children = list(children)
        if not children:
            raise ValueError("a population set needs at least one job")
        first = children[0].cfg.expt
        cycles = [int(cycle) for cycle in first.floquet_cycles]
        calibration = _job_calibration(children[0])
        for child in children:
            expt = child.cfg.expt
            if [int(cycle) for cycle in expt.floquet_cycles] != cycles:
                raise ValueError(f"{child.fname}: different Floquet cycles")
            if list(expt.swap_stors) != list(first.swap_stors):
                raise ValueError(f"{child.fname}: different swap_stors")
            if not np.allclose(expt.detunings, first.detunings, rtol=0, atol=1e-12):
                raise ValueError(f"{child.fname}: different detunings")
            other = _job_calibration(child)
            if (other is None) != (calibration is None) or (
                    other is not None and not np.allclose(other["matrix"], calibration["matrix"])):
                raise ValueError(f"{child.fname}: different readout correction")
        entries = [(child.cfg.expt.occupations, child.cfg.expt.ro_stor) for child in children]
        return entries, cycles, list(first.swap_stors), list(first.detunings), calibration

    @classmethod
    def from_children(cls, children, job_ids=(), notes="", **kwargs):
        """Assemble loaded jobs, in the order given."""
        children = list(children)
        entries, cycles, swap_stors, detunings, calibration = cls.shared_settings(children)
        assembled = cls(entries, cycles, swap_stors, detunings, calibration=calibration,
                        notes=notes)
        assembled.children = children
        assembled.job_ids = list(job_ids)
        assembled._check_children()
        return assembled

    def _ordered(self, children, job_ids):
        """-> children and job IDs sorted into ``self.entries`` order."""
        children, job_ids = list(children), list(job_ids)
        keys = [(tuple(int(n) for n in child.cfg.expt.occupations), int(child.cfg.expt.ro_stor))
                for child in children]
        if sorted(keys) != sorted(self.entries):
            raise ValueError("the jobs must cover each (occupation, read mode) exactly once")
        order = [keys.index(entry) for entry in self.entries]
        return ([children[index] for index in order],
                [job_ids[index] for index in order] if job_ids else [])

    def analyze(self, **kwargs):
        """Stack the jobs' saved results. Jobs without results are analyzed if they have shots."""
        self._check_children()
        for child in self.children:
            if "n2_err" not in child.data:
                if "idata" not in child.data:
                    raise ValueError(f"{child.fname} holds no analyzed results; "
                                     "load it with its shots to analyze it")
                child.analyze()

        def stack(key):
            rows = [np.asarray(child.data[key], dtype=float).reshape(-1) for child in self.children]
            if any(row.shape != (len(self.cycles),) for row in rows):
                raise ValueError(f"{key}: one value per cycle expected in every job")
            return np.asarray(rows)

        data = dict(
            occupations=np.asarray([occupation for occupation, _ in self.entries], dtype=int),
            ro_stor=np.asarray([mode for _, mode in self.entries], dtype=int),
            cycles=np.asarray(self.cycles),
            swap_stors=np.asarray(self.swap_stors, dtype=int),
            detunings=np.asarray(self.detunings),
            shot_counts=stack("shot_counts"),
            p_mod=np.stack([stack(f"p_mod{n}") for n in range(4)], axis=-1),
            n_mean=stack("nmod4_mean"), n2_mean=stack("nmod4_second_moment"),
            n_err=stack("n_err"), n2_err=stack("n2_err"),
        )
        corrected = ["nmod4_mean_corrected" in child.data for child in self.children]
        if any(corrected) and not all(corrected):
            raise ValueError("some jobs are readout-corrected and some are not")
        if all(corrected):
            data.update(
                p_mod_corrected=np.stack([stack(f"p_mod{n}_corrected") for n in range(4)], axis=-1),
                n_mean_corrected=stack("nmod4_mean_corrected"),
                n2_mean_corrected=stack("nmod4_second_moment_corrected"),
                confusion_matrix=np.asarray(self.calibration["matrix"]),
            )
        cycle_us = {recorded_cycle_us(child) for child in self.children}
        if len(cycle_us) == 1 and None not in cycle_us:
            data["time_us"] = data["cycles"] * cycle_us.pop()
        self.data = AttrDict(data)
        return self.data

    def display(self, raw=False, **kwargs):
        """Mean read-mode number vs cycles: one line per job, or a map for many jobs."""
        if "n_mean" not in self.data:
            self.analyze()
        key, kind = (("n_mean", "raw") if raw or "n_mean_corrected" not in self.data
                     else ("n_mean_corrected", "corrected"))
        rows, cycles = np.asarray(self.data[key]), np.asarray(self.data["cycles"])
        labels = [f"{list(occupation)} read {mode_name(mode)}"
                  for occupation, mode in zip(self.data["occupations"], self.data["ro_stor"])]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        if len(rows) <= 8:
            for label, row in zip(labels, rows):
                ax.plot(cycles, row, lw=1, label=label)
            ax.set_ylabel(f"Mean n mod 4 ({kind})")
            ax.legend(fontsize=7, ncol=2)
        else:
            mesh = ax.pcolormesh(cycles, np.arange(len(rows)), rows, shading="auto")
            ax.set_yticks(range(len(rows)), labels, fontsize=7)
            fig.colorbar(mesh, ax=ax, label=f"Mean n mod 4 ({kind})")
        detunings = np.asarray(self.data["detunings"])
        ax.set(xlabel="Floquet cycles",
               title=f"{type(self).__name__}: {len(rows)} jobs, "
                     f"detuning RMS {1e3 * np.sqrt(np.mean(detunings**2)):.1f} kHz")
        return fig

    # -- persistence ------------------------------------------------------

    def manifest_parameters(self):
        return dict(entries=[[list(occupation), mode] for occupation, mode in self.entries],
                    cycles=self.cycles, swap_stors=self.swap_stors, detunings=self.detunings,
                    calibration=self.calibration)

    def assembled_arrays(self):
        return dict(self.data)


def _job_calibration(job):
    """-> the readout correction a job was analyzed with, as a record, or None."""
    matrix = job.data.get("confusion_matrix")
    if matrix is None:
        return None
    saved = job.cfg.expt.get("multiparity_confusion_matrix")
    record = calibration_record(saved) if isinstance(saved, dict) else {}
    record["matrix"] = np.asarray(matrix, dtype=float).tolist()
    return record

