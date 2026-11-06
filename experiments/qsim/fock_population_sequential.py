"""Sequential number readout after one Fock preparation and Floquet evolution.

Use ``FockPopulationSequentialReadoutExperiment`` with the ordinary runner,
``readout='multiparity'`` and ``ro_stors=[0, 4, 5, ...]`` (0 denotes M1).
Every mode retains its own two readout lanes and an explicit result axis,
including a one-mode list. Photon-number labels are modulo four.

Optional ``multiparity_confusion_matrices`` is a (4, 4, steps) array, a
record with matrices and mode order, or a sequential calibration HDF5 path.
Matrices use axes (prepared number, measured label, readout step). They are
never inferred from the legacy single-mode calibration.

``analyze`` writes all results into ``self.data`` for normal HDF5 saving.
All modes stay in the same measurement file. AGP analysis reads that file
directly. No parent class changes.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.fock_population import (
    FockPopulationExperiment,
    FockPopulationProgram,
    mode_name,
    multiparity_moment_error,
)
from experiments.qsim.qsim_base import _herald_readouts, readout_mode


def readout_layout(cfg):
    """Describe emitted lanes and reject silently incorrect readout protocols."""
    expt = cfg.expt
    modes = expt.get("ro_stors", 0)
    if isinstance(modes, np.ndarray):
        modes = modes.tolist()
    if isinstance(modes, (int, np.integer)):
        modes = [int(modes)]
    else:
        modes = list(modes)
    if readout_mode(expt) != "multiparity":
        raise ValueError("Sequential readout requires readout='multiparity'")
    if expt.get("phase_second_pulse", 180) != 180 or expt.get("cond_sec_phase", -90) != -90:
        raise ValueError("Sequential labels require phase_second_pulse=180 and cond_sec_phase=-90")
    if not modes or len(set(modes)) != len(modes) or 0 in modes[1:]:
        raise ValueError("Readout modes must be nonempty and distinct; M1 (0) can only be first")
    heralds = _herald_readouts(cfg)
    pairs = np.arange(heralds, heralds + 2 * len(modes)).reshape(-1, 2)
    preselection = MMAveragerProgram.lane_layout(cfg)["idx_pre_selection_list"]
    return dict(ro_stors=modes, read_num=heralds + 2 * len(modes),
                readout_lane_pairs=pairs.tolist(), preselection_lanes=preselection,
                readout_bits=2)


class SequentialMultiparityAnalysisMixin:
    """Analyze paired readouts without using the legacy final-lane decoder."""

    def decode_readouts(self, point_idx=0, threshold=None):
        """Return accepted labels and IQ, plus the mask in the unfiltered shot axis."""
        if threshold is None:
            threshold = self.cfg.device.readout.threshold[self.cfg.expt.qubits[0]]
        read_num = int(self.cfg.read_num)
        pairs = np.asarray(self.data["readout_lane_pairs"], dtype=int)
        heralds = np.asarray(self.data["preselection_lanes"], dtype=int)
        idata = np.asarray(self.data["idata"][point_idx]).reshape(-1, read_num)
        qdata = np.asarray(self.data["qdata"][point_idx]).reshape(-1, read_num)
        keep = np.all(idata[:, heralds] < threshold, axis=1)
        i, q = idata[keep][:, pairs], qdata[keep][:, pairs]
        first_e = i[..., 0] > threshold
        second_e = i[..., 1] > threshold
        n_mod4 = first_e.astype(int) + 2 * second_e.astype(int)
        return dict(labels=n_mod4, keep=keep, i=i, q=q)

    def analyze_multiparity(self, data=None):
        """Store raw and optional corrected populations in self.data; return it."""
        if data is not None:
            self.data = data
        modes = self.data["ro_stors"]
        calibration = self.cfg.expt.get("multiparity_confusion_matrices")
        if calibration is None and self.cfg.expt.get("multiparity_confusion_matrix") is not None:
            raise ValueError("Use explicit per-step multiparity_confusion_matrices for sequential readout")
        if isinstance(calibration, (str, Path)):
            with h5py.File(calibration, "r") as handle:
                calibration = dict(matrices=handle["multiparity_confusion_matrices"][()].tolist(),
                                   ro_stors=handle["ro_stors"][()].tolist(), fname=str(calibration))
        if calibration is not None and not isinstance(calibration, dict):
            calibration = dict(matrices=np.asarray(calibration).tolist(), ro_stors=list(modes))
        matrices = None
        if calibration is not None:
            matrices = np.asarray(calibration["matrices"], dtype=float)
            if (matrices.shape != (4, 4, len(modes))
                    or list(calibration.get("ro_stors", modes)) != list(modes)
                    or not np.isfinite(matrices).all() or np.any(matrices < 0)
                    or not np.allclose(matrices.sum(axis=1), 1)):
                raise ValueError("Confusion matrices must be row probabilities with shape (4, 4, steps) and matching ro_stors")
            self.cfg.expt.multiparity_confusion_matrices = calibration

        probabilities = np.full((len(self.data["idata"]), len(modes), 4), np.nan)
        shot_counts = np.zeros(probabilities.shape[:2], dtype=int)
        for point in range(len(probabilities)):
            n_mod4 = self.decode_readouts(point)["labels"]
            shot_counts[point, :] = len(n_mod4)
            if len(n_mod4):
                for number in range(4):
                    probabilities[point, :, number] = np.mean(n_mod4 == number, axis=0)
        for key in ("p_mod_corrected", "n_mean_corrected", "n2_mean_corrected",
                    "multiparity_confusion_matrices", "calibration_source"):
            self.data.pop(key, None)
        numbers = np.arange(4)
        self.data.update(p_mod=probabilities, shot_counts=shot_counts,
                         n_mean=probabilities @ numbers, n2_mean=probabilities @ numbers**2)
        n_err = np.zeros(probabilities.shape[:2])
        n2_err = np.zeros(probabilities.shape[:2])
        corrected = np.zeros(probabilities.shape)
        for step in range(len(modes)):
            raw_probability = probabilities[:, step]
            counts = shot_counts[:, step]
            confusion = None
            if matrices is not None:
                confusion = matrices[:, :, step]
                corrected[:, step] = np.linalg.solve(confusion.T, raw_probability.T).T
            n_err[:, step] = multiparity_moment_error(raw_probability, counts, numbers, confusion)
            n2_err[:, step] = multiparity_moment_error(raw_probability, counts, numbers**2, confusion)
        self.data.update(n_err=n_err, n2_err=n2_err)
        if matrices is not None:
            self.data.update(p_mod_corrected=corrected, n_mean_corrected=corrected @ numbers,
                             n2_mean_corrected=corrected @ numbers**2,
                             multiparity_confusion_matrices=matrices)
            if calibration.get("fname"):
                self.data["calibration_source"] = np.asarray(str(calibration["fname"]).encode("utf-8"))
        return self.data


class FockPopulationSequentialReadoutProgram(FockPopulationProgram):
    """Read each requested mode once; all measurements belong to multiparity."""

    def initialize(self):
        layout = readout_layout(self.cfg)
        self.cfg.expt.ro_stors = layout["ro_stors"]
        self.cfg.expt.ro_stor = layout["ro_stors"][0]
        super().initialize()

    @classmethod
    def readouts_per_shot(cls, cfg):
        return readout_layout(cfg)["read_num"]

    def multiparity(self, *, name, register_label, man_idx=1, fast=False, final=False):
        """Define both Ramsey stages and complete one mode's two-bit readout."""
        cfg = self.cfg
        qTest = cfg.expt.qubits[0]
        self.r_cond_phase, self.r_read_q, self.r_thresh_q = 8, 9, 11
        second_phase = cfg.expt.get("phase_second_pulse", 180)
        cond_sec_phase = cfg.expt.get("cond_sec_phase", -90)
        revival_time = cfg.device.manipulate.revival_time[man_idx - 1]
        first_phase = second_phase
        transition = ["qubit", "ge"]
        if fast:
            # Preserve the existing first Ramsey's calibrated Stark phase convention.
            stark_shift = cfg.device.manipulate.revival_stark_shift[man_idx - 1]
            first_phase = (second_phase + 360 * stark_shift * revival_time) % 360
            revival_time = cfg.device.manipulate.revival_time_fast[man_idx - 1]
            transition = ["multiphoton", "g0-e0"]

        # First bit: parity Ramsey, measurement, and conditional qubit reset.
        gates = [[*transition, "hpi", 0], ["wait", revival_time],
                 [*transition, "hpi", first_phase]]
        creator = self.get_prepulse_creator(gates)
        self.custom_pulse(cfg, creator.pulse.tolist(), prefix=name)
        self._measure_multiparity_bit(register_label + "_bit0")

        # Second bit: half the Ramsey wait, with phase conditioned on the first bit.
        wait_time = revival_time / 2
        phase = second_phase
        frequency, gain, waveform = self.f_ge, self.hpi_ge_gain, "hpi_qubit_ge"
        if fast:
            phase = (phase + 360 * stark_shift * wait_time) % 360
            frequency, gain = self.f_ge_hpi_fast, self.hpi_ge_gain_fast
            waveform = "hpi_qubit_ge_fast"
        qubit_ch = self.qubit_chs[qTest]
        reg_page = self.ch_page(qubit_ch)
        reg_phase = self.sreg(qubit_ch, "phase")
        self.safe_regwi(reg_page, self.r_cond_phase, self.deg2reg(phase % 360, gen_ch=qubit_ch))
        cond_op = "<" if second_phase > 90 else ">"
        self.condj(0, self.r_read_q, cond_op, self.r_thresh_q, register_label + "_phase")
        self.safe_regwi(reg_page, self.r_cond_phase,
                        self.deg2reg((phase + cond_sec_phase) % 360, gen_ch=qubit_ch))
        self.label(register_label + "_phase")
        self.set_pulse_registers(ch=qubit_ch, freq=frequency, style="arb",
                                 phase=self.deg2reg(0), gain=gain, waveform=waveform)
        self.pulse(ch=qubit_ch)
        self.sync_all()
        self.sync_all(self.us2cycles(wait_time))
        self.mathi(reg_page, reg_phase, self.r_cond_phase, "+", 0)
        self.pulse(ch=qubit_ch)
        self.sync_all()
        self._measure_multiparity_bit(register_label + "_bit1", final=final)

    def _measure_multiparity_bit(self, register_label, final=False):
        """Measure once, then reset the qubit or apply the terminal relaxation."""
        cfg = self.cfg
        qTest = cfg.expt.qubits[0]
        if final:
            self.sync_all(10)
            self.measure(pulse_ch=self.res_chs[qTest], adcs=[self.adc_chs[qTest]],
                         adc_trig_offset=cfg.device.readout.trig_offset[qTest], wait=True,
                         syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest]))
        else:
            self.safe_regwi(0, self.r_read_q, 0)
            threshold = cfg.device.readout.threshold[qTest] * self.readout_lengths_adc[qTest]
            self.safe_regwi(0, self.r_thresh_q, int(threshold))
            self.measure(pulse_ch=self.res_chs[qTest], adcs=[self.adc_chs[qTest]],
                         adc_trig_offset=cfg.device.readout.trig_offset[qTest], t="auto", wait=True)
            self.wait_all(self.us2cycles(0.10))
            self.read(0, 0, "lower", self.r_read_q)
            self.condj(0, self.r_read_q, "<", self.r_thresh_q, register_label)
            self.set_pulse_registers(ch=self.qubit_chs[qTest], freq=self.f_ge_reg[qTest],
                                     style="arb", phase=self.deg2reg(0), gain=self.pi_ge_gain,
                                     waveform="pi_qubit_ge")
            self.pulse(ch=self.qubit_chs[qTest])
            self.label(register_label)
            self.sync_all(self.us2cycles(2.0))

    def multi_parity_readout(self, name="multiparity_readout", register_label="mpreadout",
                             man_idx=1, final_sync=False, fast=False):
        """The inherited body has already routed the first mode into M1."""
        modes = self.cfg.expt.ro_stors
        for step, mode in enumerate(modes):
            if step:
                # man_stor_swap's fixed waveform name is already used during preparation.
                self._play_gate_list([["storage", f"M1-S{mode}", "pi", 0]],
                                     prefix=f"sequential_readout_swap_{step}_")
                self.sync_all(self.us2cycles(0.2))
            self.multiparity(name=f"{name}_{step}", register_label=f"{register_label}_{step}",
                             man_idx=man_idx, fast=fast, final=step == len(modes) - 1)

    def measure_wrapper(self):
        # All measurements and terminal relaxation were completed above.
        pass


class FockPopulationSequentialReadoutExperiment(SequentialMultiparityAnalysisMixin,
                                              FockPopulationExperiment):
    """One preparation/evolution per shot, with an explicit readout-step axis."""

    default_program = FockPopulationSequentialReadoutProgram

    def shape_data(self, axes, points):
        self.data = super().shape_data(axes, points)
        self.data.update({key: np.asarray(value, dtype=int)
                          for key, value in readout_layout(self.cfg).items()})
        shape = (len(self.data["idata"]), len(self.data["ro_stors"]), 2)
        avgi, avgq = np.full(shape, np.nan), np.full(shape, np.nan)
        for point in range(shape[0]):
            decoded = self.decode_readouts(point)
            if len(decoded["labels"]):
                avgi[point] = decoded["i"].mean(axis=0)
                avgq[point] = decoded["q"].mean(axis=0)
        self.data.update(avgi=avgi, avgq=avgq, amps=np.abs(avgi + 1j * avgq),
                         phases=np.angle(avgi + 1j * avgq))
        return self.data

    def analyze(self, data=None, **kwargs):
        self.analyze_multiparity(data)
        params = self.recorded_derived_params() or {}
        if "floquet_cycle_us" in params:
            self.data["time_us"] = np.asarray(self.data["xpts"]) * params["floquet_cycle_us"]
        return self.data

    def display(self, data=None, raw=False, **kwargs):
        if data is not None:
            self.data = data
        if "n_mean" not in self.data:
            self.analyze()
        corrected = "n_mean_corrected" in self.data
        key = "n_mean_corrected" if corrected and not raw else "n_mean"
        fig, ax = plt.subplots(figsize=(9, 4))
        cycles = np.asarray(self.data["xpts"])
        for step, mode in enumerate(self.data["ro_stors"]):
            mean = self.data[key][:, step]
            line, = ax.plot(cycles, mean, ".-", label=mode_name(mode))
            if not raw or not corrected:
                error = self.data["n_err"][:, step]
                ax.fill_between(cycles, mean - error, mean + error, color=line.get_color(), alpha=.2)
        kind = "corrected" if corrected and not raw else "raw"
        ax.set(xlabel="Floquet cycles", ylabel=f"Mean n mod 4 ({kind})",
               title=f"Sequential readout: {list(self.cfg.expt.occupations)}")
        ax.legend()
        fig.tight_layout()
        return fig, ax



