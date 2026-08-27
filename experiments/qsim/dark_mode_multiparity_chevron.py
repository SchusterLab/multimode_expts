# -*- coding: utf-8 -*-
"""Man-storage multiparity chevron: hardware frequency sweep per swap length.

Split out of ``floquet_dark_mode_readout.py`` unchanged. ``DarkBaseRProgram``
comes along because this is its only subclass, so the whole measurement now
reads in one place; it is not a general base layer.
"""
import numpy as np
from slab import AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import MMRAveragerProgram
from experiments.qsim.utils import ensure_list_in_cfg
from experiments.qsim.qsim_base import QsimBaseProgram
from experiments.qsim.floquet_dark_mode_readout import (
    DarkBaseExperiment,
    DarkBaseProgram,
)

class DarkBaseRProgram(MMRAveragerProgram):
    """RAverager counterpart of ``DarkBaseProgram``.

    The pulse-building methods below do not depend on the AveragerProgram
    software loop, so both program types use the same implementations.
    Concrete RAverager programs only need to define ``core_pulses`` and
    ``update``.
    """

    _pre_selection_filtering = True

    retrieve_swap_parameters = QsimBaseProgram.retrieve_swap_parameters #borrowing methods
    prep_man_fock_state = DarkBaseProgram.prep_man_fock_state #borrowing methods
    multi_parity_readout = DarkBaseProgram.multi_parity_readout #borrowing methods
    body = DarkBaseProgram.body #borrowing methods

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()

        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()

        self.storage_phase_matrix = self.cfg.expt.get(
            "storage_phase_matrix", None)
        if self.storage_phase_matrix is not None:
            self.storage_phase_matrix = np.asarray(
                self.storage_phase_matrix, dtype=float)

        man_mode_no = self.cfg.expt.get("man_mode_no", 1)
        self.man_mode_idx = man_mode_no - 1

        for stor in range(7):
            if self.m1s_style[stor] != "arb":
                continue

            stor_name = f"M1-S{stor + 1}"
            ch = self.m1s_ch[stor]
            sigma_us = self.cfg.expt.get("floquet_gauss_sigma", None)
            if sigma_us is None:
                sigma_us = self.swap_ds.get_gauss_sigma(stor_name)
            n_sigma = self.swap_ds.get_gauss_n_sigma(stor_name)
            sigma = self.us2cycles(sigma_us, gen_ch=ch)

            self.add_gauss(
                ch=ch,
                name=self.m1s_wf_name[stor],
                sigma=sigma,
                length=sigma * n_sigma,
            )

        self.m1s_kwargs = []
        for stor in range(7):
            pulse_kwargs = {
                "ch": self.m1s_ch[stor],
                "style": self.m1s_style[stor],
                "freq": self.m1s_freq[stor],
                "phase": 0,
                "gain": self.m1s_gain[stor],
                "waveform": self.m1s_wf_name[stor],
            }
            if self.m1s_style[stor] != "arb":
                pulse_kwargs["length"] = self.m1s_length[stor]
            self.m1s_kwargs.append(pulse_kwargs)

        self.sync_all(200)


class ManStorMultiparityChevronRProgram(DarkBaseRProgram):
    """Hardware frequency sweep for one fixed M1-storage pulse length."""

    def initialize(self):
        super().initialize()

        cfg = AttrDict(self.cfg)
        swap_stor = int(cfg.expt.swap_stor)
        pulse_creator = self.get_prepulse_creator([
            ["storage", f"M1-S{swap_stor}", "pi", 0.0],
        ])
        pulse = pulse_creator.pulse

        self.chevron_ch = int(pulse[4][0])
        self.chevron_page = self.ch_page(self.chevron_ch)
        self.chevron_freq_register = self.sreg(
            self.chevron_ch, "freq")
        self.chevron_freq_sweep_register = 4
        self.chevron_start_freq = self.freq2reg(
            cfg.expt.start, gen_ch=self.chevron_ch)
        self.chevron_step_freq = self.freq2reg(
            cfg.expt.step, gen_ch=self.chevron_ch)
        self.safe_regwi(
            self.chevron_page,
            self.chevron_freq_sweep_register,
            self.chevron_start_freq,
        )

        self.chevron_gain = int(cfg.expt.get(
            "custom_scramble_gain", pulse[1][0]))
        self.chevron_phase = self.deg2reg(
            cfg.expt.get("custom_scramble_phase", 0.0),
            gen_ch=self.chevron_ch,
        )
        self.chevron_length = self.us2cycles(
            cfg.expt.custom_scramble_length,
            gen_ch=self.chevron_ch,
        )

        if self.chevron_ch == self.flux_low_ch[0]:
            self.chevron_waveform = "pi_m1si_low"
        elif self.chevron_ch == self.flux_high_ch[0]:
            self.chevron_waveform = "pi_m1si_high"
        else:
            raise ValueError(
                f"unexpected M1-storage channel: {self.chevron_ch}")

        self.sync_all(200)

    def core_pulses(self):
        # Prep/reset pulses may use the same flux channel, so restore every
        # fixed pulse register immediately before the swept swap.
        self.set_pulse_registers(
            ch=self.chevron_ch,
            style="flat_top",
            freq=0,
            phase=self.chevron_phase,
            gain=self.chevron_gain,
            length=self.chevron_length,
            waveform=self.chevron_waveform,
        )
        self.mathi(
            self.chevron_page,
            self.chevron_freq_register,
            self.chevron_freq_sweep_register,
            "+",
            0,
        )
        self.pulse(ch=self.chevron_ch)
        self.sync_all(self.us2cycles(0.2))

    def update(self):
        self.mathi(
            self.chevron_page,
            self.chevron_freq_sweep_register,
            self.chevron_freq_sweep_register,
            "+",
            self.chevron_step_freq,
        )


class ManStorMultiparityChevronRExperiment(DarkBaseExperiment):
    """Software length sweep with a hardware frequency sweep on every line."""

    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        lengths = np.asarray(
            self.cfg.expt.custom_scramble_lengths, dtype=float)
        frequency_count = int(self.cfg.expt.expts)
        reps = int(self.cfg.expt.reps)
        rounds = int(self.cfg.expt.get("rounds", 1))
        if rounds != 1:
            raise ValueError(
                "ManStorMultiparityChevronRExperiment requires rounds=1 "
                "so every multiparity shot remains associated with its "
                "length point"
            )

        if not self.cfg.expt.get("perform_wigner", False):
            self.cfg.expt.perform_wigner = False

        read_num = 1
        if self.cfg.expt.get("active_reset", False):
            params = MMRAveragerProgram.get_active_reset_params(self.cfg)
            read_num += MMRAveragerProgram.active_reset_read_num(**params)
        if self.cfg.expt.get("multiparity_readout", False):
            read_num += 1
        self.cfg.read_num = read_num

        avgi_lines = []
        avgq_lines = []
        idata_lines = []
        qdata_lines = []
        frequency_points = None

        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.custom_scramble_length = float(length)
            self.prog = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)

            xpts, avgi, avgq = self.prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
                debug=debug,
                readouts_per_experiment=read_num,
            )
            frequency_points = np.asarray(xpts, dtype=float)
            avgi_lines.append(np.asarray(avgi[0][-1], dtype=float))
            avgq_lines.append(np.asarray(avgq[0][-1], dtype=float))

            idata, qdata = self.prog.collect_shots()
            expected_values = rounds * frequency_count * reps * read_num
            if np.asarray(idata).size != expected_values:
                raise RuntimeError(
                    "unexpected RAverager I-buffer size: "
                    f"got {np.asarray(idata).size}, expected "
                    f"{expected_values}"
                )
            if np.asarray(qdata).size != expected_values:
                raise RuntimeError(
                    "unexpected RAverager Q-buffer size: "
                    f"got {np.asarray(qdata).size}, expected "
                    f"{expected_values}"
                )

            idata = np.asarray(idata).reshape(
                rounds, frequency_count, reps, read_num)
            qdata = np.asarray(qdata).reshape(
                rounds, frequency_count, reps, read_num)
            idata_lines.append(
                idata.transpose(1, 0, 2, 3).reshape(frequency_count, -1)
            )
            qdata_lines.append(
                qdata.transpose(1, 0, 2, 3).reshape(frequency_count, -1)
            )

        avgi = np.asarray(avgi_lines).T
        avgq = np.asarray(avgq_lines).T
        idata_lines = np.asarray(idata_lines)
        qdata_lines = np.asarray(qdata_lines)

        data = {
            "xpts": lengths,
            "ypts": frequency_points,
            "avgi": avgi,
            "avgq": avgq,
            "amps": np.abs(avgi + 1j * avgq),
            "phases": np.angle(avgi + 1j * avgq),
            "idata": [],
            "qdata": [],
        }

        # DarkBaseExperiment.analyze_multiparity expects frequency-major rows.
        for frequency_index in range(frequency_count):
            for length_index in range(len(lengths)):
                data["idata"].append(
                    idata_lines[length_index, frequency_index])
                data["qdata"].append(
                    qdata_lines[length_index, frequency_index])

        self.data = data
        return data
