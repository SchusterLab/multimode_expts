"""
Pulse-probe qubit spectroscopy, NDAverager (hardware freq sweep) variant.

Cross-validation harness for MMNDAveragerProgram. The pulse and readout match
PulseProbeSpectroscopyProgram (flat_top probe on qubit_ch, gauss ramp of sigma
cfg.expt.sigma, standard MM_base readout), so running this with the same
cfg.expt as the MMRAveragerProgram version should produce the same spectrum
up to averaging noise. Mismatch = ND wiring bug.

Differences vs the R-averager version:
  - Freq sweep is registered via add_sweep(QickSweep(...)) on the gen's freq
    register and advances in hardware between body iterations.
  - No update() method, no r_freq2 dummy register, no mathi in body.
  - acquire() returns (expt_pts, avg_di, avg_dq); we use expt_pts[0] as xpts.

Intentionally minimal: no prepulse, no active reset, no pre-selection. Add
those after the bare comparison works.
"""

import matplotlib.pyplot as plt
import numpy as np
from qick import QickConfig
from qick.averager_program import QickSweep
from slab import AttrDict, Experiment

import fitting.fitting as fitter
from experiments.MM_base import MMNDAveragerProgram


class PulseProbeSpectroscopyNDProgram(MMNDAveragerProgram):
    def __init__(self, soccfg: QickConfig, cfg: AttrDict):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.get('rounds', 1)

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        qTest = cfg.expt.get('qubits', [0])[0]
        qubit_ch = self.qubit_chs[qTest]

        ramp_cycles = self.us2cycles(cfg.expt.sigma, gen_ch=qubit_ch)
        self.add_gauss(ch=qubit_ch, name="ramp_nd",
                       sigma=ramp_cycles, length=ramp_cycles * 4)

        freq_start_MHz = float(cfg.expt.start)
        freq_step_MHz = float(cfg.expt.step)
        freq_npts = int(cfg.expt.expts)
        freq_stop_MHz = freq_start_MHz + freq_step_MHz * (freq_npts - 1)

        self.set_pulse_registers(
            ch=qubit_ch, style="flat_top",
            freq=self.freq2reg(freq_start_MHz, gen_ch=qubit_ch),
            phase=0, gain=int(cfg.expt.gain),
            length=self.us2cycles(cfg.expt.length, gen_ch=qubit_ch),
            waveform="ramp_nd",
        )

        self.swept_freq_reg = self.get_gen_reg(qubit_ch, "freq")
        self.add_sweep(QickSweep(
            self, self.swept_freq_reg,
            freq_start_MHz, freq_stop_MHz, freq_npts,
        ))

        self.qubit_ch_used = qubit_ch
        self.sync_all(self.us2cycles(0.2))

    def body(self):
        self.pulse(ch=self.qubit_ch_used)
        self.measure_wrapper()


class PulseProbeSpectroscopyNDExperiment(Experiment):
    """
    NDAverager version of PulseProbeSpectroscopyExperiment. Same cfg.expt
    schema so the two can be run with identical configs for comparison.

    Experimental Config:
        start: Qubit frequency [MHz]
        step:  step [MHz]
        expts: number of freq points
        reps:  averages per point
        rounds: soft averages of the whole sweep
        length: probe flat-segment length [us]
        gain:   probe gain [DAC]
        sigma:  probe ramp gauss sigma [us]
        qubits: list, e.g. [0]
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeSpectroscopyND',
                 config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not isinstance(value3, list):
                                value2.update({key3: [value3] * num_qubits_sample})
                elif not isinstance(value, list):
                    subcfg.update({key: [value] * num_qubits_sample})

        prog = PulseProbeSpectroscopyNDProgram(soccfg=self.soccfg, cfg=self.cfg)
        self.prog = prog
        expt_pts, avg_di, avg_dq = prog.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None, load_pulses=True, progress=progress,
        )
        xpts = np.asarray(expt_pts[0])
        avgi = np.asarray(avg_di[0]).squeeze()
        avgq = np.asarray(avg_dq[0]).squeeze()
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {'xpts': xpts, 'avgi': avgi, 'avgq': avgq,
                'amps': amps, 'phases': phases}
        self.data = data
        return data

    def analyze(self, data=None, fit=True, signs=(1, 1, 1), **kwargs):
        if data is None:
            data = self.data
        if fit:
            xdata = data['xpts'][1:-1]
            data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(
                xdata, signs[0] * data['amps'][1:-1])
            data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(
                xdata, signs[1] * data['avgi'][1:-1])
            data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(
                xdata, signs[2] * data['avgq'][1:-1])
        return data

    def display(self, data=None, fit=True, signs=(1, 1, 1), **kwargs):
        if data is None:
            data = self.data

        if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq + data['xpts'][1:-1]
        else:
            xpts = data['xpts'][1:-1]

        plt.figure(figsize=(9, 11))
        plt.subplot(311,
                    title=f"Qubit ND Spectroscopy (gain {self.cfg.expt.gain})",
                    ylabel="Amplitude [ADC units]")
        plt.plot(xpts, data["amps"][1:-1], 'o-')
        if fit and 'fit_amps' in data:
            plt.plot(xpts, signs[0] * fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
            print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')

        plt.subplot(312, ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1], 'o-')
        if fit and 'fit_avgi' in data:
            plt.plot(xpts, signs[1] * fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
            print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')

        plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
        plt.plot(xpts, data["avgq"][1:-1], 'o-')
        if fit and 'fit_avgq' in data:
            plt.plot(xpts, signs[2] * fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
            print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
