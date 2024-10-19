import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import experiments.fitting as fitter

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
"""


class DisplacementEnhancedSideband(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits

        qTest = self.qubits[0]

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch[cfg.expt.cavity_name]
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type

        # get register page for qubit_chs
        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]
        self.f_ge_reg = self.freq2reg(
            cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])

        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(
            cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(
            self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(
            self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []

        # declare res dacs
        mask = None
        mixer_freq = 0  # MHz
        mux_freqs = None  # MHz
        mux_gains = None
        ro_ch = None
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest],
                         mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest],
                             freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(
                    ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])



        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pi_sigma = self.us2cycles(
            cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ge value
        self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest]/2, gen_ch=self.qubit_chs[qTest])

        # add qubit pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi2", sigma=self.pi2sigma, length=self.pi2sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test",
                       sigma=self.us2cycles(self.cfg.expt.ramp_sigma), length=self.us2cycles(self.cfg.expt.ramp_sigma)*4)

        # cavity pulses 
        ##print(cfg.device.manipulate.f_ge[cfg.expt.cavity_name])
        ##print(self.man_ch)
        self.man_freq = self.freq2reg(cfg.device.manipulate.f_ge[cfg.expt.cavity_name], gen_ch=self.man_ch)
        self.man_sigma = self.us2cycles(cfg.expt.cavity_disp_pulse[1], gen_ch=self.man_ch) 
        self.man_gain = int(self.cfg.expt.cavity_disp_pulse[2] /self.cfg.expt.cavity_disp_pulse[3])
        self.add_gauss(ch=self.man_ch, name="gauss_cav", sigma=self.man_sigma, length=self.man_sigma*4)
        
        # readout resonator pulses 
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(
            cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        self.sync_all(self.us2cycles(0.2))

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        #qubit prepulse 
        if cfg.qubit_ge_prep:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[0], waveform="pi_qubit")
            self.sync_all()

        # first pi/2 pulse 
        ##self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[0], waveform="pi_qubit")
        if cfg.expt.hadamard[0]:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg, phase=0, gain = cfg.device.qubit.pulses.pi_ge.gain[0], waveform="pi2")
            self.sync_all()

        # cavity displacement 
        if cfg.expt.cavity_disp_pulse[0]:
            self.setup_and_pulse(ch=self.man_ch, style="arb", 
                                        freq=self.freq2reg(cfg.device.manipulate.f_ge[cfg.expt.cavity_name], gen_ch=self.man_ch),
                                        phase=0, gain=self.man_gain,
                                        waveform="gauss_cav")
            self.sync_all()

        # wait 
        self.setup_and_pulse(
                    ch=self.qubit_chs[qTest],
                    style="flat_top",
                    freq=self.freq2reg(cfg.device.qubit.f_ge[qTest] + cfg.expt.wait[1], gen_ch=self.qubit_chs[qTest]),
                    length=self.us2cycles(self.cfg.expt.length_placeholder),
                    phase=self.deg2reg(90, gen_ch=self.qubit_chs[qTest]),
                    gain=cfg.expt.wait[0], 
                    waveform="pi_test")
        self.sync_all()
        
        ##self.sync_all(self.us2cycles(cfg.expt.length_placeholder))
        


        # cavity displacement back 
        if cfg.expt.cavity_disp_pulse[0]:
            self.setup_and_pulse(ch=self.man_ch, style="arb", 
                                    freq=self.freq2reg(cfg.device.manipulate.f_ge[cfg.expt.cavity_name], gen_ch=self.man_ch),
                                    phase=self.deg2reg(180 + cfg.expt.cavity_disp_pulse[4], gen_ch=self.man_ch), gain=self.man_gain,
                                    waveform="gauss_cav")
            self.sync_all()

        # second -pi/2 pulse
        if cfg.expt.hadamard[0]:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg, 
                                phase=self.deg2reg(cfg.expt.hadamard[1], gen_ch=self.qubit_chs[qTest]) ,gain = cfg.device.qubit.pulses.pi_ge.gain[0], waveform="pi2")
            self.sync_all()

        # align channels and wait 50ns and measure
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )


class DisplacementEnhancedSidebandExperiment(Experiment):
    """
    Length Rabi Experiment
    Experimental Config
    expt = dict(
        start: start length [us],
        step: length step, 
        expts: number of different length experiments, 
        reps: number of reps,
        gain: gain to use for the qubit pulse
        pulse_type: 'gauss' or 'const'
        checkZZ: True/False for putting another qubit in e (specify as qA)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qA in e , qB sweeps length rabi]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiGeneral', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update(
                                    {key3: [value3]*num_qubits_sample})
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        lengths = self.cfg.expt["start"] + \
            self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}

        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            lengthrabi = DisplacementEnhancedSideband(
                soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(
                self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq)  # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq)  # Calculating the phase
            data["xpts"].append(length)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data = self.data
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = [None, 1/max(data['xpts']), None, None]
            # fitparams = None
            p_avgi, pCov_avgi = fitter.fitdecaysin(
                data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(
                data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(
                data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        xpts_ns = data['xpts']*1e3

        # plt.figure(figsize=(12, 8))
        # plt.subplot(111, title=f"Length Rabi", xlabel="Length [ns]", ylabel="Amplitude [ADC units]")
        # plt.plot(xpts_ns[1:-1], data["amps"][1:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     plt.plot(xpts_ns[1:-1], fitter.sinfunc(data["xpts"][1:-1], *p))

        plt.figure(figsize=(10, 8))
        if 'gain' in self.cfg.expt:
            gain = self.cfg.expt.gain
        else:
            # gain of the pulse we are trying to calibrate
            gain = self.cfg.device.qubit.pulses.pi_ge.gain[self.cfg.expt.qubits[-1]]
        plt.subplot(
            211, title=f"Length Rabi (Qubit Gain {gain})", ylabel="I [adc level]")
        plt.plot(xpts_ns[1:-1], data["avgi"][1:-1], 'o-')
        if fit:
            p = data['fit_avgi']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_length = (1/2 - p[2]/180)/2/p[1]
            else:
                pi_length = (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print('Decay from avgi [us]', p[3])
            print(f'Pi length from avgi data [us]: {pi_length}')
            print(f'\tPi/2 length from avgi data [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')

        print()
        plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[1:-1], data["avgq"][1:-1], 'o-')
        if fit:
            p = data['fit_avgq']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_length = (1/2 - p[2]/180)/2/p[1]
            else:
                pi_length = (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print('Decay from avgq [us]', p[3])
            print(f'Pi length from avgq data [us]: {pi_length}')
            print(f'Pi/2 length from avgq data [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
