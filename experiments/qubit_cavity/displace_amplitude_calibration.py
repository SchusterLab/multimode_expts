import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter

class DisplaceCalibrationProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
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

        self.man_chs = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_types = cfg.hw.soc.dacs.manipulate_in.type

        # self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        # self.man_rps = self.ch_page(self.man_chs)  # get register page for man_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []
        
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_types[qTest] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]

        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest], freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])
        
        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.hpisigma_ge = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default hpi_ge value
        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.gain_hge_init = self.cfg.device.qubit.pulses.hpi_ge.gain[qTest]
        # define pi2sigma as the pulse that we are calibrating with ramsey
        self.f_pi_test_reg = self.f_ge_reg[qTest] # freq we are trying to calibrate

        # define the displace sigma for calibration
        self.displace_sigma = cfg.expt.displace_sigma

        
        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit", sigma=self.hpisigma_ge, length=self.hpisigma_ge*4)

        if cfg.expt.cavity_name == 0:
            self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[0], gen_ch=self.man_chs[0])
        else:
            self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[1], gen_ch=self.man_chs[0])

        self.add_gauss(ch=self.man_chs[0], name="displace", sigma=self.displace_sigma, length=self.displace_sigma*4)


        # add readout pulses to respective channels
        # if self.res_ch_types[qTest] == 'mux4':
        #     self.set_pulse_registers(ch=self.res_chs[qTest], style="const", length=self.readout_lengths_dac[qTest], mask=mask)
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])
        self.tp = cfg.expt.tp   # in cycles, this is waiting time


        self.chi_shift = cfg.expt.guessed_chi
        self.ratio = np.cos(np.pi*2*self.chi_shift/4*(2*self.cycles2us(self.tp)+3*self.cycles2us(self.displace_sigma*4)))/np.cos(np.pi*2*self.chi_shift/4*self.cycles2us(self.displace_sigma*4))


        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        # initializations qubit state to g+e
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_hge_init, waveform="hpi_qubit")
        self.sync_all(self.us2cycles(0.05))

        # 4 pulses enclosing cavity IQ space

        for i in range(cfg.expt.repeat_time):

            self.set_pulse_registers(
                ch=self.man_chs[0],
                style="arb",
                freq=self.f_cavity,
                phase=self.deg2reg(0),
                gain=self.cfg.expt.amp_placeholder, # gain set by update
                waveform="displace")
            self.pulse(ch=self.man_chs[0])
            self.sync_all(self.tp)

            self.set_pulse_registers(
                ch=self.man_chs[0],
                style="arb",
                freq=self.f_cavity,
                phase=self.deg2reg(180),
                gain=int(self.cfg.expt.amp_placeholder*self.ratio), # gain set by update
                waveform="displace")
            self.pulse(ch=self.man_chs[0])
            self.sync_all()

            self.set_pulse_registers(
                ch=self.man_chs[0],
                style="arb",
                freq=self.f_cavity,
                phase=self.deg2reg(180),
                gain=int(self.cfg.expt.amp_placeholder*self.ratio), # gain set by update
                waveform="displace")
            self.pulse(ch=self.man_chs[0])
            self.sync_all(self.tp)

            self.set_pulse_registers(
                ch=self.man_chs[0],
                style="arb",
                freq=self.f_cavity,
                phase=self.deg2reg(0),
                gain=self.cfg.expt.amp_placeholder, # gain set by update
                waveform="displace")
            self.pulse(ch=self.man_chs[0])
            self.sync_all(self.us2cycles(0.05))


        # post qubit rotation
        if cfg.expt.check_type == 'X':
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_hge_init, waveform="hpi_qubit")
        else:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=self.deg2reg(-90), gain=self.gain_hge_init, waveform="hpi_qubit")

        # align channels and measure
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

# ====================================================== #
                      
class DisplaceCalibrationExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac level]
        step: gain step [dac level]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='DisplaceCalibration', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        qTest = self.cfg.expt.qubits[0]

        amp_list = self.cfg.expt["gain_start"] + self.cfg.expt["gain_step"] * np.arange(self.cfg.expt["gain_expts"])

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        for amps_value in tqdm(amp_list, disable=not progress):
            self.cfg.expt.amp_placeholder = int(amps_value)
            lengthrabi = DisplaceCalibrationProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
            data["xpts"].append(amps_value)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)


          
        
        for k, a in data.items():
            data[k]=np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data=self.data
        
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            xdata = data['xpts']

            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps
        return data

    def display(self, data=None, fit=True, fitparams=None, vline = None, **kwargs):
        if data is None:
            data=self.data 

        plt.figure(figsize=(10,10))
        plt.subplot(211, title=f"Displace amplitude calibration (Pulse Length {self.cfg.expt.displace_sigma})", ylabel="I [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'o-')
        if fit:
            p = data['fit_avgi']
            plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgi data [dac units]: {int(pi_gain)}')
            # print(f'\tPi/2 gain from avgi data [dac units]: {int(pi2_gain)}')
            print(f'\tPi/2 gain from avgi data [dac units]: {int(1/4/p[1])}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
            if not(vline==None):
                plt.axvline(vline, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1],'o-')
        if fit:
            p = data['fit_avgq']
            plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgq data [dac units]: {int(pi_gain)}')
            # print(f'\tPi/2 gain from avgq data [dac units]: {int(pi2_gain)}')
            print(f'\tPi/2 gain from avgq data [dac units]: {int(1/4/p[1])}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #
                      