import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from qick import *

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
import time

import experiments.fitting as fitter

class TestOptProgram2(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type

        self.q_rp=self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_freq=self.sreg(self.qubit_ch, "freq") # get frequency register for qubit_ch    
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.res_ch_type == 'mux4':
            assert self.res_ch == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
            ro_ch=self.adc_ch
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.f_= self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch) # get start/step frequencies

        # add qubit and readout pulses to respective channels
        # self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_, phase=0, gain=cfg.expt.gain, length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch))
        self.add_opt_pulse(ch=self.qubit_ch, name="test_opt", pulse_location=cfg.expt.opt_file_path)


        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(cfg.device.readout.phase),
                                        gain=cfg.device.readout.gain, length=self.readout_length_dac)
    


        self.synci(200) # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)
        if cfg.expt.cavity_drive:
            self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.freq2reg(cfg.device.manipulate.f_ge[cfg.expt.cavity_name], gen_ch=self.man_ch), phase=0, gain=cfg.expt.cavity_gain, length=self.us2cycles(cfg.expt.cavity_length, gen_ch=self.man_ch))
            # self.setup_and_pulse(ch=self.man_ch, style="const", freq=cfg.device.manipulate.f_ge[cfg.expt.cavity_name], phase=0, gain=cfg.expt.cavity_gain, length=self.us2cycles(cfg.expt.cavity_length))
        self.sync_all(cfg.device.manipulate.cycles_add_to_Q)    

        # test optimal controlled pulse
        self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_, phase=0, gain=cfg.expt.opt_gain, waveform="test_opt")
        if cfg.expt.wait_qubit:
            self.sync_all(cfg.device.qubit.cycles_add_to_R) # align channels and wait designated time
        else:
            self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def collect_shots(self):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg = self.cfg
        # print(self.di_buf[0])
        shots_i0 = self.di_buf[0].reshape((1, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]
        # print(shots_i0)
        shots_q0 = self.dq_buf[0].reshape((1, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]

        return shots_i0, shots_q0
    
    # def update(self):
    #     self.mathi(self.q_rp, self.r_freq, self.r_freq, '+', self.f_step) # update frequency list index
 
# ====================================================== #

class TestOptExperiment2(Experiment):
    """
    PulseProbe Spectroscopy Experiment
    Experimental Config:
        start: Qubit frequency [MHz]
        step
        expts: Number of experiments stepping from start
        reps: Number of averages per point
        rounds: Number of start to finish sweeps to average over
        length: Qubit probe constant pulse length [us]
        gain: Qubit pulse gain [DAC units]
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})                                

        data = {"xpts": [], "avgi": [], "avgq": [], "i0":[], "i1":[], "q0":[], "q1":[]}
        prog = TestOptProgram2(soccfg=self.soccfg, cfg=self.cfg)
        self.prog = prog
        avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=False)


        data["avgi"].append(avgi)
        data["avgq"].append(avgq)

        i0, q0 = prog.collect_shots()
        data["i0"].append(i0)
        data["q0"].append(q0)

        for k, a in data.items():
            data[k]=np.array(a)

        self.data = data

        return data


    # def analyze(self, data=None, fit=True, signs=[1,1,1], **kwargs):
    #     if data is None:
    #         data=self.data
    #     if fit:
    #         xdata = data['xpts'][1:-1]
    #         data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
    #         data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(xdata, signs[1]*data['avgi'][1:-1])
    #         data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(xdata, signs[2]*data['avgq'][1:-1])
    #     return data

    # def display(self, data=None, fit=True, signs=[1,1,1], **kwargs):
    #     if data is None:
    #         data=self.data 

    #     if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
    #         xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq + data['xpts'][1:-1]
    #     else: 
    #         xpts = data['
    #         '][1:-1]

    #     plt.figure(figsize=(9, 11))
    #     plt.subplot(311, title=f"Qubit {self.cfg.expt.qubit} Spectroscopy (Gain {self.cfg.expt.gain})", ylabel="Amplitude [ADC units]")
    #     plt.plot(xpts, data["amps"][1:-1],'o-')
    #     if fit:
    #         plt.plot(xpts, signs[0]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
    #         print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')

    #     plt.subplot(312, ylabel="I [ADC units]")
    #     plt.plot(xpts, data["avgi"][1:-1],'o-')
    #     if fit:
    #         plt.plot(xpts, signs[1]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
    #         print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
    #     plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
    #     plt.plot(xpts, data["avgq"][1:-1],'o-')
    #     # plt.axvline(3476, c='k', ls='--')
    #     # plt.axvline(3376+50, c='k', ls='--')
    #     # plt.axvline(3376, c='k', ls='--')
    #     if fit:
    #         plt.plot(xpts, signs[2]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
    #         # plt.axvline(3593.2, c='k', ls='--')
    #         print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')

    #     plt.tight_layout()
    #     plt.show()

    # def save_data(self, data=None):
    #     print(f'Saving {self.fname}')
    #     super().save_data(data=data)

