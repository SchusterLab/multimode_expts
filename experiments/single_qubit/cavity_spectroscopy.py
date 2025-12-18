import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import fitting.fitting as fitter
from experiments.MM_base import MMAveragerProgram

"""
Measures the cavity frequency when the qubit is in its ground state: sweep readout pulse frequency and look for the frequency with the maximum measured amplitude.

The cavity frequency is stored in the parameter cfg.device.cavity.frequency.

Note that harmonics of the clock frequency (6144 MHz) will show up as "infinitely"  narrow peaks!
"""
class CavitySpectroscopyProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        # print('Using the readout length of ', 5*cfg.device.readout.readout_length, 'us')

        if self.cfg.expt.cavity_name == 'manipulate':
            
            self.adc_ch = cfg.hw.soc.adcs.cavity_out.ch
            self.res_ch = cfg.hw.soc.dacs.manipulate_in.ch
            self.res_ch_type = cfg.hw.soc.dacs.manipulate_in.type
            self.res_gain = cfg.expt.drive_gain
            self.readout_length_dac = self.us2cycles(5*cfg.device.readout.readout_length[0], gen_ch=self.res_ch)
            self.readout_length_adc = self.us2cycles(5*cfg.device.readout.readout_length[0], ro_ch=self.adc_ch)
            self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer
            self.adc_trig_offset=cfg.device.readout.trig_offset
        else:
            self.adc_ch = cfg.hw.soc.adcs.cavity_out.ch
            self.res_ch = cfg.hw.soc.dacs.storage_in.ch
            self.res_ch_type = cfg.hw.soc.dacs.storage_in.type
            self.res_gain = cfg.expt.drive_gain
            self.readout_length_dac = self.us2cycles(5*cfg.device.readout.readout_length[0], gen_ch=self.res_ch)
            self.readout_length_adc = self.us2cycles(5*cfg.device.readout.readout_length[0], ro_ch=self.adc_ch)
            self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer
            self.adc_trig_offset=cfg.device.readout.trig_offset

        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.frequency = cfg.expt.frequency
        self.freqreg = self.freq2reg(self.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        if self.cfg.expt.pulse_f: 
            self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        

        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = self.adc_ch
        if self.cfg.expt.cavity_name == 'manipulate':
            self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.manipulate_in.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        else:
            self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.storage_in.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        mixer_freq = 0
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=self.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        if self.cfg.expt.pulse_f:
            self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
            self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain
        
        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        if self.cfg.expt.pulse_f:
            self.add_gauss(ch=self.qubit_ch, name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        self.frequency1 = cfg.expt.RF_modulation[1]

        if self.cfg.expt.RF_modulation[3] == 'low':
            self.rf_ch = cfg.hw.soc.dacs.flux_low.ch
            self.declare_gen(ch=self.rf_ch, nqz=cfg.hw.soc.dacs.flux_low.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=self.rf_ch)
            self.freqreg1 = self.freq2reg(self.frequency1, gen_ch=self.rf_ch)
        elif self.cfg.expt.RF_modulation[3] == 'flux_storage':
            self.rf_ch = cfg.hw.soc.dacs.flux_storage.ch
            self.declare_gen(ch=self.rf_ch, nqz=cfg.hw.soc.dacs.flux_storage.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=self.rf_ch)
            self.freqreg1 = self.freq2reg(self.frequency1, gen_ch=self.rf_ch)
        else:
            self.rf_ch = cfg.hw.soc.dacs.flux_high.ch
            self.declare_gen(ch=self.rf_ch, nqz=cfg.hw.soc.dacs.flux_high.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=self.rf_ch)
            self.freqreg1 = self.freq2reg(self.frequency1, gen_ch=self.rf_ch)


        self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.freqreg, phase=0, gain=self.res_gain, length=self.readout_length_dac)
        self.synci(200) # give processor some time to configure pulses

    def body(self):
        # pass
        cfg=AttrDict(self.cfg)
        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
            self.sync_all() # align channels
        if self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=self.pi_ef_gain, waveform="pi_ef_qubit")
            self.sync_all() # align channels
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        

        if self.cfg.expt.RF_modulation[0]:
            self.setup_and_pulse(ch=self.rf_ch, style="const", freq=self.freqreg1, phase=0, gain=self.cfg.expt.RF_modulation[2], length=self.readout_length_dac)
        self.measure(
            pulse_ch=self.res_ch,
            adcs=[self.adc_ch],
            adc_trig_offset=self.adc_trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[0]))

# ====================================================== #

class CavitySpectroscopyExperiment(Experiment):
    """
    Cavity Spectroscopy Experiment
    Experimental Config
    expt = dict(
        start: start frequency (MHz), 
        step: frequency step (MHz), 
        expts: number of experiments, 
        pulse_e: boolean to add e pulse prior to measurement
        pulse_f: boolean to add f pulse prior to measurement
        reps: number of reps
        )
    """

    def __init__(self, soccfg=None, path='', prefix='CavitySpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        q_ind = self.cfg.expt.qubit
        if self.cfg.expt.cavity_name == 'manipulate':
            for subcfg in (self.cfg.device.manipulate, self.cfg.device.qubit, self.cfg.hw.soc):
                for key, value in subcfg.items() :
                    if isinstance(value, list):
                        subcfg.update({key: value[q_ind]})
                    elif isinstance(value, dict):
                        for key2, value2 in value.items():
                            for key3, value3 in value2.items():
                                if isinstance(value3, list):
                                    value2.update({key3: value3[q_ind]})      
        else:
            for subcfg in (self.cfg.device.storage, self.cfg.device.qubit, self.cfg.hw.soc):
                for key, value in subcfg.items() :
                    if isinstance(value, list):
                        subcfg.update({key: value[q_ind]})
                    elif isinstance(value, dict):
                        for key2, value2 in value.items():
                            for key3, value3 in value2.items():
                                if isinstance(value3, list):
                                    value2.update({key3: value3[q_ind]})   
                                  

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for f in tqdm(xpts, disable=not progress):
            self.cfg.expt.frequency = f
            rspec = CavitySpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            # print(rspec)
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase

            data["xpts"].append(f)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
        
        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data

        return data

    def analyze(self, data=None, fit=False, findpeaks=False, verbose=True, fitparams=None, **kwargs):
        if data is None:
            data=self.data
            
        if fit:
            # fitparams = [f0, Qi, Qe, phi, scale]
            xdata = data["xpts"][1:-1]
            # ydata = data["avgi"][1:-1] + 1j*data["avgq"][1:-1]
            ydata = data['amps'][1:-1]
            fitparams=fitparams
            data['fit'], data['fit_err'] = fitter.fithanger(xdata, ydata, fitparams=fitparams)
            if isinstance(data['fit'], (list, np.ndarray)):
                f0, Qi, Qe, phi, scale, a0, slope = data['fit']
                if verbose:
                    print(f'\nFreq with minimum transmission: {xdata[np.argmin(ydata)]}')
                    print(f'Freq with maximum transmission: {xdata[np.argmax(ydata)]}')
                    print('From fit:')
                    print(f'\tf0: {f0}')
                    print(f'\tQi: {Qi}')
                    print(f'\tQe: {Qe}')
                    print(f'\tQ0: {1/(1/Qi+1/Qe)}')
                    print(f'\tkappa [MHz]: {f0*(1/Qi+1/Qe)}')
                    print(f'\tphi [radians]: {phi}')
            
        if findpeaks:
            maxpeaks, minpeaks = dsfit.peakdetect(data['amps'][1:-1], x_axis=data['xpts'][1:-1], lookahead=30, delta=5*np.std(data['amps'][:5]))
            data['maxpeaks'] = maxpeaks
            data['minpeaks'] = minpeaks
            
        return data

    def display(self, data=None, fit=True, findpeaks=False, **kwargs):
        if data is None:
            data=self.data 

        xpts = data['xpts'][1:-1]

        plt.figure(figsize=(16,16))
        plt.subplot(311, title=f"Cavity Spectroscopy at gain {self.cfg.expt.drive_gain}",  ylabel="Amps [ADC units]")
        plt.plot(xpts, data['amps'][1:-1],'o-')
        if fit:
            plt.plot(xpts, fitter.hangerS21func_sloped(data["xpts"][1:-1], *data["fit"]))
        if findpeaks:
            # for peak in np.concatenate((data['maxpeaks'], data['minpeaks'])):
            for peak in data['minpeaks']:
                plt.axvline(peak[0], linestyle='--', color='0.2')
                print(f'Found peak [MHz]: {peak[0]}')
        # plt.axvline(float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + 812.37), c='k', ls='--')
        # plt.axvline(7687.5, c='k', ls='--')

        plt.subplot(312, xlabel="Cavity Frequency [MHz]", ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1],'o-')

        plt.subplot(313, xlabel="Cavity Frequency [MHz]", ylabel="Phases [ADC units]")
        plt.plot(xpts, data["phases"][1:-1],'o-')
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


# ====================================================== #