import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter

class T1RingdownProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)

        if self.cfg.expt.cavity_name == 'manipulate':
            
            self.adc_ch = cfg.hw.soc.adcs.cavity_out.ch
            self.qubit_nyquist = cfg.hw.soc.dacs.manipulate_in.nyquist
            self.qubit_ch = cfg.hw.soc.dacs.manipulate_in.ch
            self.qubit_ch_type = cfg.hw.soc.dacs.manipulate_in.type

            self.res_ch = cfg.hw.soc.dacs.manipulate_in.ch
            self.res_ch_type = cfg.hw.soc.dacs.manipulate_in.type
            self.res_gain = cfg.device.manipulate.gain
            self.readout_length_dac = self.us2cycles(cfg.device.manipulate.readout_length, gen_ch=self.res_ch)
            self.readout_length_adc = self.us2cycles(cfg.device.manipulate.readout_length, ro_ch=self.adc_ch)
            self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer
            self.adc_trig_offset=cfg.device.manipulate.trig_offset
        else:
            self.adc_ch = cfg.hw.soc.adcs.cavity_out.ch
            self.qubit_nyquist = cfg.hw.soc.dacs.storage_in.nyquist
            self.qubit_ch = cfg.hw.soc.dacs.storage_in.ch
            self.qubit_ch_type = cfg.hw.soc.dacs.storage_in.type
            self.res_ch = cfg.hw.soc.dacs.storage_in.ch
            self.res_ch_type = cfg.hw.soc.dacs.storage_in.type
            self.res_gain = cfg.device.storage.gain
            self.readout_length_dac = self.us2cycles(cfg.device.storage.readout_length, gen_ch=self.res_ch)
            self.readout_length_adc = self.us2cycles(cfg.device.storage.readout_length, ro_ch=self.adc_ch)
            self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer
            self.adc_trig_offset=cfg.device.storage.trig_offset

        
        self.frequency = cfg.expt.freq
        self.freqreg = self.freq2reg(self.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        

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
        self.declare_gen(ch=self.qubit_ch, nqz=self.qubit_nyquist, mixer_freq=mixer_freq)

        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=self.frequency, gen_ch=self.res_ch)



        self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.freqreg, phase=0, gain=self.res_gain, length=self.readout_length_dac)
        self.synci(200) # give processor some time to configure pulses

    def body(self):
        # pass
        cfg=AttrDict(self.cfg)

        self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.freqreg, phase=0, 
                            gain=cfg.expt.gain, length=self.us2cycles(self.cfg.expt.displace_length))
        self.sync_all(self.us2cycles(self.cfg.expt.length))
        self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.freqreg, phase=0, gain=self.res_gain, length=self.readout_length_dac)
        self.measure(
            pulse_ch=self.res_ch,
            adcs=[self.adc_ch],
            adc_trig_offset=self.adc_trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

# ====================================================== #


class T1RingdownExperiment(Experiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1Ringdown', config_file=None, progress=None):
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
            self.cfg.expt.length = f
            rspec = T1RingdownProgram(soccfg=self.soccfg, cfg=self.cfg)
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

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
        data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
        data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        # plt.figure(figsize=(12, 8))
        # plt.subplot(111,title="$T_1$", xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
        # plt.plot(data["xpts"][:-1], data["amps"][:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     pCov = data['fit_err_amps']
        #     captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
        #     plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_amps"]), label=captionStr)
        #     plt.legend()
        #     print(f'Fit T1 amps [us]: {data["fit_amps"][3]}')

        plt.figure(figsize=(10,15))
        plt.subplot(311, title="$T_1$", ylabel="I [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(312, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')
        plt.subplot(313, xlabel="Wait Time [us]", ylabel="Amp [ADC units]")
        plt.plot(data["xpts"][:-1], data["amps"][:-1],'o-')
        if fit:
            p = data['fit_amps']
            pCov = data['fit_err_amps']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_amps"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 amp [us]: {data["fit_amps"][3]}')

        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname