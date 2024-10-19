import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class T1CavityProgram(RAveragerProgram):
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
        
        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type

        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type

        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_wait = 3
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        # self.f0g1 = self.freq2reg(cfg.device.qubit.f0g1, gen_ch=self.f0g1_ch)
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer
        # self.f0g1_length = self.us2cycles(cfg.device.qubit.pulses.f0g1.length, gen_ch=self.f0g1_ch)

        

        self.f_man1 = self.freq2reg(cfg.device.manipulate.f_ge[0], gen_ch=self.man_ch)
        self.f_man2 = self.freq2reg(cfg.device.manipulate.f_ge[1], gen_ch=self.man_ch)

        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        self.pief_gain = cfg.device.qubit.pulses.pi_ef.gain
        self.pif0g1_gain = cfg.device.qubit.pulses.f0g1.gain

        if cfg.expt.cavity > 0:
            ii = 0
            jj = 0
            self.man_gain = self.cfg.expt.cavity_prepulse[1]
            self.man_length = self.us2cycles(self.cfg.expt.cavity_prepulse[2], gen_ch=self.man_ch)           
            
            if cfg.expt.cavity==1: 
                ii=1
                jj=0
                self.f_man = self.f_man1
            if cfg.expt.cavity==2: 
                ii=0
                jj=1
                self.f_man = self.f_man2
            # systematic way of adding qubit pulse under chi shift
            # self.pif0g1_gain = self.cfg.device.QM.pulses.f0g1.gain[cfg.expt.cavity-1]
            self.pif0g1_gain = cfg.expt.f0g1_param[1]
            # self.f_pi_test_reg = self.freq2reg(self.cfg.device.QM.chi_shift_matrix[0][cfg.expt.f0g1_cavity]+self.cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest]) # freq we are trying to calibrate
            self.gain_pi_test = self.cfg.device.QM.pulses.qubit_pi_ge.gain[ii][jj] # gain of the pulse we are trying to calibrate
            # self.pi2sigma_test = self.cfg.device.QM.pulses.qubit_pi_ge.sigma[ii][jj]
            # self.add_gauss(ch=self.qubit_chs[qTest], name="pi2_test", sigma=self.pi2sigma, length=self.pi2sigma*4)
            # self.f0g1 = self.freq2reg(cfg.device.QM.pulses.f0g1.freq[cfg.expt.cavity-1], gen_ch=self.f0g1_ch)
            # self.f0g1_length = self.us2cycles(cfg.device.QM.pulses.f0g1.length[cfg.expt.cavity-1], gen_ch=self.f0g1_ch)
            self.f0g1 = self.freq2reg(cfg.expt.f0g1_param[0], gen_ch=self.f0g1_ch)
            self.f0g1_length = self.us2cycles(cfg.expt.f0g1_param[2], gen_ch=self.f0g1_ch)
            self.pi_resolved_sigma = self.us2cycles(cfg.device.QM.pulses.qubit_pi_ge_resolved.sigma[ii][jj], gen_ch=self.qubit_ch)
            self.flat_length = self.us2cycles(cfg.device.QM.pulses.qubit_pi_ge_resolved.length[ii][jj], gen_ch=self.qubit_ch)
            self.pi_gain_resolved = self.cfg.device.QM.pulses.qubit_pi_ge_resolved.gain[ii][jj]
            
            self.f_ge_resolved = self.freq2reg(self.cfg.device.QM.chi_shift_matrix[0][cfg.expt.cavity]+self.cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)

            # self.set_pulse_registers(ch=self.man_ch, style="const", freq=self.f_man, phase=0, gain=self.man_gain, length=self.man_length)

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = self.adc_ch
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
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)
        self.declare_gen(ch=self.man_ch, nqz=cfg.hw.soc.dacs.manipulate_in.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        # self.pi_resolved_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.qubit_ch)

        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        self.add_gauss(ch=self.qubit_ch, name="pi_resolved_qubit", sigma=self.pi_resolved_sigma, length=self.pi_resolved_sigma*4)
        self.set_pulse_registers(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        self.add_gauss(ch=self.qubit_ch, name="pief_qubit", sigma=self.pief_sigma, length=self.pief_sigma*4)

        self.add_gauss(ch=self.f0g1_ch, name="f0g1_pi_test",
                       sigma=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma), length=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma)*4)

        # if self.res_ch_type == 'mux4':
        #     self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        # if cfg.expt.resolved_pi:
        #     self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_resolved_qubit")
        #     self.sync_all() # align channels

        if cfg.expt.f0g1_prep:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
            self.sync_all() # align channels
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=self.pief_gain, waveform="pief_qubit")
            self.sync_all() # align channels
            self.setup_and_pulse(
                    ch=self.f0g1_ch,
                    style="flat_top",
                    freq=self.f0g1,
                    length=self.f0g1_length,
                    phase=0,
                    gain=self.pif0g1_gain, 
                    waveform="f0g1_pi_test")
            # self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f0g1, phase=0, gain=self.pif0g1_gain, length=self.f0g1_length)
            self.sync_all() # align channels
        
        if cfg.expt.cavity_prepulse[0]:
            self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.f_man, phase=0, gain=self.man_gain, length=self.man_length)
            self.sync_all(self.us2cycles(0.05)) # align cavity pulse and qubit pulse, remember to calibrate!!!! 
        
        self.sync(self.q_rp, self.r_wait) # wait for the time stored in the wait variable register
       

        if cfg.expt.resolved_pi:
            self.setup_and_pulse(ch=self.qubit_ch, style="flat_top", freq=self.f_ge_resolved, phase=0, length=self.flat_length, gain=self.pi_gain_resolved, waveform="pi_resolved_qubit")
            self.sync_all() # align channels
        elif cfg.expt.f0g1_prep:
            self.setup_and_pulse(
                    ch=self.f0g1_ch,
                    style="flat_top",
                    freq=self.f0g1,
                    length=self.f0g1_length,
                    phase=0,
                    gain=self.pif0g1_gain, 
                    waveform="f0g1_pi_test")
            # self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f0g1, phase=180, gain=self.pif0g1_gain, length=self.f0g1_length)
            self.sync_all() # align channels            
            # self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=180, gain=self.pief_gain, waveform="pief_qubit")
            # self.sync_all() # align channels
            # self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=180, gain=self.pi_gain, waveform="pi_qubit")
            # self.sync_all() # align channels
            # self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_resolved_qubit")
            # self.sync_all() # align channels
        
        
            
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update wait time


class T1CavityExperiment(Experiment):
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

    def __init__(self, soccfg=None, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

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

        t1 = T1CavityProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}

        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)
            
            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]
        
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

        plt.figure(figsize=(10,10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')

        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname