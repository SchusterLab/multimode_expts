import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class RamseyEFProgram(RAveragerProgram):
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

        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_wait = 3
        self.r_phase2 = 4
        if self.qubit_ch_type == 'int4':
            self.r_phase = self.sreg(self.qubit_ch, "freq")
            self.r_phase3 = 5 # for storing the left shifted value
        else: self.r_phase = self.sreg(self.qubit_ch, "phase")

        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
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

        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.pi2sigma_ef = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        self.pi2sigma_ef_new = self.us2cycles(cfg.device.qubit.pulses.pi_ef_new.sigma, gen_ch=self.qubit_ch)

        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        self.add_gauss(ch=self.qubit_ch, name="pi2_ef", sigma=self.pi2sigma_ef, length=self.pi2sigma_ef*4)
        self.add_gauss(ch=self.qubit_ch, name="pi2_ef_new", sigma=self.pi2sigma_ef_new, length=self.pi2sigma_ef_new*4)

        # if self.res_ch_type == 'mux4':
        #     self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        # initialize wait registers
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.q_rp, self.r_phase2, 0) 
        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        # init to qubit excited state
        self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")

        # play pi/2 ef pulse
        # play gaussian pulse
        self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain, waveform="pi2_ef")
        # play flat pulse
        # self.setup_and_pulse(ch=self.qubit_ch, style="flat_top",length=self.us2cycles(self.cfg.device.qubit.pulses.hpi_ef_new.length), freq=self.f_ef, phase=0, gain=cfg.device.qubit.pulses.hpi_ef_new.gain, waveform="pi2_ef_new")
        
        # wait advanced wait time
        self.sync_all()
        self.sync(self.q_rp, self.r_wait)

        # play pi/2 ef pulse with advanced phase (all regs except phase already set by previous pulse)
        if self.qubit_ch_type == 'int4':
            self.bitwi(self.q_rp, self.r_phase3, self.r_phase2, '<<', 16)
            self.bitwi(self.q_rp, self.r_phase3, self.r_phase3, '|', self.f_ef)
            self.mathi(self.q_rp, self.r_phase, self.r_phase3, "+", 0)
        else: self.mathi(self.q_rp, self.r_phase, self.r_phase2, "+", 0)
        self.pulse(ch=self.qubit_ch)

        # map excited back to qubit ground state for measurement
        self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        # self.setup_and_pulse(ch=self.qubit_ch, style="flat_top",length=self.us2cycles(self.cfg.device.qubit.pulses.hpi_ef_new.length), freq=self.f_ef, phase=0, gain=cfg.device.qubit.pulses.hpi_ef_new.gain, waveform="pi2_ef_new")

        # align channels and wait 50ns
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

    def update(self):
        phase_step = self.deg2reg(360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step, gen_ch=self.qubit_ch) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update the time between two π/2 pulses
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+', phase_step) # advance the phase of the LO for the second π/2 pulse


class RamseyEFExperiment(Experiment):
    """
    Ramsey EF Experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(self, soccfg=None, path='', prefix='RamseyEF', config_file=None, progress=None):
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

        ramsey_ef = RamseyEFProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = ramsey_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        
 
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}        
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=None)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=None)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=None)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps
            data['f_ef_adjust_ramsey_avgi'] = (self.cfg.expt.ramsey_freq - p_avgi[1], -self.cfg.expt.ramsey_freq - p_avgi[1])
            data['f_ef_adjust_ramsey_avgq'] = (self.cfg.expt.ramsey_freq - p_avgq[1], -self.cfg.expt.ramsey_freq - p_avgq[1])
            data['f_ef_adjust_ramsey_amps'] = (self.cfg.expt.ramsey_freq - p_amps[1], -self.cfg.expt.ramsey_freq - p_amps[1])
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        # plt.figure(figsize=(10, 6))
        # plt.subplot(111,title=f"EF Ramsey (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
        #             xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
        # plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     pCov = data['fit_err_amps']
        #     captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
        #     plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
        #     plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
        #     plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
        #     plt.legend()
        #     print(f'Current EF frequency: {self.cfg.device.qubit.f_ef}')
        #     print(f'Fit frequency from amps [MHz]: {p[1]}')
        #     print('Suggested new EF frequencies from fit amps [MHz]:\n',
        #           f'\t{self.cfg.device.qubit.f_ef + data["f_ef_adjust_ramsey_amps"][0]}\n',
        #           f'\t{self.cfg.device.qubit.f_ef + data["f_ef_adjust_ramsey_amps"][1]}')
        #     print(f'T2 Ramsey EF from fit amps [us]: {p[3]}')

        plt.figure(figsize=(10,9))
        plt.subplot(211, 
            title=f"EF Ramsey (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
            ylabel="I [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
            plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.legend()
            print(f'Current EF frequency: {self.cfg.device.qubit.f_ef}')
            print(f'Fit frequency from I [MHz]: {p[1]}')
            print('Suggested new EF frequencies from fit avgi [MHz]:\n',
                  f'\t{self.cfg.device.qubit.f_ef + data["f_ef_adjust_ramsey_avgi"][0]}\n',
                  f'\t{self.cfg.device.qubit.f_ef + data["f_ef_adjust_ramsey_avgi"][1]}')
            print(f'T2 Ramsey EF from fit avgi [us]: {p[3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
            plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.legend()
            print(f'Fit frequency from Q [MHz]: {p[1]}')
            print('Suggested new EF frequencies from fit avgq [MHz]:\n',
                  f'\t{self.cfg.device.qubit.f_ef + data["f_ef_adjust_ramsey_avgq"][0]}\n',
                  f'\t{self.cfg.device.qubit.f_ef + data["f_ef_adjust_ramsey_avgq"][1]}')
            print(f'T2 Ramsey EF from fit avgq [us]: {p[3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)