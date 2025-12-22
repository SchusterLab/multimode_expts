import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.MM_base import MMRAveragerProgram

from fitting.fit_utils import guess_sinusoidal_params


class RamseyProgram(MMRAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        cfg = AttrDict(self.cfg)
        self.checkEF = self.cfg.expt.checkEF

        # self.num_qubits_sample = len(self.cfg.device.qubit.f_ge_idle)
        qTest = self.qubits[0]

        # declare registers for phase incrementing
        self.r_wait = 3
        self.r_phase2 = 4
        if self.qubit_ch_types[qTest] == 'int4':
            self.r_phase = self.sreg(self.qubit_chs[qTest], "freq")
            self.r_phase3 = 5 # for storing the left shifted value
        else: self.r_phase = self.sreg(self.qubit_chs[qTest], "phase")

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        # self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        # define pi2sigma as the pulse that we are calibrating with ramsey
        self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # -------------<--
        self.f_test_reg = self.f_ge_reg[0] # freq we are trying to calibrate
        self.gain_test = self.cfg.device.qubit.pulses.hpi_ge.gain[qTest] # gain of the pulse we are trying to calibrate ------------<

        if cfg.expt.f0g1_cavity > 0:
            ii = 0
            jj = 0
            if cfg.expt.f0g1_cavity==1: 
                ii=1
                jj=0
            if cfg.expt.f0g1_cavity==2: 
                ii=0
                jj=1
            # systematic way of adding qubit pulse under chi shift
            self.pif0g1_gain = self.cfg.device.QM.pulses.f0g1.gain[cfg.expt.f0g1_cavity-1]
            # self.f_pi_test_reg = self.freq2reg(self.cfg.device.QM.chi_shift_matrix[0][cfg.expt.f0g1_cavity]+self.cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest]) # freq we are trying to calibrate
            # self.gain_pi_test = self.cfg.device.QM.pulses.qubit_pi_ge.gain[ii][jj] # gain of the pulse we are trying to calibrate
            self.pi2sigma_test = self.cfg.device.QM.pulses.qubit_pi_ge.sigma[ii][jj]
            # self.add_gauss(ch=self.qubit_chs[qTest], name="pi2_test", sigma=self.pi2sigma, length=self.pi2sigma*4)

            self.f0g1 = self.freq2reg(cfg.device.QM.pulses.f0g1.freq[cfg.expt.f0g1_cavity-1], gen_ch=self.f0g1_ch[0])
            self.f0g1_length = self.us2cycles(cfg.device.QM.pulses.f0g1.length[cfg.expt.f0g1_cavity-1], gen_ch=self.f0g1_ch[0])
            self.add_gauss(ch=self.f0g1_ch[0], name="f0g1",
                       sigma=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma), length=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma)*4)

        if self.checkEF:
            self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])
            self.f_test_reg = self.f_ef_reg[qTest] # freq we are trying to calibrate
            self.gain_test = self.cfg.device.qubit.pulses.hpi_ef.gain[qTest] # gain of the pulse we are trying to calibrate

        if self.cfg.expt.user_defined_freq[0]:
            self.f_test_reg = self.freq2reg(self.cfg.expt.user_defined_freq[1], gen_ch=self.qubit_chs[qTest])
            self.gain_test = self.cfg.expt.user_defined_freq[2]
            self.pi2sigma = self.us2cycles(self.cfg.expt.user_defined_freq[3], gen_ch=self.qubit_chs[qTest])

        # add qubit pulses to respective channels
        # print(f"Calibrating pi/2 pulse on qubit {qTest} with freq {self.f_pi_test_reg} MHz")
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi2_test_ram", sigma=self.pi2sigma, length=self.pi2sigma*4)
        # if self.checkEF:
        # self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge_ram", sigma=self.pisigma_ge, length=self.pisigma_ge*4)

        # add readout pulses to respective channels
        # self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        # initialize wait registers
        self.safe_regwi(self.q_rps[qTest], self.r_wait, self.us2cycles(cfg.expt.start, gen_ch=self.qubit_chs[qTest])) # wait time register
        self.safe_regwi(self.q_rps[qTest], self.r_phase2, 0) 

        ## print pule parameters 
        print('fge is ', cfg.device.qubit.f_ge[qTest])
        print('fef is ', cfg.device.qubit.f_ef[qTest])

        self.sync_all(200)


    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0] 

        # initializations as necessary
        self.reset_and_sync()

        if cfg.expt.pre_active_reset_pulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_active_reset_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_ar_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_active_reset_sweep_pulse, prefix = 'pre_ar_')

        if self.cfg.expt.active_reset: 
            self.active_reset( man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)

        #prepulse : 
        self.sync_all(self.us2cycles(0.1))

        if cfg.expt.prepulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')

        if self.cfg.expt.qubit_ge_init:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[0], phase=0, gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            # self.wait_all(self.us2cycles(0.01))
            self.sync_all(self.us2cycles(0.01))

        # play pi/2 pulse with the freq that we want to calibrate
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_test_reg, phase=0, gain=self.gain_test, waveform="pi2_test_ram")

        # self.wait_all(self.us2cycles(0.01))
        self.sync_all(self.us2cycles(0.01))

        # wait advanced wait time
        self.sync_all()
        self.sync(self.q_rps[qTest], self.r_wait)

        # play echoes 
        # echoes 
        if cfg.expt.echoes[0]:
            for i in range(cfg.expt.echoes[1]):
                # even if ef, we still need just a pi pulse within that space
                self.pulse(ch=self.qubit_chs[qTest]) # this is ge or ef depedning on last hpi pulse
                self.pulse(ch=self.qubit_chs[qTest])

                    #print('Echo Only implemented for ge qubit')
                self.sync_all()
                self.sync(self.q_rps[qTest], self.r_wait)
                self.sync_all()

        # play pi/2 pulse with advanced phase (all regs except phase are already set by previous pulse)
        self.set_pulse_registers(ch=self.qubit_chs[qTest], style="arb", freq=self.f_test_reg, phase=self.deg2reg(cfg.advance_phase, gen_ch=self.qubit_chs[qTest]),
                                  gain=self.gain_test, waveform="pi2_test_ram")

        # self.wait_all(self.us2cycles(0.01))
        if self.qubit_ch_types[qTest] == 'int4':
            self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase2, '<<', 16)
            self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase3, '|', self.f_test_reg)
            self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase3, "+", 0)
            self.sync_all(self.us2cycles(0.01))
        else: 
            self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase2, "+", 0)
            self.sync_all(self.us2cycles(0.01)) # need this for mathi to finish before next pulse
        self.pulse(ch=self.qubit_chs[qTest])

        #postpulse :
        self.sync_all()
        if cfg.expt.postpulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.post_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'post_')
            else: 
                self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix = 'post_')

        if self.cfg.expt.qubit_ge_after: # map excited back to qubit ground state for measurement
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[0], phase=0, gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            # self.wait_all(self.us2cycles(0.01))
            self.sync_all(self.us2cycles(0.01))

        # align channels and measure
        self.measure_wrapper()

    def update(self):
        qTest = self.qubits[0]

        phase_step = self.deg2reg(360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step, gen_ch=self.qubit_chs[qTest]) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        self.mathi(self.q_rps[qTest], self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step, gen_ch = self.qubit_chs[qTest])) # update the time between two π/2 pulses
        self.sync_all(self.us2cycles(0.01))
        self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '+', phase_step) # advance the phase of the LO for the second π/2 pulse
        self.sync_all(self.us2cycles(0.01))


class RamseyExperiment(Experiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        checkZZ: True/False for putting another qubit in e (specify as qA)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qA in e , qB sweeps length rabi]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Ramsey', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)

        self.format_config_before_experiment( num_qubits_sample)

        read_num = 4 if self.cfg.expt.active_reset else 1

        ramsey = RamseyProgram(soccfg=self.soccfg, cfg=self.cfg)

        x_pts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc],
                                           threshold=None,
                                           load_pulses=True,
                                           progress=progress,
                                        #    debug=debug,
                                           readouts_per_experiment=read_num)
 
        avgi = avgi[0][-1] # when using active reset, second index selects which out of the 4 readouts per exp
        avgq = avgq[0][-1]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases} 
        data['idata'], data['qdata'] = ramsey.collect_shots()      

        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        self.data=data
        return data

    def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
        if data is None:
            data=self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            if fitparams is None:
                freq, amp, offset = guess_sinusoidal_params(data['xpts'], data['avgi'])
                fitparams=[amp,  freq, 0, data['xpts'][-1]/2, offset, None]
            # Remove the first and last point from fit in case weird edge measurements
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            if isinstance(p_avgi, (list, np.ndarray)): data['f_adjust_ramsey_avgi'] = sorted((self.cfg.expt.ramsey_freq - p_avgi[1], self.cfg.expt.ramsey_freq + p_avgi[1]), key=abs)
            if isinstance(p_avgq, (list, np.ndarray)): data['f_adjust_ramsey_avgq'] = sorted((self.cfg.expt.ramsey_freq - p_avgq[1], self.cfg.expt.ramsey_freq + p_avgq[1]), key=abs)
            if isinstance(p_amps, (list, np.ndarray)): data['f_adjust_ramsey_amps'] = sorted((self.cfg.expt.ramsey_freq - p_amps[1], self.cfg.expt.ramsey_freq + p_amps[1]), key=abs)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        self.qubits = self.cfg.expt.qubits
        self.checkEF = self.cfg.expt.checkEF

        q = self.qubits[0]

        f_pi_test = self.cfg.device.qubit.f_ge[q]
        if self.checkEF: f_pi_test = self.cfg.device.qubit.f_ef[q]
        if self.cfg.expt.f0g1_cavity > 0:
            ii = 0
            jj = 0
            if self.cfg.expt.f0g1_cavity==1: 
                ii=1
                jj=0
            if self.cfg.expt.f0g1_cavity==2: 
                ii=0
                jj=1
            # systematic way of adding qubit pulse under chi shift
            f_pi_test = self.cfg.device.QM.chi_shift_matrix[0][self.cfg.expt.f0g1_cavity]+self.cfg.device.qubit.f_ge[0] # freq we are trying to calibrate

        title = ('EF' if self.checkEF else '') + 'Ramsey' 

        plt.figure(figsize=(10,9))
        plt.subplot(211, 
            title=f"{title} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
            ylabel="I [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgi']
                try:
                    captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                except ValueError:
                    print('Fit Failed ; aborting')
                plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Current pi pulse frequency: {f_pi_test}')
                print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequency from fit I [MHz]:\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][0]}\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][1]}')
                print(f'T2 Ramsey from fit I [us]: {p[3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgq']
                try:
                    captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                except ValueError:
                    print('Fit Failed ; aborting')

                plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequencies from fit Q [MHz]:\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}')
                print(f'T2 Ramsey from fit Q [us]: {p[3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

