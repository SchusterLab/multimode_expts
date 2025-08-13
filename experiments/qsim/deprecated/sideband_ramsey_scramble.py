import matplotlib.pyplot as plt
import numpy as np
from qick import QickConfig
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from dataset import storage_man_swap_dataset
from experiments.qsim.utils import (
    ensure_list_in_cfg,
    guess_freq,
    post_select_raverager_data,
)
from MM_base import MMRAveragerProgram


class SidebandRamseyProgram(MMRAveragerProgram):
    """
    First initialize a photon into man1 by qubit ge, qubit ef, f0g1 
    Then do a Ramsey experiment on M1-Sx swap 
    """
    def __init__(self, soccfg: QickConfig, cfg: AttrDict):
        super().__init__(soccfg, cfg)


    def retrieve_swap_parameters(self):
        """
        retrieve pulse parameters for the M1-Sx swap
        """
        qTest = self.qubits[0]
        stor_no = self.cfg.stor_no
        stor_name = f'M1-S{stor_no}'
        self.m1s_freq_MHz = self.swap_ds.get_freq(stor_name) + self.cfg.expt.detune
        self.m1s_is_low_freq = True if self.m1s_freq_MHz < 1000 else False
        self.m1s_ch = self.flux_low_ch[qTest] if self.m1s_is_low_freq else self.flux_high_ch[qTest]
        self.m1s_freq = self.freq2reg(self.m1s_freq_MHz, gen_ch=self.m1s_ch)
        self.m1s_length = self.us2cycles(self.swap_ds.get_h_pi(stor_name), gen_ch=self.m1s_ch)
        self.m1s_gain = self.swap_ds.get_gain(stor_name)
        self.m1s_wf_name = "pi_m1si_low" if self.m1s_is_low_freq else "pi_m1si_high"


    def initialize(self):
        """
        Retrieves ch, freq, length, gain from csv for M1-Sx π/2 pulse and
        sets the waiting time and phase advance registers for the tau sweep
        """
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        cfg = self.cfg # should be AttrDict already if experiment class init was run properly
        self.swap_ds = storage_man_swap_dataset()
        self.retrieve_swap_parameters()

        # declare registers for waiting time and phase incrementing
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase = self.sreg(self.m1s_ch, "phase")

        # initialize wait and phase registers
        self.m1s_ch_page = self.ch_page(self.m1s_ch)
        self.safe_regwi(self.m1s_ch_page, self.r_wait, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.m1s_ch_page, self.r_phase2, 0) 

        self.sync_all(200)


    def body(self):
        cfg=AttrDict(self.cfg)

        # initializations as necessary
        self.reset_and_sync()

        if self.cfg.expt.active_reset: 
            self.active_reset(
                man_reset=self.cfg.expt.man_reset,
                storage_reset= self.cfg.expt.storage_reset)

        # prepulse: ge -> ef -> f0g1
        prepules_cfg = [
            ['qubit', 'ge', 'pi', 0,],
            ['qubit', 'ef', 'pi', 0,],
            ['man', 'M1', 'pi', 0,],
        ]
        pulse_creator = self.get_prepulse_creator(prepules_cfg)
        self.sync_all(self.us2cycles(0.1))
        self.custom_pulse(cfg, pulse_creator.pulse, prefix='pre_')
        self.sync_all(self.us2cycles(0.1))

        # first pi/2 pulse 
        self.setup_and_pulse(ch=self.m1s_ch, 
                             style="flat_top", 
                             freq=self.m1s_freq, 
                             phase=0, 
                             gain=self.m1s_gain,
                             length=self.m1s_length,
                             waveform=self.m1s_wf_name)
        self.sync_all(self.us2cycles(0.01))

        # wait advanced wait time
        self.sync_all()
        self.sync(self.m1s_ch_page, self.r_wait)

        self.setup_and_pulse(ch=self.m1s_ch, 
                             style="flat_top", 
                             freq=self.m1s_freq+self.freq2reg(100, gen_ch=self.m1s_ch), 
                             phase=0, 
                             gain=0, #self.m1s_gain,
                             length=self.m1s_length,
                             waveform=self.m1s_wf_name)

        self.sync(self.m1s_ch_page, self.r_wait)

        # play second pi/2 pulse with advanced phase (all regs except phase are already set by previous pulse)
        self.set_pulse_registers(ch=self.m1s_ch, 
                                 style="flat_top", 
                                 freq=self.m1s_freq, 
                                 phase=self.deg2reg(cfg.advance_phase),
                                 gain=self.m1s_gain,
                                 length=self.m1s_length,
                                 waveform=self.m1s_wf_name)
        self.mathi(self.m1s_ch_page, self.r_phase, self.r_phase2, "+", 0)
        self.sync_all(self.us2cycles(0.01))
        self.pulse(ch=self.m1s_ch)

        # postpulse
        postpules_cfg = [
            ['man', 'M1', 'pi', 0,],
        ]
        pulse_creator = self.get_prepulse_creator(postpules_cfg)
        self.sync_all(self.us2cycles(0.1))
        self.custom_pulse(cfg, pulse_creator.pulse, prefix='post_')
        self.sync_all(self.us2cycles(0.1))

        self.measure_wrapper()


    def update(self):
        # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        phase_step = self.deg2reg(360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step,
                                  gen_ch=self.m1s_ch) 

        # update the time between two π/2 pulses
        self.mathi(self.m1s_ch_page, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step/2))
        self.sync_all(self.us2cycles(0.01))

        # update the phase for the second π/2 pulse
        self.mathi(self.m1s_ch_page, self.r_phase2, self.r_phase2, '+', phase_step) # advance the phase of the LO for the second π/2 pulse
        self.sync_all(self.us2cycles(0.01))


class SidebandRamseyExperiment(Experiment):
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
        qubits: this is just 0 for the purpose of the currrent multimode sample
    )
    """
    def __init__(self, soccfg=None, path='', prefix='SidebandRamsey',
                 config_file=None, expt_params=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)
        self.cfg.expt = AttrDict(expt_params)


    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        read_num = 4 if self.cfg.expt.active_reset else 1

        ramsey = SidebandRamseyProgram(soccfg=self.soccfg, cfg=self.cfg)
        self.qick_program = ramsey

        x_pts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc],
                                           threshold=None,
                                           load_pulses=True,
                                           progress=progress,
                                           debug=debug,
                                           readouts_per_experiment=read_num)
 
        avgi = avgi[0][-1]
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
        # works poorly now: visibly sinusoidal curves fail to fit
        if data is None:
            data=self.data

        if self.cfg.expt.active_reset:
            data['avgi'], data['avgq'] = post_select_raverager_data(data, self.cfg)

        if fit:
            # if fitparams is None:
            #     fitparams=[200,  0.2, 0, 200, None, None]
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'], data["avgi"], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'], data["avgq"], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'], data["amps"], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            if isinstance(p_avgi, (list, np.ndarray)):
                data['f_adjust_ramsey_avgi'] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgi[1],
                     self.cfg.expt.ramsey_freq + p_avgi[1]),
                    key=abs)
            if isinstance(p_avgq, (list, np.ndarray)):
                data['f_adjust_ramsey_avgq'] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgq[1],
                     self.cfg.expt.ramsey_freq + p_avgq[1]),
                    key=abs)
            if isinstance(p_amps, (list, np.ndarray)):
                data['f_adjust_ramsey_amps'] = sorted(
                    (self.cfg.expt.ramsey_freq - p_amps[1],
                     self.cfg.expt.ramsey_freq + p_amps[1]),
                    key=abs)
        return data


    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        self.qubits = self.cfg.expt.qubits

        q = self.qubits[0]

        f_pi_test = self.cfg.device.qubit.f_ge[q]
        # if self.cfg.expt.f0g1_cavity > 0:
        #     f_pi_test = self.cfg.device.QM.chi_shift_matrix[0][self.cfg.expt.f0g1_cavity] \
        #         + self.cfg.device.qubit.f_ge[0] # freq we are trying to calibrate

        title = 'Sideband Ramsey' 

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
                plt.plot(data["xpts"][:-1], 
                         fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]),
                         color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1],
                         fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]),
                         color='0.2', linestyle='--')
                plt.legend()
                print(f'Current pi pulse frequency: {f_pi_test}')
                print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*self.cfg.expt.ramsey_freq:
                    print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
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
                plt.plot(data["xpts"][:-1],
                         fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]),
                         color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1],
                         fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]),
                         color='0.2', linestyle='--')
                plt.legend()
                print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*self.cfg.expt.ramsey_freq: 
                    print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequencies from fit Q [MHz]:\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}')
                print(f'T2 Ramsey from fit Q [us]: {p[3]}')

        plt.tight_layout()
        plt.show()


    def save_data(self, data=None):
        # do we really need to ovrride this?
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


class SidebandChevronExperiment(SidebandRamseyExperiment):
    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        read_num = 4 if self.cfg.expt.active_reset else 1

        y_pts = np.linspace(-1,1,51)

        data = {'avgi': [], 'avgq': [], 'amps': [], 'phases': [], 'idata': [], 'qdata': []}

        for detune in tqdm(y_pts):
            self.cfg.expt.detune = detune
            ramsey = SidebandRamseyProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.qick_program = ramsey

            x_pts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc],
                                            threshold=None,
                                            load_pulses=True,
                                            progress=False,
                                            debug=debug,
                                            readouts_per_experiment=read_num)
    
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase

            data['avgi'].append(avgi)
            data['avgq'].append(avgq)
            data['amps'].append(amps)
            data['phases'].append(phases)
            idata, qdata = ramsey.collect_shots()
            data['idata'].append(idata)
            data['qdata'].append(qdata)

        data['xpts'] = x_pts
        data['ypts'] = y_pts + ramsey.m1s_freq_MHz
        for key in ['avgi', 'avgq', 'amps', 'phases', 'idata', 'qdata']:
            data[key] = np.array(data[key])

        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        self.data=data
        return data

