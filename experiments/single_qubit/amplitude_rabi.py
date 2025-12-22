import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

from copy import deepcopy

import fitting.fitting as fitter
from fitting.fit_display_classes import AmplitudeRabiFitting
from experiments.MM_base import MMRAveragerProgram

class AmplitudeRabiProgram(MMRAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = 0

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        # define pi2sigma as the pulse that we are calibrating with ramsey
        self.pi_test_sigma = self.us2cycles(cfg.expt.sigma_test, gen_ch=self.qubit_chs[qTest])
        self.flat_length = self.us2cycles(cfg.expt.flat_length, gen_ch=self.qubit_chs[qTest])
        self.f_pi_test_reg = self.f_ge_reg[qTest] # freq we are trying to calibrate

        if self.cfg.expt.checkEF:
            self.f_pi_test_reg = self.f_ef_reg[qTest] # freq we are trying to calibrate

        if cfg.expt.user_defined_freq[0]:
            self.f_pi_test_reg = self.freq2reg(cfg.expt.user_defined_freq[1], gen_ch=self.qubit_chs[0])

        # add qubit and readout pulses to respective channels
        if cfg.expt.pulse_type.lower() == "gauss" and self.pi_test_sigma > 0:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test", sigma=self.pi_test_sigma, length=self.pi_test_sigma*4)
        if cfg.expt.pulse_type.lower() == "flat_top" and self.pi_test_sigma > 0:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test", sigma=self.pi_test_sigma, length=self.pi_test_sigma*4)

        if self.cfg.expt.checkEF:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge", sigma=self.pisigma_ge, length=self.pisigma_ge*4)

        # initialize registers
        if self.qubit_ch_types[qTest] == 'int4':
            self.r_gain = self.sreg(self.qubit_chs[qTest], "addr") # get gain register for qubit_ch
        else:
            if cfg.expt.pulse_type == "flat_top":
                self.r_gain = self.sreg(self.qubit_chs[qTest], "gain") # get gain register for qubit_ch
                self.r_gain2 = self.sreg(self.qubit_chs[qTest], "gain2") # get gain register for qubit_ch
            else:
                self.r_gain = self.sreg(self.qubit_chs[qTest], "gain") # get gain register for qubit_ch

        self.r_gain3 = 4
        self.safe_regwi(self.q_rps[qTest], self.r_gain3, self.cfg.expt.start)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)

        qTest = self.qubits[0]

        # initializations as necessary
        if self.cfg.expt.pulse_ge_init:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[0], phase=0, gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            self.sync_all(0.05)

        # pre pulse
        if cfg.expt.prepulse:
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else:
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')

        if self.pi_test_sigma > 0:
            if cfg.expt.pulse_type.lower() == "gauss":
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest],
                    style="arb",
                    freq=self.f_pi_test_reg,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_test")
            elif cfg.expt.pulse_type == "flat_top":
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest],
                    style="flat_top",
                    freq=self.f_pi_test_reg,
                    length=self.flat_length,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_test")
            elif cfg.expt.pulse_type == "drag":
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest],
                    style="arb",
                    freq=self.f_pi_test_reg,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_test_drag")
            else:
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest],
                    style="const",
                    freq=self.f_pi_test_reg,
                    phase=0,
                    gain=0, # gain set by update
                    length=self.pi_test_sigma)
        self.mathi(self.q_rps[qTest], self.r_gain, self.r_gain3, "+", 0)
        if cfg.expt.pulse_type == "flat_top":
            self.mathi(self.q_rps[qTest], self.r_gain2, self.r_gain3, "+", 0)
        self.pulse(ch=self.qubit_chs[qTest])
        # self.sync_all()

        # if self.checkEF: # map excited back to qubit ground state for measurement
        #     self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")

        #postpulse :
        self.sync_all()
        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='post')

        if self.cfg.expt.pulse_ge_after:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[0], phase=0, gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            self.sync_all(0.05)
        # align channels and measure
        self.measure_wrapper()

    def update(self):
        if self.cfg.expt.checkZZ: qA, qTest = self.qubits
        else: qTest = self.qubits[0]

        step = self.cfg.expt.step
        if self.qubit_ch_types[qTest] == 'int4': step = step << 16
        self.mathi(self.q_rps[qTest], self.r_gain3, self.r_gain3, '+', step) # update test gain

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        # print(np.average(self.di_buf[0]))
        self.readout_length_adc = self.readout_lengths_adc[0]
        shots_i0 = self.di_buf[0] / self.readout_length_adc
        shots_q0 = self.dq_buf[0] / self.readout_length_adc
        return shots_i0, shots_q0
        # return shots_i0[:5000], shots_q0[:5000]

# ====================================================== #

class AmplitudeRabiExperiment(Experiment):
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
        pulse_type: 'gauss' or 'flat_top' or 'drag' or 'const' (implementation not checked)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabi', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        from copy import deepcopy
        base_cfg = AttrDict(deepcopy(self.cfg)) # unedited config file for the histogram experiment

        #expand entries in config that are length 1 to fill all qubits
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


        if self.cfg.expt.checkZZ:
            assert len(self.cfg.expt.qubits) == 2
            qA, qTest = self.cfg.expt.qubits
            assert qA != 1
            assert qTest == 1
        else: qTest = self.cfg.expt.qubits[0]

        if 'sigma_test' not in self.cfg.expt:
            if not self.cfg.expt.checkZZ:
                self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ge.sigma[qTest]
            else: self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qA]

        if not self.cfg.expt.single_shot:
            amprabi = AmplitudeRabiProgram(soccfg=self.soccfg, cfg=self.cfg)

            xpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)

            # shots_i = amprabi.di_buf[adc_ch].reshape((self.cfg.expt.expts, self.cfg.expt.reps)) / amprabi.readout_length_adc
            # shots_i = np.average(shots_i, axis=1)
            # print(len(shots_i), self.cfg.expt.expts)
            # shots_q = amprabi.dq_buf[adc_ch] / amprabi.readout_length_adc
            # print(np.std(shots_i), np.std(shots_q))

            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase

            # data={'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
            data={'xpts': xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
            if self.cfg.expt.normalize:
                from experiments.single_qubit.normalize import normalize_calib
                g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

                data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
                data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
                data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        else:
            from experiments.single_qubit.single_shot import hist, HistogramProgram
            from copy import deepcopy

            # ----------------- Single Shot Calibration -----------------
            data=dict()
            sscfg = AttrDict(deepcopy(base_cfg))

            # setup sscfg (formatting)
            q_ind = sscfg.expt.qubit
            for subcfg in (sscfg.device.readout, sscfg.device.qubit, sscfg.hw.soc):
                for key, value in subcfg.items() :
                    if isinstance(value, list):
                        subcfg.update({key: value[q_ind]})
                    elif isinstance(value, dict):
                        for key2, value2 in value.items():
                            for key3, value3 in value2.items():
                                if isinstance(value3, list):
                                    value2.update({key3: value3[q_ind]})

            sscfg.expt.reps = sscfg.expt.singleshot_reps
            # Ground state shots
            # cfg.expt.reps = 10000
            sscfg.expt.qubit = 0
            sscfg.expt.rounds = 1
            sscfg.expt.pulse_e = False
            sscfg.expt.pulse_f = False
            # print(sscfg)

            data['Ig'] = []
            data['Qg'] = []
            data['Ie'] = []
            data['Qe'] = []
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
            data['Ig'], data['Qg'] = histpro.collect_shots()

            # Excited state shots
            sscfg.expt.pulse_e = True
            sscfg.expt.pulse_f = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
            data['Ie'], data['Qe'] = histpro.collect_shots()
            # print(data)

            fids, thresholds, angle = hist(data=data, plot=False, verbose=False, span=self.cfg.expt.span)
            data['fids'] = fids
            data['angle'] = angle
            data['thresholds'] = thresholds

            print(f'ge fidelity (%): {100*fids[0]}')
            print(f'rotation angle (deg): {angle}')
            print(f'threshold ge: {thresholds[0]}')

            # ------------------- Experiment -------------------

            data['I_data']= []
            data['Q_data']= []
            data['avgi'] = [] # for debugging
            data['avgq'] = []
            # Do single round experiments since collecting shots for all rounds is not supported
            rounds = self.cfg.expt.rounds

            for round in range(rounds):
                print(f'Round {round}')
                rcfg = AttrDict(deepcopy(self.cfg))
                rcfg.expt.rounds = 1

                prog = AmplitudeRabiProgram(soccfg=self.soccfg, cfg=rcfg)
                x_pts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)
                II, QQ = prog.collect_shots()
                # save data for each round
                data['I_data'].append(II)
                data['Q_data'].append(QQ)
                data['avgi'].append(avgi) # for debugging
                data['avgq'].append(avgq)
                data['xpts'] = x_pts # same for all rounds

        self.data=data
        return data

    def single_shot_analysis(self, data=None, **kwargs):
        '''
        Bin shots in g and e state s
        '''
        threshold = self.cfg.device.readout.threshold[0] # for i data
        theta = self.cfg.device.readout.phase[0] * np.pi / 180 # degrees to rad
        I = data['I']
        Q = data['Q']

        # """Rotate the IQ data"""
        # I_new = I*np.cos(theta) - Q*np.sin(theta)
        # Q_new = I*np.sin(theta) + Q*np.cos(theta)
        I_new = I
        Q_new = Q

        # """Threshold the data"""
        shots = np.zeros(len(I_new))
        #shots[I_new < threshold] = 0 # ground state
        shots[I_new > threshold] = 1 # excited state

        # Reshape data into 2D array: reps x expts
        shots = shots.reshape(self.cfg.expt.expts, self.cfg.expt.reps)

        # Average over reps
        probs_ge = np.mean(shots, axis=1)

        data['probs_ge'] = probs_ge
        return data


    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        """
        Analyze amplitude rabi data using the canonical AmplitudeRabiFitting implementation.

        This method delegates to AmplitudeRabiFitting to ensure consistent analysis
        between standalone data analysis and experiment-based workflows.

        Note: When data=None, this modifies self.data in place (standard behavior).
        """
        if data is None:
            data = self.data

        # Use the canonical implementation from fitting_display_classes
        fitter_instance = AmplitudeRabiFitting(
            data=data,
            config=self.cfg,
            fitparams=fitparams
        )

        # Run analysis - this modifies data in place and returns it
        data = fitter_instance.analyze(data=data, fit=fit, fitparams=fitparams, **kwargs)

        return data

    def display(self, data=None, fit=True, fitparams=None, vline = None, **kwargs):
        if data is None:
            data=self.data

        plt.figure(figsize=(10,10))
        plt.subplot(211, title=f"Amplitude Rabi (Pulse Length {self.cfg.expt.sigma_test})", ylabel="I [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'o-')
        if fit:
            p = data['fit_avgi']
            plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            pi_gain = data['pi_gain_avgi']
            hpi_gain = data['hpi_gain_avgi']
            print(f'Pi gain from avgi data [dac units]: {pi_gain}')
            print(f'\tPi/2 gain from avgi data [dac units]: {hpi_gain}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(hpi_gain, color='0.2', linestyle='--')
            if not(vline==None):
                plt.axvline(vline, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1],'o-')
        if fit:
            p = data['fit_avgq']
            plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            pi_gain = data['pi_gain_avgq']
            hpi_gain = data['hpi_gain_avgq']
            print(f'Pi gain from avgq data [dac units]: {pi_gain}')
            print(f'\tPi/2 gain from avgq data [dac units]: {hpi_gain}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(hpi_gain, color='0.2', linestyle='--')

        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #

class AmplitudeRabiChevronExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        flat_length: flat top length [us] (implementation not checked)
        pulse_type: 'gauss' or 'flat_top' or 'drag' or 'const' (implementation not checked)
        checkEF: bool
        checkZZ: bool
        prepulse: bool
        postpulse: bool
        pulse_ge_init: bool
        pulse_ge_after: bool
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiChevron', config_file=None, progress=None):
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

        if 'sigma_test' not in self.cfg.expt:
            self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ge.sigma

        freqpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        data={"xpts":[], "freqpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[0]

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain
        for freq in tqdm(freqpts):
            self.cfg.device.qubit.f_ge = [freq]
            amprabi = AmplitudeRabiProgram(soccfg=self.soccfg, cfg=self.cfg)

            xpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)

            avgi = avgi[adc_ch][0]
            avgq = avgq[adc_ch][0]
            amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)

        data['xpts'] = xpts
        data['freqpts'] = freqpts
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        x_sweep = data['xpts']
        y_sweep = data['freqpts']
        avgi = data['avgi']
        avgq = data['avgq']

        plt.figure(figsize=(10,8))
        plt.subplot(211, title="Amplitude Rabi", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='I [ADC level]')
        plt.clim(vmin=None, vmax=None)

        plt.subplot(212, xlabel="Gain [dac units]", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='Q [ADC level]')
        plt.clim(vmin=None, vmax=None)

        if fit: pass

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
