import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.MM_base import *

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse.
This is a preliminary measurement to prove that we see Rabi oscillations.
This measurement is followed up by the Amplitude Rabi experiment.
"""

class SidebandGeneralProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        qTest = self.cfg.expt.qubits[0]
        self.MM_base_initialize()

        self.rf_ch = self.flux_low_ch if self.cfg.expt.flux_drive[0] == 'low' else self.flux_high_ch
        print('Using RF channel:', self.rf_ch[qTest])

        if "freq" in self.cfg.expt:
            self.cfg.expt.flux_drive[1] = self.cfg.expt.freq
        if "gain" in self.cfg.expt:
            self.cfg.expt.flux_drive[2] = self.cfg.expt.gain

        self.test_pulse_str = [
            [self.cfg.expt.flux_drive[1]], # freq (MHz)
            [self.cfg.expt.flux_drive[2]], # gain
            [self.cfg.expt.length_placeholder], # length (us)
            [0], # phase
            [self.rf_ch[qTest]], # ch
            ["flat_top"],
            [self.cfg.device.storage.ramp_sigma]]
        # flux drive = [low/high (ch), freq, gain, ramp_sigma(us)]
        # RF flux modulation, gaussian flat top pulse

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.cfg.expt.qubits[0]

        # phase reset
        self.reset_and_sync()

        #do the active reset
        if cfg.expt.active_reset:
            self.active_reset( man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)

        #  prepulse
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

        # RF flux modulation
        # self.wait_all(self.us2cycles(cfg.expt.length_placeholder))

        if self.cfg.expt.length_placeholder>0:
            self.custom_pulse(cfg, self.test_pulse_str, prefix='flux')

        self.sync_all()  # align channels

        # post pulse
        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='post')
            

        # align channels and wait 50ns and measure
        self.sync_all(self.us2cycles(0.05))

        self.measure_wrapper()



class SidebandGeneralExperiment(Experiment):
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

    def __init__(self, soccfg=None, path='', prefix='SidebandGeneral', config_file=None, progress=None):
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

        data = {"xpts": [], "idata": [], "qdata": [], "avgi": [], "avgq": [], "amps": []}

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        for length in tqdm(lengths, disable=not progress):
            #### update phase of post pulse (only for cavity ramsey in presence of coupler drive)
            if self.cfg.expt.update_post_pulse_phase[0]:
                wait_freq = self.cfg.expt.update_post_pulse_phase[1]
                wait_phase = length * wait_freq  * 360
                self.cfg.expt.post_sweep_pulse[3][-1] = wait_phase
                # print(f'Updated post pulse phase to {wait_phase} deg')
                # print(self.cfg.expt.post_sweep_pulse)

            self.cfg.expt.length_placeholder = float(length)
            lengthrabi = SidebandGeneralProgram(
                soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(
                self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug, readouts_per_experiment=read_num)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            idata, qdata = lengthrabi.collect_shots()
            # amp = np.abs(avgi+1j*avgq)  # Calculating the magnitude
            # phase = np.angle(avgi+1j*avgq)  # Calculating the phase
            data["xpts"].append(length)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(np.mean(np.abs(idata + 1j*qdata)))  # mean amplitude
            if self.cfg.expt.active_reset:
                data["idata"].append(idata)
                data["qdata"].append(qdata)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, fit=False, fitparams=None, **kwargs):
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
        gain = self.cfg.expt.flux_drive[2] 
        plt.subplot(
            211, title=f"RF modulation (RF Gain {gain})", ylabel="I [adc level]")
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
