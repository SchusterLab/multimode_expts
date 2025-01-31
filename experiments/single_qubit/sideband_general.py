import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import experiments.fitting as fitter
from MM_base import *

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
"""


class SidebandGeneralProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

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
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
        self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        if self.cfg.expt.flux_drive[0] == 'low':
            self.rf_ch = cfg.hw.soc.dacs.flux_low.ch
            self.rf_ch_types = cfg.hw.soc.dacs.flux_low.type
        else:
            self.rf_ch = cfg.hw.soc.dacs.flux_high.ch
            self.rf_ch_types = cfg.hw.soc.dacs.flux_high.type

        # get register page for qubit_chs
        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]
        self.rf_rps = [self.ch_page(ch) for ch in self.rf_ch]

        self.f_ge_reg = [self.freq2reg(
            cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])]
        self.f_ef_reg = [self.freq2reg(
            cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])]

        # self.f_ge_resolved_reg = [self.freq2reg(
        #     self.cfg.expt.qubit_resolved_pi[0], gen_ch=self.qubit_chs[qTest])]

        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(
            cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.f_rf_reg = [self.freq2reg(self.cfg.expt.flux_drive[1], gen_ch=self.rf_ch[0])]

        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(
            self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(
            self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []

        # declare res dacs
        mask = None
        mixer_freq = 0  # MHz
        mux_freqs = None  # MHz
        mux_gains = None
        ro_ch = None
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest],
                         mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest],
                             freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(
                    ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # define pi_test_ramp as the pulse that we are calibrating with ramsey, update in outer loop over averager program
        self.pi_test_ramp = self.us2cycles(
            cfg.device.qubit.ramp_sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.rf_gain_test = self.cfg.expt.flux_drive[2]  # gain we are trying to play

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(
            cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ge value
        self.pisigma_ef = self.us2cycles(
            cfg.device.qubit.pulses.pi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ef value
        
        self.sideband_sigma_low = self.us2cycles(
            cfg.device.active_reset.M1_S_sigma, gen_ch=self.flux_low_ch[qTest]) 
        self.sideband_sigma_high = self.us2cycles(
            cfg.device.active_reset.M1_S_sigma, gen_ch=self.flux_high_ch[qTest])  


        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.f_ef_init_reg = self.f_ef_reg[qTest]
        # self.f_ge_resolved_int_reg = self.f_ge_resolved_reg[qTest]
        self.rf_freq_reg = self.f_rf_reg[qTest]

        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.gain_ef_init = self.cfg.device.qubit.pulses.pi_ef.gain[qTest]

        # add qubit pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test_ramp", sigma=self.pi_test_ramp,
                       length=self.pi_test_ramp*2*cfg.device.qubit.ramp_sigma_num[qTest])
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge",
                       sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ef",
                       sigma=self.pisigma_ef, length=self.pisigma_ef*4)
        self.add_gauss(ch=self.flux_low_ch[qTest], name="ramp_low",
                       sigma=self.sideband_sigma_low, length=self.sideband_sigma_low*4)
        self.add_gauss(ch=self.flux_high_ch[qTest], name="ramp_high",
                       sigma=self.sideband_sigma_high, length=self.sideband_sigma_high*4)
        # self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_resolved",
        #                sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
        self.add_gauss(ch=self.rf_ch[0], name="rf_test",
                       sigma=self.us2cycles(self.cfg.expt.flux_drive[3]), length=self.us2cycles(self.cfg.expt.flux_drive[3])*6)

        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(
            cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        ## ALL ACTIVE RESET REQUIREMENTS
        # read val definition
        self.r_read_q = 3   # ge active reset register
        self.r_read_q_ef = 4   # ef active reset register
        self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0
        self.safe_regwi(0, self.r_read_q_ef, 0)  # init read val to be 0

        # threshold definition
        self.r_thresh_q = 5  # Define a location to store the threshold info

        # # multiplication bc the readout is summed, so need common thing to compare to
        self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold[qTest] * self.readout_lengths_adc[qTest]))

        # Define a location to store a counter for how frequently the condj is triggered
        self.r_counter = 7
        self.safe_regwi(0, self.r_counter, 0)  # init counter val to 0
        
        
        self.sync_all(self.us2cycles(0.2))


    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        # phase reset
        self.reset_and_sync()

        #do the active reset
        if cfg.expt.active_reset:
            self.active_reset( man_reset= self.cfg.expt.man_reset, storage_reset= self.cfg.expt.storage_reset)

        #  prepulse
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

        # RF flux modulation
        self.wait_all(self.us2cycles(cfg.expt.length_placeholder))

        if self.cfg.expt.length_placeholder>0:

            self.setup_and_pulse(
                    ch=self.rf_ch[0],
                    style="flat_top",
                    freq=self.rf_freq_reg,
                    length=self.us2cycles(self.cfg.expt.length_placeholder),
                    phase=0,
                    gain=self.rf_gain_test, 
                    waveform="rf_test")
            self.sync_all(self.us2cycles(0.01))

        self.sync_all()  # align channels

        # post pulse
        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='post')
            

        # align channels and wait 50ns and measure
        self.sync_all(self.us2cycles(0.05))

        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

    # def collect_shots(self):
    #     # collect shots for the relevant adc and I and Q channels
    #     qTest = 0
    #     cfg=AttrDict(self.cfg)
    #     # print(np.average(self.di_buf[0]))
    #     shots_i0 = self.di_buf[0] / self.readout_lengths_adc[qTest]
    #     shots_q0 = self.dq_buf[0] / self.readout_lengths_adc[qTest]
    #     return shots_i0, shots_q0


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

        data = {"xpts": [], "idata": [], "qdata": [], "avgi": [], "avgq": []}

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
            if self.cfg.expt.active_reset:
                data["idata"].append(idata)
                data["qdata"].append(qdata)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
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
