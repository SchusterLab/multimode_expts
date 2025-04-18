import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import yaml

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import experiments.fitting as fitter
from MM_base import *

from dataset import storage_man_swap_dataset


"""
Do a prepulse, then floquet drives, then a postpulse, then measure
"""

def get_swap_params(storage_no, dataset):
    df = dataset.df
    pulse_params = df[df['stor_name']==f'M1-S{storage_no}']
    pulse_freq = pulse_params['freq (MHz)'].values[0]  
    pulse_gain = pulse_params['gain (DAC units)'].values[0]
    pulse_pi = pulse_params['pi (mus)'].values[0]
    pulse_hpi = pulse_params['h_pi (mus)'].values[0]
    return pulse_freq, pulse_gain, pulse_pi, pulse_hpi

class FloquetGeneralProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)


    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.dataset = storage_man_swap_dataset(self.cfg.device.storage.storage_man_file)

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
        # self.flux_storage_ch = cfg.hw.soc.dacs.flux_storage.ch
        # self.flux_storage_ch_type = cfg.hw.soc.dacs.flux_storage.type
        self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
        self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type


        # get register page for qubit_chs
        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]

        self.f_ge_reg = [self.freq2reg(
            cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])]
        self.f_ef_reg = [self.freq2reg(
            cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])]

        # self.f_ge_resolved_reg = [self.freq2reg(
        #     self.cfg.expt.qubit_resolved_pi[0], gen_ch=self.qubit_chs[qTest])]

        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(
            cfg.device.readout.frequency, self.res_chs, self.adc_chs)]

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

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(
            cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ge value
        self.pisigma_ef = self.us2cycles(
            cfg.device.qubit.pulses.pi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ef value
        # self.pisigma_resolved = self.us2cycles(
        #     self.cfg.expt.qubit_resolved_pi[3], gen_ch=self.qubit_chs[qTest])  # default resolved pi value

        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.f_ef_init_reg = self.f_ef_reg[qTest]
        # self.f_ge_resolved_int_reg = self.f_ge_resolved_reg[qTest]

        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.gain_ef_init = self.cfg.device.qubit.pulses.pi_ef.gain[qTest]

        # add qubit pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test_ramp", sigma=self.pi_test_ramp,
                       length=self.pi_test_ramp*2*cfg.device.qubit.ramp_sigma_num[qTest])
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge",
                       sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ef",
                       sigma=self.pisigma_ef, length=self.pisigma_ef*4)
        # self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_resolved",
        #                sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
        
        # add all gauss pulses for floquet drives
        
        self.add_gauss(ch=1, name=f'M1S_low', 
                        sigma=self.us2cycles(0.05), length=self.us2cycles(0.3))
        self.add_gauss(ch=3, name=f'M1S_high', 
                        sigma=self.us2cycles(0.05), length=self.us2cycles(0.3)) 

        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(
            cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        self.sync_all(self.us2cycles(0.2))
    

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        # actiev reset 
        if self.cfg.expt.active_reset:
            self.active_reset(man_reset=self.cfg.expt.man_reset, storage_reset=self.cfg.expt.storage_reset)

        #  prepulse
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')
            self.sync_all()

        # do something in the middle
        current_stor_no = self.cfg.expt.init_stor_no
        for floquet_reps in range(self.cfg.expt.floquet_cycles):
            # self.sync_all(self.us2cycles(0.02*7))

            # swap back into man
            for storage_no in range(1,7): 
                floquet_pulse_freq, floquet_pulse_gain, floquet_pi_length, floquet_hpi_length = get_swap_params(storage_no, self.dataset)
                if floquet_pulse_freq < 1000:
                    floquet_ch = 1
                    wf_name = 'M1S_low'
                else:
                    floquet_ch = 3
                    wf_name = 'M1S_high'
                self.setup_and_pulse(ch=floquet_ch, style="flat_top", 
                                    waveform=wf_name,
                                    freq=self.freq2reg(floquet_pulse_freq, gen_ch=floquet_ch),
                                    length=self.us2cycles(floquet_pi_length/10),
                                    gain=0,
                                    phase=0, phrst=1) 
                self.sync_all()  # align channels

            # floquet_pulse_freq, floquet_pulse_gain, floquet_pi_length, floquet_hpi_length = get_swap_params(current_stor_no%7+1, self.dataset)
            # if floquet_pulse_freq < 1000:
            #     floquet_ch = 1
            #     wf_name = 'M1S_low'
            # else:
            #     floquet_ch = 3
            #     wf_name = 'M1S_high'
            # self.setup_and_pulse(ch=floquet_ch, style="flat_top", 
            #                     waveform=wf_name,
            #                     freq=self.freq2reg(floquet_pulse_freq, gen_ch=floquet_ch),
            #                     length=self.us2cycles(floquet_pi_length/10),
            #                     gain=floquet_pulse_gain,
            #                     phase=0, phrst=1) 
            # self.sync_all()  # align channels

            # current_stor_no = current_stor_no%7 + 1
        
        # post pulse
        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='post')
            self.sync_all()

        # qubit resolved pi pulse
        # self.setup_and_pulse(ch=self.qubit_chs[qTest], style="flat_top", freq=self.f_ge_resolved_int_reg, length=self.us2cycles(self.cfg.expt.qubit_resolved_pi[2]),
        #                          phase=0, gain=self.cfg.expt.qubit_resolved_pi[1], waveform="pi_qubit_resolved")
        self.sync_all()

        # align channels and wait 50ns and measure
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )


class FloquetGeneralExperiment(Experiment):
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

    def __init__(self, soccfg=None, path='', prefix='FloquetGeneral', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix,
                         config_file=config_file, progress=progress)
        
        with open(self.config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        self.yaml_cfg = AttrDict(yaml_cfg)
        self.mm_base = MM_base(cfg = self.yaml_cfg) # so that we can use the pre/post-pulse creator


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


        data = {"xpts": [], "ypts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4


        # init_stor_no = self.cfg.expt.init_storage
        # print('Init storage number: ', init_stor_no)

        data['xpts'] = np.arange(1, 8)
        data['ypts'] = np.arange(1, 8)
        for init_stor_no in tqdm(range(1,8), disable=not progress):
            self.cfg.expt.init_stor_no = init_stor_no
            pre_sweep_pulse_str = [
                ['qubit', 'ge', 'pi',0],
                ['qubit', 'ef', 'pi',0],
                ['man', 'M1' , 'pi',0],
                ['storage', f'M1-S{init_stor_no}', 'pi',0]]
            self.cfg.expt.pre_sweep_pulse = self.mm_base.get_prepulse_creator(pre_sweep_pulse_str).pulse.tolist()
            
            for ro_stor_no in range(1,8):
                self.cfg.expt.ro_stor_no = ro_stor_no
                post_sweep_pulse_str = [
                    ['storage', f'M1-S{ro_stor_no}', 'pi',0],
                    ['man', 'M1' , 'pi',0]]
                self.cfg.expt.post_sweep_pulse = self.mm_base.get_prepulse_creator(post_sweep_pulse_str).pulse.tolist()

                qprog = FloquetGeneralProgram(
                    soccfg=self.soccfg, cfg=self.cfg)
                self.prog = qprog
                avgi, avgq = qprog.acquire(
                    self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug,
                    readouts_per_experiment=read_num)
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq)  # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq)  # Calculating the phase
                idata, qdata = qprog.collect_shots()
                
                data["avgi"].append(avgi)
                data["avgq"].append(avgq)
                data["amps"].append(amp)
                data["phases"].append(phase)

        for k, a in data.items():
            data[k] = np.array(a)
            if 'pts' not in k:
                data[k] = data[k].reshape((7,7))

        self.data = data

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
