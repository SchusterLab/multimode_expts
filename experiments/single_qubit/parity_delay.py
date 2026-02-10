import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from fitting.fit_display_classes import GeneralFitting
from experiments.MM_base import *
class ParityDelayProgram(MMAveragerProgram):
    _pre_selection_filtering = True

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

        self.sync_all(200)


    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0] 

        # phase reset
        self.reset_and_sync()

        # active reset
        if cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        self.sync_all(self.us2cycles(0.2))

        # if cfg.expt.prepulse:
        #     creator = self.get_prepulse_creator(cfg.expt.pre_gate_sweep_pulse)
        #     self.custom_pulse(cfg, creator.pulse.tolist(), prefix = '')
        #     # self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

        if cfg.expt.prepulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')

        if cfg.expt.parity_fast:
            f_ge = cfg.device.multiphoton.hpi['gn-en']['frequency'][0]
            gain = cfg.device.multiphoton.hpi['gn-en']['gain'][0]
            sigma = cfg.device.multiphoton.hpi['gn-en']['sigma'][0]
            
            f_ge_reg = self.freq2reg(f_ge, gen_ch=self.qubit_chs[qTest])
            _sigma = self.us2cycles(sigma, gen_ch=self.qubit_chs[qTest])

            theta_2 =180 + cfg.expt.length_placeholder*2*np.pi*cfg.device.manipulate.revival_stark_shift[qTest]*180/np.pi # 180 degrees phase shift for the second half of the parity pulse
            # define the angle modulo 360
            theta_2 = theta_2 % 360
            print('theta_2:', theta_2)
            theta_2_reg = self.deg2reg(theta_2, self.qubit_chs[qTest])
            self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit_ge", sigma=_sigma, length=_sigma*4)
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=f_ge_reg, phase=self.deg2reg(0), gain=gain, waveform="hpi_qubit_ge")
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="const", freq=f_ge_reg, phase=self.deg2reg(0), gain=0, length=self.us2cycles(cfg.expt.length_placeholder, gen_ch=self.qubit_chs[qTest]))
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=f_ge_reg, phase=theta_2_reg, gain=gain, waveform="hpi_qubit_ge")

        else:
            # self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[qTest], phase=self.deg2reg(0), gain=self.hpi_ge_gain, waveform="hpi_qubit_ge")
            # self.setup_and_pulse(ch=self.qubit_chs[qTest], style="const", freq=self.f_ge_reg[qTest], phase=self.deg2reg(0), gain=0, length=self.us2cycles(cfg.expt.length_placeholder, gen_ch=self.qubit_chs[qTest]))
            # self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[qTest], phase=self.deg2reg(180, self.qubit_chs[qTest]), gain=self.hpi_ge_gain, waveform="hpi_qubit_ge")

            parity_str = self.get_parity_str()
            parity_str[1][1] = cfg.expt.length_placeholder  # set the wait time in the parity string
            parity_pulse = self.get_prepulse_creator(parity_str)
            self.custom_pulse(cfg, parity_pulse.pulse.tolist())
        # self.wait_all(self.us2cycles(0.01)) # wait for the time stored in the wait variable register

        self.measure_wrapper()


class ParityDelayExperiment(Experiment):
    """
    ParityDelay Experiment
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
        q_ind = self.cfg.expt.qubits[0]
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


        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

        # Calculate read_num to account for active_reset measurements
        read_num = 1
        if self.cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[], "idata":[], "qdata":[]}

        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            lengthrabi = ParityDelayProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug,
                                            readouts_per_experiment=read_num )      
            idata, qdata = lengthrabi.collect_shots()
            data["idata"].append(idata)
            data["qdata"].append(qdata)

            if self.cfg.expt.active_reset and self.cfg.expt.get('pre_selection_reset', False):
                avgi_val, avgq_val = GeneralFitting.filter_shots_per_point(
                    idata, qdata, read_num,
                    threshold=self.cfg.device.readout.threshold[self.cfg.expt.qubits[0]],
                    pre_selection=True)
            else:
                avgi_val = avgi[0][-1]
                avgq_val = avgq[0][-1]

            amp = np.abs(avgi_val+1j*avgq_val)
            phase = np.angle(avgi_val+1j*avgq_val)
            data["xpts"].append(length)
            data["avgi"].append(avgi_val)
            data["avgq"].append(avgq_val)
            data["amps"].append(amp)
            data["phases"].append(phase)                             

        
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

        plt.figure(figsize=(10,10))
        plt.subplot(211, title="Parity Delay", ylabel="I [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')

        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
