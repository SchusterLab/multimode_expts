import os
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
from MM_base import MMAveragerProgram


class SidebandScrambleProgram(MMAveragerProgram):
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
        stor_names = [f'M1-S{stor_no}' for stor_no in range(1,8)]
        self.m1s_freq_MHz = [self.swap_ds.get_freq(stor_name)+self.cfg.expt.detune for stor_name in stor_names]
        self.m1s_is_low_freq = [True]*4 + [False]*3
        self.m1s_ch = [self.flux_low_ch[qTest]]*4 + [self.flux_high_ch[qTest]]*3
        self.m1s_freq = [self.freq2reg(freq_MHz, gen_ch=ch) for freq_MHz, ch in zip(self.m1s_freq_MHz, self.m1s_ch)]
        self.m1s_length = [self.us2cycles(self.swap_ds.get_pi(stor_name), gen_ch=ch)
            for stor_name, ch in zip(stor_names, self.m1s_ch)]
        self.m1s_gain = [self.swap_ds.get_gain(stor_name)
            # if int(stor_name[-1])==self.cfg.expt.init_stor else 0
            for stor_name in stor_names]
        self.m1s_wf_name = ["pi_m1si_low"]*4 + ["pi_m1si_high"]*3


    def initialize(self):
        """
        Retrieves ch, freq, length, gain from csv for M1-Sx π/2 pulse and
        sets the waiting time and phase advance registers for the tau sweep
        """
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        self.swap_ds = storage_man_swap_dataset()
        self.retrieve_swap_parameters()

        self.m1s_kwargs = [{
                'ch': self.m1s_ch[stor],
                'style': 'flat_top',
                'freq': self.m1s_freq[stor],
                'phase': 0,
                'gain': self.m1s_gain[stor],
                'length': self.m1s_length[stor],
                'waveform': self.m1s_wf_name[stor],
        } for stor in range(7)]

        self.sync_all(200)


    def body(self):
        cfg=AttrDict(self.cfg)

        # initializations as necessary
        self.reset_and_sync()

        if self.cfg.expt.active_reset: 
            self.active_reset(
                man_reset=self.cfg.expt.man_reset,
                storage_reset= self.cfg.expt.storage_reset)

        init_stor = self.cfg.expt.init_stor
        ro_stor = self.cfg.expt.ro_stor

        # prepulse: ge -> ef -> f0g1
        prepules_cfg = [
            ['qubit', 'ge', 'pi', 0,],
            ['qubit', 'ef', 'pi', 0,],
            ['man', 'M1', 'pi', 0,],
            ['storage', f'M1-S{init_stor}', 'pi', 0,],
        ]
        pulse_creator = self.get_prepulse_creator(prepules_cfg)
        self.sync_all(self.us2cycles(0.1))
        self.custom_pulse(cfg, pulse_creator.pulse, prefix='pre_')
        self.sync_all(self.us2cycles(0.1))

        for kk in range(self.cfg.expt.floquet_cycle):
            for jj in range(7):
                if jj+1==self.cfg.expt.init_stor:
                    self.setup_and_pulse(**self.m1s_kwargs[jj])
                    self.sync_all(self.us2cycles(0.1))
                # else:
                #     self.sync_all(self.us2cycles(1.2))

        # postpulse
        if ro_stor > 0:
            postpules_cfg = [
                ['storage', f'M1-S{ro_stor}', 'pi', 0,],
                ['man', 'M1', 'pi', 0,],
            ]
        else:
            postpules_cfg = [
                ['man', 'M1', 'pi', 0,],
            ]
        pulse_creator = self.get_prepulse_creator(postpules_cfg)
        self.sync_all(self.us2cycles(0.1))
        self.custom_pulse(cfg, pulse_creator.pulse, prefix='post_')
        self.sync_all(self.us2cycles(0.1))

        self.measure_wrapper()



class SidebandScrambleExperiment(Experiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        expts: number experiments should be 1 here as we do soft loops
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qubits: this is just 0 for the purpose of the currrent multimode sample
        init_stor: storage to initialize the photon into (1-7)
        ro_stor: storage to readout the photon from (1-7)
        floquet_cycles: list of Floquet cycles to run
    )
    """
    def __init__(self, soccfg=None, path='', prefix='SidebandScramble',
                 config_file=None, expt_params=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)
        self.cfg.expt = AttrDict(expt_params)


    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        read_num = 4 if self.cfg.expt.active_reset else 1

        data = {
            'xpts': self.cfg.expt.floquet_cycles,
            'avgi': [],
            'avgq': [],
            'amps': [],
            'phases': [],
            'idata': [],
            'qdata': [],
        }
        for self.cfg.expt.floquet_cycle in tqdm(self.cfg.expt.floquet_cycles, disable=not progress):
            self.prog = SidebandScrambleProgram(soccfg=self.soccfg, cfg=self.cfg)

            avgi, avgq = self.prog.acquire(self.im[self.cfg.aliases.soc],
                                            threshold=None,
                                            load_pulses=True,
                                            progress=False,
                                            debug=debug,
                                            readouts_per_experiment=read_num)
            avgi, avgq = avgi[0][-1], avgq[0][-1]
            data['avgi'].append(avgi)
            data['avgq'].append(avgq)
            data['amps'].append(np.abs(avgi+1j*avgq)) # Calculating the magnitude
            data['phases'].append(np.angle(avgi+1j*avgq)) # Calculating the phase

            idata, qdata = self.prog.collect_shots()      
            data['idata'].append(idata)
            data['qdata'].append(qdata)

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
            pass
            # needs a post_select_averager to take care of the fact that expts=1 for averager
            # data['avgi'], data['avgq'] = post_select_raverager_data(data, self.cfg)

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

        title = self.fname.split(os.path.sep)[-1]

        plt.figure(figsize=(10,9))
        plt.subplot(211, 
            title=f"{title}",
            ylabel="I [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        plt.subplot(212, xlabel="# of cycles", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')

        plt.tight_layout()
        plt.show()


    def save_data(self, data=None):
        # do we really need to ovrride this?
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


