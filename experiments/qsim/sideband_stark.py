import os

import matplotlib.pyplot as plt
import numpy as np
from qick import QickConfig
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from dataset import storage_man_swap_dataset
from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.qsim.utils import (
    ensure_list_in_cfg,
    guess_freq,
    post_select_raverager_data,
)


class SidebandStarkProgram(QsimBaseProgram):
    """
    First initialize a photon into man1 by qubit ge, qubit ef, f0g1 
    Then do a rabi on the sideband
    """

    def core_pulses(self):
        m1s_kwarg = self.m1s_kwargs[self.cfg.expt.init_stor-1]
        m1s_kwarg['freq'] += self.freq2reg(self.cfg.expt.detune, gen_ch=m1s_kwarg['ch'])

        # first hpi
        self.setup_and_pulse(**m1s_kwarg)

        self.sync_all(self.us2cycles(self.cfg.expt.wait))

        # second hpi with updated phase
        m1s_kwarg.update({
            'phase': self.deg2reg(self.cfg.expt.phase),
        })
        self.setup_and_pulse(**m1s_kwarg)

        self.sync_all(self.us2cycles(0.1))


class SidebandStarkExperiment(QsimBaseExperiment):
    """
    Sweep amplitude vs detuning
    Experimental Config:
    expt = dict(
        expts: number experiments should be 1 here as we do soft loops
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qubits: this is just 0 for the purpose of the currrent multimode sample
        init_stor: storage to initialize the photon into (1-7)
        ro_stor: storage to readout the photon from (1-7)
    )
    """
    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        read_num = 4 if self.cfg.expt.active_reset else 1

        assert len(self.cfg.expt.swept_params) in {1,2}, "can only handle 1D and 2D sweeps for now"
        sweep_dim = 2 if len(self.cfg.expt.swept_params) == 2 else 1

        outer_param = self.cfg.expt.swept_params[0]
        outer_params = self.cfg.expt[outer_param+'s']
        if sweep_dim == 2:
            inner_param = self.cfg.expt.swept_params[1]
            inner_params = self.cfg.expt[inner_param+'s']
        else:
            inner_param = 'dummy'
            inner_params = [None]  # Dummy value for single parameter sweep

        data = {
            'xpts': inner_params,
            'ypts': outer_params,
            'avgi': [],
            'avgq': [],
            'amps': [],
            'phases': [],
            'idata': [],
            'qdata': [],
        }

        for self.cfg.expt[outer_param] in tqdm(outer_params, disable=not progress):
            for self.cfg.expt[inner_param] in inner_params:
                self.prog = SidebandStarkProgram(soccfg=self.soccfg, cfg=self.cfg)

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
        for key in 'avgi avgq amps phases'.split():
            data[key] = np.array(data[key])
            if sweep_dim == 2:
                data[key] = np.reshape(data[key], (len(outer_params), len(inner_params)))

        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        self.data=data
        return data


