import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qick import QickConfig
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.dataset import FloquetStorageSwapDataset
from experiments.qsim.utils import (
    ensure_list_in_cfg,
    post_select_raverager_data,
)
from fitting.fit_utils import guess_freq
from experiments.MM_base import MMAveragerProgram


class QsimBaseProgram(MMAveragerProgram):
    """
    First initialize a photon into man1 by qubit ge, qubit ef, f0g1 
    Then (optionally) swap into init_stor
    Then do whatever in the core_pulses() that you override
    Finally swap ro_stor back into man and then man into qb and readout
    """
    def __init__(self, soccfg: QickConfig, cfg: AttrDict):
        super().__init__(soccfg, cfg)


    def retrieve_swap_parameters(self):
        """
        retrieve pulse parameters for the M1-Sx swap
        """
        qTest = self.qubits[0]
        stor_names = [f'M1-S{stor_no}' for stor_no in range(1,8)]
        self.m1s_pi_fracs = [self.swap_ds.get_pi_frac(stor_name) for stor_name in stor_names]
        self.m1s_freq_MHz = [self.swap_ds.get_freq(stor_name) for stor_name in stor_names]
        self.m1s_is_low_freq = [True]*4 + [False]*3
        self.m1s_ch = [self.flux_low_ch[qTest]]*4 + [self.flux_high_ch[qTest]]*3
        self.m1s_freq = [self.freq2reg(freq_MHz, gen_ch=ch) for freq_MHz, ch in zip(self.m1s_freq_MHz, self.m1s_ch)]
        self.m1s_length = [self.us2cycles(self.swap_ds.get_len(stor_name), gen_ch=ch)
            for stor_name, ch in zip(stor_names, self.m1s_ch)]
        self.m1s_gain = [self.swap_ds.get_gain(stor_name) for stor_name in stor_names]
        self.m1s_wf_name = ["pi_m1si_low"]*4 + ["pi_m1si_high"]*3


    def displace_man(self, alpha=None, setup=False, play=False):
        # This function must be called first with setup before calling it with play
        if setup:
            self.gain2alpha = self.cfg.device.manipulate.gain_to_alpha[self.man_mode_idx]
            self.f_man = self.freq2reg(
                self.cfg.device.manipulate.f_ge[self.man_mode_idx],
                gen_ch=self.man_ch[self.man_mode_idx]
                )

            self.displace_sigma = self.us2cycles(
                self.cfg.device.manipulate.displace_sigma[self.man_mode_idx],
                gen_ch=self.man_ch[self.man_mode_idx]
                )
            self.add_gauss(
                ch=self.man_ch[self.man_mode_idx],
                name="displace",
                sigma=self.displace_sigma,
                length=self.displace_sigma*4
                )
            
        if play:
            assert alpha is not None

            _alpha = np.conj(alpha) # convert to conjugate to respect qick convention
            gain =  int(np.abs(_alpha)/self.gain2alpha)
            phase = np.angle(_alpha)/np.pi*180 - 90 # 90 is needed since da/dt = -i*drive
            self.setup_and_pulse(
                ch=self.man_ch[self.man_mode_idx],
                style="arb",
                freq=self.f_man,
                phase=self.deg2reg(phase, gen_ch=self.man_ch[self.man_mode_idx]),
                gain=gain,
                waveform="displace")


    def initialize(self):
        """
        MM_base_init to pull basic info 
        Retrieves ch, freq, length, gain from csv for M1-Sx π/2 pulses
        """
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        #TODO: this should use a config key to determine whether
        # to use floquet or gate (pi or pi/2) datasets
        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()

        self.man_mode_idx = 1
        self.m1s_kwargs = [{
                'ch': self.m1s_ch[stor],
                'style': 'flat_top',
                'freq': self.m1s_freq[stor],
                'phase': 0,
                'gain': self.m1s_gain[stor],
                'length': self.m1s_length[stor],
                'waveform': self.m1s_wf_name[stor],
        } for stor in range(7)]

        if self.cfg.expt.perform_wigner:
            self.displace_man(setup=True, play=False)

        self.sync_all(200)


    def core_pulses(self):
        """
        Override this method to control what happens in between pre and post pulses
        so that for most experiments we don't need to override the body method
        """
        # eg:
        # self.setup_and_pulse(**self.m1s_kwargs[0])
        self.sync_all(self.us2cycles(0.1))
    

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
        # TODO: make this overridable from cfg
        if cfg.expt.prepulse:

            if type(init_stor) is int:
                init_stor = [init_stor]
            if type(init_stor) is not list:
                raise ValueError("init_stor must be int or list of int")

            if cfg.expt.init_fock:

                prepulse_cfg = []
                for each_init_stor in init_stor:
                    prepulse_cfg += [
                        ['qubit', 'ge', 'pi', 0,],
                        ['qubit', 'ef', 'pi', 0,], # qubit in f
                        ['man', 'M1', 'pi', 0,], # f0-g1 --> man in 1
                    ]
                    if each_init_stor > 0:
                        prepulse_cfg.append(['storage', f'M1-S{each_init_stor}', 'pi', 0,])

                pulse_creator = self.get_prepulse_creator(prepulse_cfg)
                self.sync_all()
                self.custom_pulse(cfg, pulse_creator.pulse, prefix='pre_')
                self.sync_all()


            else: # init in coherent state

                assert 'init_alpha' in cfg.expt and cfg.expt.init_alpha

                for each_init_stor in init_stor:
                    self.displace_man(
                        alpha=cfg.expt.init_alpha,
                        setup=False,
                        play=True,
                    ) 
            
                    if each_init_stor > 0:
                        prepulse_cfg = ['storage', f'M1-S{each_init_stor}', 'pi', 0,]

                        pulse_creator = self.get_prepulse_creator(prepulse_cfg)
                        self.sync_all()
                        self.custom_pulse(cfg, pulse_creator.pulse, prefix=f'pre_{each_init_stor}_')
                        self.sync_all()

        # core pulses: override the method to define your own expeirment
        self.core_pulses()

        # postpulse
        if cfg.expt.postpulse:

            # Move ro_stor to man
            postpulse_cfg = [ ['storage', f'M1-S{ro_stor}', 'pi', 0,] ] if ro_stor > 0 else []

            if not self.cfg.expt.perform_wigner:
                # Move man to qubit for population measurement
                postpulse_cfg.append(['man', 'M1', 'pi', 0,])

            pulse_creator = self.get_prepulse_creator(postpulse_cfg)
            self.sync_all()
            self.custom_pulse(cfg, pulse_creator.pulse, prefix='post_')
            self.sync_all()

            if self.cfg.expt.perform_wigner:
                # Population is still in man, perform displacement + parity measurement

                # Displacement
                self.displace_man(
                    alpha=cfg.expt.wigner_alpha,
                    setup=False,
                    play=True,
                    )
                
                # Parity pulse on qubit
                self.play_parity_pulse(self.man_mode_idx, second_phase=self.cfg.expt.phase_second_pulse, fast=self.cfg.expt.parity_fast)

        self.measure_wrapper()


class QsimBaseExperiment(Experiment):
    """
    Sweep 1 or 2 parameters in cfg.expt
    Experimental Config:
    expt = dict(
        expts: number experiments should be 1 here as we do soft loops
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qubits: this is just 0 for the purpose of the currrent multimode sample
        init_stor: storage to initialize the photon into (0-7)
        ro_stor: storage to readout the photon from (0-7)
        active_reset, man_reset, storage_reset: bool
        swept_params: list of parameters to sweep, e.g. ['detune', 'gain']
    )
    In principle this overlaps with qick.NDAveragerProgram, but this allows you to
    skip writing new expeirment classes or at least acquire() while doing 
    more general sweeps than just a qick register, incl nonlinear steps.
    Consider doing NDAverager or RAverager if there's speed advantage.
    See notebook for usage.
    """
    def __init__(self, soccfg=None, path='', prefix=None,
                 config_file=None, expt_params=None,
                 program=None, progress=None, **kwargs):
        """
        program can be:
        - A class object (the class you imported, not an instance)
        - A tuple of (module_path, class_name) strings
        - None (defaults to QsimBaseProgram)
        """
        if not prefix:
            prefix = self.__class__.__name__
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, **kwargs)
        self.cfg.expt = AttrDict(expt_params)

        # Store program class info as strings (pickle-safe)
        if program is None:
            # Default to QsimBaseProgram
            self.program_module = QsimBaseProgram.__module__
            self.program_class = QsimBaseProgram.__name__
        elif isinstance(program, tuple) and len(program) == 2:
            # Program passed as (module, class_name) tuple
            self.program_module, self.program_class = program
        else:
            # Program passed as class object - extract module and name
            self.program_module = program.__module__
            self.program_class = program.__name__

        self.cfg.expt.QickProgramName = self.program_class

        # ProgramClass is loaded lazily when needed
        self._ProgramClass = None

    @property
    def ProgramClass(self):
        """Lazy load the program class when first accessed."""
        if self._ProgramClass is None:
            import importlib
            module = importlib.import_module(self.program_module)
            self._ProgramClass = getattr(module, self.program_class)
        return self._ProgramClass


    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        read_num = 4 if self.cfg.expt.active_reset else 1

        assert len(self.cfg.expt.swept_params) in {1,2}, "can only handle 1D and 2D sweeps for now"
        sweep_dim = 2 if len(self.cfg.expt.swept_params) == 2 else 1

        if 'perform_wigner' not in self.cfg.expt:
            self.cfg.expt.perform_wigner = False

        outer_param = self.cfg.expt.swept_params[0]
        outer_params = self.cfg.expt[outer_param+'s']
        if sweep_dim == 2:
            inner_param = self.cfg.expt.swept_params[1]
            inner_params = self.cfg.expt[inner_param+'s']
        else:
            inner_param = 'dummy'
            inner_params = [None]  # Dummy value for single parameter sweep
        self.outer_param, self.inner_param = outer_param, inner_param

        data = {
            'avgi': [], 'avgq': [],
            'amps': [], 'phases': [],
            'idata': [], 'qdata': [],
        }
        if sweep_dim == 2:
            data['xpts'] = inner_params
            data['ypts'] = outer_params
        else:
            data['xpts'] = outer_params

        for self.cfg.expt[outer_param] in tqdm(outer_params, disable=not progress):
            for self.cfg.expt[inner_param] in inner_params:
                self.prog = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)

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


    # def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
    #     pass


    def display(self, data=None, fit=True, **kwargs):
        #TODO: might want to add capability to plot custom keys
        # such as extra_data_keys=['best_fit']
        if data is None:
            data=self.data

        title = self.fname.split(os.path.sep)[-1]

        if 'ypts' in data.keys(): # guess if this is 2D or 1D
            fig, axs = plt.subplots(2, 1, figsize=(10, 9))
            axs[0].set_title(title)
            mesh = axs[0].pcolormesh(data['xpts'], data['ypts'], data['avgi'])
            fig.colorbar(mesh, ax=axs[0], label='I [ADC level]')
            mesh = axs[1].pcolormesh(data['xpts'], data['ypts'], data['avgq'])
            fig.colorbar(mesh, ax=axs[1], label='Q [ADC level]')
            try:
                xlabel, ylabel = self.inner_param, self.outer_param
            except AttributeError:
                try:
                    ylabel, xlabel = self.cfg.swept_params
                except AttributeError:
                    try:
                        ylabel, xlabel = self.cfg.expt.swept_params
                    except Exception as e:
                        print("Couldn't get x and y labels automatically:", e)
                        xlabel, ylabel = None, None
            for ax in axs:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
        else:
            try:
                xlabel = self.outer_param
            except AttributeError:
                xlabel = self.cfg.expt.swept_params[0]
            except Exception as e:
                print("Couldn't get x label automatially", e)
            fig, axs = plt.subplots(2, 1, figsize=(10, 9))
            ax = axs[0]
            ax.set_title(title)
            ax.set_ylabel("I [ADC level]")
            ax.plot(data["xpts"], data["avgi"],'o-')
            ax = axs[1]
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Q [ADC level]")
            ax.plot(data["xpts"], data["avgq"],'o-')
        return fig, axs


    def save_data(self, data=None):
        # do we really need to ovrride this?
        # TODO: at least make this save line-by-line
        temp_cfg = deepcopy(self.cfg)
        if "ds_floquet" in self.cfg:
            self.cfg.pop('ds_floquet')  # remove the dataset object from cfg before saving otherwise json gets mad
        if "ds_floquet" in self.cfg.expt:
            self.cfg.expt.pop('ds_floquet')  # remove the dataset object from cfg before saving otherwise json gets mad
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        self.cfg = temp_cfg
        return self.fname
