# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Debug and scratch lines (dormant)
#
# Relocated verbatim from `measurement_notebooks/jonginn/qsim_experiments.ipynb`
# cells 263-285 ("Test lines" and "Sideband Chevron (TBD)") by the stage-2
# notebook decomposition. **Dormant**: hardware-poking scratch cells, a bare qick
# `LoopbackProgram`/`StoppingProgram` pair, register/frequency conversions, and an
# unfinished sideband chevron. No active caller.
#
# Like its sibling, this does not stand alone: it reads
# `sideband_stark_error_amp_runner` and `phase_expts` from what is now
# `floquet_calibration.py`, and `dm_sideband_scramble_defaults` /
# `sideband_scramble_preproc` from the bare-readout section. The runner in
# particular is a live object built against a connected station, so there is no
# import that would make these cells runnable on their own.
#
# Relocation only, per the stage-2 instructions. Sibling: `dark_mode.py`.

# %% [markdown]
# # Prepare

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy

import experiments as meas
from slab import AttrDict
from experiments import MultimodeStation, CharacterizationRunner, SweepRunner

from job_server import JobClient
from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager

# Initialize database and config manager
db = get_database()
config_dir = 'C:/python/multimode_expts/configs'
config_manager = ConfigVersionManager(config_dir)

# Initialize job client (handle submitting and waiting for jobs)
client = JobClient()

# Check server health
health = client.health_check()
print(f"Server status: {health['status']}")
print(f"Pending jobs: {health['pending_jobs']}")

# Initialize database and config manager
db = get_database()
config_dir = 'C:/python/multimode_expts/configs'
config_manager = ConfigVersionManager(config_dir)

# Initialize job client (handle submitting and waiting for jobs)
client = JobClient()

# Check server health
health = client.health_check()
client.print_queue()

# Who is running these experiments??
user = 'jonginn'

print(f"Welcome {user}!")

# The four aggregate MBR stages. `EncodingHamiltonianSpectroscopyExperiment`
# is still the loading layer and the shared numerics, and is still the class
# every job here was acquired under -- so it stays, and these four sit beside
# it. See analysis_notebooks/guan/MBR_analysis.py for the worked example.
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.deprecated.legacy_mbr import MBRSpectrumExperiment
from experiments.qsim.deprecated.legacy_mbr import MBROrthogonalityExperiment
from experiments.qsim.deprecated.legacy_mbr import MBRPropagatorExperiment

# %%
# Initialize station to retrieve soc and configs
config_dict = {'hardware_config': 'CFG-HW-20260904-00019',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20260904-00014',
 'floquet_storage_swap': 'CFG-FL-20260904-00042'}


station = MultimodeStation(
    user = user,
    experiment_name = "260818_qsim_spectroscopy",
    project = "EncSpec",
    log_measurements=True,
    
    storage_man_file = config_dict['man1_storage_swap'],
    hardware_config= config_dict['hardware_config'],
    floquet_file=config_dict['floquet_storage_swap'],
)

# %% [markdown]
# # Global Exp Config Setting

# %%
station.ds_storage.df

# %%
measurement_config_default_dict = {
    'avoid_yoko': False,
    'use_multiphoton_swap': False,
    
    
}

active_reset_default_dict = {
    "reset_dump_mode": 2,
    "dump_reset_iter_num": 1,
}

floquet_default_dict = {
    #For phase accumulation:
    "include_10cycles_buffer": True,
    "include_10cycles_buffer_in_pi_half" : True,
    
    # flat_top: legacy 3-segment pulse; preload_flattop: one preloaded arb envelope
    "floquet_waveform": "preload_flattop",
    "floquet_hardware_loop" : True,
    "scramble_sync_cycles" : 1,
    "palindrome_scramble": False,    
}

# %% [markdown]
# # Test lines

# %%
station.soccfg.reg2freq(station.soccfg.freq2reg(7600, gen_ch=4), gen_ch=4)

# %%
6881.28-718.72

# %%
station.soccfg.freq2reg(1210.24, gen_ch=4)

# %%
station.soccfg.reg2freq(station.soccfg.freq2reg(7600, gen_ch=1), gen_ch=1)

# %%
station.soccfg.reg2freq(station.soccfg.freq2reg(7000, gen_ch=1), gen_ch=1)+6389.76

# %%
station.hardware_cfg.hw.soc.dacs.flux_high.ch

# %%
station.hardware

# %%
from qick import AveragerProgram

# %%
class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg=self.cfg 
        res_ch = cfg["res_ch"]

        # set the nyquist zone
        self.declare_gen(ch=cfg["res_ch"], nqz=cfg["nyquist_zone"])
        
        # configure the readout lengths and downconversion frequencies (ensuring it is an available DAC frequency)
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=self.cfg["readout_length"],
                                 freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        # convert frequency to DAC frequency (ensuring it is an available ADC frequency)
        freq = self.freq2reg(cfg["pulse_freq"],gen_ch=res_ch) #, ro_ch=cfg["ro_chs"][0])
        phase = self.deg2reg(cfg["res_phase"], gen_ch=res_ch)
        gain = cfg["pulse_gain"]
        
        # self.default_pulse_registers(ch=res_ch, freq=freq, phase=phase, gain=gain)

        # self.set_pulse_registers(ch=res_ch, style=style, length=cfg["length"],freq=freq, phase=phase, gain=gain)
        self.synci(200)  # give processor some time to configure pulses
    
    def body(self):
        # fire the pulse
        # trigger all declared ADCs
        # pulse PMOD0_0 for a scope trigger
        # pause the tProc until readout is done
        # increment the time counter to give some time before the next measurement
        # (the syncdelay also lets the tProc get back ahead of the clock)
        cfg=self.cfg 
        res_ch = cfg["res_ch"]
        freq = self.freq2reg(cfg["pulse_freq"],gen_ch=res_ch)#, ro_ch=cfg["ro_chs"][0])
        phase = self.deg2reg(cfg["res_phase"], gen_ch=res_ch)
        gain = cfg["pulse_gain"]
        self.setup_and_pulse(ch = self.cfg["res_ch"],
                             style="const", 
                             length=cfg["length"],
                             freq=freq, 
                             phase=phase, 
                             gain=gain)
                             #,mode = "periodic")
        self.sync_all(10)
        self.measure(pulse_ch=self.cfg["res_ch"], 
                     adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


class StoppingProgram(LoopbackProgram):
    def body(self):
        self.set_pulse_registers(ch=self.cfg["res_ch"], style="const", length=16, mode = "oneshot")
        self.pulse(ch = self.cfg["res_ch"])

# %%
config={"res_ch":1, # --Fixed
        "ro_chs":[0], # --Fixed
        "reps":100, # --Fixed
        "relax_delay": 0,# --us
        "res_phase":0, # --degrees
        "pulse_style": "const", # --Fixed
        
        "length":1000, # [Clock ticks]
        "readout_length":10, # [Clock ticks]

        "pulse_gain":0, # [DAC units]
        "pulse_freq": 2250, # [MHz]
        "nyquist_zone": 1,
        
        "adc_trig_offset": 100, # [Clock ticks]
        "soft_avgs":50
       }

###################
# Try it yourself !
###################
prog =LoopbackProgram(station.soccfg, config)
iq_list = prog.acquire(station.im["Qick101"],
                       progress=True,
                       threshold=None,
                       load_pulses=True)
# prog.run_rounds(im["Qick101"], 
#                         progress=True)
# iq_list = prog.acquire_decimated(soc, progress=True)

# %%
station.ds_floquet.update_gain('M1-S1', 30000)

# %%
station.ds_floquet.df

# %%
stor_modes_to = [1] #list(range(1,8))
stor_modes_from = [6]
for iA, init_storA in enumerate(stor_modes_to): #range(1,8):
    for iB, init_storB in enumerate(stor_modes_from): #range(1,8):
        if init_storA == init_storB:
            continue
        print("Starting experiment for storage modes:", init_storA, "from", init_storB)

        qbe = sideband_stark_error_amp_runner.execute(
            stor_A=init_storA,
            stor_B=init_storB,
            relax_delay=0,
            reps=5000,
            include_10cycles_buffer = floquet_default_dict["include_10cycles_buffer"],
            advance_phases=np.linspace(-40, 40, 51).tolist(),
            n_pulses=[30],
            coupler_current = 0.0
        )
        phase_expts[init_storA - 1][init_storB - 1] = qbe

# %%
dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgram,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles = np.arange(0, 1001, step=100)

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2]
detunings = [0, 0] # None/False/unspecified all default to all zeros

scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

for meas_stor in meas_stors:
    scramble = dmscramble_runner.execute(
        reps=1000,
        init_fock=True,
        init_stor=0,
        ro_stor = meas_stor,
        relax_delay=0,
        active_reset=False,
        pre_relax_delay = 500,
        man_reset=True, 
        storage_reset = [2, 3], 
        reset_dump_mode = 1,
        dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
        swap_stors=swap_stors,
        update_phases=True,
        detunings=detunings,
        floquet_cycles=floquet_cycles,  
        swept_params=['floquet_cycle'],
        custom_prepulse = False,
        custom_postpulse = False,
        debug = False,
        swap_man_dark = True,
        dark_swap_order = [2, 3],
        second_rel_phase = 180,
        map_to_qubit_ge = True,
        prepulse = False, #for debugging. Should always be true
        postpulse = False,  # TODO(stage2): source cell Q278 omitted this comma
        coupler_current = 0#or debugging. Should always be true
        )
    scramble_expts.append(scramble)
    # scramble.display()

# %% [markdown]
# # Sideband Chevron (TBD)

# %%
# Storage Spectroscopy - New Pattern with CharacterizationRunner
from experiments.MM_dual_rail_base import MM_dual_rail_base

def get_storage_mode_parameters(ds_storage, config_thisrun, man_mode_no, stor_mode_no, man_photon_no = 1):
    """Get pulse parameters for a given storage mode."""
    stor_name = 'M' + str(man_mode_no) + '-S' + str(stor_mode_no)
    freq = ds_storage.get_freq(stor_name)
    gain = ds_storage.get_gain(stor_name)
    pi_len = ds_storage.get_pi(stor_name)
    h_pi_len = ds_storage.get_h_pi(stor_name)
    # flux_low_ch = config_thisrun.hw.soc.dacs.flux_low.ch
    # flux_high_ch = config_thisrun.hw.soc.dacs.flux_high.ch
    # ch = flux_low_ch if freq < 1000 else flux_high_ch
    ch = 'low' if freq < 1800 else 'high'

    mm_base_dummy = MM_dual_rail_base(config_thisrun, soccfg=station.soccfg)
    if man_photon_no == 1:
        prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
    else:
        prep_man_pi = []
        for i in range(int(man_photon_no)):
            prep_man_pi += [['multiphoton', f'g{i}-e{i}', 'pi', 0]]
            prep_man_pi += [['multiphoton', f'e{i}-f{i}', 'pi', 0]]
            prep_man_pi += [['multiphoton', f'f{i}-g{i + 1}', 'pi', 0]]
    prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist()

    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse

# %%
from datetime import datetime

# Configuration defaults for sideband sweep
sideband_chevron_defaults = AttrDict(dict(
    start=0.007, # start time in us
    pi_len_sweep=2.0, # total sweep length in us
    expts=25, # num steps of time
    reps=50,
    rounds=1,
    qubit=0,
    qubits=[0],
    man_mode_no=1,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    update_post_pulse_phase=[False, 0],
    relax_delay=2500,
))

def sideband_chevron_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    # Get storage mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_storage_mode_parameters(
        station.ds_storage, station.hardware_cfg, expt_cfg.man_mode_no, expt_cfg.stor_mode_no, expt_cfg.get("man_photon_no", 1)
    )
    print(expt_cfg.get("man_photon_no", 1))
    print(expt_cfg.get("debug", False))
    if expt_cfg.get("debug", False):
        print(prepulse)
    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'
    
    pi_len_sweep = expt_cfg.pi_len_sweep
    expt_cfg.step = pi_len_sweep / (expt_cfg.expts - 1)
    if 'gain' in expt_cfg and expt_cfg.gain is not None:
        gain = expt_cfg.gain  # Override gain if provided
    
    expt_cfg.flux_drive = [ch, freq, gain, 0]
    expt_cfg.pre_sweep_pulse = prepulse
    expt_cfg.post_sweep_pulse = postpulse
    
    print(f'Sideband chevron for {stor_name}: freq={freq:.3f} MHz, gain={gain}')
    
    return expt_cfg

def sideband_chevron_postproc(station, mother_expt):
    expt_cfg = mother_expt.cfg.expt
    print('mom', expt_cfg)
    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'

    from fitting.fit_display_classes import ChevronFitting

    chevron_analysis = ChevronFitting(
        frequencies=mother_expt.data['freq_sweep'],
        time=mother_expt.data['xpts'][0],
        response_matrix=mother_expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )

    chevron_analysis.analyze()
    
    best_freq = chevron_analysis.results.get('best_frequency_contrast')
        
    if best_freq and not expt_cfg.get("skip_update", False):
        print(f"Best frequency found: {best_freq:.4f} MHz")
        station.ds_storage.update_freq(stor_name, best_freq)
        print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz")
        pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
        station.ds_storage.update_pi(stor_name, pi_len)
        print('Updated the pi length to:', pi_len)
        station.ds_storage.update_h_pi(stor_name, pi_len / 2)
        print('Updated the h_pi length to:', pi_len / 2)
        station.ds_storage.update_gain(stor_name, expt_cfg.flux_drive[2])
        print('Updated gain to:', expt_cfg.flux_drive[2])
    mother_expt.analysis = chevron_analysis
    # station.snapshot_man1_storage_swap(update_main=False)

sideband_chevron_runner = SweepRunner(
    station=station,
    ExptClass=meas.single_qubit.sideband_general.SidebandGeneralExperiment,
    default_expt_cfg=sideband_chevron_defaults,
    sweep_param='freq',
    preprocessor=sideband_chevron_preproc,
    postprocessor=sideband_chevron_postproc,
    job_client=client
)

# %%
station.ds_storage.get_freq('M1-S2') - freq_span/2,

# %%
for stor_i in [2]: #range(1,8):
    stor_name = f'M1-S{stor_i}'

    freq_span = 0.03  # MHz
    freq_step = 0.003
    pi_len_sweep = 100
    # gain = station.ds_storage.get_gain(stor_name)
    gain = 995
    
    print(f'Running sideband chevron for {stor_name}')
    chevron_analysis = sideband_chevron_runner.execute(
        reps=50,
        stor_mode_no=stor_i,
        sweep_start= 518.6 - freq_span/2,#station.ds_storage.get_freq(stor_name) - freq_span/2,
        sweep_stop= 518.6 + freq_span/2,#station.ds_storage.get_freq(stor_name) + freq_span/2,
        sweep_npts=int(freq_span//freq_step + 1),
        pi_len_sweep=pi_len_sweep,
        expts=50, # num steps of time
        gain=gain,
        skip_update = True,
        man_photon_no = 1,
        batch = True,
        debug = True,
        
    )
    chevron_analysis.analysis.display_results(title = f"{chevron_analysis.fname}")

# %%
station.update_all_station_snapshots()

# %%
import inspect
print(inspect.getsource(sideband_chevron_runner.preprocessor))
