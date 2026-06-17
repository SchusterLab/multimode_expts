# ---
# jupyter:
#   jupytext:
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
# # Prepare

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy
import yaml

import experiments as meas
from slab import AttrDict
from experiments import MultimodeStation, CharacterizationRunner, SweepRunner
from experiments.branch_manager import BranchManager

from job_server import JobClient
from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager

# %%
# Initialize database and config manager
db = get_database()
config_dir = 'C:/python/multimode_expts/configs'
config_manager = ConfigVersionManager(config_dir)

# Initialize job client (handle submitting and waiting for jobs)
client = JobClient()

# Check server health
health = client.health_check()
client.print_queue()

# %%
# Who is running these experiments??
user = 'guan'

print(f"Welcome {user}!")

# %%
# Initialize station to retrieve soc and configs
station = MultimodeStation(
    user = user,
    experiment_name = "260522_qsim",

    hardware_config="CFG-HW-20260528-00032",
    storage_man_file="CFG-M1-20260528-00055",
    floquet_file="C:/python/multimode_expts/configs/versions/floquet_storage_swap/CFG-FL-20260216-00024.csv",
    # multiphoton_config="C:/python/multimode_expts/configs/versions/multiphoton_config/CFG-MP-20260115-00001.yml",
    log_measurements=True,
)

# %% [markdown]
# ## Config branches

# %%
bm = BranchManager(station)

# %%
bm.list()

# %%
bm.commit('coupler0.5', note='found f0g1')

# %%
bm.branch('coupler0.5', note='branching off from 0.3')

# %%
bm.checkout('coupler0.05', force=True)

# %% [markdown]
# ## Manually update entries

# %% [markdown]
# ### Readout 

# %%
with open(station.config_dir / 'versions/hardware_config/CFG-HW-20260527-00047.yml') as f:
    rocfg = AttrDict(yaml.safe_load(f))
rocfg.device.readout

# %%
station.hardware_cfg.device.readout = rocfg.device.readout

# %%
station.snapshot_hardware_config()

# %% [markdown]
# ### Coupler current

# %%
station.hardware_cfg.hw.yoko_coupler.current = 0.5e-3

station.snapshot_hardware_config()

# %%
station.yoko_coupler.ramp_current(0, 2e-4)

# %% [markdown]
# ## Experiments to run
#
# Depending on stage of cooldown, we will run a different sequence of calibration experiments. For example, amplitude rabi don't need to be updated every time, but the frequency correction from T2 is important to do every day. In the dictionary experiments to run, we will speciy the experiments we want to run. 

# %%
expts_to_run = {# readout 
                'res_spec': True, # Readout spectroscopy
                'single_shot': True, 
                # qubit ge 
                'pulse_probe_ge': True,
                't2_ge': True, 
                'amplitude_ge': True,
                't1_ge': True,
                # qubit ef
                'pulse_probe_ef': True,
                't2_ef': True,
                'amplitude_ef': True,
                't1_ef': True,

                # manipulate 
                'man_modes': [1], # [1,2] if also want to run mode 2
                'pulse_probe_f0g1': True,
                'length_rabi_sweep':True,
                'length_rabi':False, # this will run automatically if the length_rabi_sweep is set to True
                'chi_ge': True, 
                'chi_ef': True,
                'RB': False,

                #storage
                'stor_modes': [1,2,3,4,5,6,7], # [1,2, .., 7] if also want to run  all modes 
                # 'stor_modes': [3, 4, 5], # [1,2, .., 7] if also want to run  all modes 
                'stor_spectroscopy': True,
                'sideband_freq_sweep': True,
                'sideband_length_rabi': True,
                # 'storage_t1': True
                }

# %% [markdown]
# # Qubit characterization

# %% [markdown]
# ## Resonator spectroscopy

# %% [markdown]
# Fitting parameters are wrong because of using the hanger function (more or less reflection/2), instead of transmission. Is this an easy fix?

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
resspec_defaults = AttrDict(dict(
    # start = center - span / 2, 
    # step = span / expts, # min step ~1 Hz
    # center and span can be user supplied or use defaults in preproc
    expts = 250, # Number experiments stepping from start
    reps = 500, # Number averages per point
    relax_delay = 50, # us
    pulse_e = False, # add ge pi pulse prior to measurement
    pulse_f = False, # add ef pi pulse prior to measurement
    pulse_cavity = False,  # prepulse on cavity prior to measurement (False also disables next line)
    cavity_pulse = [4984.373226159381, 800, 2, 0], # [frequency, gain, length, phase]  const pulse
    qubit = 0,
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def resspec_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)

    span = kwargs.pop('span', 1.5)  # MHz
    center = kwargs.pop('center', station.hardware_cfg.device.readout.frequency[0])
    expts = kwargs.get('expts', default_expt_cfg.expts)

    expt_cfg.start = center - span / 2
    expt_cfg.step = span / expts
    expt_cfg.update(kwargs)
    station.snapshot_hardware_config(update_main=False)
    return expt_cfg

def resspec_postproc(station, expt):
    old_freq = station.hardware_cfg.device.readout.frequency[0]
    station.hardware_cfg.device.readout.frequency = [expt.data['fit'][0]]
    print(f'Updated readout frequency from {old_freq} to {expt.data["fit"][0]}!')



# %%
# Execute
# =================================
rspec_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.ResonatorSpectroscopyExperiment,
    default_expt_cfg = resspec_defaults,
    preprocessor = resspec_preproc,
    postprocessor = None, # resspec_postproc, # uncomment if you want to update the readout freq from this expt
    job_client=client,
)

if expts_to_run['res_spec']:
    rspec = rspec_runner.execute(span=5)
    rspec.display()

# %%
# station.preview_config_update()

# %% [markdown]
# ## Single Shot

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
singleshot_defaults = AttrDict(dict(    
    reps=5000,
    relax_delay=500,
    check_f=False,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    qubit=0,
    pulse_manipulate=False,
    cavity_freq=4984.373226159381,
    cavity_gain=400,
    cavity_length=2,
    prepulse=False,
    pre_sweep_pulse=None,
    gate_based=True,
    qubits=[0],
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def singleshot_postproc(station, expt):
    expt.analyze(plot=False, station=station, subdir=station.autocalib_path)
    fids = expt.data['fids']
    confusion_matrix = expt.data['confusion_matrix']
    thresholds_new = expt.data['thresholds']
    angle = expt.data['angle']
    print(fids)

    hardware_cfg = station.hardware_cfg
    hardware_cfg.device.readout.phase = [hardware_cfg.device.readout.phase[0] + angle]
    hardware_cfg.device.readout.threshold = thresholds_new
    hardware_cfg.device.readout.threshold_list = [thresholds_new]
    hardware_cfg.device.readout.Ie = [np.median(expt.data['Ie_rot'])]
    hardware_cfg.device.readout.Ig = [np.median(expt.data['Ig_rot'])]
    if expt.cfg.expt.active_reset:
        hardware_cfg.device.readout.confusion_matrix_with_active_reset = confusion_matrix
    else:
        hardware_cfg.device.readout.confusion_matrix_without_reset = confusion_matrix
    print('Updated readout!')


# %%
# Execute
# =================================
ss_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.HistogramExperiment,
    default_expt_cfg = singleshot_defaults,
    postprocessor = singleshot_postproc,
    job_client=client,
)

ss = ss_runner.execute(
    go_kwargs=dict(analyze=False, display=False),
    check_f=False,
    active_reset=False, # on recalibration of readout, turn off active reset because it will be wrong for selecting when to apply the qubit pulse
    relax_delay=2000,
    # active_reset=True,
    # relax_delay=200,
    # coupler_current=coupler_current,
    # priority=1,
)

# %%
station.preview_config_update()

# %%
station.snapshot_hardware_config(update_main=False)


# %% [markdown]
# ## Qubit ge

# %% [markdown]
# ### Pulse-probe

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
gespec_defaults = AttrDict(dict(    
    # start=center-span/2,  # [MHz]
    # step=span/expts,  # min step ~1 MHz
    expts=200,  # Number of experiments stepping from start
    reps=100,  # Number of averages per point
    rounds=1,  # Number of start to finish sweeps to average over
    length=1,  # Qubit probe constant pulse length [us]
    gain=100,  # Qubit pulse gain
    sigma=0.1,  # Qubit flat top sigma
    qubit=[0],
    qubits=[0],
    prepulse=False,
    pre_sweep_pulse=[],
    gate_based=False,
    relax_delay=250,  # Wait time between experiments [us]
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values


def gespec_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)

    span = kwargs.pop('span', 10)  # MHz
    center = kwargs.pop('center', station.hardware_cfg.device.qubit.f_ge[0])
    expts = kwargs.pop('expts', default_expt_cfg.expts)

    expt_cfg.start = center - span / 2
    expt_cfg.step = span / expts
    
    expt_cfg.update(kwargs)
    return expt_cfg

def gespec_postproc(station, expt):
    old_freq = station.hardware_cfg.device.qubit.f_ge[0]
    station.hardware_cfg.device.qubit.f_ge = [expt.data['fit_avgi'][2]]
    print(f'Updated qubit frequency from {old_freq} to {station.hardware_cfg.device.qubit.f_ge[0]}!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
gespec_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.PulseProbeSpectroscopyExperiment,
    default_expt_cfg = gespec_defaults,
    preprocessor = gespec_preproc,
    # postprocessor = gespec_postproc,
    job_client=client,
)

gespec = gespec_runner.execute(
    # coupler_current=coupler_current,
)
gespec.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# ### NDAverager version

# %%
# ND-averager comparison: same defaults/preproc as gespec, just swap ExptClass.
# =================================
gespec_nd_runner = CharacterizationRunner(
  station=station,
  ExptClass=meas.PulseProbeSpectroscopyNDExperiment,
  default_expt_cfg=gespec_defaults,
  preprocessor=gespec_preproc,
  # postprocessor=gespec_postproc,  # leave off until we trust the ND result
  job_client=client,
)

gespec_nd = gespec_nd_runner.execute(
  # same kwargs as the R-averager call above so span/center are identical
)
gespec_nd.display()

# %%
# Visual comparison: R-averager vs NDAverager pulse-probe.
# Run cell 28 (gespec) and the cell above (gespec_nd) first.

d_r  = gespec.data
d_nd = gespec_nd.data

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for ax, key, label in zip(axes, ('amps', 'avgi', 'avgq'),
                        ('|IQ|', 'I', 'Q')):
  ax.plot(d_r['xpts'],  d_r[key],  'o-', label='R-averager', alpha=0.7)
  ax.plot(d_nd['xpts'], d_nd[key], 's-', label='NDAverager', alpha=0.7)
  ax.set_ylabel(f'{label} [ADC]')
  ax.legend(loc='best')
axes[-1].set_xlabel('Probe freq [MHz]')
axes[0].set_title('Pulse-probe spectroscopy: R-averager vs NDAverager')
plt.tight_layout()

# %%

# %% [markdown]
# ### T2 Ramsey

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
from sqlalchemy import false


geramsey_defaults = AttrDict(dict(
    ramsey_freq=0.2,  # [MHz]
    start=0.01, # us
    step=0.5,
    expts=101,
    reps=200,
    rounds=1,
    pre_sweep_pulse=None,
    post_sweep_pulse=None,
    if_ef=False,
    ef_init=True, # redundant
    qubits=[0],
    user_defined_freq=[False, 3568.2038290468167, 5304, 0.035],
    f0g1_cavity=0,
    normalize=False,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    prepulse=None,
    postpulse=None,
    pre_active_reset_pulse=False,
    gate_based=False,
    advance_phase=0,
    echoes=[False, 0],
    relax_delay=200
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def geramsey_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    # Copied over but what's all this below????
    # Which ones are actually used? 
    # Can we straighten out all the nested boolean logic???
    checkEF = False
    qubit_ge_init = False
    qubit_ge_after = False
    if expt_cfg.if_ef:
        checkEF = True
        qubit_ge_init = True if expt_cfg.ef_init else False
        qubit_ge_after = True if expt_cfg.ef_init else False
    expt_cfg.checkEF = checkEF
    expt_cfg.qubit_ge_init = qubit_ge_init
    expt_cfg.qubit_ge_after = qubit_ge_after

    expt_cfg.prepulse = False if expt_cfg.pre_sweep_pulse is None else True if expt_cfg.prepulse is None else expt_cfg.prepulse,
    expt_cfg.postpulse = False if expt_cfg.post_sweep_pulse is None else True if expt_cfg.postpulse is None else expt_cfg.postpulse,
    
    return expt_cfg

def geramsey_postproc(station, expt):
    old_freq = station.hardware_cfg.device.qubit.f_ge[0]
    station.hardware_cfg.device.qubit.f_ge = [
        station.hardware_cfg.device.qubit.f_ge[0] + min(expt.data['f_adjust_ramsey_avgi'])
    ]
    print(f'Updated qubit frequency from {old_freq} to {station.hardware_cfg.device.qubit.f_ge[0]}!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
geramsey_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.RamseyExperiment,
    default_expt_cfg = geramsey_defaults,
    preprocessor = geramsey_preproc,
    postprocessor = geramsey_postproc,
    job_client=client,
)

geramsey = geramsey_runner.execute(
    ramsey_freq = 0.2,
    step=0.2,
    active_reset=False,
    relax_delay=2500,
    postprocess=True,
    if_ef=False,
    # coupler_current=coupler_current,
)
# geramsey.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# ### Amplitude Rabi

# %% [markdown]
# We should probably use a cosine fit with fixed phase=0 instead of decaying sine with varying phase?

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
amprabi_defaults = AttrDict(dict(
    start=0,
    step=150,
    expts=151,
    reps=200,
    rounds=1,
    sigma_test=None,
    qubit=0,
    pulse_type='gauss',
    drag_beta=0.0,
    pulse_ge_init=False,
    pulse_ge_after=False,
    checkZZ=False,
    checkEF=False,
    qubits=[0],
    flat_length=0,
    normalize=False,
    single_shot=False,
    singleshot_reps=10000,
    span=50,
    user_defined_freq=[False, 3568.203829046816],
    prepulse=False,
    postpulse=False, 
    if_ef=False,  # If true, will check ef frequency and update it
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def amprabi_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    # Copied over but what's all this below????
    # Which ones are actually used? 
    # Can we straighten out all the nested boolean logic???
    pulse_ge = station.hardware_cfg.device.qubit.pulses.pi_ge
    if expt_cfg.sigma_test is None:
        expt_cfg.sigma_test = pulse_ge.sigma[0]
    if expt_cfg.step is None:
        expt_cfg.step = int(pulse_ge.gain[0] / (expt_cfg.expts - 1))
    
    expt_cfg.checkEF = False
    expt_cfg.pulse_ge_init = False
    expt_cfg.pulse_ge_after = False
    if expt_cfg.if_ef:
        expt_cfg.checkEF = True
        expt_cfg.pulse_ge_init = True
        expt_cfg.pulse_ge_after = True
    
    return expt_cfg

def amprabi_postproc(station, expt):
    station.hardware_cfg.device.qubit.pulses.pi_ge.gain = [expt.data['pi_gain_avgi']]
    station.hardware_cfg.device.qubit.pulses.hpi_ge.gain = [expt.data['hpi_gain_avgi']]
    print('Updated qubit ge pi and hpi gaussian gain!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
amprabi_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.AmplitudeRabiExperiment,
    default_expt_cfg = amprabi_defaults,
    preprocessor = amprabi_preproc,
    postprocessor = amprabi_postproc,
    job_client=client,
)

amprabi = amprabi_runner.execute(
    relax_delay=2500,
    postprocess=True,
)
amprabi.display()

#After amplitude calibration, do another T2 Ramsey to fine tune frequency
#Added by Jonginn, as a part of practice
#Please remove if the below code causes any problem.
t2_ramsey_ge_after_amp = geramsey_runner.execute(
    ramsey_freq = 0.2,
    step = 0.4,
    active_reset = False,
    relax_delay=2500,
)
t2_ramsey_ge_after_amp.display()


# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### T1

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
t1_ge_defaults = AttrDict(dict(
    start=0,
    step=20,
    expts=100,
    reps=100,
    rounds=1,
    qubit=0,
    qubit_ef=False,
    normalize=False,
    relax_delay=2500,
))

def t1_ge_postproc(station, expt):
    station.hardware_cfg.device.qubit.T1 = [expt.data['fit_avgi'][3]]
    print('Updated qubit T1!')
    station.snapshot_hardware_config(update_main=False)



# %%
# Execute
# =================================
t1_ge_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.t1.T1Experiment,
    default_expt_cfg=t1_ge_defaults,
    postprocessor=t1_ge_postproc,
    job_client=client,
)

if expts_to_run['t1_ge']:
    t1_ge = t1_ge_runner.execute()
    t1_ge.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# ## Qubit ef

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Pulse-probe

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
efspec_defaults = AttrDict(dict(
    start=3415,
    step=0.05,
    expts=500,
    reps=200,
    rounds=1,
    length=1,
    gain=100,
    qubit_f=False,
    qubit=0,
    cavity_drive=False,
    wait_qubit=False,
    relax_delay=500,
))

def efspec_postproc(station, expt):
    old_freq = station.hardware_cfg.device.qubit.f_ef[0]
    station.hardware_cfg.device.qubit.f_ef = [expt.data['fit_avgi'][2]]
    print(f'Updated qubit ef frequency from {old_freq} to {station.hardware_cfg.device.qubit.f_ef[0]}!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
efspec_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.PulseProbeEFSpectroscopyExperiment,
    default_expt_cfg=efspec_defaults,
    postprocessor=efspec_postproc,
    job_client=client,
)

if expts_to_run['pulse_probe_ef']:
    qspec_ef = efspec_runner.execute(
        relax_delay=250,
    )
    qspec_ef.display()

# %%
station.hardware_cfg.device.qubit.f_ef = [3419.058641409324]

# %% [markdown]
# ### T2 Ramsey

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
# Reuse ge defaults but with if_ef=True
eframsey_defaults = AttrDict(deepcopy(geramsey_defaults))
eframsey_defaults.if_ef = True
eframsey_defaults.ef_init = True
eframsey_defaults.ramsey_freq = 3  # Typical ef ramsey frequency

def eframsey_postproc(station, expt):
    old_freq = station.hardware_cfg.device.qubit.f_ef[0]
    station.hardware_cfg.device.qubit.f_ef = [
        station.hardware_cfg.device.qubit.f_ef[0] + min(expt.data['f_adjust_ramsey_avgi'])
    ]
    print(f'Updated qubit ef frequency from {old_freq} to {station.hardware_cfg.device.qubit.f_ef[0]}!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
eframsey_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.RamseyExperiment,
    default_expt_cfg=eframsey_defaults,
    preprocessor=geramsey_preproc,  # Reuse ge preprocessor
    postprocessor=eframsey_postproc,
    job_client=client,
)

t2ramsey_ef = eframsey_runner.execute(
    ramsey_freq=0.8,
    step = 0.05,
    relax_delay=800,
    active_reset=False,
)
# t2ramsey_ef.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# ### Amplitude Rabi

# %% [markdown]
# We should probably use a cosine fit with fixed phase=0 instead of decaying sine with varying phase?

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
# Reuse ge defaults but with if_ef=True
efamprabi_defaults = AttrDict(deepcopy(amprabi_defaults))
efamprabi_defaults.if_ef = True
efamprabi_defaults.step = 100  # Typical ef amplitude step
efamprabi_defaults.expts = 150

def efamprabi_postproc(station, expt):
    station.hardware_cfg.device.qubit.pulses.pi_ef.gain = [expt.data['pi_gain_avgi']]
    station.hardware_cfg.device.qubit.pulses.hpi_ef.gain = [expt.data['hpi_gain_avgi']]
    print('Updated qubit ef pi and hpi gaussian gain!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
efamprabi_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.AmplitudeRabiExperiment,
    default_expt_cfg=efamprabi_defaults,
    preprocessor=amprabi_preproc,  # Reuse ge preprocessor
    postprocessor=efamprabi_postproc,
    job_client=client,
)

amprabi_ef = efamprabi_runner.execute(
            relax_delay=2500,
)
# amprabi_ef.display()

# After amplitude calibration, do another T2 Ramsey to fine tune frequency
t2_ramsey_ef_after_amp = eframsey_runner.execute(
    ramsey_freq=0.5,
    step = 0.1,
    relax_delay=2500,
)
# t2_ramsey_ef_after_amp.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %%
station.snapshot_hardware_config(update_main=False)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### T1

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
t1_ef_defaults = AttrDict(dict(
    start=0,
    step=5,
    expts=100,
    reps=50,
    rounds=1,
    qubit=0,
    qubit_ef=True,
    normalize=False,
    relax_delay=2500,
))

def t1_ef_postproc(station, expt):
    station.hardware_cfg.device.qubit.T1_ef = [expt.data['fit_avgq'][3]]
    print('Updated qubit T1 ef!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
t1_ef_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.T1Experiment,
    default_expt_cfg=t1_ef_defaults,
    postprocessor=t1_ef_postproc,
    job_client=client,
)

if expts_to_run['t1_ef']:
    t1_ef = t1_ef_runner.execute()
    t1_ef.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# # Manipulate

# %% [markdown]
# ## Spectroscopy

# %%
f0g1spec_defaults = AttrDict(dict(
    start=None,  # Will be computed in preprocessor from ds_storage
    step=0.2,
    expts=200,
    reps=100,
    rounds=1,
    length=1,
    gain=3000,
    pulse_type='gaussian',
    qubit_f=True,
    qubits=[0],
    prepulse=False,
    relax_delay=200,
    man_mode_no=1,
))

def f0g1spec_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    man_mode_no = expt_cfg.man_mode_no
    expt_cfg.update(kwargs)
    
    # Compute start frequency from dataset if not provided
    if expt_cfg.start is None:
        expt_cfg.start = station.ds_storage.get_freq('M' + str(man_mode_no)) - 20
    
    return expt_cfg

def f0g1spec_postproc(station, expt):
    man_mode_no = expt.cfg.expt.man_mode_no
    station.ds_storage.update_freq('M' + str(man_mode_no), expt.data['fit_avgi'][2])
    station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['frequency'][0] = expt.data['fit_avgi'][2]
    print(f"Updated man f0g1 freq to: {station.ds_storage.get_freq('M' + str(man_mode_no))}")
    station.snapshot_hardware_config(update_main=False)
    station.snapshot_man1_storage_swap(update_main=False)


# %%
station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']

# %%
# Execute
# =================================
f0g1spec_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.PulseProbeF0g1SpectroscopyExperiment,
    default_expt_cfg=f0g1spec_defaults,
    preprocessor=f0g1spec_preproc,
    postprocessor=f0g1spec_postproc,
    job_client=client,
)

 
man_spec = f0g1spec_runner.execute(
    man_mode_no=1,
    start=1965,
    expts=300,
    step=0.2,
    # gain=20000,
    # length=5,
    # pulse_type='gaussian',
    go_kwargs=dict(progress=True),
    active_reset=False,
    relax_delay=800,
    # coupler_current=coupler_current,
)
# man_spec.display()


# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)
# station.snapshot_multiphoton_config(update_main=True)

# %%
station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['frequency']

# %%
station.snapshot_hardware_config(update_main=False)


# %% [markdown]
# ## Find Frequency (Chevron)
#

# %%
chevron_defaults = AttrDict(dict(
    start=0, # time start in us
    step=0.1, # time step in us
    expts=25, # number of time points
    reps=100,
    rounds=1,
    qubits=[0],
    gain=None, # Leave as None to use current value in dataset
    ramp_sigma=0.005,
    use_arb_waveform=False,
    pi_ge_before=True,
    pi_ef_before=True,
    pi_ge_after=False,
    normalize=False,
    active_reset=False,
    check_man_reset=[False, 0],
    check_man_reset_pi=[],
    prepulse=False,
    pre_sweep_pulse=[],
    err_amp_reps=0,
    swap_lossy=False,
))


def f0g1_chevron_postproc(station, mother_expt):
    expt_cfg = mother_expt.cfg.expt
    expt_cfg.man_mode_no = 1
    stor_name = f'M{expt_cfg.man_mode_no}'

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
        
    if best_freq:
        print(f"Best frequency found: {best_freq:.4f} MHz")
        station.ds_storage.update_freq(stor_name, best_freq)
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['frequency'][0] = best_freq
        print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz (ds_storage and hardware_cfg)")

        pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
        station.ds_storage.update_pi(stor_name, pi_len)
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['length'][0] = pi_len
        print('Updated the pi length to:', pi_len, "(ds_storage and hardware_cfg)")

        station.ds_storage.update_h_pi(stor_name, pi_len / 2)
        print('Updated the h_pi length to:', pi_len / 2, "(ds_storage)")

        gain = expt_cfg.get('gain', station.ds_storage.get_gain(stor_name))
        station.ds_storage.update_gain(stor_name, gain)
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['gain'][0] = gain
        print('Updated gain to:', gain)
    mother_expt.analysis = chevron_analysis
    chevron_analysis.display_results()
    station.snapshot_man1_storage_swap(update_main=False)
    station.snapshot_hardware_config(update_main=False)


runner = SweepRunner(
    station=station,
    ExptClass=meas.LengthRabiGeneralF0g1Experiment,
    default_expt_cfg=chevron_defaults,
    sweep_param='freq',
    live_plot=False,
    # preprocessor=my_preproc,
    postprocessor=f0g1_chevron_postproc,
    job_client=client,
)

# %%
# coarse sweep
f0g1_chevron = runner.execute(
    sweep_start=station.ds_storage.get_freq('M1') - 3,
    sweep_stop=station.ds_storage.get_freq('M1') + 3,
    sweep_npts=11,
    gain = station.ds_storage.get_gain('M1'),
    start = 1, # time start in us
    # coupler_current=coupler_current,
    batch = True,
)

# %%
# fine sweep
f0g1_chevron = runner.execute(
    sweep_start=station.ds_storage.get_freq('M1') - 0.5,
    sweep_stop=station.ds_storage.get_freq('M1') + 0.5,
    sweep_npts=11,
    gain = station.ds_storage.get_gain('M1'),
    # start = 1, # time start in us
    # coupler_current=coupler_current,
)

# %% [markdown]
# ## Error amplification

# %%
# Manipulate/Error Amplification - New Pattern with CharacterizationRunner

# Configuration defaults
error_amp_defaults = AttrDict(dict(
    start=1998,
    expts=40,
    step=0.025,
    reps=100,
    rounds=1,
    qubit=0,
    qubits=[0],
    n_pulses=7,
    sideband = 'f0-g1', #should be in the format of 'fn-gn+1' 
    parameter_to_test='frequency',
    pulse_type= None, # if this is None, will be set to ['multiphoton', sideband, 'pi', 0] in preproc
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=2500,
))

def error_amp_preproc(station, 
                      default_expt_cfg, 
                      **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    if expt_cfg.pulse_type is None:
        expt_cfg.pulse_type = ['multiphoton', expt_cfg.sideband, 'pi', 0]
    return expt_cfg

def error_amp_postproc(station, expt):
    """
    Postprocessor for error amplification.
    Analyze with custom parameters.
    """
    sideband = expt.cfg.expt.sideband
    _sideband = sideband[0] + 'n' + '-' + sideband[3] + 'n+1'
    i = int(sideband[1])
    # Perform analysis with state_fin='e' as in original code
    expt.analyze(data=expt.data, state_fin='e')
    # expt.display(data=expt.data, state_fin='e')
    print('Error amplification analysis complete')

    print(f"Man {sideband} pi frequency before update:", 
          station.hardware_cfg.device.multiphoton['pi'][_sideband]['frequency'][i])
    station.hardware_cfg.device.multiphoton['pi'][_sideband]['frequency'][i] = expt.data['fit_avgi'][2]
    print(f"Man {sideband} pi frequency after update:", 
          station.hardware_cfg.device.multiphoton['pi'][_sideband]['frequency'][i])
    if i > 0:
        print("WARNING! No update will occur! The update in this cell was meant for the csv which does not have multiphoton params. To update the multiphoton params, please run the multiphoton calibration notebook instead.")
    else:
        station.ds_storage.update_freq('M1', expt.data['fit_avgi'][2])
        print("Updated the ds_storage frequency to:", station.ds_storage.get_freq('M1'))
    station.snapshot_hardware_config(update_main=False)
    station.snapshot_multiphoton_config(update_main=False)
    station.snapshot_man1_storage_swap(update_main=False)  # persist ds_storage to disk

# Create runner
error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_defaults,
    preprocessor=error_amp_preproc,
    postprocessor=error_amp_postproc,
    job_client=client,
)

# Example execution

# Run with analyze=False, display=False initially
# Postprocessor will handle custom analysis
error_amp_exp = error_amp_runner.execute(
    start=station.hardware_cfg.device.multiphoton.pi['fn-gn+1'].frequency[0]-0.5,
    go_kwargs = dict(analyze=False, display=False),
    postprocess=True  # This will call postprocessor which does the custom analysis
)

# error_amp_exp.display()

# %%
station.preview_config_update()

# %% [markdown]
# ## Length Rabi f0g1 (Update time)

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
lenrabi_f0g1_defaults = AttrDict(dict(
    man_mode_no=1,
    start=None,  # Will be computed in preprocessor (soc.cycles2us(3))
    step=0.01,
    qubits=[0],
    expts=150,
    reps=100,
    rounds=1,
    gain=8000,
    freq=None,  # Will be set from ds_storage in preprocessor
    use_arb_waveform=False,
    pi_ge_before=True,
    pi_ef_before=True,
    pi_ge_after=True,
    normalize=False,
    active_reset=False,
    man_reset=True,
    stor_reset=True,
    check_man_reset=[False, 0],
    swap_lossy=False,
    check_man_reset_pi=[],
    prepulse=False,
    pre_sweep_pulse=[],
    err_amp_reps=0,
    relax_delay=5000,
))

def lenrabi_f0g1_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    man_mode_no = default_expt_cfg.man_mode_no

    # Compute start and freq from soc and ds_storage if not provided
    if expt_cfg.start is None:
        expt_cfg.start = station.soccfg.cycles2us(3)
    if expt_cfg.freq is None:
        expt_cfg.freq = station.ds_storage.get_freq('M' + str(man_mode_no))
    
    return expt_cfg

def lenrabi_f0g1_postproc(station, expt):
    man_mode_no = expt.cfg.expt.man_mode_no
    
    # Get analysis results from the LengthRabiFitting object stored in expt
    if hasattr(expt, '_length_rabi_analysis'):
        analysis = expt._length_rabi_analysis
        pi_length = analysis.results['pi_length']
        pi2_length = analysis.results['pi2_length']
        gain = expt.cfg.expt.gain
        freq = expt.cfg.expt.freq

        # Update ds_storage
        station.ds_storage.update_freq('M' + str(man_mode_no), freq)
        station.ds_storage.update_pi('M' + str(man_mode_no), pi_length)
        station.ds_storage.update_h_pi('M' + str(man_mode_no), pi2_length)
        station.ds_storage.update_gain('M' + str(man_mode_no), gain)
        print(f'Updated ds_storage M{man_mode_no}: freq={freq:.4f}, pi_length={pi_length:.4f}, pi2_length={pi2_length:.4f}, gain={gain}')

        # Keep hardware_cfg.multiphoton in sync (freq, length, gain — no h_pi field there)
        idx = man_mode_no - 1
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['frequency'][idx] = freq
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['length'][idx] = pi_length
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['gain'][idx] = gain
        print(f'Updated hardware_cfg multiphoton[{idx}]: freq={freq:.4f}, length={pi_length:.4f}, gain={gain}')

        # Persist both
        station.snapshot_hardware_config(update_main=False)
        station.snapshot_man1_storage_swap(update_main=False)



# %%
# Execute
# =================================
lenrabi_f0g1_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.LengthRabiGeneralF0g1Experiment,
    default_expt_cfg=lenrabi_f0g1_defaults,
    preprocessor=lenrabi_f0g1_preproc,
    postprocessor=lenrabi_f0g1_postproc,
    job_client=client,
)

len_rabis_man = lenrabi_f0g1_runner.execute(
    man_mode_no=1,
    gain=station.ds_storage.get_gain('M1'),
    step=0.01,
    go_kwargs=dict(progress=True, analyze=False),
)
# len_rabis_man.display()ArithmeticError


# %%
station.preview_config_update()

# %%
station.update_all_station_snapshots(update_main=False)

# %%
# Safety-net snapshot — postprocs now keep ds_storage and hardware_cfg.multiphoton
# in sync themselves, so this is no longer needed as a fix.
# Kept here as a belt-and-suspenders full sync if desired.
# station.update_all_station_snapshots(update_main=False)

# %%
# len_rabis_mans[0].active_reset = True
# len_rabis_mans[0].analyze()
# len_rabis_mans[0].display(title_str='Length Rabi General F0g1')

# %% [markdown]
# ## --- Manipulate sections below not refactored ---

# %% [markdown]
# ## Chi between qubit and Manipulate 

# %% [markdown]
# ### ge

# %%
mm_base_dummy = MM_dual_rail_base(config_thisrun, soc)
prep_man_pi = mm_base_dummy.prep_man_photon(1)
mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse

# %%
from experiments.MM_dual_rail_base import MM_dual_rail_base
import numpy as np

# the do function contains 2 calls to ramsey, one with no prepulse and one with prepulse 
# that initializes manipulation mode to 1 state
# do_t2_ramsey_ge is already defined in previous cells and can be used directly

# Add active_reset, relax_delay, expts as arguments and pass them to do_t2_ramsey_ge
def do_chi(config_thisrun, expt_path, config_path, prepulse=None, standard_ramsey=True,
           man_mode_no=1, active_reset=False, relax_delay=2500, expts=100):
    """
    Run two Ramsey experiments: one standard, one with a prepulse that initializes manipulation mode 1.
    Returns both experiment objects.
    """
    # Standard Ramsey (no prepulse)
    if standard_ramsey:
        t2ramsey_no_prepulse = do_t2_ramsey_ge(
            config_thisrun, expt_path, config_path,
            pre_sweep_pulse=None, post_sweep_pulse=None,
            step_size=0.1,
            active_reset=active_reset,
            relax_delay=relax_delay,
            expts=expts
        )
    else: 
        t2ramsey_no_prepulse = None

    # Ramsey with prepulse (initialize manipulation mode 1)
    if prepulse is None:
        mm_base_dummy = MM_dual_rail_base(config_thisrun, soc)
        prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
        prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()

    t2ramsey_with_prepulse = do_t2_ramsey_ge(
        config_thisrun, expt_path, config_path,
        pre_sweep_pulse=prepulse, post_sweep_pulse=None,
        step_size=0.1,
        active_reset=active_reset,
        relax_delay=relax_delay,
        expts=expts
    )

    return t2ramsey_no_prepulse, t2ramsey_with_prepulse


def update_chi(t2_ramsey_original, t2_ramsey_prepulse, config_thisrun, man_mode_no=1):
    """
    Update config_thisrun.device.qubit.chi for the given manipulation mode.
    """
    f_without_prepulse = t2_ramsey_original.data['f_adjust_ramsey_avgi'][0]
    f_with_prepulse = t2_ramsey_prepulse.data['f_adjust_ramsey_avgi'][0]
    chi = f_with_prepulse - f_without_prepulse
    print('Chi:', chi)
    config_thisrun.device.manipulate.chi_ge[man_mode_no - 1] = chi
    config_thisrun.device.manipulate.revival_time[man_mode_no-1] = np.abs(np.pi/(2 * np.pi * chi))
    print('Delay time (mus):', config_thisrun.device.manipulate.revival_time[man_mode_no-1] )



# %%
t2_ramsey_original, t2_ramsey_prepulse = None, None
# Run the chi experiments/__pycache__/
if expts_to_run['chi_ge']:
    t2_ramsey_original, t2_ramsey_prepulse = do_chi(config_thisrun, expt_path, config_path, standard_ramsey=True)
    # analyze and display the results
    t2_ramsey_original.analyze(fitparams=[300, None, None, None, None, None])
    t2_ramsey_original.display()
    t2_ramsey_prepulse.analyze(fitparams=[300, None, None, None, None, None])
    t2_ramsey_prepulse.display() 
    # update the config_thisrun with the chi value
    update_chi(t2_ramsey_original, t2_ramsey_prepulse, config_thisrun, man_mode_no=1)
    print('Only doing it for mode 1')

# %%
# config_thisrun.device.manipulate.revival_time[0] = np.pi/(2 * np.pi * config_thisrun.device.manipulate.chi[0])

# %% [markdown]
# ### ef

# %%
from experiments.MM_dual_rail_base import MM_dual_rail_base

# the do function contains 2 calls to ramsey, one with no prepulse and one with prepulse 
# that initializes manipulation mode to 1 state
def do_chi_f(config_thisrun, expt_path, config_path, 
           man_mode_no=1):
    """
    Run two Ramsey experiments: one standard, one with a prepulse that initializes manipulation mode 1.
    Returns both experiment objects.
    """

    
    mm_base_dummy = MM_dual_rail_base(config_thisrun, soc)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
    # Add qubit ge prepulse and postpulse for ef (ge init)
    qubit_ge_prepulse = [['qubit', 'ge', 'pi', 0]]
    prep_man_pi_prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    prepulse =  mm_base_dummy.get_prepulse_creator(prep_man_pi + qubit_ge_prepulse ).pulse.tolist() 
    postpulse  = mm_base_dummy.get_prepulse_creator(qubit_ge_prepulse).pulse.tolist()

    # Add chi to frequency of qubit ge 
    # prepulse[0][-1] += config_thisrun.device.manipulate.chi[man_mode_no - 1]
    # postpulse[0][-1] += config_thisrun.device.manipulate.chi[man_mode_no - 1]

    # Do an ef and ge ramsey with this prepulse 
    # Run ge Ramsey 
    
    
    t2ramsey_no_prepulse_ge = do_t2_ramsey_ge(config_thisrun,
        expt_path,
        config_path,
        step_size=0.1,
        pre_sweep_pulse=prep_man_pi_prepulse,
        post_sweep_pulse=None)
    # Run ef Ramsey with prepulse and postpulse
    t2ramsey_with_prepulse_ef = do_t2_ramsey_ef(
        config_thisrun,
        expt_path,
        config_path,
        pre_sweep_pulse=prepulse,
        post_sweep_pulse=postpulse,
        step_size=0.1,
        ef_init=False  # Do not initialize ef, we are already in ef state
    )
    

    return t2ramsey_no_prepulse_ge, t2ramsey_with_prepulse_ef

def update_chi_ef(t2_standard_ramsey, t2_prepulsed_ramsey, config_thisrun, man_mode_no=1):
    """
    Update config_thisrun.device.qubit.chi for the given manipulation mode.
    """
    f_with_prepulse = t2_prepulsed_ramsey.data['f_adjust_ramsey_avgi'][0]
    f_without_prepulse = t2_standard_ramsey.data['f_adjust_ramsey_avgi'][0]
    chi_ef = f_with_prepulse - f_without_prepulse
    print('Chi:', chi_ef)
    config_thisrun.device.manipulate.chi_ef[man_mode_no - 1] = chi_ef



# %%
man_mode_no = 1
t2ge_ramsey_forchief, t2ef_ramsey_forchief = None, None
t2ef_standard_ramsey_forchief = None
if expts_to_run['chi_ef']:
    
    #get standard ef ramsey 
    t2ef_standard_ramsey_forchief = do_t2_ramsey_ef(config_thisrun, expt_path, config_path)
    # analyze and display the results
    t2ef_standard_ramsey_forchief.analyze(fitparams=[300, None, None, None, None, None])
    t2ef_standard_ramsey_forchief.display(title_str='T2_ef_standard_for_chief')

    # Now ramseys with man photon prepulse
    config_thisrun_chief = deepcopy(config_thisrun)
    config_thisrun_chief.device.qubit.f_ge[0] += config_thisrun.device.manipulate.chi_ge[man_mode_no - 1]
    t2ge_ramsey_forchief, t2ef_ramsey_forchief = do_chi_f(config_thisrun_chief, expt_path, config_path)
    # analyze and display the results
    t2ge_ramsey_forchief.analyze(fitparams=[300, None, None, None, None, None])
    t2ge_ramsey_forchief.display(title_str='T2_ge_for_chief_tocheck_this_is_correctly_at_ramsey_freq')
    t2ef_ramsey_forchief.analyze(fitparams=[300, None, None, None, None, None])
    t2ef_ramsey_forchief.display(title_str='T2_ef_for_chief')
    # update the config_thisrun with the chi value
    update_chi_ef(t2ef_standard_ramsey_forchief, t2ef_ramsey_forchief, config_thisrun)


# %% [markdown]
# ## Parity Delay
# NOT Implemented yet: We can use chi to estimate parity waiting time pi/chi but can also fine tune it using this experiment
#
# Basically Length rabi analysis 

# %% [markdown]
# ## T1

# %%
def do_t1_manipulate(config_thisrun, expt_path, config_path, man_mode_no=1):
    """
    Run T1 experiment for the specified manipulate mode (man_mode_no).
    """
    t1_man = meas.single_qubit.t1_cavity.T1CavityExperiment(
        soccfg=soc, path=expt_path, prefix='T1CavityExperiment', config_file=config_path
    )

    t1_man.cfg = AttrDict(deepcopy(config_thisrun))

    # Set experiment parameters for the specified manipulate mode
    t1_man.cfg.expt = dict(
        start=0,
        step=15,
        expts=60,
        reps=300,
        rounds=1,
        cavity_prepulse=[False, 300, 1.5],
        f0g1_prep=True,
        f0g1_param=[ds_storage.get_freq(f'M{man_mode_no}'), ds_storage.get_gain(f'M{man_mode_no}'), ds_storage.get_pi(f'M{man_mode_no}')],
        resolved_pi=False,
        cavity=man_mode_no,
        qubit=0,
        normalize=False
    )

    t1_man.cfg.device.readout.relax_delay = [2500]
    t1_man.go(analyze=True, display=True, progress=True, save=True)
    return t1_man



# %%
t1_man = do_t1_manipulate(config_thisrun, expt_path, config_path, 1)


# %% [markdown]
# ## T2
# 06/19/2025: The code below should use man ramsey directly instead of user defined; user defined is for ramsey where you directly displace manipulate mode  - Eesh

# %%
def do_cavity_ramsey(config_thisrun, expt_path, config_path, man_mode_no=1):
    """
    Run the Cavity Ramsey experiment using the specified configuration.
    """
    cavity_ramsey = meas.single_qubit.t2_cavity.CavityRamseyExperiment(
        soccfg=soc, path=expt_path, prefix='CavityRamseyExperiment', config_file=config_path
    )

    cavity_ramsey.cfg = AttrDict(deepcopy(config_thisrun))

    # Prepulse and postpulse

    # Set experiment parameters as in the YAML block above
    cavity_ramsey.cfg.expt = dict(
        start=0.01,
        step=0.02*7.5,
        expts=600,
        # ramsey_freq=-3.5,
        ramsey_freq=1.5,
        reps=100,
        rounds=1,
        qubits=[0],
        checkEF=False,
        f0g1_cavity=0,
        init_gf=False,
        active_reset=False,
        man_reset=True,
        storage_reset=True,
        user_defined_pulse=[False, ds_storage.get_freq(stor_name='M'+ str(man_mode_no)), 
                            ds_storage.get_gain(stor_name='M'+ str(man_mode_no)), 0.005, 
                            ds_storage.get_pi(stor_name='M'+ str(man_mode_no)), 0],
        parity_meas=False,
        man_mode_no=man_mode_no ,
        storage_ramsey=[False, 2, True],
        man_ramsey=[True, man_mode_no],
        coupler_ramsey=False,
        custom_coupler_pulse=[[944.25], [1000], [0.316677658], [0], [1], ['flat_top'], [0.005]],
        echoes=[True, 1],
        prepulse=True,
        postpulse=True,
        gate_based = True,
        pre_sweep_pulse= [['qubit', 'ge', 'hpi', 0], ['qubit', 'ef', 'pi', 0]],
        post_sweep_pulse=[['qubit', 'ef', 'pi', 0], ['qubit', 'ge', 'hpi', 0]]
    )

    cavity_ramsey.cfg.device.readout.relax_delay = [2500]
    cavity_ramsey.go(analyze=False, display=False, progress=True, save=True)
    return cavity_ramsey

cavity_ramsey = do_cavity_ramsey(config_thisrun, expt_path, config_path, man_mode_no=1)

# %%
cavity_ramsey.analyze(fitparams=[300, None, None, None, None, None])
cavity_ramsey.display()


# %% [markdown]
# # Storage

# %%
def get_mode_parameters(ds_storage, config_thisrun, mode_name):
    """
    Get sideband pulse parameters for any mode from ds_storage.
    Works for storage (e.g. 'M1-S4'), dump ('M1-D2'), and coupler ('M1-C').
    Returns: freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse
    """
    freq     = ds_storage.get_freq(mode_name)
    gain     = ds_storage.get_gain(mode_name)
    pi_len   = ds_storage.get_pi(mode_name)
    h_pi_len = ds_storage.get_h_pi(mode_name)
    ch = 'low'

    prep_man_pi = [
        ['qubit', 'ge', 'pi', 0],
        ['qubit', 'ef', 'pi', 0],
        ['man', 'M1', 'pi', 0],
    ]
    from experiments.MM_base import MM_base
    mm_base_dummy = MM_base(config_thisrun, soccfg=station.soccfg)
    prepulse  = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist()

    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse


# %% [markdown]
# ## Man-Stor Spectroscopy

# %%
# ── Sideband spectroscopy — unified for storage (M1-S*), dump (M1-D*), coupler (M1-C) ──
# Pass mode_name= at execute time, e.g.:
#   sideband_spec_runner.execute(mode_name='M1-S4', ...)
#   sideband_spec_runner.execute(mode_name='M1-D2', relax_delay=2500, ...)
#   sideband_spec_runner.execute(mode_name='M1-C',  bw=200, ...)

sideband_spec_defaults = AttrDict(dict(
    mode_name=None,         # REQUIRED at execute time, e.g. 'M1-S4', 'M1-D2', 'M1-C'
    bw=20,                  # sweep bandwidth in MHz, centred on ds_storage frequency
    expts=250,
    step=None,              # computed as bw/expts in preproc; do not set manually
    reps=200,
    qubit=[0],
    flux_drive_ch=None,     # overrides ds_storage ch if set
    flux_drive_gain=None,   # overrides ds_storage gain if set
    flux_drive_duration=5,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    relax_delay=500,
))

def sideband_spec_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    mode_name = expt_cfg.mode_name
    assert mode_name is not None, \
        "Pass mode_name= at execute time, e.g. mode_name='M1-S4'"

    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_mode_parameters(
        station.ds_storage, station.hardware_cfg, mode_name
    )

    expt_cfg.start = freq - expt_cfg.bw / 2
    expt_cfg.step  = expt_cfg.bw / expt_cfg.expts

    if expt_cfg.flux_drive_gain is not None:
        gain = expt_cfg.flux_drive_gain
    if expt_cfg.flux_drive_ch is not None:
        ch = expt_cfg.flux_drive_ch
    expt_cfg.flux_drive       = [ch, 1, gain, expt_cfg.flux_drive_duration]
    expt_cfg.pre_sweep_pulse  = prepulse
    expt_cfg.post_sweep_pulse = postpulse

    print(f'Sideband spectroscopy for {mode_name}: centre={freq:.3f} MHz, bw={expt_cfg.bw} MHz, gain={gain}')
    return expt_cfg

def sideband_spec_postproc(station, expt):
    mode_name = expt.cfg.expt.mode_name
    new_freq  = expt.data['fit_avgi'][2]
    station.ds_storage.update_freq(mode_name, new_freq)
    station.snapshot_man1_storage_swap(update_main=False)
    print(f'Updated {mode_name} frequency → {new_freq:.4f} MHz')

sideband_spec_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.FluxSpectroscopyF0g1Experiment,
    default_expt_cfg=sideband_spec_defaults,
    preprocessor=sideband_spec_preproc,
    postprocessor=sideband_spec_postproc,
    job_client=client,
)

# %%
station.ds_storage.append_dataset(
    'M1-D1', **{'freq (MHz)': 2313.294828, 
                'pi (mus)': 20.0, 
                'h_pi (mus)': 10.0, 
                'gain (DAC units)': 8000
               })

# %%
station.ds_storage.update_freq('M1-C', 950)

# %%
stor_specs = {}
for stor_mode_no in [2]:  # range(1, 8): # for all storage modes
    mode_name = f'M1-S{stor_mode_no}'
    print(f'Running storage spectroscopy for {mode_name}')
    stor_specs[stor_mode_no] = sideband_spec_runner.execute(
        mode_name=mode_name,
        postproc=False,
        relax_delay=800,
        reps=100,
        bw=200,
        expts=100,
        flux_drive_gain=6000,
        flux_drive_duration=5,
    )
    # stor_specs[stor_mode_no].display()

# %%
new_gains = [2000, 6000, 8000, 8000, 8000, 8000, 8000]

for i, new_gain in zip(list(range(1,8)), new_gains):
    station.ds_storage.update_gain(f'M1-S{i}', new_gain)

# %%
new_freqs = [340, 517.5, 702.5, 882.5, 1053.5, 1250, 1420]

for i, new_freq in zip(list(range(1,8)), new_freqs):
    station.ds_storage.update_freq(f'M1-S{i}', new_freq)

# %% [markdown]
# ### Save dataset and update to main if desired

# %%
station.snapshot_man1_storage_swap(update_main=False)
# station.snapshot_man1_storage_swap(update_main=True)

# %% [markdown]
# ## Man-dump

# %%
station.ds_storage.update_freq('M1-D1', 2245)

# %% [markdown]
# ### Spectroscopy

# %%
# get_dump_mode_parameters has been consolidated into get_mode_parameters above.
# Dump spectroscopy uses sideband_spec_runner — see execute cell below.

# %%
# dump_spec_defaults/preproc/postproc/runner have been consolidated into
# sideband_spec_defaults/preproc/postproc/runner above.
# Notable difference vs storage: longer relax_delay (dump modes decay faster).

# %%
station.ds_storage.update_freq('M1-D1', 2262.6886)
station.ds_storage.update_pi('M1-D1', 10.0)
station.ds_storage.update_h_pi('M1-D1', 5.0)

# station.ds_storage.update_pi('M1-S5', 0.0)
# station.ds_storage.update_h_pi('M1-S5', 0.0)

# %%
dump_specs = {}
for dump_mode_no in [1,2]:  # specify which dump modes to run
    mode_name = f'M1-D{dump_mode_no}'
    print(f'Running dump spectroscopy for {mode_name}')
    dump_specs[dump_mode_no] = sideband_spec_runner.execute(
        mode_name=mode_name,
        flux_drive_gain=5000,
        flux_drive_duration=2,
        bw=10.0,
        expts=100,
        reps=100,
        relax_delay=1500,
    )

# %%
station.update_all_station_snapshots(update_main=False)

# %%

# %% [markdown]
# ### Dump chevron

# %%
from fitting.fit_display_classes import ChevronFitting
from datetime import datetime

# Configuration defaults for dump sideband sweep
dump_sideband_chevron_defaults = AttrDict(dict(
    start=0.007, # start time in us
    pi_len_sweep=2.0, # total sweep length in us
    expts=30, # num steps of time
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
 
def dump_sideband_chevron_preproc(station, default_expt_cfg, **kwargs):
    assert 'dump_mode_no' in kwargs
    
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    # Get dump mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_dump_mode_parameters(
        station.ds_storage, station.hardware_cfg, expt_cfg.man_mode_no, expt_cfg.dump_mode_no
    )

    print('Prepulse:', prepulse)
    print('Postpulse:', postpulse)

    dump_name = f'M{expt_cfg.man_mode_no}-D{expt_cfg.dump_mode_no}'
    
    pi_len_sweep = expt_cfg.pi_len_sweep
    expt_cfg.step = pi_len_sweep / (expt_cfg.expts - 1)
    if 'gain' in expt_cfg and expt_cfg.gain is not None:
        gain = expt_cfg.gain  # Override gain if provided
    
    # expt_cfg.flux_drive = [ch, freq, gain, 0]
    expt_cfg.flux_drive = ['high', freq, gain, 0]
    expt_cfg.pre_sweep_pulse = prepulse
    expt_cfg.post_sweep_pulse = postpulse
    
    print(f'Dump sideband chevron for {dump_name}: freq={freq:.3f} MHz, gain={gain}')
    
    return expt_cfg

# def dump_sideband_chevron_postproc(station, mother_expt):
#     expt_cfg = mother_expt.cfg.expt
#     dump_name = f'M{expt_cfg.man_mode_no}-D{expt_cfg.dump_mode_no}'

#     from fitting.fit_display_classes import ChevronFitting

#     chevron_analysis = ChevronFitting(
#         frequencies=mother_expt.data['freq_sweep'],
#         time=mother_expt.data['xpts'][0],
#         response_matrix=mother_expt.data['avgi'],
#         config=station.hardware_cfg,
#         station=station,
#     )

#     # chevron_analysis.analyze()
    
#     # best_freq = chevron_analysis.results.get('best_frequency_contrast')
        
#     # if best_freq:
#     #     print(f"Best frequency found: {best_freq:.4f} MHz")
#     #     station.ds_storage.update_freq(dump_name, best_freq)
#     #     print(f"Updated {dump_name} frequency to {best_freq:.4f} MHz")
#     #     pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
#     #     station.ds_storage.update_pi(dump_name, pi_len)
#     #     print('Updated the pi length to:', pi_len)
#     #     station.ds_storage.update_h_pi(dump_name, pi_len / 2)
#     #     print('Updated the h_pi length to:', pi_len / 2)
#     #     station.ds_storage.update_gain(dump_name, expt_cfg.flux_drive[2])
#     #     print('Updated gain to:', expt_cfg.flux_drive[2])
#     # mother_expt.analysis = chevron_analysis
#     # station.snapshot_man1_storage_swap(update_main=False)

def dump_sideband_chevron_postproc(station, mother_expt):
    expt_cfg = mother_expt.cfg.expt
    expt_cfg.man_mode_no = 1
    stor_name = f'M1-D{expt_cfg.dump_mode_no}'

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
        
    if best_freq:
        print(f"Best frequency found: {best_freq:.4f} MHz")
        station.ds_storage.update_freq(stor_name, best_freq)
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['frequency'][0] = best_freq
        print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz (ds_storage and hardware_cfg)")

        pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
        station.ds_storage.update_pi(stor_name, pi_len)
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['length'][0] = pi_len
        print('Updated the pi length to:', pi_len, "(ds_storage and hardware_cfg)")

        station.ds_storage.update_h_pi(stor_name, pi_len / 2)
        print('Updated the h_pi length to:', pi_len / 2, "(ds_storage)")

        gain = expt_cfg.get('gain', station.ds_storage.get_gain(stor_name))
        station.ds_storage.update_gain(stor_name, gain)
        station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['gain'][0] = gain
        print('Updated gain to:', gain)
    mother_expt.analysis = chevron_analysis
    chevron_analysis.display_results()
    station.snapshot_man1_storage_swap(update_main=False)
    station.snapshot_hardware_config(update_main=False)


dump_sideband_chevron_runner = SweepRunner(
    station=station,
    ExptClass=meas.single_qubit.sideband_general.SidebandGeneralExperiment,
    default_expt_cfg=dump_sideband_chevron_defaults,
    sweep_param='freq',
    # sweep_param='gain',
    preprocessor=dump_sideband_chevron_preproc,
    # postprocessor=dump_sideband_chevron_postproc,
    job_client=client,
)


# %%
# Run dump chevron for specified dump mode
dump_mode_no = 2
man_mode_no = 1

print(f'Running dump sideband chevron for M{man_mode_no}-D{dump_mode_no}')

# Get current frequency for sweep range
dump_name = f'M{man_mode_no}-D{dump_mode_no}'
center_freq = station.ds_storage.get_freq(dump_name)
# freq_bw = 0.6  # MHz bandwidth for frequency sweep
freq_bw = 4  # MHz bandwidth for frequency sweep

dump_chevron = dump_sideband_chevron_runner.execute(
    dump_mode_no=dump_mode_no,
    man_mode_no=man_mode_no,
    sweep_start=center_freq - freq_bw/2,
    sweep_stop=center_freq + freq_bw/2,
    sweep_npts=30,
    pi_len_sweep=20.0,
    expts=30,
    reps=100,
    gain=5000,
)

# dump_chevron.analysis.display_analysis()


# %%
dump_chevron.data.keys()


# %%
def get_dump_mode_parameters(ds_storage, config_thisrun, man_mode_no, dump_mode_no):
    """
    Get pulse parameters for a given storage mode. 
    Also returns prepulse and postpulse (single photon prep and meas for ge meas)

    Args:
        ds_storage: Dataset object for managing frequency data.
        config_thisrun: Configuration dictionary for the current run.
        man_mode_no: Manipulation mode number.
        dump_mode_no: Dump mode number.

    Returns:
        A tuple containing freq, gain, ch, prepulse, and postpulse.
    """
    stor_name = 'M' + str(man_mode_no) + '-D' + str(dump_mode_no)
    freq = ds_storage.get_freq(stor_name)
    gain = ds_storage.get_gain(stor_name)
    pi_len = ds_storage.get_pi(stor_name)
    h_pi_len = ds_storage.get_h_pi(stor_name)
    ch = 'low' if freq < 1000 else 'high'

    from experiments.MM_dual_rail_base import MM_dual_rail_base
    mm_base_dummy = MM_dual_rail_base(config_thisrun, soccfg=soc)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
    prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist() # for ge meas, only do f0g1 and ef pi

    prepulse_overwrite = [['multiphoton', 'g0-e0', 'pi', 0],
                            ['multiphoton', 'e0-f0', 'pi', 0],
                            ['multiphoton', 'f0-g1', 'pi', 0]
                        ]
    postpulse_overwrite = [ ['multiphoton', 'f0-g1', 'pi', 0],
                            ['multiphoton', 'e0-f0', 'pi', 0]]
    prepulse = mm_base_dummy.get_prepulse_creator(prepulse_overwrite).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(postpulse_overwrite).pulse.tolist()

    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse


def do_dump_spectroscopy(config_thisrun, 
                         ds_storage, 
                         expt_path, 
                         config_path, 
                         man_mode_no = 1, 
                         dump_no = 1,
                         flux_gain = 5000,
                         flux_length = 1):
    """
    Run the Flux Spectroscopy F0g1 Experiment.

    This function performs a flux spectroscopy experiment to measure the transition frequency
    between the f0 and g1 states of a qubit. It configures the experiment parameters, executes
    the experiment, and saves the results.

    Args:
        config_thisrun (AttrDict): Configuration dictionary for the current run.
        ds_storage (dataset.storage_man_swap_dataset): Dataset object for managing frequency data.
        expt_path (str): Path to save the experiment results.
        config_path (str): Path to the configuration file.
        man_mode_no (int, optional): Manipulation mode number (default is 1).
        dump_no (int, optional): Storage mode number (default is 1).

    Returns:
        FluxSpectroscopyF0g1Experiment: The experiment object containing the results.
    """
    flux_spec = meas.single_qubit.rf_flux_spectroscopy_f0g1.FluxSpectroscopyF0g1Experiment(
        soccfg=soc, path=expt_path, prefix='FluxSpectroscopyF0g1Experiment', config_file=config_path
    )

    flux_spec.cfg = AttrDict(deepcopy(config_thisrun))

    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_dump_mode_parameters(ds_storage, config_thisrun,
                                                                      man_mode_no, dump_no)

    freq_start = 2300 #2350 
    freq_stop = 2400 #2400
    bw = freq_stop - freq_start
    expts = 200
    step = bw/expts



    flux_spec.cfg.expt = dict(
        # start=freq - 100,  # Start RF frequency [MHz]
        start=freq_start,  # Start RF frequency [MHz]
        step=step,  # Step size [MHz]
        expts=expts,  # Number of experiments
        reps=200,  # Number of averages per point
        qubit=[0],
        flux_drive=[ch, 1, flux_gain, flux_length],  # RF flux modulation parameters [low/high (ch), freq (will be overwritten), gain, length(us)]
        prepulse=True,
        postpulse=True,
        active_reset=False,
        man_reset=False,
        storage_reset=False,
        pre_sweep_pulse= prepulse,
        post_sweep_pulse= postpulse,
    )

    flux_spec.cfg.device.readout.relax_delay = [2500]  # Wait time between experiments [us]
    flux_spec.go(analyze=False, display=False, progress=True, save=True)
    return flux_spec


def update_dump_spectroscopy(flux_spec, ds_storage, man_mode_no = 1, dump_no = 1):
    """Update the configuration based on Flux Spectroscopy F0g1 experiment results."""
    # Update the dataset with the new frequency
    ds_storage.update_freq('M' + str(man_mode_no) + '-D' + str(dump_no), flux_spec.data['fit'][2])
    print(f"Updated frequency for M{man_mode_no}-D{dump_no}: {flux_spec.data['fit'][2]}")



# %%
spec = do_dump_spectroscopy(config_thisrun, ds_storage, expt_path, config_path, 
                            man_mode_no=1, dump_no=1,
                            flux_gain=8000, flux_length=5)
# analyze_and_display_stor_spectroscopy(spec)
# update_dump_spectroscopy(spec, ds_storage, 1, 1)
spec.analyze(fit=True)
spec.display()

# %%
spec.analyze(fit=True, fitparams=[800, 100, 40, 0, 60, 0, 0])
spec.display()

# %%

# %% [markdown]
# ## Man-coupler

# %%
# get_coupler_parameters has been consolidated into get_mode_parameters above.
# Coupler spectroscopy uses sideband_spec_runner — see execute cell below.

# %%
# coupler_spec_defaults/preproc/postproc/runner have been consolidated into
# sideband_spec_defaults/preproc/postproc/runner above.
# Notable difference vs storage: much wider bw (200 MHz) needed.

# %% 
station.ds_storage.df

# %%
coupler_spec = sideband_spec_runner.execute(
    mode_name='M1-C',
    postproc=False,
    relax_delay=2500,
    reps=100,
    bw=100,
    expts=100,
    flux_drive_gain=1000,
    flux_drive_duration=5,
)
# coupler_spec.display()

# %%

# %% [markdown]
# ## Freq Chevron

# %%
station.ds_storage.update_freq('M1-S4', 883.6)

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

    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'
    # Get storage mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_mode_parameters(
        station.ds_storage, station.hardware_cfg, stor_name
    )

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
        
    if best_freq:
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
    station.snapshot_man1_storage_swap(update_main=False)

sideband_chevron_runner = SweepRunner(
    station=station,
    ExptClass=meas.single_qubit.sideband_general.SidebandGeneralExperiment,
    default_expt_cfg=sideband_chevron_defaults,
    sweep_param='freq',
    preprocessor=sideband_chevron_preproc,
    postprocessor=sideband_chevron_postproc,
    job_client=client,
)

# %%
station.ds_storage.update_gain('M1-S2', 8000)

# %%

# %%
for stor_i in [2]: # range(1,8):
    stor_name = f'M1-S{stor_i}'

    freq_span = 3 # MHz
    freq_step = 0.3
    pi_len_sweep = 2
    gain = station.ds_storage.get_gain(stor_name)
    
    print(f'Running sideband chevron for {stor_name}')
    chevron_analysis = sideband_chevron_runner.execute(
        reps=50,
        stor_mode_no=stor_i,
        sweep_start=station.ds_storage.get_freq(stor_name) - freq_span/2,
        sweep_stop=station.ds_storage.get_freq(stor_name) + freq_span/2,
        sweep_npts=int(freq_span//freq_step + 1),
        pi_len_sweep=pi_len_sweep,
        expts=25,
        # gain=4000,
        # batch=True,
        # debug=True,
        relax_delay=8000,
    )
    # chevron_analysis.analysis.display_results()

# %%
station.ds_storage.update_pi('M1-S1', 1)

# %%
station.snapshot_man1_storage_swap(update_main=False)

# %%
station.ds_storage.df

# %% [markdown]
# ### Coupler

# %%
stor_name = f'M1-C'

freq_span = 12 # MHz
freq_step = 1.2
pi_len_sweep = 0.06
gain = station.ds_storage.get_gain(stor_name)

print(f'Running sideband chevron for {stor_name}')
chevron_analysis = sideband_chevron_runner.execute(
    reps=50,
    stor_mode_no=stor_i,
    sweep_start=station.ds_storage.get_freq(stor_name) - freq_span/2,
    sweep_stop=station.ds_storage.get_freq(stor_name) + freq_span/2,
    sweep_npts=int(freq_span//freq_step + 1),
    pi_len_sweep=pi_len_sweep,
    expts=25,
    # gain=2000,
    # batch=True,
    # debug=True,
    relax_delay=2500,
)

# %%

# %%

# %% [markdown]
# ## Length Rabi

# %%
# Configuration defaults for storage sideband length-rabi (1D time sweep).
# Mirrors the f0g1 length-rabi block (manipulate section), but uses
# SidebandGeneralExperiment driven by (man_mode_no, stor_mode_no) and pulls
# freq/gain from ds_storage (M1-S{stor_mode_no}). Run AFTER the storage
# sideband chevron has refined the centre freq.
#
# NOTE: do NOT put 'freq' or 'gain' in this defaults dict. SidebandGeneralProgram.initialize()
# does `if "freq"/"gain" in cfg.expt: flux_drive[1/2] = cfg.expt.<key>`, so having
# a default-None key would overwrite the value we set in preproc with None.
# Pass gain=<int> at execute() time to override the ds_storage gain.
sideband_lenrabi_defaults = AttrDict(dict(
    start=None,         # default: soc.cycles2us(3); set in preproc
    step=None,          # default: 2 * current pi_len / (expts-1); set in preproc
    expts=50,
    reps=200,
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
    relax_delay=8000,
))


def sideband_lenrabi_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs, "Pass stor_mode_no=<int> at execute time."
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_mode_parameters(
        station.ds_storage, station.hardware_cfg, stor_name
    )

    # Mirror chevron preproc: only honour 'gain' override if explicitly passed.
    if 'gain' in expt_cfg and expt_cfg.gain is not None:
        gain = expt_cfg.gain
    if expt_cfg.start is None:
        expt_cfg.start = station.soccfg.cycles2us(5)
    if expt_cfg.step is None:
        # default sweep range: out to ~2 current pi lengths
        expt_cfg.step = (2 * pi_len - expt_cfg.start) / (expt_cfg.expts - 1)

    expt_cfg.flux_drive       = [ch, freq, gain, 0]
    expt_cfg.pre_sweep_pulse  = prepulse
    expt_cfg.post_sweep_pulse = postpulse

    print(f'Sideband length-rabi for {stor_name}: freq={freq:.3f} MHz, gain={gain}')
    return expt_cfg


def sideband_lenrabi_postproc(station, expt):
    # SidebandGeneralExperiment.analyze() now defaults to fit=True (class
    # patched), so expt.go(analyze=True) populates fit_avgi during go().
    # This explicit call stays as a defensive measure in case someone runs
    # execute(..., go_kwargs={'analyze': False}).
    if 'fit_avgi' not in expt.data:
        expt.analyze(fit=True)

    expt_cfg = expt.cfg.expt
    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'
    p = expt.data['fit_avgi']

    # Same phase-wrap + pi-length math as SidebandGeneralExperiment.display()
    if p[2] > 180:
        p[2] -= 360
    elif p[2] < -180:
        p[2] += 360
    if p[2] < 0:
        pi_length = (1/2 - p[2]/180) / 2 / p[1]
    else:
        pi_length = (3/2 - p[2]/180) / 2 / p[1]
    pi2_length = pi_length / 2

    # Guard against degenerate fits (e.g. mock zeros, noisy real data).
    if not (np.isfinite(pi_length) and np.isfinite(pi2_length)):
        print(f'{stor_name}: fit produced non-finite pi length '
              f'(pi={pi_length}, pi/2={pi2_length}); skipping ds_storage update.')
        return

    station.ds_storage.update_pi(stor_name, pi_length)
    station.ds_storage.update_h_pi(stor_name, pi2_length)

    # Only push gain back when the user explicitly passed it at execute time —
    # otherwise it's a no-op write of the same value the chevron just set.
    gain_overridden = (
        'gain' in expt.cfg.expt
        and expt.cfg.expt.get('gain') is not None
        and expt.cfg.expt.gain == expt_cfg.flux_drive[2]
    )
    if gain_overridden:
        station.ds_storage.update_gain(stor_name, expt_cfg.flux_drive[2])
        print(f'Updated {stor_name}: pi={pi_length:.4f} us, pi/2={pi2_length:.4f} us, '
              f'gain={expt_cfg.flux_drive[2]} (overridden)')
    else:
        print(f'Updated {stor_name}: pi={pi_length:.4f} us, pi/2={pi2_length:.4f} us')

    station.snapshot_man1_storage_swap(update_main=False)


sideband_lenrabi_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.sideband_general.SidebandGeneralExperiment,
    default_expt_cfg=sideband_lenrabi_defaults,
    preprocessor=sideband_lenrabi_preproc,
    postprocessor=sideband_lenrabi_postproc,
    job_client=client,
)

# %%
sideband_lenrabis = {}
for stor_mode_no in [1,2]: # range(1, 8):  # storage modes M1-S1..M1-S7
    print(f'Running sideband length-rabi for M1-S{stor_mode_no}')
    sideband_lenrabis[stor_mode_no] = sideband_lenrabi_runner.execute(
        stor_mode_no=stor_mode_no,
    )

# %%

# %% [markdown]
# ## Error amplification

# %%
error_amp_gain1 = [None] * 7 # len(expts_to_run['stor_modes'])
error_amp_freq1 = [None] * 7 # len(expts_to_run['stor_modes'])
error_amp_gain2 = [None] * 7 # len(expts_to_run['stor_modes'])
error_amp_freq2 = [None] * 7 # len(expts_to_run['stor_modes'])

# %%
error_amp_stor_defaults = AttrDict(dict(
    reps=50,
    rounds=1,
    qubits=[0],
    man_mode_no=1,
    stor_is_dump=False,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=8000, 
    expts=30,
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

error_amp_gain_stor_coarse_defaults = AttrDict(dict(
    n_start=0,
    n_step=3,
    n_pulses=15,
    expts=50,
))

error_amp_freq_stor_coarse_defaults = AttrDict(dict(
    n_start=0,
    n_step=1,
    n_pulses=5,
    expts=50,
))


def error_amp_stor_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    assert 'parameter_to_test' in kwargs 

    # construct the defaults
    expt_cfg = deepcopy(default_expt_cfg)
    if kwargs['parameter_to_test'] == 'gain':
        expt_cfg.update(error_amp_gain_stor_coarse_defaults)
    elif kwargs['parameter_to_test'] == 'frequency':
        expt_cfg.update(error_amp_freq_stor_coarse_defaults)

    # override with the passed kwargs
    expt_cfg.update(kwargs)

    stor_mode_no = expt_cfg.stor_mode_no    
    man_mode_no = expt_cfg.man_mode_no

    stor_name = f'M{man_mode_no}-S{stor_mode_no}'
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_mode_parameters(
        station.ds_storage, station.hardware_cfg, stor_name
        )
    print(f"Previous freq {freq}, gain {gain}, pi_len {pi_len}")

    if 'span' not in expt_cfg:
        if expt_cfg.parameter_to_test == 'gain':
            expt_cfg.span = int(gain * 0.3)
        elif expt_cfg.parameter_to_test == 'frequency':
            expt_cfg.span = 0.15 # MHz
  
    stor_name = f'M{man_mode_no}-S{stor_mode_no}'
    pulse_type = ['storage', stor_name, 'pi', 0]

    if expt_cfg.parameter_to_test == 'frequency':
        start = freq - expt_cfg.span / 2
        step = expt_cfg.span / (expt_cfg.expts - 1)
    elif expt_cfg.parameter_to_test == 'gain':
        start = int(gain - expt_cfg.span / 2)
        step = int(expt_cfg.span / (expt_cfg.expts - 1))
    else:
        raise ValueError("parameter_to_test must be either 'frequency' or 'gain'.")
    expt_cfg.start = start
    expt_cfg.step = step
    expt_cfg.pulse_type = pulse_type 
    return expt_cfg

def error_amp_stor_postproc(station, expt):
    expt.analyze(data=expt.data, state_fin='e')

    opt_gain = expt.data['fit_avgi'][2]
    stor_name = 'M1-S' + str(expt.cfg.expt.stor_mode_no)
    if expt.cfg.expt.parameter_to_test == 'gain':
        station.ds_storage.update_gain(stor_name, opt_gain)
    elif expt.cfg.expt.parameter_to_test == 'frequency':
        station.ds_storage.update_freq(stor_name, opt_gain)
    station.snapshot_man1_storage_swap(update_main=False)


error_amp_stor_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_stor_defaults,
    preprocessor=error_amp_stor_preproc,
    postprocessor=error_amp_stor_postproc,
    job_client=client,
)

# %%
for i, stor_i in enumerate([2]):
    stor_name = 'M1-S' + str(stor_i)
    print("Running", stor_name)
    error_amp_freq1[i] = error_amp_stor_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='frequency',
        span=0.2,
        n_step=2,
        n_pulses=10,
    )
    # error_amp_freq1[i].display()

    error_amp_gain1[i] = error_amp_stor_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='gain',
        # span=5000,
    )
    # error_amp_gain1[i].display()

# %%
station.snapshot_man1_storage_swap(update_main=False)

# %%
station.snapshot_hardware_config(update_main=False)

# %% [markdown]
# ## f0g1 spectroscopy vs flux (2D)
# This section wraps the new `PulseProbeF0g1SpectroscopyFluxSweepExperiment` into a convenience function using the same config approach as the 1D f0g1 example.

# %%
# Helper: do_f0g1_versus_flux_spectroscopy()
from pathlib import Path
from typing import Optional
from copy import deepcopy
import yaml
from slab import AttrDict
from experiments.single_qubit.pulse_probe_f0g1_spectroscopy import (
    PulseProbeF0g1SpectroscopyFluxSweepExperiment,
)

def do_f0g1_versus_flux_spectroscopy(
    *,
    config_thisrun,
    ds_storage,
    man_mode_no: int = 1,
    prefix: str = "PulseProbeF0g1SpectroscopyFluxSweep",
    # DC current sweep
    curr_start: float = -1,
    curr_step: float = 0.001,
    curr_expts: int = 21,
    yokogawa_address: str = "192.168.137.148",
    sweeprate: float = 2,
    safety_limit: float = 10,
    # frequency sweep overrides (optional, follow the 1D example)
    freq_start: Optional[float] = None,
    freq_step: Optional[float] = None,
    freq_expts: Optional[int] = None,
    # book-keeping
    save: bool = True,
    progress: bool = False,
    qubit_f: bool = True,
    qubits: list = [0],
    prepulse: bool = False,
    pre_sweep_pulse: Optional[list] = None,
):
    """
    Configure and run the 2D f0g1 spectroscopy vs flux using the same conventions
    as the 1D PulseProbeF0g1SpectroscopyExperiment above. Returns the saved H5 filename.
    """
    # Resolve frequency defaults from dataset if not provided
    if freq_start is None:
        try:
            base_f = ds_storage.get_freq('M' + str(man_mode_no))
        except Exception:
            base_f = ds_storage.get_freq('M1')
        if base_f is None:
            raise ValueError("Could not resolve base frequency from ds_storage.")
        freq_start = base_f - 5  # MHz offset similar to 1D convention
    if freq_step is None:
        freq_step = 1.0
    if freq_expts is None:
        freq_expts = 51

    # Build optional prepulse if requested and not provided
    print(pre_sweep_pulse, prepulse)
    if prepulse and pre_sweep_pulse is None:
        # mm_base_calib is prepared earlier in the notebook alongside config_thisrun/ds_storage
        mm_base_dummy = MM_dual_rail_base(config_thisrun, soccfg=soc)
        pre = mm_base_dummy.prep_man_photon(man_no=man_mode_no, photon_no=0)
        pre.append(['multiphoton', 'g0-e0', 'pi', 0])
        pre.append(['multiphoton', 'e0-f0', 'pi', 0])
        pre_sweep_pulse = mm_base_dummy.get_prepulse_creator(pre, config_thisrun).pulse.tolist()
        print('prep pulse', pre_sweep_pulse)


    # Instantiate the 2D experiment. Use global paths/config used elsewhere in the notebook
    exp = PulseProbeF0g1SpectroscopyFluxSweepExperiment(
        soccfg=soc if 'soc' in globals() else None,
        path=str(expt_path),
        prefix=prefix,
        config_file=str(config_file),
    )

    # Attach this-run configuration and expt overrides
    exp.cfg = AttrDict(deepcopy(config_thisrun))

    ex_overrides = dict(
        # Frequency sweep (inner axis)
        start=freq_start,
        step=freq_step,
        expts=freq_expts,
        reps=250,
        rounds=1,
        length=1,
        gain=5000,
        pulse_type='gaussian',
        qubit_f=bool(qubit_f),
        qubits=list(qubits),
        prepulse=bool(prepulse),
        pre_sweep_pulse=(pre_sweep_pulse if prepulse else []),
        # Current sweep (outer axis)
        curr_start=curr_start,
        curr_step=curr_step,
        curr_expts=curr_expts,
        yokogawa_address=yokogawa_address,
        sweeprate=sweeprate,
        safety_limit=safety_limit,
    )

    # Ensure an expt section exists and populate overrides
    if not hasattr(exp.cfg, 'expt') or exp.cfg.expt is None:
        exp.cfg.expt = AttrDict()
    for k, v in ex_overrides.items():
        try:
            exp.cfg.expt[k] = v
        except Exception:
            setattr(exp.cfg.expt, k, v)

    exp.go(analyze=False, display=True, progress=progress, save=save)
    return exp

# %%
# Call example: run the 2D f0g1 vs flux experiment

f_start = 1970
f_stop = 2025
f_expt = 500
freq_step = (f_stop - f_start)/f_expt

c_start = 0.30
c_stop = 1
# c_stop = 1 
c_expt = 10
c_step = np.abs(c_stop - c_start)/c_expt


prefix_2d = "PulseProbeF0g1SpectroscopyFluxSweep"
fname_2d = do_f0g1_versus_flux_spectroscopy(
    config_thisrun=config_thisrun,
    ds_storage=ds_storage,
    man_mode_no=1,
    prefix=prefix_2d,
    # DC current sweep
    curr_start=c_start,  # -10 mA
    curr_step=c_step,    # 1 mA steps
    curr_expts=c_expt,      # total points
    # Frequency sweep (mirror the 1D example defaults; override if needed)
    # freq_start=..., freq_step=..., freq_expts=...,
    save=True,
    progress=True,
    qubit_f=True,
    qubits=[0],
    prepulse=True,
    freq_start=f_start,
    freq_step = freq_step,
    freq_expts=f_expt,
)
# print("Saved:", fname_2d)

# %% [markdown]
# # Coupler

# %% [markdown]
# ## Length Rabi

# %%
station.ds_storage.get_freq('M1-C') 

# %%
# === LengthRabiCouplerFreqsweep: defaults + preproc ========================
lenrabi_coupler_defaults = AttrDict(dict(
    # Outer axis: flat-segment length (software loop, us)
    start=0.007,
    step=5,
    expts=21,
    # Inner axis: drive freq (hardware NDAverager). Filled in by preproc.
    # freq_start, freq_stop, freq_expts <- preproc
    gain=20000,         # DAC, fixed
    ramp_sigma=0.005,    # [us] gauss ramp sigma of swept pulse (fixed)
    reps=100,
    rounds=1,
    qubits=[0],
    relax_delay=800,    # [us]
))

def lenrabi_coupler_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)


    fge, fef =  station.hardware_cfg.device.qubit.f_ge[0], station.hardware_cfg.device.qubit.f_ef[0]
    ff0g1 = station.ds_storage.get_freq('M1')
    fman = fge+fef-ff0g1
    fm1c = station.ds_storage.get_freq('M1-C')
    fcoupler_guess = fman-fm1c
    
    freq_center = kwargs.pop('freq_center', fcoupler_guess)
    freq_span = kwargs.pop('freq_span', 5.0)          # [MHz]
    freq_expts = kwargs.pop('freq_expts', 51)
    
    expt_cfg.freq_start = freq_center - freq_span / 2
    expt_cfg.freq_stop  = freq_center + freq_span / 2
    expt_cfg.freq_expts = freq_expts
    
    expt_cfg.update(kwargs)
    return expt_cfg

# %%
# === LengthRabiCouplerFreqsweep: runner + execute ==========================
lenrabi_coupler_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.LengthRabiCouplerFreqsweepExperiment,
    default_expt_cfg=lenrabi_coupler_defaults,
    preprocessor=lenrabi_coupler_preproc,
    job_client=client,
)

lenrabi_coupler = lenrabi_coupler_runner.execute(
    # freq_center = 4448,
    freq_span = 50,
    freq_expts = 26,
    start=0.007,
    step=0.02,
    expts=31,  # start=0.0, step=0.05, expts=21,
    gain=20000,         # DAC, fixed
    reps=100,
)
# lenrabi_coupler.display()

# %%
plt.plot(lenrabi_coupler.data['ypts'], lenrabi_coupler.data['avgi'][:,1])

# %%

# %% [markdown]
# ## Amplitude Rabi

# %%
# === AmplitudeRabiCouplerFreqsweep: defaults + preproc =====================
amprabi_coupler_defaults = AttrDict(dict(
    # Outer axis: gain (software loop, DAC units)
    start=0,
    step=2000,
    expts=11,
    # Inner axis: drive freq (hardware NDAverager). Filled in by preproc.
    # freq_start, freq_stop, freq_expts <- preproc
    flat_length=5,    # [us] flat segment of swept pulse (fixed)
    ramp_sigma=0.005,    # [us] gauss ramp sigma of swept pulse (fixed)
    reps=100,
    rounds=1,
    qubits=[0],
    relax_delay=800,    # [us]
))


# %%
# === AmplitudeRabiCouplerFreqsweep: runner + execute =======================
amprabi_coupler_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.AmplitudeRabiCouplerFreqsweepExperiment,
    default_expt_cfg=amprabi_coupler_defaults,
    preprocessor=lenrabi_coupler_preproc,
    job_client=client,
)

amprabi_coupler = amprabi_coupler_runner.execute(
    freq_span = 100, 
    freq_expts = 201,
    flat_length = 1,
    start=30000,
    step=200,
    expts=1,
)

# %%
station.ds_storage.df

# %%

# %% [markdown]
# # Cooling

# %% [markdown]
# ## setup

# %%
cool_spec_defaults = AttrDict(dict(
    reps=100,              # repetitions
    rounds=1,             # rounds
    qubits=[0],           # qubits
    init_stor = 7,
    ro_stor = 7,
    init_fock=True,
    # cooling_freq=7150,      # Ramsey frequency
    cooling_gain=5000,
    cooling_length=1,
    swept_params=['cooling_freq'], #, 'cooling_gain'],
    cooling_freqs = np.linspace(2500, 2600, 101).tolist(),
    charge_freq = 4450,
    charge_gain = 2000,
    active_reset=False,   # active reset
    man_reset=True,       # manipulate reset
    storage_reset=True,   # storage reset
    relax_delay=8000,

    normalize=False,
    preloaded_pulses=False,
    perform_wigner=False,
    prepulse=True,
    postpulse=True,
    pre_sweep_pulse=None, # pre sweep pulse
    post_sweep_pulse=None,# post sweep pulse
))

cool_spec_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.qsim.cooling.CoolingSpectroscopyProgram,
    default_expt_cfg=cool_spec_defaults,
    # preprocessor=kerr_ramsey_preproc,
    # postprocessor=kerr_ramsey_postproc,
    job_client=client,
)


# %% [markdown]
# ## flux sweep

# %%
def calibrate_f0g1(coupler_current):
    branch_name = f'f0g1cal_coupler{coupler_current*1e3:.2f}'
    bm.branch(branch_name)

    # f0g1 spectroscopy
    man_spec = f0g1spec_runner.execute(
        man_mode_no=1,
        start=1965,
        expts=300,
        step=0.2,
        active_reset=False,
        relax_delay=1500,
        reps=50,
    )

    # coarse chevron
    f0g1_chevron_coarse = runner.execute(
        sweep_start=station.ds_storage.get_freq('M1') - 3,
        sweep_stop=station.ds_storage.get_freq('M1') + 3,
        sweep_npts=11,
        gain = station.ds_storage.get_gain('M1'),
        start = 1, # time start in us
        batch = True,
    )
    
    # fine chevron
    f0g1_chevron_fine = runner.execute(
        sweep_start=station.ds_storage.get_freq('M1') - 0.5,
        sweep_stop=station.ds_storage.get_freq('M1') + 0.5,
        sweep_npts=11,
        gain = station.ds_storage.get_gain('M1'),
        # start = 1, # time start in us
        # coupler_current=coupler_current,
        batch=True
    )

    # freq erramp
    error_amp_exp = error_amp_runner.execute(
        start=station.hardware_cfg.device.multiphoton.pi['fn-gn+1'].frequency[0]-0.5,
        go_kwargs = dict(analyze=False, display=False),
        postprocess=True  # This will call postprocessor which does the custom analysis
    )

    # length rabi
    len_rabis_man = lenrabi_f0g1_runner.execute(
        man_mode_no=1,
        gain=station.ds_storage.get_gain('M1'),
        step=0.01,
        go_kwargs=dict(analyze=False),
    )

    bm.commit(branch_name)
    print(f'{branch_name} committed')

    return man_spec, f0g1_chevron_coarse, f0g1_chevron_fine, error_amp_exp, len_rabis_man



# %%
all_expt_objs = []
for coupler_current in np.linspace(-0.1e-3,0.2e-3,31):
    station.hardware_cfg.hw.yoko_coupler.current = coupler_current
    expt_objs = calibrate_f0g1(coupler_current)
    all_expt_objs.append(expt_objs)

# %%
all_specs = {}

for charge_gain in [0,8000,15000]:
    for cooling_gain in [3000,8000,15000]:
        cool_specs = []
        for coupler_current in np.linspace(-0.1e-3,0.2e-3,31):
            branch_name = f'f0g1cal_coupler{coupler_current*1e3:.2f}'
            bm.checkout(branch_name, force=True)
            cool_spec = cool_spec_runner.execute(
                cooling_gain = cooling_gain,
                charge_freq = 4600,
                charge_gain = charge_gain,
                cooling_length = 10,
                cooling_freqs = np.linspace(2500,3000,251),
                ramp_sigma = 0.01,
                swept_params = ['cooling_freq'],
                init_stor = 0,
                prepulse=True,
                ro_stor = 0,
                reps = 50,
                relax_delay = 2500,
            )
            cool_specs.append(cool_spec)
        
        avgis = np.array([cs.data['avgi'] for cs in cool_specs])
        fnames = [cs.fname for cs in cool_specs]
        
        all_specs[f'charge{charge_gain}flux{cooling_gain}'] = {'fnames': fnames, 'avgi': avgis}


# %% [markdown]
# ## Run

# %% [markdown]
# Cooling requires
#
# $$
# H_\text{4WM} \propto c^\dagger c^\dagger s_{i-1}^\dagger s_i
# $$
#
# This is strongest 4WM that involves two cavity modes. 
#
# But coupler freq $f^{gf}_\text{coupler}-\text{FSR} \gg $ on chip filter cutoff (might give 7 to 8GHz a shot?)
#
# To probe the Rabi rate and linewidth of similar processes with accessible freq drives, try
#
# $$
# H_\text{4WM} \propto c^\dagger c^\dagger m_1 s_i
# $$
#
# drive at $f_\text{M1} + f_{\text{S}i} - f^{gf}_\text{coupler}$
#
# Another nice property is this requires a two-photon state to be prepared and both photons to go away so we can throw out DUST more easily

# %%
cool_spec = cool_spec_runner.execute(
    cooling_gain = 8000,
    charge_gain = 0,
    charge_freq = 4642,
    cooling_length = 100,
    # cooling_lengths = np.linspace(0.01, 1, 51).tolist(),
    cooling_freqs = np.linspace(1650,1750,101).tolist(),
    ramp_sigma = 0.01,
    # swept_params = ['cooling_freq', 'cooling_length'],
    swept_params = ['cooling_freq'],
    init_stor = [1], 
    prepulse=True,
    ro_stor = 0,
    reps = 50,
    relax_delay = 2500,
)

# %%
cool_spec = cool_spec_runner.execute(
    cooling_gain = 8000,
    charge_gain = 0,
    charge_freq = 4642,
    cooling_length = 100,
    # cooling_lengths = np.linspace(0.01, 1, 51).tolist(),
    cooling_freqs = np.linspace(1650,1750,101).tolist(),
    ramp_sigma = 0.01,
    # swept_params = ['cooling_freq', 'cooling_length'],
    swept_params = ['cooling_freq'],
    init_stor = [1], 
    prepulse=False,
    ro_stor = 0,
    reps = 50,
    relax_delay = 2500,
)

# %%
cool_specs = []
for ro_stor in [0,1,2]:
    cool_spec = cool_spec_runner.execute(
        cooling_gain = 8000,
        charge_gain = 0,
        charge_freq = 4642,
        # cooling_length = 60,
        cooling_lengths = np.linspace(5, 150, 30).tolist(),
        cooling_freqs = np.linspace(1200,1275,51).tolist(),
        ramp_sigma = 0.01,
        swept_params = ['cooling_freq', 'cooling_length'],
        # swept_params = ['cooling_freq'],
        init_stor = [1,0], 
        prepulse=True,
        ro_stor = ro_stor,
        reps = 200,
        relax_delay = 8000,
    )
    cool_specs.append(cool_spec)

# %%
cool_specs = []
for ro_stor in [0,1,2]:
    cool_spec = cool_spec_runner.execute(
        cooling_gain = 8000,
        charge_gain = 0,
        charge_freq = 4642,
        # cooling_length = 60,
        cooling_lengths = np.linspace(5, 150, 30).tolist(),
        cooling_freqs = np.linspace(1360,1420,51).tolist(),
        ramp_sigma = 0.01,
        swept_params = ['cooling_freq', 'cooling_length'],
        # swept_params = ['cooling_freq'],
        init_stor = [2,0], 
        prepulse=True,
        ro_stor = ro_stor,
        reps = 200,
        relax_delay = 8000,
    )
    cool_specs.append(cool_spec)

# %%
cool_specs = []
for ro_stor in [0,1,2]:
    cool_spec = cool_spec_runner.execute(
        cooling_gain = 8000,
        charge_gain = 0,
        charge_freq = 4642,
        # cooling_length = 60,
        cooling_lengths = np.linspace(5, 150, 30).tolist(),
        cooling_freqs = np.linspace(1680,1750,51).tolist(),
        ramp_sigma = 0.01,
        swept_params = ['cooling_freq', 'cooling_length'],
        # swept_params = ['cooling_freq'],
        init_stor = [2,1], 
        prepulse=True,
        ro_stor = ro_stor,
        reps = 200,
        relax_delay = 8000,
    )
    cool_specs.append(cool_spec)


# %% [markdown]
# ## Plotting


# %%
f0g1_specs = [eo[0] for eo in all_expt_objs]
f0g1_avgis = np.array([fs.data['avgi'] for fs in f0g1_specs])

plt.figure(figsize=(12,4))
plt.pcolormesh(f0g1_specs[0].data['xpts'], 1e3*coupler_currents, f0g1_avgis)
plt.colorbar(label='avgi')
plt.xlabel('f0g1 (MHz)')
plt.ylabel('Coupler current (mA)')

# %%
coupler_currents = np.linspace(-0.1e-3,0.2e-3,31)

# %%
plt.figure(figsize=(12,4))
plt.pcolormesh(cool_specs[0].data['xpts'], coupler_currents*1e3, avgis)
plt.colorbar(label='avgi')
plt.xlabel('Coupler drive (MHz)')
plt.ylabel('Coupler current (mA)')

# %%
avgis = np.array([cs.data['avgi'] for cs in cool_specs])

plt.figure(figsize=(8,4))
plt.pcolormesh(cs.data['xpts'], coupler_currents*1e3, avgis)
plt.colorbar(label='avgi')
plt.xlabel('Coupler drive (MHz)')
plt.ylabel('Coupler current (mA)')
plt.xlim([2550, 2600])

# %%

for key, result in all_specs.items():
    fnames = result['fnames']
    avgis = result['avgi']
    plt.figure(figsize=(8,4))
    plt.pcolormesh(cool_specs[0].data['xpts'], coupler_currents*1e3, avgis)
    plt.colorbar(label='avgi')
    plt.xlabel('Coupler drive (MHz)')
    plt.ylabel('Coupler current (mA)')
    plt.title(key)

# %%
coupler_current = 0.02e-3
branch_name = f'f0g1cal_coupler{coupler_current*1e3:.2f}'
bm.checkout(branch_name, force=True)


# %%
for coupler_current in [0]:
    station.hardware_cfg.hw.yoko_coupler.current = coupler_current
    calibrate_f0g1()

    cool_spec = cool_spec_runner.execute(
        cooling_gain = 8000,
        charge_gain = 0,
        cooling_length = 1,
        cooling_freqs = np.linspace(2300,2700,201),
        ramp_sigma = 0.01,
        swept_params = ['cooling_freq'],
        init_stor = 7,
        prepulse=False,
        ro_stor = 0,
        reps = 50,
        relax_delay = 800,
    )

# %%
for cooling_gain in [5000, 10000, 15000]:
    for init_stor in [0, 1, 2]:
        for ro_stor in [0, 1, 2]:
            cool_spec = cool_spec_runner.execute(
                cooling_gain = cooling_gain,
                charge_gain = 0,
                charge_freq = 4642,
                # cooling_length = 10,
                cooling_lengths = np.linspace(0.01, 5, 51).tolist(),
                cooling_freqs = np.linspace(2775,2800,51).tolist(),
                ramp_sigma = 0.01,
                swept_params = ['cooling_freq', 'cooling_length'],
                # swept_params = ['cooling_freq'],
                init_stor = init_stor,
                prepulse=True,
                ro_stor = ro_stor,
                reps = 100,
                relax_delay = 8000,
            )

# %%
for cooling_gain in [5000]:
    for init_stor in [0, 1, 2]:
        for ro_stor in [0, 1, 2]:
            cool_spec = cool_spec_runner.execute(
                cooling_gain = cooling_gain,
                charge_gain = 5000,
                charge_freq = 4324,
                # cooling_length = 10,
                cooling_lengths = np.linspace(0.01, 5, 51).tolist(),
                cooling_freqs = np.linspace(2775,2800,51).tolist(),
                ramp_sigma = 0.01,
                swept_params = ['cooling_freq', 'cooling_length'],
                # swept_params = ['cooling_freq'],
                init_stor = init_stor,
                prepulse=True,
                ro_stor = ro_stor,
                reps = 100,
                relax_delay = 8000,
            )



