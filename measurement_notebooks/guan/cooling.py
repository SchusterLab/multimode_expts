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
# # Cooling
#
# Split out of `qsim_experiments.py`, where it had been copied from an
# earlier experiment of Guan's. The surface map is explicit that this is not
# a migration target -- it is here to stay runnable and findable, not to be
# rewritten. The library side is `experiments/qsim/cooling.py`.
#
# The scratch cells at the end (register/frequency conversions and a bare
# qick `LoopbackProgram`) came along with it: they are hardware-poking
# one-liners with no project of their own.
#
# Its neighbours: `floquet_calibration.py`, `dark_mode.py`,
# `flux_excursion.py`, `mbramsey.py`.

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
from experiments import MultimodeStation, CharacterizationRunner

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

# %%
# Who is running these experiments??
user = 'guan'

print(f"Welcome {user}!")

# %% editable=true slideshow={"slide_type": ""}
# Initialize station to retrieve soc and configs
station = MultimodeStation(
    user = user,
    experiment_name = "260513_qsim",
    
    hardware_config="CFG-HW-20260709-00019",
    storage_man_file="CFG-M1-20260707-00020",
    floquet_file="CFG-FL-20260707-00053",
    # multiphoton_config="versions/multiphoton_config/CFG-MP-20260115-00001.yml",
    log_measurements=True,
)

# %%
# The swap-parameter reader and the readout-calibration postprocessor used to
# be defined in this cell. Seven notebooks had byte-identical copies, and both
# of them write or read calibration, so they are modules now.
from experiments.qsim.utils import floquet_pulse_parameters
from experiments.readout_calibration import apply_singleshot_calibration

get_floquet_parameters = floquet_pulse_parameters  # old name, same function



# %% [markdown]
# ## Datset for Sidebands

# %%
station.ds_storage.df

# %%
# Override/initalize using the storage dataset
station.ds_floquet.import_from_swap_dataset(station.ds_storage, gain_div=1, 
                                    pi_div=15)
station.snapshot_floquet_storage_swap(update_main=False)

# %%
station.ds_floquet.df

# %% [markdown]
# # Single shot

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
    # cavity_freq=4984.373226159381,
    # cavity_gain=400,
    # cavity_length=2,
    prepulse=False,
    pre_sweep_pulse=None,
    gate_based=True,
    qubits=[0],
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

# `apply_singleshot_calibration` replaces the local
# `singleshot_postproc` this cell used to define.


# %%
# Execute
# =================================
ss_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.HistogramExperiment,
    default_expt_cfg = singleshot_defaults,
    postprocessor = apply_singleshot_calibration,
    job_client=client,
)

ss = ss_runner.execute(
    check_f=False,
    active_reset=False, # on recalibration of readout, turn off active reset because it will be wrong for selecting when to apply the qubit pulse
    relax_delay=2000,
    # active_reset=True,
    # relax_delay=200,
    priority=1,
)
# ss.display(station)

# %%
station.hardware_cfg.data_management['vault_root'] = 'H:/Shared drives/Slab/Multimode'

# %%
station.snapshot_hardware_config(update_main=False)
station.preview_config_update()

# %% [markdown]
# # Cooling

# %% [markdown]
# ##  Spectroscopy

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

# %%
cool_spec = cool_spec_runner.execute(
    # cooling_gains=[5000,10000,15000,20000,25000,30000],
    cooling_gain = 8000,
    charge_gain = 0,
    # cooling_lengths=np.linspace(0.1,10,51),
    cooling_length = 1,
    cooling_freqs = np.linspace(2540,2640,101),
    ramp_sigma = 0.1,
    # swept_params = ['cooling_length', 'cooling_freq'],
    swept_params = ['cooling_freq'],
    init_stor = 7,
    prepulse=False,
    ro_stor = 0,
    reps = 50,
)

# %%
cool_spec = cool_spec_runner.execute(
    # cooling_gains=[5000,10000,15000,20000,25000,30000],
    cooling_gain = 25000,
    cooling_lengths=np.linspace(0.1,10,51),
    # cooling_length = 10,
    cooling_freqs = np.linspace(7097,7100,101),
    swept_params = ['cooling_length', 'cooling_freq'],
    # swept_params = ['cooling_freq'],
    init_stor = 3,
    prepulse=False,
    ro_stor = 3,
    reps = 100,
)

# %% [markdown]
# ###### cool_specs = []
# for init_stor in [0]:
#     for ro_stor in [3]:
#         cool_spec = cool_spec_runner.execute(
#             cooling_gain=25000,
#             # cooling_lengths=np.linspace(0.1,10,51),
#             cooling_length = 2,
#             # cooling_freq = 7098.3,
#             cooling_freqs = np.linspace(7097,7100,101),
#             # swept_params = ['cooling_length', 'cooling_freq'],
#             swept_params = ['cooling_freq'],
#             # swept_params = ['cooling_length'],
#             prepulse=False,
#             postpulse=True,
#             init_stor = init_stor,
#             ro_stor = ro_stor,
#             reps = 100,
#         )
#         cool_specs.append(cool_spec)

# %% editable=true slideshow={"slide_type": ""}
cool_specs = []
for cooling_gain in [25000]:
    cool_spec = cool_spec_runner.execute(
        cooling_gain=cooling_gain,
        cooling_lengths=np.linspace(0.1,10,51),
        # cooling_length = 10,
        cooling_freq = 7098.7,
        # cooling_freqs = np.linspace(7050,7150,201),
        # swept_params = ['cooling_length', 'cooling_freq'],
        # swept_params = ['cooling_freq'],
        swept_params = ['cooling_length'],
        prepulse=False,
        postpulse=True,
        init_stor = 3,
        ro_stor = 3,
        reps = 100,
    )
    cool_specs.append(cool_spec)

# %% editable=true slideshow={"slide_type": ""}
cool_spec.display()

# %%
css.append(cool_spec)

# %%

# %%
for cs in cool_specs:
    plt.plot(cs.data['xpts'], cs.data['avgi'])

plt.legend(['M1', 'S1', 'S2', 'S3', 'S7'])

# %%

# %%

# %%
css = [cool_spec]

# %%
fig, axs = plt.subplots(nrows=5, figsize=(8,12))
colors = plt.get_cmap('coolwarm')([0,0.25,0.5,0.75,1])

for num, cs in enumerate(cool_specs):
    init_mode = ['M1', 'S1', 'S2', 'S3', 'S7'][num//5]
    ro_mode = ['M1', 'S1', 'S2', 'S3', 'S7'][num%5]
    axs[num//5].plot(cs.data['xpts'], cs.data['avgi'], alpha=0.8, color=colors[num%5], label=f'{ro_mode}')
    ax.set_title(f'init: {init_mode}')

for ax in axs:
    ax.set_ylabel('avgi')
    ax.legend(ncols=3, fontsize=10)

axs[4].set_xlabel('freq (MHz)')
fig.tight_layout()

# %%
from experiments import QsimBaseExperiment

# %%
cs = QsimBaseExperiment.from_h5file(station.data_path / 'JOB-20260427-00508_QsimBaseExperiment.h5')

# %%
cs.display()

# %%
np.array(cool_spec.data['idata']).shape

# %%
plt.scatter(np.array(cool_spec.data['idata'])[30:40], 
            np.array(cool_spec.data['qdata'])[30:40], 
            alpha=0.4, marker='.')

# %%

# %%
import pickle
with open(r'D:/experiments/260410_qsim/expt_objs/JOB-20260416-00260_expt.pkl', 'rb') as f:
    tp = pickle.load(f)

# %%
tp.display()

# %%

# %%
[cs.fname.split('\\')[-1] for cs in cool_specs]

# %%
avgis = np.array([cs.data['avgi'] for cs in cool_specs])
xpts = cool_specs[0].data['xpts']
plt.pcolormesh(xpts, [1000,3000,10000,30000], avgis)
plt.xlabel('flux drive  freq (MHz)')
plt.ylabel('flux drive gain')
plt.colorbar(label='avgi')

# %%
cool_spec.display()

# %%

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
6389.76-1210.24

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
        "reps":1000, # --Fixed
        "relax_delay":2.0, # --us
        "res_phase":0, # --degrees
        "pulse_style": "const", # --Fixed
        
        "length":1000, # [Clock ticks]
        "readout_length":10, # [Clock ticks]

        "pulse_gain":20000, # [DAC units]
        "pulse_freq": 2250, # [MHz]
        "nyquist_zone": 1,
        
        "adc_trig_offset": 100, # [Clock ticks]
        "soft_avgs":5000
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

# %%

# %%
