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
# # Floquet pulse calibration (preloaded flat-top)
#
# What the MBR campaigns need before they run: the Floquet swap pulses of the
# storage modes, calibrated in three steps, and a bare-readout check.
#
# 1. **Gain chevron** (detuning x gain): the frequency and the pi/N gain.
# 2. **Error amplification**: frequency and gain, coarse then fine.
# 3. **Phase accumulation**: the Stark phase each swap leaves on the others.
# 4. **Bare readout check**: one photon scrambled and read out of every mode.
#
# The Floquet pulse is the preloaded flat-top (`docs/qsim/mbr_step9_plan.md`,
# decision 0.3). The synthesized 3-segment flat-top and the Gaussian procedures
# are in `dormant/floquet_calibration_all_envelopes.py`. The `ds_floquet` dataset
# is initialized at the tail of `multiphoton_calibration.py`.
#
# Each step is defaults -> pre/post hooks -> runner -> `execute` with kwarg
# overrides, as in the single-qubit autocalibrate notebooks: the hooks are in
# the cells, the fits are the experiments' own `analyze`. The defaults of source
# cells 2-6 are in `experiments/qsim/notebook_helpers/defaults.py`.
#
# Split out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
# 58-60, 100-122 and 150-158 by the stage-2 notebook decomposition.
#
# Its neighbours: `multiphoton_calibration.py`, `mbr.py`, `mbr_disorder.py`,
# `mbr_tomography.py`, `floquet_displacement_kerr.py`.

# %%
# %load_ext autoreload
# %autoreload 2

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import experiments as meas
from slab import AttrDict
from experiments import CharacterizationRunner, MultimodeStation

from job_server import JobClient
# Imported under the names the relocated cells already use.
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS as active_reset_default_dict,
    FLOQUET_DEFAULTS as floquet_default_dict,
    MEASUREMENT_CONFIG_DEFAULTS as measurement_config_default_dict,
)
from experiments.qsim.notebook_helpers.run_mode import run_settings
from experiments.qsim.bare_readout_check import display_bare_scramble
from experiments.qsim.floquet_gain_chevron import (
    FloquetGainChevronExperiment,
    FloquetGainChevronProgram,
)

# Set by tools/run_qsim_suite.py. Unset: through the queue in the main
# checkout, directly on this kernel in a worktree (see run_mode.py).
RUN = run_settings()

# %%
# The config versions this campaign ran against.
config_dict = {
    "hardware_config": "CFG-HW-20260915-00001",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260909-00032",
    "floquet_storage_swap": "CFG-FL-20260909-00043",
}

station = MultimodeStation(
    user="guan",
    experiment_name="260915_qsim_migration",
    project="test_migration",
    log_measurements=not RUN.smoke,
    **RUN.station_configs(config_dict),
)
client = JobClient()

# %% [markdown] jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
# # Single shot

# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
singleshot_defaults = AttrDict(dict(
    reps=RUN.pick(5000, smoke=1000),
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


# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
# Execute
# =================================
ss_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.HistogramExperiment,
    default_expt_cfg = singleshot_defaults,
    postprocessor = singleshot_postproc,
    job_client=client,
    use_queue=RUN.use_queue,
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
# ss.display()

# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
station.update_all_station_snapshots()

# %% [markdown]
# # Floquet pulse calibrations
#
# ## 1. Gain chevron

# %%
floquet_gain_chev_defaults = AttrDict(dict(
    expts = 1,
    reps = 100,
    rounds = 1,
    qubits = [0],
    ro_stor = 0, # storage mode number that gets read out in the end
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    # if 0, this means to read out man instead
    detunes=np.linspace(-0.3, 0.3, 11).tolist(),
    swept_params = ['detune', 'gain'],
    normalize = False,
    active_reset = False,
    man_reset = False,
    storage_reset = False,
    prepulse=True,
    postpulse=True,
    init_fock=True,
)) # Shouldn't be modifying this on the fly!
floquet_gain_chev_defaults.update(active_reset_default_dict)
floquet_gain_chev_defaults.update(floquet_default_dict)
# You can use kwargs in the run function to override these values

def floquet_gain_chev_preproc(station, default_expt_cfg, **kwargs):
    """Gains from 0 up to ``max_gain`` (default: 10x the current gain, at most
    15000), in ``gain_expts`` points."""
    assert 'init_stor' in kwargs
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    max_gain = expt_cfg.get("max_gain", 15000)
    gain_expts = expt_cfg.get("gain_expts", 11)
    if max_gain == 15000:
        max_gain = np.minimum(10.0 * station.ds_floquet.get_gain(f'M1-S{expt_cfg.init_stor}'), max_gain)
    expt_cfg.gains = np.linspace(0, max_gain, gain_expts).astype(int).tolist()
    return expt_cfg

def floquet_gain_chev_postproc(station, expt):
    """Accept the chevron: the best detuning into the frequency, pi/(omega) into the gain."""
    stor_name = f'M{expt.cfg.expt.f0g1_cavity}-S{expt.cfg.expt.init_stor}'
    expt.analyze(station=station)
    results = expt.chevron_analysis.results
    best_detune = results.get('best_frequency_contrast')
    if best_detune is not None:
        new_freq = station.ds_floquet.get_freq(stor_name) + best_detune
        station.ds_floquet.update_freq(stor_name, new_freq)
        print(f"Best detune {best_detune:.4f} MHz; {stor_name} frequency -> {new_freq:.4f} MHz")
        pi_frac = station.ds_floquet.get_pi_frac(stor_name)
        old_gain = station.ds_floquet.get_gain(stor_name)
        frac_pi_gain = abs(np.pi / results['best_fit_params_contrast']['omega'])
        station.ds_floquet.update_gain(stor_name, frac_pi_gain)
        print(f'pi/{pi_frac} gain {old_gain:.4f} -> {frac_pi_gain:.4f}')
    expt.chevron_analysis.display_results()
    station.snapshot_floquet_storage_swap(update_main=False)

# %%
stor_modes_to_run = [4] #list(range(1,8))
you_have_to_do_amp_chev = True
if you_have_to_do_amp_chev == True:
    # Note the postprocessor is intentionally not attached here; cell 102
    # below applies it by hand after inspecting the result.
    floquet_gain_chev_runner = CharacterizationRunner(
        station=station,
        ExptClass=FloquetGainChevronExperiment,
        ExptProgram=FloquetGainChevronProgram,
        default_expt_cfg=floquet_gain_chev_defaults,
        preprocessor=floquet_gain_chev_preproc,
        # postprocessor=floquet_gain_chev_postproc,
        job_client=client,
        use_queue=RUN.use_queue,
    )

    detune_span = 0.5

    freq_len_expt = [None] * len(stor_modes_to_run)
    for i, init_stor in enumerate(stor_modes_to_run):
        print(f'Running Floquet Frequency vs Gain Chevron for Storage Mode {init_stor}')
        freq_len_expt[i] = floquet_gain_chev_runner.execute(
            init_stor=init_stor,
            detunes=np.linspace(-detune_span, detune_span, 21).tolist(),
            reps=50,
            gain_expts=21,
            max_gain=14000,
            relax_delay=200,  # 8000 without active reset
            active_reset=True,
            man_reset=True,
            storage_reset=[init_stor],
            reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            debug=False,
        )

# %%
floquet_gain_chev_postproc(station, freq_len_expt[0])

# %% tags=["suite-skip"]
# Suite: skipped. Hand-entered values.
# station.ds_floquet.update_gain("M1-S6", 13648)
station.ds_floquet.update_gain("M1-S2", 3700)
# station.ds_floquet.update_freq("M1-S4", 878.2990264499939)

# station.ds_floquet.update_freq("M1-S4", 873.5242589230924)

# %%
station.ds_floquet.df

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ## 2. Error amplification

# %%
error_amp_floquet_defaults = AttrDict(dict(
    reps=100,
    rounds=1,
    qubits=[0],
    active_reset=False,
    man_mode_no=1,
    stor_is_dump=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=2500,
    expts=25,
    qubit_start_storage='g',
    floquet_hardware_loop=floquet_default_dict["floquet_hardware_loop"],
    scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
)) # Shouldn't be modifying this on the fly!
error_amp_floquet_defaults.update(active_reset_default_dict)
error_amp_floquet_defaults.update(floquet_default_dict)
# You can use kwargs in the run function to override these values

error_amp_gain_floquet_coarse_defaults = AttrDict(dict(
    n_pulses=8,
    span=4000,#4000
    expts=30,#30
))

error_amp_freq_floquet_coarse_defaults = AttrDict(dict(
    n_pulses=7,
    span=0.25,
    expts=50,
))

def error_amp_floquet_preproc(station, default_expt_cfg, **kwargs):
    """The coarse block of the scanned parameter, then the kwargs; the scan is
    centred on the dataset's current value, ``span`` wide."""
    assert 'stor_mode_no' in kwargs
    assert 'parameter_to_test' in kwargs
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update({'gain': error_amp_gain_floquet_coarse_defaults,
                     'frequency': error_amp_freq_floquet_coarse_defaults}[kwargs['parameter_to_test']])
    expt_cfg.update(kwargs)
    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'
    pi_frac = station.ds_floquet.get_pi_frac(stor_name)
    expt_cfg.pulse_type = ['floquet', f'M{expt_cfg.man_mode_no}-{"D" if expt_cfg.stor_is_dump else "S"}'
                                      f'{expt_cfg.stor_mode_no}', f'pi/{pi_frac}', 0]
    if expt_cfg.parameter_to_test == 'frequency':
        expt_cfg.start = station.ds_floquet.get_freq(stor_name) - expt_cfg.span / 2
        expt_cfg.step = expt_cfg.span / (expt_cfg.expts - 1)
    else:
        expt_cfg.start = int(station.ds_floquet.get_gain(stor_name) - expt_cfg.span / 2)
        expt_cfg.step = int(expt_cfg.span / (expt_cfg.expts - 1))
    return expt_cfg

def error_amp_floquet_postproc(station, expt):
    """Accept the fitted centre into ds_floquet.

    Uses analyze's default periodic=True fit, as before; see
    docs/qsim/mbr_step9_plan.md 0.4 before changing it.
    """
    expt.analyze(data=expt.data, state_fin='e')
    opt_val = expt.data['fit_avgi'][2]
    stor_name = f'M1-S{expt.cfg.expt.stor_mode_no}'
    if expt.cfg.expt.parameter_to_test == 'gain':
        station.ds_floquet.update_gain(stor_name, opt_val)
    else:
        station.ds_floquet.update_freq(stor_name, opt_val)
    print(f'Updated {expt.cfg.expt.parameter_to_test} for {stor_name} to {opt_val}')
    station.snapshot_floquet_storage_swap(update_main=False)

error_amp_floquet_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.error_amplification.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_floquet_defaults,
    preprocessor=error_amp_floquet_preproc,
    postprocessor=error_amp_floquet_postproc,
    job_client=client,
    use_queue=RUN.use_queue,
)

# %%
# stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
# stor_modes_to_run = [4, 5, 6, 7] #list(range(1,8))

stor_modes_to_run = [1, 2, 3, 4, 5, 7]

# Both span lists are indexed by stor-1; None falls back to the default below.
freq_span_list = [0.1,  0.07, 0.07, 0.1,  0.07,   0.1, 0.05] # set for the Coarse
gain_span_list = [None, None, None, None,  0.25,  0.25, 0.25]
freq_span_default = 0.1
gain_span_default = 0.3

# %%
# Source cell 109. Unlike the flat-top coarse pass, this one derives
# relax_delay from whether active reset is on.
do_active_reset = True
relax_delay = 200
if not do_active_reset:
    relax_delay = 8000

stor_modes_to_run = [4]

span_divisor = 1
error_amp_freq1 = [None] * len(stor_modes_to_run)
error_amp_gain1 = [None] * len(stor_modes_to_run)
for i, stor_i in enumerate(stor_modes_to_run):
    stor_name = f'M1-S{stor_i}'
    freq_span = freq_span_list[stor_i - 1]
    if freq_span is None:
        freq_span = freq_span_default
    gain_span = gain_span_list[stor_i - 1]
    if gain_span is None:
        gain_span = gain_span_default

    error_amp_freq1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='frequency',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.2 is the program default, but we are already close.
        span=freq_span / span_divisor,
        relax_delay=relax_delay,
        active_reset=do_active_reset,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )

    error_amp_gain1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='gain',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.7 is the program default.
        span=int(station.ds_floquet.get_gain(stor_name) * gain_span / span_divisor),
        expts=60,
        relax_delay=relax_delay,
        active_reset=do_active_reset,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )

# %%
# station.ds_floquet.update_freq("M1-S4", 878.27)

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ### Fine
#
# Source cell 113: spans halved, and it reuses whatever `stor_modes_to_run`
# the cell above left bound.

# %% tags=["suite-skip"]
# Suite: skipped. A second pass of the coarse loop above, with halved spans.
span_divisor = 2
error_amp_freq1 = [None] * len(stor_modes_to_run)
error_amp_gain1 = [None] * len(stor_modes_to_run)
for i, stor_i in enumerate(stor_modes_to_run):
    stor_name = f'M1-S{stor_i}'
    freq_span = freq_span_list[stor_i - 1]
    if freq_span is None:
        freq_span = freq_span_default
    gain_span = gain_span_list[stor_i - 1]
    if gain_span is None:
        gain_span = gain_span_default

    error_amp_freq1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='frequency',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.2 is the program default, but we are already close.
        span=freq_span / span_divisor,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )

    error_amp_gain1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='gain',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.7 is the program default.
        span=int(station.ds_floquet.get_gain(stor_name) * gain_span / span_divisor),
        expts=60,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )

# %% [markdown]
# ## 3. Phase accumulation

# %%
phase_expts = [[None for _ in range(7)] for _ in range(7)]

sideband_stark_error_amp_defaults = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    f0g1_cavity=1,  #  1/2 name of manipulate cavity
    init_stor=0, # storage mode number to initialize to n=1 Fock state (0=man)
    ro_stor=0, # storage mode number that gets read out in the end (0=man)
    advance_phases=np.linspace(-15, 15, 51).tolist(),
    n_pulses=np.arange(0, 24, 4).tolist(),
    swept_params=['n_pulse', 'advance_phase'],
    normalize=False, # not sure what this does
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    prepulse=True,
    postpulse=True,
    init_fock=True,
))
sideband_stark_error_amp_defaults.update(active_reset_default_dict)
sideband_stark_error_amp_defaults.update(floquet_default_dict)

def sideband_stark_error_amp_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_A' in kwargs
    assert 'stor_B' in kwargs
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    return expt_cfg

def sideband_stark_error_amp_postproc(station, expt):
    """Accept the Stark phase that a swap of stor_B leaves on stor_A."""
    stor_name = f'M1-S{expt.cfg.expt.stor_A}'
    from_stor_name = f'M1-S{expt.cfg.expt.stor_B}'
    expt.analyze(fit=True)
    expt.display(fit=True)
    # Open question (jonginn): the sequence plays pi/12, -pi/12 on the from_stor
    # swap, so /2 was expected; the undivided value calibrates correctly.
    opt_phase = expt.data['fit_avgi'][2]
    print("Opt phase on", stor_name, "from", from_stor_name, ":", opt_phase)
    station.ds_floquet.update_phase_from(stor_name, from_stor_name, opt_phase)

sideband_stark_error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.SidebandStarkAmplificationExperiment,
    # ExptProgram=meas.qsim.sideband_stark_shift_cal.StorageSwapStarkPhaseProgram,
    ExptProgram=meas.SidebandStarkAmplificationProgram, #meas.SidebandStarkAmplificationProgram,
    default_expt_cfg=sideband_stark_error_amp_defaults,
    preprocessor=sideband_stark_error_amp_preproc,
    postprocessor=sideband_stark_error_amp_postproc,
    job_client=client,
    use_queue=RUN.use_queue,
)

# %%
# Source cell 116. From here on the buffer flag and sync cycles are forwarded.
# stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
stor_modes_to_run = RUN.pick([1, 2, 3, 4], smoke=[1, 2]) #list(range(1,8))

do_active_reset = True
relax_delay = 200
if not do_active_reset:
    relax_delay = 8000

for init_storA in stor_modes_to_run:
    for init_storB in stor_modes_to_run:
        if init_storA == init_storB:
            continue
        print("Starting experiment for storage modes:", init_storA, "from", init_storB)
        phase_expts[init_storA - 1][init_storB - 1] = sideband_stark_error_amp_runner.execute(
            stor_A=init_storA,
            stor_B=init_storB,
            reps=100,
            relax_delay=relax_delay,
            active_reset=do_active_reset,
            man_reset=True,
            storage_reset=[init_storA, init_storB],
            reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
            include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
            scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
            advance_phases=np.linspace(-10, 10, 51).tolist(),
        )

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ### Redo or add calibration
#
# Source cell 119: one directed pair, on a wider and finer phase grid.

# %% tags=["suite-skip"]
# Suite: skipped. A redo of one pair from the loop above.
init_storA, init_storB = 2, 6

print("Starting experiment for storage modes:", init_storA, "from", init_storB)
phase_expts[init_storA - 1][init_storB - 1] = sideband_stark_error_amp_runner.execute(
    stor_A=init_storA,
    stor_B=init_storB,
    reps=100,
    relax_delay=100,
    active_reset=True,
    man_reset=True,
    storage_reset=[init_storA, init_storB],
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
    include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
    scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
    advance_phases=np.linspace(-30, 30, 301).tolist(),
)

# %% [markdown]
# ### Add modes
#
# Source cell 121: measure only the pairs that involve a newly added mode,
# skipping those where both modes were already calibrated.

# %% tags=["suite-skip"]
# Suite: skipped. Extends a campaign already calibrated by hand.
stor_modes_to_add = [2] #list(range(1,8))
pre_existing_store_modes = [4, 5, 6]

total_stor_modes = list(set(pre_existing_store_modes + stor_modes_to_add))

for init_storA in total_stor_modes:
    for init_storB in total_stor_modes:
        if init_storA == init_storB:
            continue
        # Skip the pairs that were already calibrated before the new modes.
        if (init_storA in pre_existing_store_modes
                and init_storB in pre_existing_store_modes):
            continue
        print("Starting experiment for storage modes:", init_storA, "from", init_storB)
        phase_expts[init_storA - 1][init_storB - 1] = sideband_stark_error_amp_runner.execute(
            stor_A=init_storA,
            stor_B=init_storB,
            reps=100,
            relax_delay=100,
            active_reset=True,
            man_reset=True,
            storage_reset=[init_storA, init_storB],
            reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
            include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
            scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
            advance_phases=np.linspace(-20, 20, 101).tolist(),
        )

# %%
station.update_all_station_snapshots()

# %% [markdown]
# # 4. Bare readout check
#
# Source cells 150-158, the "Bare" subsection. This is the prerequisite check
# the MBR campaigns need; the displace/multiparity dark-mode work that follows
# it in the source (cells 159-262) is dormant, in
# `dormant/dark_mode.py`.

# %%
dm_sideband_scramble_defaults = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    ro_stor=0, # storage mode number that gets read out in the end

    init_fock=True,

    normalize=False,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    prepulse=True,
    postpulse=True,
)) # Shouldn't be modifying this on the fly!
dm_sideband_scramble_defaults.update(active_reset_default_dict)
dm_sideband_scramble_defaults.update(floquet_default_dict)
# You can use kwargs in the run function to override these values

def sideband_scramble_preproc(station, default_expt_cfg, **kwargs):
    assert kwargs.get('swept_params')
    assert 'init_stor' in kwargs
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    if not expt_cfg.init_fock:
        assert 'init_alpha' in kwargs or 'init_man_fock_state' in kwargs
    return expt_cfg

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimExperiment,
    ExptProgram=meas.DarkModeScrambleProgram,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
    use_queue=RUN.use_queue,
)

# %%
# One job per chunk of Floquet cycles (the tProc instruction memory).
max_cycle = RUN.pick(2000, smoke=300)
cycles_per_job = RUN.pick(2000, smoke=300)
floquet_cycles_list = [np.arange(first, min(first + cycles_per_job, max_cycle), 15)
                       for first in range(0, max_cycle, cycles_per_job)]

swap_stors = [1, 2, 3, 4]
# meas_stors = [0, 2, 6]
# meas_stors = [0]
meas_stors = RUN.pick([0] + swap_stors, smoke=[0, 1])
# meas_stors = [0, 1, 2, 3, 4]
dark_swaps = [4, 5]

detunings = [0] * len(swap_stors)
# The storages that get reset are meas_stors without the leading 0, which
# reads out the manipulate mode.
reset_stors = meas_stors[1:]

scramble_expts = []
for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in floquet_cycles_list:
        scramble_sub_expts.append(dmscramble_runner.execute(
            reps=RUN.pick(300, smoke=100),
            init_fock=True,
            init_stor=0,
            ro_stor=meas_stor,
            relax_delay=200,  # 8000 without active reset
            active_reset=True,
            pre_relax_delay=100,
            man_reset=True,
            storage_reset=reset_stors,
            reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num=active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse=False,
            custom_postpulse=False,
            debug=False,
            swap_man_dark=False,
            dark_swap_order=dark_swaps,
            second_rel_phase=180,
            map_to_qubit_ge=True,
            prepulse=True,   # for debugging. Should always be true
            postpulse=True,  # for debugging. Should always be true
            palindrome_scramble=floquet_default_dict["palindrome_scramble"],
            scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
        ))
    scramble_expts.append(scramble_sub_expts)

# %%
# Time axis: the Floquet cycle each job compiled (experiments/floquet_timing.py).
fig, bare_readout, cycle_us = display_bare_scramble(scramble_expts, plot_time=True)

# %%
station.update_all_station_snapshots()
