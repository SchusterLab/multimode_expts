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
#     display_name: Multimode (direct remote)
#     language: python
#     name: multimode-direct
# ---

# %% [markdown]
# # Floquet pulse calibration and quantum walk (preloaded flat-top)
#
# What the MBR campaigns need before they run: the Floquet swap pulses of the
# storage modes, calibrated in three steps, and a bare-readout check.
#
# 1. **Gain chevron** (detuning x gain): the frequency and the pi/N gain.
# 2. **Error amplification**: frequency and gain, coarse then fine.
# 3. **Phase accumulation**: the Stark phase each swap leaves on the others.
# 4. **Quantum walk**: prepare both manipulate modes, evolve for the same
#    Floquet cycles and read out each mode in a separate repetition.
#
# The Floquet pulse is the preloaded flat-top (`docs/qsim/mbr_step9_plan.md`,
# decision 0.3). The synthesized 3-segment flat-top and the Gaussian procedures
# are in `dormant/floquet_calibration_all_envelopes.py`.
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

# %% [markdown]
# For a same-shot walk with M1 on S5, S6 and S7 and M2 on S2, S3 and S4,
# set `man_modes = [1, 2]` and `storage_modes_by_man = {1: [5, 6, 7], 2: [2, 3, 4]}`.
# The six preloaded pulse settings span two tProc register pages.
#
# **Current workflow in pseudocode**
#
# ```text
# For each selected storage, assign one manipulate and select its M-S row.
# Initialize missing or explicitly reselected links for the chosen manipulates.
# Choose one manipulate and the storages to scan in the gain chevron cell.
# Run the save cell to apply the selected chevron frequencies and gains.
# Choose one manipulate and the storages to scan in the error amplification section.
# Run coarse gain, then fine frequency and gain error amplification.
# Measure and save Stark phase corrections for the selected directions.
#
# For each selected manipulate or storage readout, repeat the same walk:
#     Reset the selected manipulates and empty each storage via its assigned manipulate.
#     Prepare the requested photon counts, filling M1 before M2.
#     Play the selected M-S pulses in each Floquet cycle.
#     For a storage readout, swap the storage into its assigned manipulate.
#     Swap the readout manipulate into the qubit and measure I.
# Plot the same average I traces as curves and a color map.
# ```

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
)
from experiments.qsim.notebook_helpers.run_mode import main_config_ids, run_settings
from experiments.qsim.bare_readout_check import floquet_cycle_us, pool_by_readout
from experiments.floquet_timing import station_floquet_hardware
from experiments.qsim.floquet_gain_chevron import (
    FloquetGainChevronExperiment,
    FloquetGainChevronProgram,
)

# Set by tools/run_qsim_suite.py. Unset: through the queue in the main
# checkout, directly on this kernel in a worktree (see run_mode.py).
RUN = run_settings()

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
man_modes = [1, 2]  # Manipulates used in the walk.
storage_modes_by_man = {1: [5, 6, 7], 2: [2, 3, 4]}
initial_m1_photon_number = 1 if 1 in man_modes else 0
initial_m2_photon_number = 1 if 2 in man_modes else 0

# Each storage belongs to one manipulate for the duration of a walk.
storage_man_modes = {}
for man in man_modes:
    for storage in storage_modes_by_man[man]:
        if storage in storage_man_modes:
            raise ValueError(f"S{storage} appears more than once in the walk")
        storage_man_modes[storage] = man
swap_stors = list(storage_man_modes)
if not swap_stors or any(man not in (1, 2) for man in man_modes):
    raise ValueError("Select M1 or M2 and at least one storage for the walk")
if any(storage not in range(1, 8) for storage in swap_stors):
    raise ValueError("Storage mode numbers run from 1 to 7")

floquet_default_dict = dict(floquet_default_dict, scramble_sync_cycles=3)

# %%
# Start from the configured main versions, or use the versions selected by RUN.
# To resume a calibration, enter its accepted snapshot IDs here.
config_dict = main_config_ids() if RUN.configs is None else {}
# config_dict['man1_storage_swap'] = 'CFG-M1-...'
# config_dict['floquet_storage_swap'] = 'CFG-FL-...'

station = MultimodeStation(
    user="jonginn",
    experiment_name="261111_calibration_after_recooldown",
    project="FloquetCalibration",
    log_measurements=not RUN.smoke,
    **RUN.station_configs(config_dict),
)
client = JobClient()

# %% [markdown]
# ## Initialize Floquet swap pulses
#
# New or empty rows for the chosen manipulates take their starting frequency
# and gain from the same M–S link in ds_storage.
# Existing Floquet fits are kept unless the link is listed for reseeding.

# %%
man_to_initialize_floquet_from_stor_ds = [2]  # Use [1], [2], [1, 2], or [] to skip.
reinitialize_links = []  # Add already calibrated rows to reseed, e.g. ['M2-S3'].
preload_flat_length_us = 0.01
preload_ramp_sigma_us = 0.002
preload_pi_frac = 20

ds_floquet = station.ds_floquet
links_to_initialize = [(man, storage) for man in man_to_initialize_floquet_from_stor_ds
                       for storage in storage_modes_by_man[man]]
for man, storage in links_to_initialize:
    link = f'M{man}-S{storage}'
    phase_column = f'phase_from_{link} (deg)'
    if phase_column not in ds_floquet.df:
        ds_floquet.df[phase_column] = 0.0  # initial correction until phase calibration

    new_link = link not in ds_floquet.df['stor_name'].values
    if new_link or ds_floquet.get_pi_frac(link) == -1 or link in reinitialize_links:
        freq = station.ds_storage.get_freq(link)
        gain = station.ds_storage.get_gain(link)
        if not (np.isfinite(freq) and freq > 0 and np.isfinite(gain) and gain > 0):
            raise ValueError(f'Calibrate the storage frequency and gain for {link} first')
        if new_link:
            phase_defaults = {column: 0.0 for column in ds_floquet.df if column.startswith('phase_from_')}
            ds_floquet.append_dataset(link, pi_frac=preload_pi_frac,
                                      **{'gain (DAC units)': int(gain)}, **phase_defaults)
        ds_floquet.update_freq(link, freq)
        ds_floquet.update_gain(link, gain)
        ds_floquet.update_pi_frac(link, preload_pi_frac)
        ds_floquet.update_waveform(link, 'preload_flattop')
        ds_floquet.update_len(link, preload_flat_length_us)
        ds_floquet.update_ramp_sigma(link, preload_ramp_sigma_us)
        print(f'Initialized {link} at {freq:.4f} MHz, gain {gain}')
ds_floquet.df

# %%
# To keep these starting values as a version before running any calibration:
# station.snapshot_floquet_storage_swap(update_main=False)
floquet_timing = station_floquet_hardware(
    station, swap_stors, floquet_default_dict['scramble_sync_cycles'],
    storage_man_modes=storage_man_modes,
)
print(f'Floquet cycle: {floquet_timing.floquet_cycle_us:.6f} us')
print('Effective couplings (kHz):', 1000 * np.asarray(floquet_timing.couplings_MHz))

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
    expt.analyze(station=station, subdir=station.autocalib_path)
    station.hardware_cfg.device.readout.update(expt.readout_cfg)


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
    f0g1_cavity = 1,  # overridden by the selected manipulate
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
        max_gain = np.minimum(10.0 * station.ds_floquet.get_gain(f'M{expt_cfg.man_mode_no}-S{expt_cfg.init_stor}'), max_gain)
    expt_cfg.gains = np.linspace(0, max_gain, gain_expts).astype(int).tolist()
    return expt_cfg

def floquet_gain_chev_postproc(station, expt):
    """Accept the chevron: the best detuning into the frequency, pi/(omega) into the gain."""
    stor_name = f'M{expt.cfg.expt.man_mode_no}-S{expt.cfg.expt.init_stor}'
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
floquet_gain_chev_runner = CharacterizationRunner(
    station=station,
    ExptClass=FloquetGainChevronExperiment,
    ExptProgram=FloquetGainChevronProgram,
    default_expt_cfg=floquet_gain_chev_defaults,
    preprocessor=floquet_gain_chev_preproc,
    job_client=client,
    use_queue=RUN.use_queue,
)

man_mode_for_gain_chevron = 2
stor_modes_for_gain_chevron = [2, 3, 4]
detune_span = 2.0
freq_len_expt = {}
for stor in stor_modes_for_gain_chevron:
    man = man_mode_for_gain_chevron
    print(f'Running Floquet frequency vs gain chevron for M{man}-S{stor}')
    freq_len_expt[man, stor] = floquet_gain_chev_runner.execute(
        init_stor=stor,
        detunes=np.linspace(-detune_span, detune_span, 21).tolist(),
        reps=50,
        gain_expts=21,
        max_gain=30000,
        debug=False,
        man_mode_no=man,
        f0g1_cavity=man,
        readout_man_mode=man,
        storage_man_modes={str(stor): man},
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        man_reset_modes=[man],
        storage_reset=[stor],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )

# %%
# Replace freq_len_expt with a list of (manipulate, storage) pairs to save a subset.
for pair in freq_len_expt:
    floquet_gain_chev_postproc(station, freq_len_expt[pair])

# %%
station.ds_floquet.df

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
    pi_frac = int(station.ds_floquet.get_pi_frac(stor_name))
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
    stor_name = f'M{expt.cfg.expt.man_mode_no}-S{expt.cfg.expt.stor_mode_no}'
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
man_mode_for_error_amp = 2
stor_modes_for_error_amp = [2, 3, 4]

# Span per storage S1 through S7. Gain spans are fractions of the current gain.
freq_span_list = [0.1, 0.07, 0.07, 0.3, 0.1, 0.1, 0.1]
gain_span_list = [None, None, None, 0.2, 0.25, 0.25, 0.25]
freq_span_default = 0.1
gain_span_default = 0.3

# %% [markdown]
# ### Coarse

# %%
do_active_reset = True
relax_delay = 200 if do_active_reset else 8000
span_divisor = 1
RUN_ERROR_AMP_COARSE_FREQ = False
RUN_ERROR_AMP_COARSE_GAIN = True

error_amp_freq_coarse, error_amp_gain_coarse = {}, {}
for stor in stor_modes_for_error_amp:
    man = man_mode_for_error_amp
    freq_span = freq_span_list[stor - 1]
    if freq_span is None:
        freq_span = freq_span_default
    gain_span = gain_span_list[stor - 1]
    if gain_span is None:
        gain_span = gain_span_default
    routing_and_reset = dict(
        man_mode_no=man,
        f0g1_cavity=man,
        readout_man_mode=man,
        storage_man_modes={str(stor): man},
        relax_delay=relax_delay,
        active_reset=do_active_reset,
        man_reset=True,
        man_reset_modes=[man],
        storage_reset=[stor],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )

    if RUN_ERROR_AMP_COARSE_FREQ:
        error_amp_freq_coarse[man, stor] = error_amp_floquet_runner.execute(
            stor_mode_no=stor,
            parameter_to_test='frequency',
            go_kwargs=dict(analyze=False, progress=True, display=False),
            span=freq_span / span_divisor,
            **routing_and_reset,
        )
    if RUN_ERROR_AMP_COARSE_GAIN:
        gain = station.ds_floquet.get_gain(f'M{man}-S{stor}')
        error_amp_gain_coarse[man, stor] = error_amp_floquet_runner.execute(
            stor_mode_no=stor,
            parameter_to_test='gain',
            go_kwargs=dict(analyze=False, progress=True, display=False),
            span=int(gain * gain_span / span_divisor),
            expts=60,
            **routing_and_reset,
        )

# %% [markdown]
# ### Fine

# %%
span_divisor = 2
error_amp_freq_fine, error_amp_gain_fine = {}, {}
for stor in stor_modes_for_error_amp:
    man = man_mode_for_error_amp
    freq_span = freq_span_list[stor - 1]
    if freq_span is None:
        freq_span = freq_span_default
    gain_span = gain_span_list[stor - 1]
    if gain_span is None:
        gain_span = gain_span_default
    routing_and_reset = dict(
        man_mode_no=man,
        f0g1_cavity=man,
        readout_man_mode=man,
        storage_man_modes={str(stor): man},
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        man_reset_modes=[man],
        storage_reset=[stor],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )

    error_amp_freq_fine[man, stor] = error_amp_floquet_runner.execute(
        stor_mode_no=stor,
        parameter_to_test='frequency',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        span=freq_span / span_divisor,
        **routing_and_reset,
    )
    gain = station.ds_floquet.get_gain(f'M{man}-S{stor}')
    error_amp_gain_fine[man, stor] = error_amp_floquet_runner.execute(
        stor_mode_no=stor,
        parameter_to_test='gain',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        span=int(gain * gain_span / span_divisor),
        expts=60,
        **routing_and_reset,
    )

# %% [markdown]
# ## 3. Phase accumulation

# %%
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
    cfg = expt.cfg.expt
    stor_name = f'M{cfg.man_mode_no}-S{cfg.stor_A}'
    from_stor_name = f'M{cfg.pulsed_man_mode}-S{cfg.stor_B}'
    expt.analyze(fit=True)
    expt.display(fit=True)
    # Open question (jonginn): the sequence plays pi/12, -pi/12 on the from_stor
    # swap, so /2 was expected; the undivided value calibrates correctly.
    opt_phase = expt.data['fit_avgi'][2]
    print("Opt phase on", stor_name, "from", from_stor_name, ":", opt_phase)
    station.ds_floquet.update_phase_from(stor_name, from_stor_name, opt_phase)

# %%
sideband_stark_error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.SidebandStarkAmplificationExperiment,
    ExptProgram=meas.SidebandStarkAmplificationProgram,
    default_expt_cfg=sideband_stark_error_amp_defaults,
    preprocessor=sideband_stark_error_amp_preproc,
    postprocessor=sideband_stark_error_amp_postproc,
    job_client=client,
    use_queue=RUN.use_queue,
)
phase_expts = {}

# %%
# Measure how a pulse on S_B shifts the phase of the M–S_A swap.
man_modes_for_phase = [2]
phase_pairs = [(stor_a, stor_b) for man in man_modes if man in man_modes_for_phase
               for stor_a in storage_modes_by_man[man]
               for stor_b in storage_modes_by_man[man] if stor_a != stor_b]
# Add any cross-group pairs to measure, in (affected storage, pulsed storage) order.
cross_group_phase_pairs = []
do_active_reset = True
relax_delay = 200 if do_active_reset else 8000

for stor_a, stor_b in phase_pairs + cross_group_phase_pairs:
    man_a = storage_man_modes[stor_a]
    man_b = storage_man_modes[stor_b]
    print(f'Measuring phase on M{man_a}-S{stor_a} from M{man_b}-S{stor_b}')
    phase_expts[stor_a, stor_b] = sideband_stark_error_amp_runner.execute(
        stor_A=stor_a,
        stor_B=stor_b,
        pulsed_man_mode=man_b,
        reps=100,
        include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
        include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
        scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
        advance_phases=np.linspace(-10, 10, 51).tolist(),
        man_mode_no=man_a,
        f0g1_cavity=man_a,
        readout_man_mode=man_a,
        storage_man_modes={str(stor_a): man_a, str(stor_b): man_b},
        relax_delay=relax_delay,
        active_reset=do_active_reset,
        man_reset=True,
        man_reset_modes=list(man_modes),
        storage_reset=[stor_a, stor_b],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )

# %%
station.snapshot_floquet_storage_swap(update_main=False)

# %% [markdown]
# ### Redo or add calibration
#
# Set one (affected storage, pulsed storage) pair and run a wider phase scan.

# %% tags=["suite-skip"]
phase_pair_to_redo = None  # e.g. (2, 3)
if phase_pair_to_redo is not None:
    stor_a, stor_b = phase_pair_to_redo
    man_a = storage_man_modes[stor_a]
    man_b = storage_man_modes[stor_b]
    print(f'Measuring phase on M{man_a}-S{stor_a} from M{man_b}-S{stor_b}')
    phase_expts[stor_a, stor_b] = sideband_stark_error_amp_runner.execute(
        stor_A=stor_a,
        stor_B=stor_b,
        pulsed_man_mode=man_b,
        reps=100,
        include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
        include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
        scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
        advance_phases=np.linspace(-30, 30, 301).tolist(),
        man_mode_no=man_a,
        f0g1_cavity=man_a,
        readout_man_mode=man_a,
        storage_man_modes={str(stor_a): man_a, str(stor_b): man_b},
        relax_delay=100,
        active_reset=True,
        man_reset=True,
        man_reset_modes=list(man_modes),
        storage_reset=[stor_a, stor_b],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )
    station.snapshot_floquet_storage_swap(update_main=False)

# %% [markdown]
# ### Add modes
#
# Use storages already assigned above. Measure both directions for pairs
# involving an added storage and reuse the stored phase for existing pairs.

# %% tags=["suite-skip"]
pre_existing_stor_modes = []  # e.g. [2, 3]
stor_modes_to_add = []  # e.g. [4]
if stor_modes_to_add:
    total_stor_modes = list(dict.fromkeys(pre_existing_stor_modes + stor_modes_to_add))
    for stor_a in total_stor_modes:
        for stor_b in total_stor_modes:
            if stor_a == stor_b:
                continue
            if stor_a in pre_existing_stor_modes and stor_b in pre_existing_stor_modes:
                continue
            man_a = storage_man_modes[stor_a]
            man_b = storage_man_modes[stor_b]
            print(f'Measuring phase on M{man_a}-S{stor_a} from M{man_b}-S{stor_b}')
            phase_expts[stor_a, stor_b] = sideband_stark_error_amp_runner.execute(
                stor_A=stor_a,
                stor_B=stor_b,
                pulsed_man_mode=man_b,
                reps=100,
                include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
                include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
                scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
                advance_phases=np.linspace(-20, 20, 101).tolist(),
                man_mode_no=man_a,
                f0g1_cavity=man_a,
                readout_man_mode=man_a,
                storage_man_modes={str(stor_a): man_a, str(stor_b): man_b},
                relax_delay=100,
                active_reset=True,
                man_reset=True,
                man_reset_modes=list(man_modes),
                storage_reset=[stor_a, stor_b],
                reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            )
    station.snapshot_floquet_storage_swap(update_main=False)

# %% [markdown]
# # 4. Quantum walk with bare readout
#
# Both groups evolve in every shot. Each readout repeats the same preparation
# and Floquet sequence, then maps one physical mode onto the qubit.

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

# %%
photon_numbers = {1: initial_m1_photon_number, 2: initial_m2_photon_number}
for man, count in photon_numbers.items():
    if not isinstance(count, (int, np.integer)) or count < 0 or (count and man not in man_modes):
        raise ValueError(f'Invalid initial photon number for M{man}: {count}')

# Keep the original one-photon preparation for a single manipulate.
fock_prep = {}
if len(man_modes) == 2 or photon_numbers[man_modes[0]] != 1:
    fock_prep = dict(initial_m1_photon_number=initial_m1_photon_number,
                     initial_m2_photon_number=initial_m2_photon_number)

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

# One job per chunk of Floquet cycles (the tProc instruction memory).
max_cycle = RUN.pick(300, smoke=3)
cycles_per_job = 300
floquet_cycles_list = [np.arange(first, min(first + cycles_per_job, max_cycle), 1)
                       for first in range(0, max_cycle, cycles_per_job)]

detunings = [0] * len(swap_stors)
readout_modes = [(man, stor) for man in man_modes for stor in [0, *storage_modes_by_man[man]]]

scramble_expts = {}
for man, meas_stor in tqdm(readout_modes):
    scramble_expts[man, meas_stor] = []
    for floquet_cycles in floquet_cycles_list:
        scramble_expts[man, meas_stor].append(dmscramble_runner.execute(
            reps=RUN.pick(300, smoke=100),
            init_fock=True,
            init_stor=0,
            ro_stor=meas_stor,
            man_mode_no=man_modes[0],
            readout_man_mode=man,
            storage_man_modes={str(storage): hub for storage, hub in storage_man_modes.items()},
            **fock_prep,
            relax_delay=200,
            active_reset=True,
            pre_relax_delay=100,
            man_reset=True,
            man_reset_modes=list(man_modes),
            storage_reset=swap_stors,
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
            map_to_qubit_ge=True,
            prepulse=True,   # for debugging. Should always be true
            postpulse=True,  # for debugging. Should always be true
            palindrome_scramble=floquet_default_dict["palindrome_scramble"],
            scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
        ))
# %% [markdown]
# ## Quantum walk curves and color map
#
# Both views show the same average I in ADC units. The bare readout does
# not resolve photon number. The time axis uses the compiled cycle duration.

# %%
cycle_us = floquet_cycle_us([job for jobs in scramble_expts.values() for job in jobs])
walk_data = {mode: pool_by_readout(jobs)[mode[1]] for mode, jobs in scramble_expts.items()}
cycles = next(iter(walk_data.values()))['floquet_cycle']
time_us = cycles * cycle_us

fig, axes = plt.subplots(len(man_modes), 1, sharex=True, squeeze=False, figsize=(9, 3 * len(man_modes)))
labels, signals = [], []
for ax, man in zip(axes[:, 0], man_modes):
    for stor in [0, *storage_modes_by_man[man]]:
        trace = walk_data[man, stor]
        if not np.array_equal(trace['floquet_cycle'], cycles):
            raise ValueError(f'M{man}, readout {stor} has a different Floquet cycle sweep')
        label = f'M{man}' if stor == 0 else f'S{stor}'
        ax.plot(time_us, trace['avgi'], label=label)
        labels.append(label)
        signals.append(trace['avgi'])
    ax.set_ylabel('Avg I (ADC units)')
    ax.legend()
    ax.grid(alpha=0.25)
axes[-1, 0].set_xlabel('Time (µs)')
fig.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(labels))))
mesh = ax.pcolormesh(time_us, np.arange(len(labels)), signals, shading='nearest', cmap='viridis')
ax.set_yticks(np.arange(len(labels)), labels)
ax.invert_yaxis()
if len(man_modes) == 2:
    ax.axhline(len(storage_modes_by_man[man_modes[0]]) + 0.5, color='white')
ax.set_xlabel('Time (µs)')
ax.set_ylabel('Readout mode')
fig.colorbar(mesh, ax=ax, label='Avg I (ADC units)')
fig.tight_layout()
plt.show()

# %%
