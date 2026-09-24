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
# # Pulse visualization scratch (dormant)
#
# Relocated from `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
# 154-166 ("Pulse Visualization Test") by the stage-2 notebook decomposition.
# **Dormant**: scratch pulse-shape plotting with no active caller.
#
# It carries six of cell 4's helpers, including three that a first pass at this
# split wrongly deleted as duplicates with no caller. This section references
# two of them by bare name -- `preprocessor=error_amp_floquet_preproc`,
# `postprocessor=error_amp_floquet_postproc` -- which a search for call syntax
# `name(` could not see, and `get_floquet_parameters` is an internal dependency
# of the first. They are copied here rather than imported, so this notebook does
# not depend on an active theme's module.
#
# `floquet_cycle_list_gen` is byte-identical to the copies in
# `qsim_experiments.ipynb` cells 151/152/287. `error_amp_floquet_postproc` is
# *not* identical to the Q copies: 11 code lines here against 19 there. This one
# is what the section actually ran.
#
# Relocation only, per the stage-2 instructions. Sibling dormant notebooks:
# `flux_excursion.py`, `wigner.py`, `dark_mode.py`.

# %%
# %load_ext autoreload
# %autoreload 2


import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle, glob
import qutip as qt
import textwrap
import experiments as meas

from copy import deepcopy
from collections import defaultdict
from tqdm.notebook import tqdm
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display_classes import GeneralFitting
from fitting.wigner import WignerAnalysis
from slab import AttrDict
from experiments import MultimodeStation, CharacterizationRunner, SweepRunner


REPO_ROOT = r"C:\python\multimode_expts" 
BASE_DIR  = r"C:\experiments"
RUN_PREFIX = "JOB-20260414"  

# The four aggregate MBR stages. `EncodingHamiltonianSpectroscopyExperiment`
# is still the loading layer and the shared numerics, and is still the class
# every job here was acquired under -- so it stays, and these four sit beside
# it. See analysis_notebooks/guan/MBR_analysis.py for the worked example.
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.legacy_mbr import MBRSpectrumExperiment
from experiments.qsim.legacy_mbr import MBROrthogonalityExperiment
from experiments.qsim.legacy_mbr import MBRPropagatorExperiment

# %% [markdown]
# Helpers this notebook uses, sliced out of the original
# `data_postprocess.ipynb` cell 4 -- a single 1563-line cell that defined all 71
# helpers for every section at once. Only this notebook's share is carried here.
# Shared helpers are copied rather than imported, so a dormant notebook never
# depends on an active theme's module.

# %%
# =====================================================================
# Helper Functions & Definitions -- run this ONE cell before any section
# below. Every function/class the notebook uses is defined here, so no
# section depends on another section's cells having been run first.
# (Requires the import cell above to have run: np, plt, meas, AttrDict, ...)
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import os, textwrap
from pathlib import Path
from collections import namedtuple, defaultdict


def normalize(z, exp_obj):
    Ig = exp_obj.cfg.device.readout.Ig[0]
    Ie = exp_obj.cfg.device.readout.Ie[0]
    return (z - Ig) / (Ie - Ig)


def error_amp_floquet_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    assert 'parameter_to_test' in kwargs 

    # construct the defaults
    expt_cfg = deepcopy(default_expt_cfg)
    if kwargs['parameter_to_test'] == 'gain':
        expt_cfg.update(error_amp_gain_floquet_coarse_defaults)
    elif kwargs['parameter_to_test'] == 'frequency':
        expt_cfg.update(error_amp_freq_floquet_coarse_defaults)
    # override with the passed kwargs
    expt_cfg.update(kwargs)

    freq, gain, length, pi_frac, ch, prepulse, postpulse = get_floquet_parameters(station, expt_cfg.man_mode_no, expt_cfg.stor_mode_no)
    pulse_type = ['floquet', f'M{expt_cfg.man_mode_no}-{"D" if expt_cfg.stor_is_dump else "S"}{expt_cfg.stor_mode_no}', f'pi/{pi_frac}', 0]
    # freq = 695.7
    # gain = 10000
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


def error_amp_floquet_postproc(station, expt):
    expt.analyze(data=expt.data, state_fin='e')

    opt_val = expt.data['fit_avgi'][2]
    stor_name = 'M1-S' + str(expt.cfg.expt.stor_mode_no)
    if expt.cfg.expt.parameter_to_test == 'gain':
        station.ds_floquet.update_gain(stor_name, opt_val)
        print(f'Updated gain for {stor_name} to {opt_val}')
    elif expt.cfg.expt.parameter_to_test == 'frequency':
        station.ds_floquet.update_freq(stor_name, opt_val)
        print(f'Updated frequency for {stor_name} to {opt_val}')
    station.snapshot_floquet_storage_swap(update_main=False)


def get_floquet_parameters(station, man_mode_no, stor_mode_no):
    """
    Get pulse parameters for a given storage mode. 
    Also returns prepulse and postpulse (single photon prep and meas for ge meas)

    Args:
        station: MultimodeStation object for managing frequency data.
        man_mode_no: Manipulation mode number.
        stor_mode_no: Storage mode number.

    Returns:
        A tuple containing freq, gain, ch, prepulse, and postpulse.
    """
    stor_name = 'M' + str(man_mode_no) + '-S' + str(stor_mode_no)
    freq = station.ds_floquet.get_freq(stor_name)
    gain = station.ds_floquet.get_gain(stor_name)
    length = station.ds_floquet.get_len(stor_name)
    pi_frac = station.ds_floquet.get_pi_frac(stor_name)
    ch = 'low' if freq < 1000 else 'high'

    mm_base_dummy = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
    prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist() # for ge meas, only do f0g1 and ef pi

    return freq, gain, length, pi_frac, ch, prepulse, postpulse


def sideband_scramble_preproc(station, default_expt_cfg, **kwargs):
    assert 'swept_params' in kwargs
    assert len(kwargs['swept_params']) > 0

    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    assert 'init_stor' in kwargs
    if not expt_cfg.init_fock:
        assert 'init_alpha' or 'init_man_fock_state' in kwargs
        
    # print(expt_cfg)
    return expt_cfg


def floquet_cycle_list_gen(start, 
                           stop, 
                           chunk,
                           step = 1):
    _to_return = []
    _q = (stop-start)//chunk
    _start_idx = start
    for _ in range(_q):
        _to_return.append(np.arange(_start_idx, 
                                    _start_idx+chunk,
                                    step))
        _start_idx += chunk
    if _start_idx < stop:
        
        _to_return.append(np.arange(_start_idx, 
                                    stop, 
                                    step))
    return _to_return

# %% [markdown]
# # Pulse Visualization Test

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

# Check server health (offline-tolerant: visualization never uses the job client)
try:
    health = client.health_check()
    print(f"Server status: {health['status']}")
    print(f"Pending jobs: {health['pending_jobs']}")
except Exception as e:
    print(f"[offline] job server not reachable ({e!r}); continuing for visualization only")

# %%
# Enable offline pulse-sequence visualization (QICK board off / no proxy).
# Structure (plot_circuit) is accurate; absolute timing is approximate.
import sys
sys.path.insert(0, r"measurement_notebooks/jonginn")
from offline_viz import enable_offline_visualization
enable_offline_visualization()

# %%
# Initialize station to retrieve soc and configs
config_dict = {'hardware_config': 'CFG-HW-20260610-00017',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20260610-00053',
 'floquet_storage_swap': 'CFG-FL-20260611-00001'}



station = MultimodeStation(
    user = "jonginn",
    mock = True,   # offline visualization (see patch cell above)
    experiment_name = "260526_qsim_darkmode",
    
    storage_man_file = config_dict['man1_storage_swap'],
    hardware_config= config_dict['hardware_config'],
    floquet_file=config_dict['floquet_storage_swap'],
    

    # storage_man_file="CFG-M1-20260218-00025",
    # hardware_config="CFG-HW-20260211-00039",
    # floquet_file="versions/floquet_storage_swap/CFG-FL-20260520-00002.csv",
    # multiphoton_config="versions/multiphoton_config/CFG-MP-20260115-00001.yml",
)

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
)) # Shouldn't be modifying this on the fly!
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





error_amp_floquet_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.error_amplification.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_floquet_defaults,
    preprocessor=error_amp_floquet_preproc,
    postprocessor=error_amp_floquet_postproc,
    job_client=client,
)

# %%
from experiments.MM_dual_rail_base import MM_dual_rail_base

# %%
import sys
sys.path.insert(0, r"measurement_notebooks/jonginn")

from pulse_sequence_visualizer import install_qick_stubs, trace_runner_point
install_qick_stubs()  

from experiments.single_qubit.error_amplification import ErrorAmplificationProgram

stor_i = 1
stor_name = f"M1-S{stor_i}"

prog, trace, cfg = trace_runner_point(
    error_amp_floquet_runner,
    program_class=ErrorAmplificationProgram,
    stor_mode_no=stor_i,
    parameter_to_test="frequency",
    span=0.03,
    expts=60,
    relax_delay=200,
    active_reset=True,
    man_reset=True,
    storage_reset=[stor_i],
    reset_dump_mode=2,
)

fig, ax = trace.plot_circuit(
    waveform_blocks=True,
    annotate_params=True,
    annotate_sweep=True,
    detail = 'full'
)

# %%
dm_sideband_scramble_defaults = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    ro_stor=0, # storage mode number that gets read out in the end

    init_fock=True,

    normalize=False,
    post_select_pre_pulse=False,
    active_reset=False,
    man_reset=False, 
    storage_reset=False, 
    prepulse=True,
    postpulse=True,
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values


import textwrap

# %%
dm_sideband_scramble_defaults = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    ro_stor=0, # storage mode number that gets read out in the end

    init_fock=True,

    normalize=False,
    post_select_pre_pulse=False,
    active_reset=False,
    man_reset=False, 
    storage_reset=False, 
    prepulse=True,
    postpulse=True,
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values


import textwrap







            



from tqdm.notebook import tqdm

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.qsim.floquet_dark_mode_readout.SidebandScrambleDarkProgramNewNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles_list = floquet_cycle_list_gen(100, 200, 200, 2)

meas_stors = [0]
# meas_stors = [0]
swap_stors = [2, 5, 6, 7]
detunings = [0, 0, 0, 0] # None/False/unspecified all default to all zeros
# detunings = [150e-3, 150e-3, 150e-3, 150e-3] # None/False/unspecified all default to all zeros
dark_swaps = [2, 5, 6, 7]
scramble_expts = []

rel_phase_list = [0]
# for phase in rel_phase_list:

prog, trace, cfg = trace_runner_point(
    dmscramble_runner,
    reps=300,
    init_fock=False,
    # init_alpha = np.sqrt(3),
    init_man_fock_state = '3',
    init_stor= 0,
    ro_stor = 0,
    relax_delay=200,
    active_reset=True,
    pre_relax_delay = 100, 
    man_reset=True, 
    storage_reset = swap_stors, 
    reset_dump_mode = 2,
    dump_reset_iter_num = 1,
    swap_stors=swap_stors,
    update_phases=True,
    detunings=detunings,
    floquet_cycles=floquet_cycles_list[0],
    swept_params=['floquet_cycle'],
    custom_prepulse = False,
    custom_postpulse = False,
    debug = False,
    load_man_dark = True,
    swap_man_dark = True,
    swap_man_large_dark = True,
    dark_swap_order = dark_swaps,
    second_rel_phase = 180,
    map_to_qubit_ge = True, 
    
    multiparity_readout = True,
    cond_sec_phase= -90,

    prepulse = True, #for debugging. Should always be true
    postpulse = True, #for debugging. Should always be true
    
    perform_wigner = False
    )
# scramble.display()

fig, ax = trace.plot_circuit(
    waveform_blocks=True,
    annotate_params=True,
    annotate_sweep=True,
)

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from experiments.qsim.floquet_dark_mode_readout import DarkBaseExperiment


spectroscopy_fnames = [
    [[r'C:\experiments\260526_qsim_darkmode\data\JOB-20260720-00330_QsimBaseExperiment.h5'],
     [r'C:\experiments\260526_qsim_darkmode\data\JOB-20260720-00331_QsimBaseExperiment.h5']],
]

occupation_strings = []
A_rows = []
cycles = None
decoder_phase_matrix = None
swap_stors = None
physical_kerr_MHz = None

for occupation_files in spectroscopy_fnames:
    Q_by_phi = {}
    saved_occupation = None

    for phi_files in occupation_files:
        cycle_parts = []
        Q_parts = []
        saved_phi = None

        for fname in phi_files:
            expt = DarkBaseExperiment.from_h5file(fname)
            expt_cfg = expt.cfg.expt

            saved_kerr_MHz = float(np.asarray(
                expt.cfg.device.manipulate.kerr
            ).reshape(-1)[0])
            saved_kerr_MHz = -10e-3
            if physical_kerr_MHz is None:
                physical_kerr_MHz = saved_kerr_MHz
            elif not np.isclose(saved_kerr_MHz, physical_kerr_MHz):
                raise ValueError('the saved M1 Kerr changed between jobs')

            occupation = tuple(int(n) for n in expt_cfg.spectroscopy_occupations)
            phi = float(expt_cfg.spectroscopy_analyzer_phase) % 360.
            measured_theta = np.asarray(expt.data['xpts'], dtype=float) % 360.
            measured_cycles = np.asarray(expt.data['ypts'], dtype=int)
            signal = np.asarray(expt.data['avgi'], dtype=float).reshape(
                len(measured_cycles),
                len(measured_theta),
            )

            theta_0 = np.flatnonzero(np.isclose(measured_theta, 0.))[0]
            theta_180 = np.flatnonzero(np.isclose(measured_theta, 180.))[0]

            readout_Ig = float(np.asarray(
                expt.cfg.device.readout.Ig
            ).reshape(-1)[0])
            readout_Ie = float(np.asarray(
                expt.cfg.device.readout.Ie
            ).reshape(-1)[0])
            Pe = (signal - readout_Ig) / (readout_Ie - readout_Ig)

            if saved_occupation is None:
                saved_occupation = occupation
            elif occupation != saved_occupation:
                raise ValueError('one occupation group contains different states')

            if saved_phi is None:
                saved_phi = phi
            elif not np.isclose(phi, saved_phi):
                raise ValueError('one analyzer group contains different phases')

            elif not np.allclose(saved_matrix, decoder_phase_matrix):
                raise ValueError('the saved decoder phase matrix changed between jobs')

            cycle_parts.append(measured_cycles)
            Q_parts.append(Pe[:, theta_0] - Pe[:, theta_180])
            del expt

        acquired_cycles = np.concatenate(cycle_parts)
        Q = np.concatenate(Q_parts)
        order = np.argsort(acquired_cycles)
        acquired_cycles = acquired_cycles[order]
        Q = Q[order]

        if cycles is None:
            cycles = acquired_cycles
        elif not np.array_equal(acquired_cycles, cycles):
            raise ValueError('the saved Floquet cycles are incomplete or different')

        Q_by_phi[int(round(saved_phi)) % 360] = Q

    if 0 not in Q_by_phi or 90 not in Q_by_phi:
        raise ValueError('each occupation needs analyzer phases 0 and 90 deg')

    occupation_strings.append(saved_occupation)
    A_rows.append(Q_by_phi[0] + 1j * Q_by_phi[90])

# %%
# expt.data.  # TODO(stage2): source cell P165 is this half-typed
# line and nothing else; the intended attribute is not recoverable.

# %%
# Plot the complex return from the two analyzer-phase jobs.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.qsim.floquet_dark_mode_readout import DarkBaseExperiment


phase_fnames = [
    r'C:\experiments\260526_qsim_darkmode\data\JOB-20260720-00330_QsimBaseExperiment.h5',
    r'C:\experiments\260526_qsim_darkmode\data\JOB-20260720-00331_QsimBaseExperiment.h5',
]
normalize_to_zero_pulse = False

quadratures = {}
pulse_counts = None

for fname in phase_fnames:
    expt = DarkBaseExperiment.from_h5file(fname)
    measured_pulse_counts = np.asarray(expt.data['ypts'], dtype=int)
    prep_phases = np.asarray(expt.data['xpts'], dtype=float) % 360.
    signal = np.asarray(expt.data['avgi'], dtype=float).reshape(
        len(measured_pulse_counts),
        len(prep_phases),
    )

    Ig = float(np.asarray(expt.cfg.device.readout.Ig).reshape(-1)[0])
    Ie = float(np.asarray(expt.cfg.device.readout.Ie).reshape(-1)[0])
    Pe = (signal - Ig) / (Ie - Ig)

    theta_0 = np.flatnonzero(np.isclose(prep_phases, 0.))[0]
    theta_180 = np.flatnonzero(np.isclose(prep_phases, 180.))[0]
    analyzer_phase = int(round(
        float(expt.cfg.expt.spectroscopy_analyzer_phase)
    )) % 360
    quadratures[analyzer_phase] = Pe[:, theta_0] - Pe[:, theta_180]

    if pulse_counts is None:
        pulse_counts = measured_pulse_counts
    elif not np.array_equal(measured_pulse_counts, pulse_counts):
        raise ValueError('the two jobs contain different pulse counts')

complex_return = quadratures[0] + 1j * quadratures[90]
if normalize_to_zero_pulse:
    plotted_return = complex_return / complex_return[0]
    axis_label = 'A / A(0)'
else:
    plotted_return = complex_return
    axis_label = 'A'

fig, ax = plt.subplots(figsize=(6.4, 5.8))
ax.plot(plotted_return.real, plotted_return.imag, color='0.7', zorder=1)
points = ax.scatter(
    plotted_return.real,
    plotted_return.imag,
    c=pulse_counts,
    cmap='viridis',
    s=48,
    zorder=2,
)
ax.scatter(
    plotted_return.real[0],
    plotted_return.imag[0],
    marker='*',
    s=180,
    color='tab:red',
    label='zero pulses',
    zorder=3,
)

theta = np.linspace(0, 2*np.pi, 361)
cpx = 0.82*np.exp(1j*theta)
ax.scatter(
    cpx.real,
    cpx.imag,
    s = 1,
    alpha = 0.5
)


ax.axhline(0., color='0.8', linewidth=1.)
ax.axvline(0., color='0.8', linewidth=1.)
ax.set_xlabel(f'Re({axis_label})')
ax.set_ylabel(f'Im({axis_label})')
ax.set_aspect('equal', adjustable='datalim')
ax.legend()
fig.colorbar(points, ax=ax, label='number of physical Floquet pulses')
# fig.suptitle(
#     'M1 from M1-S4 complex return\n'
#     + '\n'.join(Path(fname).name for fname in phase_fnames)
# )
plt.tight_layout()
plt.show()
