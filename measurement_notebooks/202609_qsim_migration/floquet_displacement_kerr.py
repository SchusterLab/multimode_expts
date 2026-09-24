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
# # Floquet-pulse Kerr from coherent displacement
#
# Split out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
# 370-374 by the stage-2 notebook decomposition. One of the adjacent extensions
# on the surface map, kept as its own recipe because its measurement identity is
# distinct from the MBR products.
#
# The section already had the shape stage 2 asks for -- a defaults dict, a
# `CharacterizationRunner`, one `execute()`, then `analyze()`/`display()` -- so
# nothing needed wrapping and it defines no helpers. Only the setup preamble
# changed: the defaults of cells 2-6 are now
# `experiments/qsim/notebook_helpers/defaults.py`.
#
# The library side is `experiments/qsim/floquet_displacement_kerr.py` and
# `FloquetDisplacementKerrExperiment` in
# `experiments/qsim/floquet_dark_mode_readout.py`.
#
# Its neighbours: `multiphoton_calibration.py`, `floquet_calibration.py`,
# `mbr.py`, `mbr_disorder.py`, `mbr_tomography.py`, `mbr_sff.py`.
# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy

import experiments as meas
from slab import AttrDict
from experiments import CharacterizationRunner, SweepRunner, MultimodeStation

from job_server import JobClient
# Imported under the names the relocated cells already use.
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS as active_reset_default_dict,
    FLOQUET_DEFAULTS as floquet_default_dict,
    MEASUREMENT_CONFIG_DEFAULTS as measurement_config_default_dict,
)
from experiments.qsim.notebook_helpers.run_mode import run_settings

# Set by tools/run_qsim_suite.py. Unset: a normal run through the queue.
RUN = run_settings()

# %%
# The config versions this campaign ran against.
config_dict = {
    "hardware_config": "CFG-HW-20260904-00019",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260904-00014",
    "floquet_storage_swap": "CFG-FL-20260904-00042",
}

station = MultimodeStation(
    user="jonginn",
    experiment_name="260818_qsim_spectroscopy",
    project="EncSpec",
    log_measurements=not RUN.smoke,
    mock=RUN.mock,
    **RUN.station_configs(config_dict),
)
client = JobClient()

# %% [markdown]
# # Floquet-pulse Kerr from coherent displacement
#
# This uses the same sequence and analysis as `CavityRamseyGainSweepExperiment`: `D(alpha) -> closed forward/backward Floquet pairs -> D(alpha) -> slow-pi vacuum readout`. `ramsey_freq` only advances the frame of the second displacement; set it to zero when the two displacement phases should also be identical. One closed pair contains two physical Floquet cycles, so the fitting time is `2 * n_cycle_pair * floquet_cycle_us`.

# %%
# Configure the coherent-state Kerr measurement.
import importlib
from experiments.qsim import floquet_dark_mode_readout

importlib.reload(floquet_dark_mode_readout)

DispKerr = floquet_dark_mode_readout.FloquetDisplacementKerrExperiment
floquet_kerr_modes = [4, 5, 6, 7]
floquet_kerr_cycle_pairs = np.arange(0, 31, RUN.pick(1, smoke=3), dtype=int)
floquet_kerr_displace_gains = np.arange(2000, 6001, RUN.pick(1000, smoke=2000))
floquet_kerr_ramsey_freq = 0.2  # MHz; keep below 1 / (4 * floquet_kerr_cycle_us)

floquet_kerr_defaults = AttrDict(dict(
    expts=1,
    rounds=1,
    reps=RUN.pick(500, smoke=100),
    qubits=[0],
    active_reset=True,
    man_reset=True,
    # Not mode 6: its M1-S6 row in CFG-M1-20260904-00014 has no pi time.
    storage_reset=[4, 5, 7],
    pre_relax_delay=100,
    relax_delay=200,
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    dump_reset_iter_num=active_reset_default_dict["dump_reset_iter_num"],
    use_qubit_man_reset=False,
    normalize=False,
    swap_stors=floquet_kerr_modes,
    scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
    floquet_hardware_loop=floquet_default_dict["floquet_hardware_loop"],
    update_phases=True,
    zero_floquet_gain=False,
    man_mode_no=1,
    perform_wigner=False,
    do_g_and_e=False,
    ramsey_freq=floquet_kerr_ramsey_freq,
    displace_gains=floquet_kerr_displace_gains,
    n_cycle_pairs=floquet_kerr_cycle_pairs,
    swept_params=['displace_gain', 'n_cycle_pair'],
))

floquet_kerr_runner = CharacterizationRunner(
    station=station,
    ExptClass=DispKerr,
    ExptProgram=floquet_dark_mode_readout.FloquetDisplacementKerrProgram,
    default_expt_cfg=floquet_kerr_defaults,
    job_client=client,
    use_queue=RUN.use_queue,
    show=False,
)

# %%
# One 2D job sweeps displacement gain and the number of closed Floquet pairs.
floquet_kerr_expt = floquet_kerr_runner.execute(
    postprocess=False,
    log=False,
    show=False,
)

# %% tags=["mock-skip"]

floquet_kerr_expt.analyze(debug=False)
floquet_kerr_expt.display(save_fig=False)
plt.show()

print('Floquet Kerr:', floquet_kerr_expt.data.Kerr * 1e3, '+/-', floquet_kerr_expt.data.Kerr_err * 1e3, 'kHz')
print('linear detuning:', floquet_kerr_expt.data.detuning_g * 1e3, '+/-', floquet_kerr_expt.data.detuning_g_err * 1e3, 'kHz')

# %% tags=["mock-skip"]
plt.plot(floquet_kerr_expt.data['avgi'][1])
