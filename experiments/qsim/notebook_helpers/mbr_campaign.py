"""The shared MBR campaign base: defaults, mode labels, and a runner per job class.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
286-317 by the stage-2 notebook decomposition. The `mbr` section was the hub
of the source notebook: cells 289-293 bound the defaults, modes and
calibrations that the disorder, tomography and SFF sections then read out of
the live kernel. `build_campaign` makes that handoff explicit:

    campaign = build_campaign(floquet_settings=...,
                              active_reset_settings=...,
                              measurement_settings=...)

so each MBR notebook builds the same base rather than depending on another
having run first. `campaign_runner` gives the `CharacterizationRunner` an
assembled class's `acquire(runner)` needs (docs/qsim/mbr_redesign.md,
section 5).

The phase calibration is an `MBRCalibrationSetExperiment` (acquire, save,
`from_manifest`), which each notebook holds itself. The old-class part --
`ensure_calibration`, `acquire_calibration` and the `EncSpec`,
`floquet_dark_mode_readout` and `calibration_*` fields -- moved to
`experiments/qsim/deprecated/mbr_campaign_legacy.py` in MBR redesign step 7b,
for the dormant notebooks.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any

from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner


@dataclass
class MBRCampaign:
    """Everything cells 289-293 left in the notebook namespace, but the calibration."""

    modes: list
    mode_labels: list
    sync_cycles: int
    defaults: Any


def build_campaign(floquet_settings, active_reset_settings, measurement_settings,
                   modes=(1, 2, 3, 4), reps=1000):
    """Build the campaign base shared by every MBR theme (cells 289-293)."""
    modes = list(modes)
    mode_labels = ["M1"] + [f"S{stor}" for stor in modes]
    sync_cycles = floquet_settings["scramble_sync_cycles"]

    defaults = AttrDict(dict(
        expts=1,
        reps=reps,
        rounds=1,
        qubits=[0],
        normalize=False,
        active_reset=True,
        man_reset=True,
        storage_reset=modes,
        pre_relax_delay=100,
        relax_delay=200,
        reset_dump_mode=active_reset_settings["reset_dump_mode"],
        dump_reset_iter_num=active_reset_settings["dump_reset_iter_num"],
        use_qubit_man_reset=False,
        prepulse=False,
        postpulse=False,
        init_fock=False,
        perform_wigner=False,
        parity_readout=False,
        multiparity_readout=False,
        load_man_dark=False,
        swap_man_dark=False,
        swap_man_large_dark=False,
        update_phases=True,
        floquet_cycle=0,
        palindrome_scramble=floquet_settings["palindrome_scramble"],
        scramble_sync_cycles=sync_cycles,
        floquet_waveform=floquet_settings["floquet_waveform"],
        floquet_hardware_loop=floquet_settings["floquet_hardware_loop"],
        swap_stors=modes,
        detunings=[0.] * len(modes),
        spectroscopy_prep_phases=[0., 180.],
        avoid_yoko=measurement_settings['avoid_yoko'],
        use_multiphoton_swap=measurement_settings['use_multiphoton_swap'],
    ))

    return MBRCampaign(
        modes=modes,
        mode_labels=mode_labels,
        sync_cycles=sync_cycles,
        defaults=defaults,
    )


def campaign_runner(campaign, station, client, ExptClass, use_queue=True):
    """-> the runner for one MBR job class, over the campaign defaults.

    Pass it to an assembled class's `acquire`, e.g.
    `MBRSpectrumExperiment(...).acquire(campaign_runner(..., MBRTimeTraceExperiment))`.
    """
    return CharacterizationRunner(
        station=station,
        ExptClass=ExptClass,
        default_expt_cfg=campaign.defaults,
        job_client=client,
        use_queue=use_queue,
        show=False,
    )


def fixed_n_occupations(N, n_modes, descending=True):
    """Every occupation of `n_modes` modes holding exactly N photons.

    Cells 293, 297, 301 and 305 each built this with the same comprehension
    over itertools.product.
    """
    occupations = [
        list(state)
        for state in product(range(N + 1), repeat=n_modes)
        if sum(state) == N
    ]
    if descending:
        occupations.sort(reverse=True)
    return occupations
