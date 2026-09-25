"""The old MBR campaign base, with the old-class calibration -- DEPRECATED.

Moved from `experiments/qsim/notebook_helpers/mbr_campaign.py` on 2026-09-24
(MBR redesign step 7b), without changes: `MBRCampaign` with its `EncSpec`,
`floquet_dark_mode_readout` and `calibration_*` fields, `build_campaign`,
`ensure_calibration` and `acquire_calibration` (old
`MBRPhaseCorrectionExperiment`). Its users are the dormant off-diagonal and
SFF notebooks and `deprecated/mbr_disorder_offdiag.py`,
`deprecated/mbr_sff_campaign.py`. The live campaign base is
`notebook_helpers/mbr_campaign.py`; see `docs/qsim/mbr_step7_plan.md`.

Not maintained; may break when live code changes. If it breaks, add a note here
and do not fix it.
"""


import importlib
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.deprecated.encoding_spectroscopy import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.deprecated.mbr_phase_correction import (
    EntireFloquetCyclePhaseCalibrationProgram,
)
from experiments.qsim.notebook_helpers.mbr_campaign import fixed_n_occupations


@dataclass
class MBRCampaign:
    """Everything cells 289-293 left in the notebook namespace.

    `calibrations` is mutable on purpose: it is the in-memory cache that the
    source kept as `encspec_calibrations`, keyed by photon number, and
    `ensure_calibration` fills it lazily.
    """

    EncSpec: Any
    floquet_dark_mode_readout: Any
    modes: list
    mode_labels: list
    sync_cycles: int
    defaults: Any
    calibration_job_ids: dict
    calibration_files: dict
    calibrations: dict = field(default_factory=dict)


def build_campaign(station, client, floquet_settings, active_reset_settings,
                   measurement_settings, modes=(1, 2, 3, 4),
                   calibration_job_ids=None, reps=1000, reload_module=True):
    """Build the campaign base shared by every MBR theme (cells 289-293).

    `calibration_job_ids` maps photon number -> list of job IDs. The source
    had it as a dict with both entries commented out, so the default here is
    empty and a notebook fills it in; that is a dataset choice, which the
    stage-2 instructions keep in the notebook.

    `reload_module` reproduces the source's `importlib.reload`, which matters
    when editing the program module in a live kernel.
    """
    from experiments.qsim import floquet_dark_mode_readout

    if reload_module:
        importlib.reload(floquet_dark_mode_readout)

    EncSpec = EncodingHamiltonianSpectroscopyExperiment

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

    calibration_job_ids = dict(calibration_job_ids or {})
    calibration_files = {
        N: [
            station.data_path / f"{job_id}_{EncSpec.__name__}.h5"
            for job_id in job_ids
        ]
        for N, job_ids in calibration_job_ids.items()
    }

    return MBRCampaign(
        EncSpec=EncSpec,
        floquet_dark_mode_readout=floquet_dark_mode_readout,
        modes=modes,
        mode_labels=mode_labels,
        sync_cycles=sync_cycles,
        defaults=defaults,
        calibration_job_ids=calibration_job_ids,
        calibration_files=calibration_files,
        calibrations={},
    )




def ensure_calibration(campaign, N, station):
    """Return the N-photon phase calibration, loading it from HDF5 if needed.

    Old class (`MBRPhaseCorrectionExperiment`); for the disorder and SFF
    notebooks until redesign step 7.

    Cells 305 and 316 had these same eight lines. Raises rather than guessing
    if no job IDs were registered for that photon number -- a missing
    calibration is a missing scientific input, which the stage-2 instructions
    say must fail visibly.
    """
    from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment

    if N in campaign.calibrations:
        return campaign.calibrations[N]

    if N not in campaign.calibration_files:
        raise KeyError(
            f"no calibration job IDs registered for N={N}; set "
            f"campaign.calibration_job_ids[{N}] and rebuild calibration_files, "
            f"or acquire a fresh calibration first"
        )

    calibration_expt = MBRPhaseCorrectionExperiment.from_job_files(
        campaign.calibration_files[N],
        station=station,
    )
    calibration_expt.batch_job_ids = campaign.calibration_job_ids[N]
    calibration_expt.analyze()
    campaign.calibrations[N] = calibration_expt
    return calibration_expt


def acquire_calibration(campaign, station, client, N, cycle_pairs, reps,
                        batch_size=10, occupations=None, use_queue=True):
    """Acquire a fresh N-photon phase calibration and cache it on the campaign.

    Old class (`MBRPhaseCorrectionExperiment`); for the disorder and SFF
    notebooks until redesign step 7.

    The "Run a new calibration" cell of `mbr.py`, as a function, so the
    notebooks that otherwise load a calibration by job ID can acquire one
    instead -- which the test suite needs, because it has no job IDs.
    `occupations` defaults to the whole fixed-N basis.
    """
    from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment

    if occupations is None:
        occupations = fixed_n_occupations(N, len(campaign.mode_labels))
    batch = MBRPhaseCorrectionExperiment.calibration_batch(
        campaign.defaults, campaign.modes, occupations, cycle_pairs,
        sync_cycles=campaign.sync_cycles, repeats=1, reps=reps,
    )
    runner = CharacterizationRunner(
        station=station,
        ExptClass=campaign.EncSpec,
        ExptProgram=EntireFloquetCyclePhaseCalibrationProgram,
        default_expt_cfg=batch.default_expt_cfg,
        job_client=client,
        use_queue=use_queue,
        show=False,
    )
    calibration_expt = MBRPhaseCorrectionExperiment._from_expts(runner.execute(
        overrides=batch.configs, batch_size=batch_size, log=True, show=False,
    ), job_ids=runner.last_job_ids, station=runner.station)
    calibration_expt.analyze()
    campaign.calibrations[N] = calibration_expt
    return calibration_expt
