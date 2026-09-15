"""The shared MBR campaign base, and the spectroscopy steps built on it.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
286-317 by the stage-2 notebook decomposition.

Primary caller: `measurement_notebooks/202609_qsim_migration/mbr.py`. But this
module exists mostly because of the *other* callers. The `mbr` section was the
hub of the source notebook: cells 289-293 bound `EncSpec`, `BatchRunner`,
`encspec_modes`, `encspec_mode_labels`, `encspec_sync_cycles`,
`encspec_defaults`, `encspec_calibration_job_ids/files` and
`encspec_calibrations`, and then the disorder, tomography and SFF sections all
read those names out of the live kernel. Splitting them into separate notebooks
means that handoff has to become explicit, which is what `build_campaign` is
for:

    campaign = build_campaign(station, client, floquet_settings=...,
                              active_reset_settings=...,
                              measurement_settings=...)

So `mbr_disorder.py`, `mbr_tomography.py` and `mbr_sff.py` each open by
building the same campaign base rather than depending on `mbr.py` having been
run first.

`ensure_calibration` was the same eight lines in cells 305 and 316; it is one
function here.

Temporary home, per the stage-2 instructions. In particular nothing here was
reconciled with the four aggregate stage classes it calls
(`MBRPhaseCorrectionExperiment`, `MBRSpectrumExperiment`,
`MBROrthogonalityExperiment`, `MBRPropagatorExperiment`) or with
`EncodingHamiltonianSpectroscopyExperiment`, which is still both the loading
layer and the class every job was acquired under.
"""

import importlib
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from slab import AttrDict


@dataclass
class MBRCampaign:
    """Everything cells 289-293 left in the notebook namespace.

    `calibrations` is mutable on purpose: it is the in-memory cache that the
    source kept as `encspec_calibrations`, keyed by photon number, and
    `ensure_calibration` fills it lazily.
    """

    EncSpec: Any
    BatchRunner: Any
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

    EncSpec = floquet_dark_mode_readout.EncodingHamiltonianSpectroscopyExperiment
    BatchRunner = floquet_dark_mode_readout.BatchRunner

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
        BatchRunner=BatchRunner,
        floquet_dark_mode_readout=floquet_dark_mode_readout,
        modes=modes,
        mode_labels=mode_labels,
        sync_cycles=sync_cycles,
        defaults=defaults,
        calibration_job_ids=calibration_job_ids,
        calibration_files=calibration_files,
        calibrations={},
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


def ensure_calibration(campaign, N, station):
    """Return the N-photon phase calibration, loading it from HDF5 if needed.

    Cells 305 and 316 had these same eight lines. Raises rather than guessing
    if no job IDs were registered for that photon number -- a missing
    calibration is a missing scientific input, which the stage-2 instructions
    say must fail visibly.
    """
    from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment

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


# --------------------------------------------------------------------------
# Cell 305, split at its own substeps.
# --------------------------------------------------------------------------


@dataclass
class SpectroscopyPlan:
    occupations: list
    final_occupations: list
    display_occupation: Any
    N: int


def select_spectroscopy_occupations(batch_encspec_N, mode_labels,
                                    occupation=None, decoder_occupation=None):
    """Resolve which matrix elements to measure (first half of cell 305).

    Two modes, exactly as the source markdown describes. With
    `batch_encspec_N=None`, `occupation` and `decoder_occupation` name the
    encoder/decoder pairs for individual matrix elements. With
    `batch_encspec_N=N`, those two are ignored and every diagonal occupation
    in the fixed-N sector is acquired, which is what the complete-basis DOS
    display needs.
    """
    if batch_encspec_N is None:
        if occupation is None or decoder_occupation is None:
            raise ValueError(
                "single-pair mode requires occupation and decoder_occupation"
            )
        N = sum(occupation[0])
        for occupation_indv, decoder_occupation_indv in zip(
                occupation, decoder_occupation):
            if len(occupation_indv) != len(mode_labels):
                raise ValueError("occupation has the wrong mode count")
            if len(decoder_occupation_indv) != len(mode_labels):
                raise ValueError("decoder_occupation has the wrong mode count")
            if sum(occupation_indv) != N:
                raise ValueError("encoder and decoder photon numbers differ")
            if sum(decoder_occupation_indv) != N:
                raise ValueError("encoder and decoder photon numbers differ")
        occupations = [list(state) for state in occupation]
        final_occupations = [list(state) for state in decoder_occupation]
        display_occupation = list(occupation[0])
    else:
        N = int(batch_encspec_N)
        if N < 0:
            raise ValueError("batch_encspec_N must be non-negative")
        occupations = fixed_n_occupations(N, len(mode_labels))
        final_occupations = [list(state) for state in occupations]
        display_occupation = None

    return SpectroscopyPlan(
        occupations=occupations,
        final_occupations=final_occupations,
        display_occupation=display_occupation,
        N=N,
    )


def build_spectroscopy_batch(campaign, station, client, plan,
                             cycle_chunks, reps, detunings=None):
    """Phase-correct the plan and build its batch and runner (cell 305 tail).

    Returns (batch, runner, calibration_expt, cycle_branches). Submits
    nothing; the notebook calls `runner.execute` so that job submission stays
    a visible, separate step.
    """
    from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
    from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment

    calibration_expt = ensure_calibration(campaign, plan.N, station)

    cycle_branches = {
        tuple(state): 0 for state in plan.final_occupations
    }
    correction = MBRPhaseCorrectionExperiment.phase_correction_from_calibration(
        calibration_expt,
        cycle_branches=cycle_branches,
    )
    missing_calibrations = [
        tuple(state)
        for state in plan.final_occupations
        if tuple(state) not in correction.phase_by_occupation
    ]
    if missing_calibrations:
        raise ValueError(f"missing calibration rows: {missing_calibrations}")

    if detunings is None:
        detunings = [0.] * len(campaign.modes)

    batch = MBRSpectrumExperiment.spectroscopy_batch(
        campaign.defaults,
        campaign.modes,
        plan.occupations,
        cycle_chunks,
        correction.phase_by_occupation,
        detunings=detunings,
        sync_cycles=campaign.sync_cycles,
        reps=reps,
        final_occupations=plan.final_occupations,
    )
    runner = campaign.BatchRunner(
        station=station,
        ExptClass=campaign.EncSpec,
        ExptProgram=batch.program,
        default_expt_cfg=batch.default_expt_cfg,
        job_client=client,
        show=False,
    )

    print("N:", plan.N)
    print("occupations:", len(plan.occupations))
    print("jobs:", len(batch.configs))
    return batch, runner, calibration_expt, cycle_branches


# --------------------------------------------------------------------------
# Cells 297 and 298: replace selected calibration occupations.
# --------------------------------------------------------------------------


def validate_recalibration_occupations(campaign, N, recalibration_occupations):
    """Check the requested rows against the active calibration (cell 297 head).

    Returns (base_calibration_expt, recalibration_keys). Every check here was
    in the source; they are the reason this is worth having as one call.
    """
    if N not in campaign.calibrations:
        raise RuntimeError(f"load or acquire the N={N} calibration first")
    base_calibration_expt = campaign.calibrations[N]
    base_occupation_keys = {
        tuple(occupation)
        for occupation in base_calibration_expt.data.occupations
    }
    recalibration_keys = [
        tuple(occupation) for occupation in recalibration_occupations
    ]
    if len(set(recalibration_keys)) != len(recalibration_keys):
        raise ValueError("recalibration occupations must be unique")
    for occupation in recalibration_occupations:
        if len(occupation) != len(campaign.mode_labels):
            raise ValueError(f"wrong mode count: {occupation}")
        if sum(occupation) != N:
            raise ValueError(f"wrong photon number: {occupation}")
        if tuple(occupation) not in base_occupation_keys:
            raise ValueError(
                f"occupation is absent from the active calibration: {occupation}"
            )
    return base_calibration_expt, recalibration_keys


def merge_replacement_calibration(campaign, station, N,
                                  replacement_calibration_expt,
                                  recalibration_keys,
                                  recalibration_cycle_pairs):
    """Swap the re-measured rows into the active calibration (cell 298).

    Refuses to mix jobs whose Floquet hardware differs, and afterwards
    verifies that no non-target row moved. Both checks were in the source and
    both raise rather than warn.

    Mutates `campaign` -- `calibrations`, `calibration_job_ids` and
    `calibration_files` for this N -- and returns
    (updated_calibration_expt, old_phase_by_occupation,
    new_phase_by_occupation).
    """
    from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment

    old_calibration_expt = campaign.calibrations[N]
    old_occupation_order = [
        tuple(occupation)
        for occupation in old_calibration_expt.data.occupations
    ]
    recalibration_key_set = set(recalibration_keys)

    if list(old_calibration_expt.data.mode_labels) != list(
            replacement_calibration_expt.data.mode_labels):
        raise ValueError("replacement calibration uses a different mode order")
    old_hardware = old_calibration_expt.data.hardware
    new_hardware = replacement_calibration_expt.data.hardware
    if (
        not np.isclose(
            old_hardware.floquet_cycle_us,
            new_hardware.floquet_cycle_us,
        )
        or not np.allclose(
            old_hardware.couplings_MHz,
            new_hardware.couplings_MHz,
        )
        or not np.isclose(
            old_hardware.physical_kerr_MHz,
            new_hardware.physical_kerr_MHz,
        )
    ):
        raise ValueError(
            "Floquet hardware changed; do not mix partial calibration jobs"
        )

    old_job_ids = list(old_calibration_expt.batch_job_ids)
    old_expts = list(old_calibration_expt.batch_expts)
    new_job_ids = list(replacement_calibration_expt.batch_job_ids)
    new_expts = list(replacement_calibration_expt.batch_expts)
    if len(old_job_ids) != len(old_expts):
        raise RuntimeError("old calibration jobs and IDs are not aligned")
    if len(new_job_ids) != len(new_expts):
        raise RuntimeError("replacement calibration jobs and IDs are not aligned")

    old_phase_by_occupation = {
        tuple(occupation): float(phase)
        for occupation, phase in zip(
            old_calibration_expt.data.occupations,
            old_calibration_expt.data.phase_mod180,
        )
    }
    merged_expts = []
    merged_job_ids = []
    for expt, job_id in zip(old_expts, old_job_ids):
        occupation_key = tuple(expt.cfg.expt.spectroscopy_occupations)
        if occupation_key in recalibration_key_set:
            continue
        merged_expts.append(expt)
        merged_job_ids.append(job_id)
    for expt, job_id in zip(new_expts, new_job_ids):
        merged_expts.append(expt)
        merged_job_ids.append(job_id)

    updated_calibration_expt = MBRPhaseCorrectionExperiment.from_job_files(
        merged_expts,
        station=station,
    )
    updated_calibration_expt.batch_job_ids = merged_job_ids
    updated_calibration_expt.analyze(
        occupations=old_occupation_order,
        cycle_pairs=recalibration_cycle_pairs,
    )
    new_phase_by_occupation = {
        tuple(occupation): float(phase)
        for occupation, phase in zip(
            updated_calibration_expt.data.occupations,
            updated_calibration_expt.data.phase_mod180,
        )
    }
    for occupation_key in old_occupation_order:
        if occupation_key in recalibration_key_set:
            continue
        if not np.isclose(
            old_phase_by_occupation[occupation_key],
            new_phase_by_occupation[occupation_key],
        ):
            raise RuntimeError(
                f"non-target calibration changed: {occupation_key}"
            )

    campaign.calibrations[N] = updated_calibration_expt
    campaign.calibration_job_ids[N] = merged_job_ids
    campaign.calibration_files[N] = [
        station.data_path / f"{job_id}_{campaign.EncSpec.__name__}.h5"
        for job_id in merged_job_ids
    ]

    for occupation_key in recalibration_keys:
        print(
            occupation_key,
            f"{old_phase_by_occupation[occupation_key]:+.6f}",
            "->",
            f"{new_phase_by_occupation[occupation_key]:+.6f}",
            "deg / cycle",
        )
    print("active calibration rows:", len(old_occupation_order))
    print("replacement jobs:", new_job_ids)
    return (updated_calibration_expt, old_phase_by_occupation,
            new_phase_by_occupation)


# --------------------------------------------------------------------------
# Cell 316: propagator setup.
# --------------------------------------------------------------------------


def build_propagator_batch(campaign, station, client, propagator_occupations,
                           propagator_cycles, reps=1000):
    """Phase-correct and build the propagator batch and runner (cell 316).

    Submits nothing. Returns (batch, runner, calibration_expt).
    """
    from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment
    from experiments.qsim.mbr_propagator import MBRPropagatorExperiment

    N = sum(propagator_occupations[0])
    calibration_expt = ensure_calibration(campaign, N, station)

    correction = MBRPhaseCorrectionExperiment.phase_correction_from_calibration(
        calibration_expt,
        cycle_branches={
            tuple(occupation): 0 for occupation in propagator_occupations
        },
    )
    batch = MBRPropagatorExperiment.propagator_batch(
        campaign.defaults,
        campaign.modes,
        propagator_occupations,
        propagator_cycles,
        phase_by_occupation=correction.phase_by_occupation,
        sync_cycles=campaign.sync_cycles,
        reps=reps,
    )
    runner = campaign.BatchRunner(
        station=station,
        ExptClass=campaign.EncSpec,
        ExptProgram=campaign.floquet_dark_mode_readout.EncodingPropagatorProgram,
        default_expt_cfg=batch.default_expt_cfg,
        job_client=client,
        show=False,
    )
    return batch, runner, calibration_expt


def plot_propagator_matrices(propagator_data):
    """|U| at each acquired cycle depth (cell 317 tail).

    Returns (fig, axes).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(propagator_data.cycles),
        figsize=(5 * len(propagator_data.cycles), 4),
        constrained_layout=True,
    )
    for axis, cycle, matrix in zip(
        np.atleast_1d(axes),
        propagator_data.cycles,
        propagator_data.matrices,
    ):
        image = axis.imshow(np.abs(matrix), origin="upper", cmap="magma")
        axis.set_title(f"cycle {cycle}: |U|")
        axis.set_xlabel("encoder")
        axis.set_ylabel("decoder")
        fig.colorbar(image, ax=axis)
    plt.show()
    return fig, axes
