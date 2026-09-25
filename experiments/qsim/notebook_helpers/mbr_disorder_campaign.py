"""Planning of the 7-1 diagonal-disorder campaign, on the new MBR classes.

The 7-1 campaign of jonginn's `qsim_experiments.ipynb` (cells 323-334): per
realization, a random zero-mean detuning direction, the fixed-N theory at
those detunings, and a max-min LDOS selection of which diagonal traces to
measure. Caller: `measurement_notebooks/202609_qsim_migration/mbr_disorder.py`.

Rewritten on the new classes in MBR redesign step 7c (2026-09-24); the old
functions are in `experiments/qsim/deprecated/mbr_disorder_campaign.py`
(docs/qsim/mbr_step7_plan.md). The notebook now does:

    plan = plan_diagonal_disorder(calibration, campaign.modes, config)
    parts = []
    for record in plan.realizations:
        spectrum = realization_spectrum(plan, record, calibration, campaign, config)
        spectrum.acquire(runner, batch_size=config.batch_size)
        spectrum.analyze(); spectrum.save()
        parts.append(spectrum)
    ensemble = MBRDisorderEnsembleExperiment.from_parts(
        parts, realizations=plan.realizations[:len(parts)], calibration=calibration)
    analyze_diagonal_disorder(ensemble, plan, config)
    ensemble.display(); ensemble.save()

The numerics are in :mod:`fitting.qsim.mbr_disorder` and the theory in
:func:`fitting.qsim.mbr_hamiltonian.fixed_n_hamiltonian`.

One difference from the old code, on purpose: the analyzer correction played
on the pulse comes from the calibration set as it is
(`MBRCalibrationSetExperiment.phase_correction`, at the calibration jobs'
Kerr). The old code rebuilt it at the campaign's self-Kerr. The analysis
applies the campaign's Kerr in the manual-Kerr frame, which undoes the played
correction first, so the analyzed data do not depend on this.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from slab import AttrDict

from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from fitting.qsim import mbr_disorder
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian


@dataclass
class DiagDisorderConfig:
    """Cell 323's `diag_disorder_*` knobs, with its values."""

    N: int = 3
    realization_count: int = 20
    strength_kHz: float = 50.0
    master_seed: int = 20260816
    selected_states: int = 10
    max_cycle: int = 200
    min_time_points: int = 100
    nyquist_margin: float = 1.35
    reps: int = 1200
    batch_size: int = 2
    edge_fraction: float = 0.10
    gap_ratio_bins: int = 15
    match_tolerance_bins: float = 1.5
    require_complete_match: bool = False

    # None uses `best_self_kerr_kHz` of `plan_diagonal_disorder` when given,
    # then the signed Kerr saved with the phase-calibration jobs.
    self_kerr_kHz: Optional[float] = None

    # Add only occupations whose calibration phase needs a nonzero 180-deg
    # branch.
    branch_overrides: dict = field(default_factory=dict)


def plan_diagonal_disorder(calibration, swap_stors, config=None, best_self_kerr_kHz=None):
    """Plan the realizations and the cycle grid (cell 325). Submits nothing.

    ``calibration`` is the `MBRCalibrationSetExperiment` the campaign runs
    with; its jobs give the Floquet cycle time, couplings and Kerr.
    ``best_self_kerr_kHz`` is a fitted Kerr (for example `fit_self_kerr` in
    `mbr_n3_reprocess`); ``config.self_kerr_kHz`` wins over it.

    Returns AttrDict with ``realizations`` (one record per realization, the
    records `MBRDisorderEnsembleExperiment` keeps), ``theory`` ({realization:
    Hamiltonian}), the ``grid`` (see :func:`fitting.qsim.mbr_disorder.cycle_grid`),
    ``cycles``, ``self_kerr_kHz``, ``kerr_source``, ``total_jobs`` and
    ``estimated_hours``.
    """
    config = config or DiagDisorderConfig()
    if "phase_mod180" not in calibration.data:
        calibration.analyze()
    hardware = calibration.data.hardware
    if config.self_kerr_kHz is not None:
        self_kerr_kHz, kerr_source = float(config.self_kerr_kHz), "config.self_kerr_kHz"
    elif best_self_kerr_kHz is not None:
        self_kerr_kHz, kerr_source = float(best_self_kerr_kHz), "best_self_kerr_kHz"
    else:
        self_kerr_kHz = 1e3 * float(hardware.physical_kerr_MHz)
        kerr_source = "phase-calibration jobs"
    self_kerr_MHz = 1e-3 * self_kerr_kHz
    swap_stors = [int(stor) for stor in swap_stors]
    mode_count = len(swap_stors) + 1

    realizations, theory = [], {}
    for realization in range(config.realization_count):
        seed = config.master_seed + realization
        direction = mbr_disorder.disorder_direction(seed, len(swap_stors))
        # The analyzer Hamiltonian uses onsite = -detunings.
        onsite_MHz = 1e-3 * config.strength_kHz * direction
        hamiltonian = fixed_n_hamiltonian(config.N, mode_count, -onsite_MHz,
                                          hardware.couplings_MHz, self_kerr_MHz)
        rows, floor, coverage = mbr_disorder.select_diagonal_rows(
            hamiltonian.basis_eigenstate_weights, config.selected_states)
        realizations.append(dict(
            realization=realization,
            seed=int(seed),
            strength_kHz=float(config.strength_kHz),
            direction=direction.tolist(),
            onsite_MHz=onsite_MHz.tolist(),
            selected_occupations=[list(hamiltonian.fock_basis[row]) for row in rows],
            selection_floor=float(floor),
            self_kerr_kHz=self_kerr_kHz,
        ))
        theory[realization] = hamiltonian

    max_abs_energy_MHz = max(np.max(np.abs(h.energies_MHz)) for h in theory.values())
    grid = mbr_disorder.cycle_grid(max_abs_energy_MHz, hardware.floquet_cycle_us,
                                   config.max_cycle, config.min_time_points,
                                   config.nyquist_margin)
    total_jobs = config.realization_count * config.selected_states
    # The old estimate: 3.76 s per cycle point per (occupation, analyzer
    # phase) job at 1200 reps. A new job holds both analyzer phases.
    estimated_hours = (2 * total_jobs * len(grid.cycles) * 3.76
                       * (config.reps / 1200.0) / 3600.0)

    print(f"diagonal disorder: N={config.N}, realizations={config.realization_count}, "
          f"jobs={total_jobs}")
    print(f"cycles=0:{config.max_cycle}:{grid.step}, points={len(grid.cycles)}, "
          f"Nyquist={1e3 * grid.nyquist_MHz:.3f} kHz, "
          f"FFT resolution={1e3 * grid.fft_resolution_MHz:.3f} kHz")
    print(f"signed M1 self-Kerr={self_kerr_kHz:.3f} kHz ({kerr_source})")
    print(f"estimated single-worker time={estimated_hours:.1f} h")
    for record in realizations:
        print(f"r={record['realization']} seed={record['seed']} | "
              f"leaf onsite={np.round(1e3 * np.asarray(record['onsite_MHz']), 3)} kHz")
        print("  diagonal occupations:",
              [tuple(o) for o in record["selected_occupations"]])

    return AttrDict(dict(
        realizations=realizations,
        theory=theory,
        grid=grid,
        cycles=grid.cycles,
        self_kerr_kHz=self_kerr_kHz,
        kerr_source=kerr_source,
        total_jobs=int(total_jobs),
        estimated_hours=float(estimated_hours),
    ))


def realization_spectrum(plan, record, calibration, campaign, config=None):
    """-> the `MBRSpectrumExperiment` that acquires one planned realization.

    Its jobs play the realization's detunings (``-onsite_MHz``) and the
    calibration set's correction with ``config.branch_overrides``. The
    calibration set must be saved, so each job records its manifest.
    """
    config = config or DiagDisorderConfig()
    return MBRSpectrumExperiment(
        record["selected_occupations"], plan.cycles, campaign.modes,
        calibration=calibration, cycle_branches=dict(config.branch_overrides),
        detunings=(-np.asarray(record["onsite_MHz"])).tolist(),
        sync_cycles=campaign.sync_cycles, reps=config.reps,
        notes=f"7-1 diagonal disorder r={record['realization']} seed={record['seed']}")


def analyze_diagonal_disorder(ensemble, plan, config=None, on_error="raise"):
    """The 7-1 analysis (cells 332-334) of an ensemble of planned realizations.

    Manual-Kerr frame at the plan's Kerr, Matrix Pencil with the 7-1 settings
    (the ensemble's defaults), theory from each part. With
    ``config.require_complete_match`` an incomplete theory match raises.
    Returns the ensemble's data.
    """
    config = config or DiagDisorderConfig()
    data = ensemble.analyze(
        phase_frame="manual_kerr",
        manual_kerr_MHz=1e-3 * plan.self_kerr_kHz,
        cycle_branches=dict(config.branch_overrides),
        match_tolerance_bins=config.match_tolerance_bins,
        edge_fraction=config.edge_fraction,
        on_error=on_error,
    )
    if config.require_complete_match:
        incomplete = [r.realization for r in data.realizations
                      if "error" in r or not r.match.complete]
        if incomplete:
            raise RuntimeError(f"incomplete theory-pole match for realizations {incomplete}")
    return data
