"""Old-class disorder loaders of mbr_saved_reanalysis -- DEPRECATED.

Moved from `experiments/qsim/notebook_helpers/mbr_saved_reanalysis.py` on
2026-09-24 (MBR redesign step 7c), without changes: `occupation_pairs`,
`load_disorder_calibrated` and `load_disorder_as_acquired`, which load the
August disorder realizations as old aggregates from job IDs. The live
versions of the two loaders take the converted `MBRDisorderEnsembleExperiment`
manifest; see `docs/qsim/mbr_step7_plan.md`.

Not maintained; may break when live code changes. If it breaks, add a note here
and do not fix it.
"""
import numpy as np

from slab import AttrDict

from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.deprecated.legacy_mbr import (
    MBRSpectrumExperiment as LegacySpectrumExperiment,
)
from experiments.saved_jobs import load_aggregate


def occupation_pairs(expt, label):
    """Group a batch's children by occupation and check both analyzer phases.

    Cells 222 and 232 defined this twice with identical bodies, under
    the names `_saved_occupation_pairs` and `_offline_occupation_pairs`.
    """
    grouped = {}
    for child in expt.batch_expts:
        cfg = child.cfg.expt
        occupation = tuple(map(int, cfg.spectroscopy_occupations))
        grouped.setdefault(occupation, []).append(
            float(cfg.spectroscopy_analyzer_phase)
        )
    for occupation, phases in grouped.items():
        if len(phases) != 2 or not np.allclose(sorted(phases), [0.0, 90.0]):
            raise RuntimeError(
                f"{label}: {occupation} has analyzer phases {phases}, expected 0/90"
            )
    return list(grouped)


def load_disorder_calibrated(saved_n3_calibration_job_ids,
                             saved_disorder_job_ids,
                             saved_n3_cycle_branches,
                             saved_fft_window="raw",
                             saved_zero_padding=1,
                             timing=None):
    """The disorder realizations *with* the N=3 phase calibration (cell 222).

    Old loaded aggregates, from job IDs, until redesign step 7: the
    calibration is loaded again here as the old `MBRPhaseCorrectionExperiment`,
    which the old `analyze(calibration=...)` needs. The body is the disorder
    half of the former `load_saved_calibrated`, unchanged.

    Returns `saved_disorder_records`.
    """
    saved_n3_calibration_expt = load_aggregate(
        saved_n3_calibration_job_ids,
        owner=MBRPhaseCorrectionExperiment,
        timing=timing,
        analyze=True,
    )

    saved_disorder_records = {}
    for realization, job_ids in sorted(saved_disorder_job_ids.items()):
        expt = load_aggregate(job_ids, owner=LegacySpectrumExperiment, timing=timing)
        cfg0 = expt.batch_expts[0].cfg.expt
        saved_realizations = {
            int(child.cfg.expt.disorder_realization)
            for child in expt.batch_expts
        }
        if saved_realizations != {int(realization)}:
            raise RuntimeError(
                f"manifest r={realization} contains saved realizations {saved_realizations}"
            )

        occupations = [
            tuple(map(int, occupation)) for occupation in cfg0.selected_occupations
        ]
        paired_occupations = occupation_pairs(expt, f"disorder r={realization}")
        if set(paired_occupations) != set(occupations):
            raise RuntimeError(f"disorder r={realization}: selected occupations differ")

        manual_kerr_MHz = float(cfg0.target_manual_kerr_MHz)
        cycle_branches = {
            occupation: saved_n3_cycle_branches.get(occupation, 0)
            for occupation in occupations
        }
        data = expt.analyze(
            calibration=saved_n3_calibration_expt,
            occupations=occupations,
            cycle_branches=cycle_branches,
            phase_frame="manual_kerr",
            manual_kerr_MHz=manual_kerr_MHz,
            fft_window=saved_fft_window,
            zero_padding=saved_zero_padding,
            spectrum_method="fft",
        )

        saved_theory_energies_MHz = np.asarray(cfg0.theory_energies_MHz, dtype=float)
        np.testing.assert_allclose(
            data.spectrum.energies_MHz,
            saved_theory_energies_MHz,
            rtol=0.0,
            atol=1e-10,
            err_msg=f"disorder r={realization}: saved theory and rebuilt theory differ",
        )
        plan = AttrDict(dict(
            realization=int(cfg0.disorder_realization),
            seed=int(cfg0.disorder_seed),
            strength_kHz=float(cfg0.disorder_strength_kHz),
            direction=np.asarray(cfg0.disorder_direction, dtype=float),
            target_onsite_MHz=np.asarray(
                cfg0.disorder_target_onsite_MHz, dtype=float
            ),
            pulse_detunings_MHz=np.asarray(
                cfg0.disorder_api_detunings_MHz, dtype=float
            ),
            occupations=[list(occupation) for occupation in occupations],
            theory_energies_MHz=saved_theory_energies_MHz,
            manual_kerr_MHz=manual_kerr_MHz,
        ))
        saved_disorder_records[realization] = AttrDict(dict(
            plan=plan,
            job_ids=list(job_ids),
            expt=expt,
            data=data,
        ))

    for realization, record in saved_disorder_records.items():
        print(
            f"disorder r={realization}: {len(record.job_ids)} jobs, "
            f"{len(record.data.reconstruction.occupations)} occupations; "
            f"K={1e3 * record.data.spectrum.physical_kerr_MHz:.4f} kHz"
        )
    return saved_disorder_records


def load_disorder_as_acquired(saved_disorder_job_ids,
                              offline_fft_window="raw",
                              offline_zero_padding=1,
                              timing=None):
    """The disorder realizations in the as-acquired frame (cell 232).

    Old loaded aggregates, from job IDs, until redesign step 7. The body is
    the disorder half of the former `load_saved_as_acquired`, unchanged.

    Returns `saved_disorder_records`.
    """
    # Disorder spectroscopy: ten selected occupations x analyzer phases 0/90.
    saved_disorder_records = {}
    for realization, job_ids in sorted(saved_disorder_job_ids.items()):
        expt = load_aggregate(job_ids, owner=LegacySpectrumExperiment, timing=timing)
        cfg0 = expt.batch_expts[0].cfg.expt
        saved_realizations = {
            int(child.cfg.expt.disorder_realization) for child in expt.batch_expts
        }
        if saved_realizations != {realization}:
            raise RuntimeError(
                f"manifest r={realization} contains saved realizations "
                f"{saved_realizations}"
            )
        occupations = [
            tuple(map(int, occupation)) for occupation in cfg0.selected_occupations
        ]
        paired = occupation_pairs(expt, f"disorder r={realization}")
        if set(paired) != set(occupations):
            raise RuntimeError(f"disorder r={realization}: occupations differ")
        data = expt.analyze(
            occupations=occupations,
            phase_frame="as_acquired",
            fft_window=offline_fft_window,
            zero_padding=offline_zero_padding,
            spectrum_method="fft",
        )
        saved_disorder_records[realization] = AttrDict(
            dict(expt=expt, data=data, cfg=cfg0)
        )

    print(
        "Disorder, as acquired: "
        + ", ".join(
            f"r={realization} ({len(record.expt.batch_job_ids)} H5 jobs)"
            for realization, record in saved_disorder_records.items()
        )
    )
    return saved_disorder_records
