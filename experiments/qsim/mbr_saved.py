# -*- coding: utf-8 -*-
"""Parameters read back from saved MBR jobs: swap modes, detunings, hardware.

Plain functions, shared by the new MBR classes and the old
``EncodingHamiltonianSpectroscopyExperiment``, which used to own them as
methods (docs/qsim/mbr_redesign.md, section 2). Moved without changes to the
arithmetic.

The Floquet timing comes from each job's ``prog``: the live compiled program
during acquisition, or the stand-in that :mod:`experiments.saved_jobs`
attaches to saved data, which carries timing recovered from the file's
``derived_params`` attribute or from the versioned config. There is no
station fallback, deliberately -- see :func:`saved_parameters`.
"""
import numpy as np
from slab import AttrDict


def first_scalar(value):
    """-> the first entry of a scalar that is saved sometimes as a list.

    The self-Kerr is stored as a list in some configs and as a float in
    others.
    """
    values = np.asarray(value).reshape(-1)
    if len(values) == 0:
        raise ValueError("saved scalar config is empty")
    return float(values[0])


def saved_detunings(ecfg, mode_count):
    """-> the saved detunings as an array, zeros if none were saved.

    Detunings are stored sometimes as a list, sometimes as an array, and
    sometimes not at all.
    """
    detunings = ecfg.get("detunings", None)
    if detunings is None or detunings is False or np.asarray(detunings).size == 0:
        detunings = [0.] * mode_count
    return np.asarray(detunings, dtype=float)


def saved_parameters(expts):
    """Swap modes, detunings, mode labels and hardware of sister jobs.

    Reads the first job's config, and the Floquet timing from the first job
    whose ``prog`` has one. Returns an AttrDict with

        - ``swap_stors``
        - ``detunings``
        - ``mode_labels``
        - ``hardware``: ``floquet_cycle_us``, ``couplings_MHz``,
          ``physical_kerr_MHz`` and ``source`` (where the timing came from)
    """
    first_cfg = expts[0].cfg
    first_expt_cfg = first_cfg.expt
    swap_stors = [int(stor) for stor in first_expt_cfg.swap_stors]
    detunings = saved_detunings(first_expt_cfg, len(swap_stors))
    physical_kerr_MHz = -abs(first_scalar(first_cfg.device.manipulate.kerr))
    if len(detunings) != len(swap_stors) or not np.all(np.isfinite(detunings)):
        raise ValueError("saved detunings do not match swap_stors")

    program_hardware = []
    for expt in expts:
        prog = getattr(expt, "prog", None)
        if prog is not None and hasattr(prog, "calculate_floquet_cycle_us") and hasattr(prog, "m1s_pi_fracs"):
            floquet_cycle_us = float(prog.calculate_floquet_cycle_us())
            pi_fracs = np.asarray([prog.m1s_pi_fracs[stor - 1] for stor in swap_stors], dtype=float)
            couplings_MHz = 1. / (4. * pi_fracs * floquet_cycle_us)
            # `source` says where the timing came from: a live compiled
            # program during acquisition, or one of the recovered sources
            # that experiments.saved_jobs resolves offline.
            program_hardware.append((floquet_cycle_us, couplings_MHz,
                                     getattr(prog, "source", "saved program")))
            break

    if program_hardware:
        floquet_cycle_us, couplings_MHz, hardware_source = program_hardware[0]
    else:
        # Deliberately no station fallback. Asking the *current* station
        # substitutes today's calibration for the historical one, and does
        # it silently: when the swap dataset moved gauss_sigma 0.04 -> 0.02
        # us between 2026-08-14 and 08-25, that fallback returned roughly
        # half the correct cycle time and every energy with it. The cycle
        # time is not a measurement -- it is computed from immutable
        # versioned config, so it is recovered exactly or not at all.
        raise RuntimeError(
            "no Floquet timing on these children. Load them through "
            "experiments.saved_jobs (from_job_ids / from_job_files), which "
            "reads the file's own 'derived_params' attribute or recomputes "
            "the timing from the versioned config, and takes timing= for "
            "files that have neither.")
    if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0. or not np.all(np.isfinite(couplings_MHz)) or np.min(couplings_MHz) <= 0.:
        raise ValueError("saved Floquet hardware parameters must be finite and positive")
    hardware = AttrDict(dict(floquet_cycle_us=float(floquet_cycle_us), couplings_MHz=np.asarray(couplings_MHz), physical_kerr_MHz=physical_kerr_MHz, source=hardware_source))
    return AttrDict(dict(swap_stors=swap_stors,
                         detunings=detunings,
                         mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors],
                         hardware=hardware))
