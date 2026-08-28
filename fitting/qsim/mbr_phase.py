"""Analyzer-phase mathematics for many-body Ramsey reconstructions.

Extracted verbatim from ``EncodingHamiltonianSpectroscopyExperiment``
(spec section 7.5, ``mbr_phase.py``): phase unwrapping, correction
construction, and the phase-frame conventions that go with them. Bodies are
unchanged; the class keeps thin wrappers under the historical names.

Sign and branch conventions, stated once because every function here depends
on them:

- Phases are in **degrees** throughout, per Floquet cycle where noted.
- The measured closed-cycle phase is only determined **modulo 180 deg/cycle**,
  because closed pairs sit two physical cycles apart. A *branch* picks one
  representative; see :func:`cycle_branches` (spec section 5).
- :func:`build_phase_correction` returns the phase the pulse program
  **subtracts** from the final analyzer, with the always-on M1 self-Kerr
  contribution removed.

:func:`saved_correction` is the odd one out: it reads ``expt.cfg.expt`` rather
than taking arrays, so by the spec's own reusability test (section 7.5) it is a
config reader, not mathematics. It moved with the rest because it is the
inverse of :func:`build_phase_correction` -- it recovers the correction that
was applied at pulse time -- and splitting the pair across two files would hide
that. Narrowing its signature to plain config mappings is a follow-up.
"""

import numpy as np

from slab import AttrDict


def cycle_branches(occupations, cycle_branches=0):
    """
    Specfy a branch for an accumulated phase for a encoding pulse per a floquet cycle
    default is 0, and possible inputs are
        - int: fix branch to int (ideally 0 or 1) for all encoding pulses
        - list: select branch per occupation; should match the length of occupation
        - dictionary: select branch for a specified occupation.

    Example:
    1. cycle_branches = 1
    2. occupations = [
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
        ]
        cycle_branches = [0, 1, 0]
    3. cycle_branches = {
            (2, 0, 0): 0,
            (1, 1, 0): 1,
            (1, 0, 1): 0,
        }

    Recommendation is 3
    """

    if isinstance(cycle_branches, dict):
        branches = np.asarray([cycle_branches.get(tuple(occupation), 0) for occupation in occupations], dtype=float)
    elif np.isscalar(cycle_branches):
        branches = np.full(len(occupations), cycle_branches, dtype=float)
    else:
        branches = np.asarray(cycle_branches, dtype=float)
    if branches.shape != (len(occupations),) or not np.all(np.isfinite(branches)) or not np.allclose(branches, np.round(branches)):
        raise ValueError("cycle_branches must give one integer branch per occupation")
    return branches.astype(int)

def build_phase_correction(occupations, 
                           phase_mod180, 
                           cycle_branches, 
                           physical_kerr_MHz,
                           floquet_cycle_us,
                           correction_sign=1.):
    """
    Compute the phase correction that the pulse program subtracts from the
    final analyzer, excluding the always-on M1 self-Kerr phase.
    """
    phase_mod180 = np.asarray(phase_mod180)
    cycle_branches = np.asarray(cycle_branches)
    if correction_sign not in (-1., 1.):
        raise ValueError("correction_sign must be +1 or -1")
    if phase_mod180.shape != (len(occupations),) or cycle_branches.shape != (len(occupations),):
        raise ValueError("phase_mod180 and cycle_branches must match occupations")
    physical_kerr_MHz = float(physical_kerr_MHz)
    if not np.isfinite(physical_kerr_MHz):
        raise ValueError("physical_kerr_MHz must be finite")
    n_M1 = np.asarray([occupation[0] for occupation in occupations])
    kerr_energy_MHz = 0.5 * physical_kerr_MHz * n_M1 * (n_M1 - 1)
    kerr_phase = -360. * kerr_energy_MHz * floquet_cycle_us
    measured_phase = phase_mod180 + 180. * cycle_branches
    analyzer_phase = correction_sign * (measured_phase - kerr_phase) 

    phase_by_occupation = {}
    for occupation, phase in zip(occupations, analyzer_phase):
        phase_by_occupation[tuple(occupation)] = float(phase)
    return AttrDict(dict(
        measured_phase=measured_phase, 
        kerr_phase=kerr_phase,
        cycle_branches=cycle_branches,
        physical_kerr_MHz=physical_kerr_MHz,
        phase_by_occupation=phase_by_occupation,
    ))

def unwrap_cycle_phase(complex_return, 
                        physical_cycles, 
                        valid_mask, 
                        closed_mask, 
                        guide_weight=0.2):
    raw_phase = np.rad2deg(np.angle(complex_return))
    fit_mask = valid_mask & closed_mask
    guide_mask = valid_mask & ~closed_mask

    # Closed-pair points are two physical cycles apart, so their slope is unique modulo 180 deg/cycle.
    slopes = np.linspace(-90., 90., 18001)
    fit_return = complex_return[fit_mask] / np.abs(complex_return[fit_mask])
    rotation = np.exp(-1j * np.deg2rad(np.outer(slopes, physical_cycles[fit_mask])))
    score = np.abs(rotation @ fit_return) / len(fit_return)

    # Odd points may have a different offset; only their internal phase progression is used as a weak guide.
    if np.count_nonzero(guide_mask) > 1:
        guide_return = complex_return[guide_mask] / np.abs(complex_return[guide_mask])
        rotation = np.exp(-1j * np.deg2rad(np.outer(slopes, physical_cycles[guide_mask])))
        score += guide_weight * np.abs(rotation @ guide_return) / len(guide_return)

    slope = slopes[np.argmax(score)]
    phase = np.full(len(complex_return), np.nan)
    for mask in [fit_mask, guide_mask]:
        if not np.any(mask):
            continue
        unit_return = complex_return[mask] / np.abs(complex_return[mask])
        rotated_return = unit_return * np.exp(-1j * np.deg2rad(slope * physical_cycles[mask]))
        intercept = np.rad2deg(np.angle(np.sum(rotated_return)))
        phase_model = intercept + slope * physical_cycles[mask]
        phase[mask] = raw_phase[mask] + 360. * np.round((phase_model - raw_phase[mask]) / 360.)
    return phase

def saved_correction(expts):
    """
    Reconstruct `phase_by_occupation` from the stored experiments.
    For now, `decoder` is the only supported

    """

    phase_by_occupation = {}
    modes = set()
    application_signs = set()
    missing_application_sign = False
    for expt in expts:
        ecfg = expt.cfg.expt
        occupation = tuple(ecfg.get(
            "offdiag_decoder_occupation",
            ecfg.get("spectroscopy_final_occupations", ecfg.spectroscopy_occupations),
        ))
        phase = float(ecfg.get(
            "offdiag_decoder_phase_correction_deg",
            ecfg.get("final_analyzer_phase_per_cycle_deg", 0.),
        ))
        if occupation in phase_by_occupation and not np.isclose(phase, phase_by_occupation[occupation]):
            raise ValueError(f"{occupation} spectroscopy chunks used different analyzer corrections")
        phase_by_occupation[occupation] = phase
        modes.add(str(ecfg.get("spectroscopy_phase_correction_mode", 
                               "decoder")))
        application_sign = ecfg.get("final_analyzer_phase_application_sign", None)
        if application_sign is None:
            missing_application_sign = True
        else:
            application_sign = float(application_sign)
            if application_sign not in (-1., 1.):
                raise ValueError("saved analyzer phase application sign must be +1 or -1")
            application_signs.add(application_sign)
    if len(application_signs) > 1:
        raise ValueError("saved jobs use different analyzer phase application signs")
    if len(modes) != 1:
        raise ValueError("saved jobs use different spectroscopy phase-correction modes")
    nonzero_correction = any(not np.isclose(phase, 0.) for phase in phase_by_occupation.values())
    if nonzero_correction and missing_application_sign and application_signs:
        raise ValueError("saved jobs mix marked and unmarked analyzer phase conventions")
    application_sign = next(iter(application_signs)) if len(application_signs) == 1 and not missing_application_sign else None
    return AttrDict(dict(phase_by_occupation=phase_by_occupation, 
                         modes=modes, 
                         application_sign=application_sign))
