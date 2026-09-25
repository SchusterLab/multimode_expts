"""Old-class N=3 reprocessing helpers -- DEPRECATED.

Moved from `experiments/qsim/notebook_helpers/mbr_n3_reprocess.py` on 2026-09-24
(MBR redesign step 7a), without changes. See `docs/qsim/mbr_step7_plan.md`.
`reprocess_n3_spectroscopy` takes the old loaded aggregates (legacy
`MBRSpectrumExperiment`); its callers moved to `dormant/`.
`fit_self_kerr_from_peak_overlap` had no caller; the new-class twin is
`mbr_n3_reprocess.fit_self_kerr`.

Not maintained; may break when live code changes. If it breaks, add a note here
and do not fix it.
"""

from experiments.qsim.notebook_helpers.mbr_n3_reprocess import _self_kerr_scan


def reprocess_n3_spectroscopy(calibration_expt, spectroscopy_expt,
                              cycle_branches=None, legacy=True,
                              manual_kerr_MHz=-19.756e-3):
    """Re-analyze the loaded N=3 jobs in the manual-Kerr phase frame (cell 175).

    `cycle_branches` defaults to empty, i.e. branch 0 for every occupation,
    as the source did. Note the source's own warning: these jobs used the old
    `+cycle*correction` analyzer convention, so a branch assignment copied
    from a `Q0 + iQ90` era notebook needs its sign flipped.

    Returns `encspec_reprocessed`.
    """
    encspec_cycle_branches = {} if cycle_branches is None else cycle_branches
    encspec_legacy = legacy
    encspec_manual_kerr_MHz = manual_kerr_MHz

    # Remove the complete analyzer correction used during acquisition. This is the uncorrected return in the current IQ convention.
    # encspec_uncorrected = spectroscopy_expt.analyze(
    #                                                 phase_frame='uncorrected',
    #                                                 cycle_branches=encspec_cycle_branches,
    #                                                 legacy=encspec_legacy,
    #                                                 fft_window='raw',
    #                                                 zero_padding=1,
    #                                                 spectrum_method='mpm')
    # print('saved analyzer correction removed')
    # spectroscopy_expt.display(data=encspec_uncorrected, spectrum_method='mpm')

    # Starting from the same saved data, undo the old correction and apply the calibration again with the selected signed Kerr.
    encspec_reprocessed = spectroscopy_expt.analyze(
                                                    calibration=calibration_expt,
                                                    phase_frame='manual_kerr',
                                                    manual_kerr_MHz=encspec_manual_kerr_MHz,
                                                    cycle_branches=encspec_cycle_branches,
                                                    legacy=encspec_legacy,
                                                    fft_window='raw',
                                                    zero_padding=1,
                                                    spectrum_method='mpm')
    spectroscopy_expt.display(data=encspec_reprocessed, spectrum_method='mpm')

    return encspec_reprocessed


def fit_self_kerr_from_peak_overlap(
        spectroscopy_expt, spectroscopy_data, calibration_expt,
        spectroscopy_occupations, cycle_branches, **scan_options):
    """The self-Kerr scan on the old loaded aggregates (see `_self_kerr_scan`).

    For the disorder campaign until redesign step 7; new code uses
    `fit_self_kerr`.
    """
    def analyze_at(kerr_MHz):
        return spectroscopy_expt.analyze(
            calibration=calibration_expt,
            occupations=spectroscopy_occupations,
            cycle_branches=cycle_branches,
            phase_frame="manual_kerr",
            manual_kerr_MHz=kerr_MHz,
            spectrum_method="fft",
        )

    return _self_kerr_scan(spectroscopy_expt, spectroscopy_data, analyze_at,
                           **scan_options)
