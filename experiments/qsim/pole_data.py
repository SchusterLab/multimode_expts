"""Registry data sets as pole-finding spectra (docs/qsim/pole_finding.md 8.2).

:func:`load_spectra` opens a registry entry's manifest, analyzes each spectrum in the frame
the entry names (``fitting.qsim.poles.registry.Analysis``), and pairs it with its model:
:class:`fitting.qsim.poles.real_benchmarks.RealSpectrum`. Here, not in ``fitting/``, because
it needs the experiment classes.
"""
import numpy as np

from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.poles.real_benchmarks import RealSpectrum
from fitting.qsim.poles.registry import Analysis, manifest_path
from fitting.qsim.poles.synthetic import distinct_levels


def load_spectra(data_set, root):
    """-> the RealSpectra of one registry entry (one per realization of an ensemble)."""
    path = manifest_path(data_set, root)
    analysis = data_set.analysis or Analysis()
    if "DisorderEnsemble" in path.name:
        ensemble = MBRDisorderEnsembleExperiment.from_manifest(path)
        parts = [(f"{data_set.label}/{record['realization']}", part, 1e-3 * float(record["self_kerr_kHz"]))
                 for record, part in zip(ensemble.realizations, ensemble.children)]
    else:
        parts = [(data_set.label, MBRSpectrumExperiment.from_manifest(path), None)]
    return [spectrum(label, data_set, part, analysis, recorded_kerr_MHz) for label, part, recorded_kerr_MHz in parts]


def spectrum(label, data_set, part, analysis, recorded_kerr_MHz):
    """-> one part analyzed in the entry's frame, with its model's distinct levels."""
    excluded = set(analysis.excluded_occupations)
    if excluded:
        kept = [c for c in part.children if c.initial_occupation not in excluded]
        part = MBRSpectrumExperiment.from_children(kept, calibration=part.calibration, notes=part.notes)
    manual_kerr_MHz = recorded_kerr_MHz if analysis.manual_kerr_MHz == "recorded" else analysis.manual_kerr_MHz
    data = part.analyze(phase_frame=analysis.phase_frame, manual_kerr_MHz=manual_kerr_MHz,
                        cycle_branches=analysis.branches(part.occupations), legacy=analysis.legacy or None,
                        spectrum_method="fft")
    kerr_MHz = recorded_kerr_MHz if analysis.model_kerr == "recorded" else data.spectrum.physical_kerr_MHz
    occupations = [tuple(o) for o in data.reconstruction.occupations]
    mode_count = len(occupations[0])
    model = fixed_n_hamiltonian(data_set.photon_number, mode_count, data.detunings, data.hardware.couplings_MHz, kerr_MHz)
    time_us = np.asarray(data.spectrum.time_us, dtype=float)
    rows = [model.fock_index[o] for o in occupations]
    levels, multiplicities, row_weights = distinct_levels(
        model.energies_MHz, model.basis_eigenstate_weights[rows], 1e-3 / (len(time_us) * (time_us[1] - time_us[0])))
    return RealSpectrum(label, data_set.label, data_set.basis == "complete", time_us,
                        np.asarray(data.reconstruction.A), levels, multiplicities, row_weights)
