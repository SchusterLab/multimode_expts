# -*- coding: utf-8 -*-
"""Gate for MBRDisorderEnsembleExperiment and its numerics (MBR redesign step 7).

docs/qsim/mbr_step7_plan.md, sections 3 and 5. Code organization, not physics:
the numerics are checked on small cases with known answers, the ensemble on
mock-acquired parts and on converted 7-1 data (``from_manifest``,
``analyze``, ``display``, ``save``; shapes and dtypes). The old-vs-new
comparison on the same raw files is the baseline, non-blocking until the
physics audit.

Run:  pixi run python -m pytest tests/test_mbr_disorder_ensemble.py -v
"""
from math import comb

import matplotlib
import numpy as np
import pytest

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.mbr_campaign import mbr_defaults, mock_station, pinned_config_set
from experiments.qsim.mbr_disorder_ensemble import MBRDisorderEnsembleExperiment
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
from fitting.qsim import mbr_disorder
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from fitting.qsim.mbr_spectrum import analyze_spectrum
from slab import AttrDict
from tests.mbr_reference import (
    DISORDER_PREVIEW_ANALYSIS,
    converted_diagonal_disorder,
    disorder_dataset,
    migration_tool,
)

matplotlib.use("Agg")

SWAP_STORS = [1, 2, 3, 4]


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


def test_gap_ratios_of_known_levels():
    ratios = mbr_disorder.adjacent_gap_ratios([0., 1., 3., 4., 8.], trim_count=0)
    np.testing.assert_allclose(ratios, [0.5, 0.5, 0.25])
    np.testing.assert_allclose(
        mbr_disorder.adjacent_gap_ratios([-9., 0., 1., 3., 4., 8., 20.], trim_count=1),
        [0.5, 0.5, 0.25])
    with pytest.raises(ValueError, match="duplicate"):
        mbr_disorder.adjacent_gap_ratios([0., 1., 1., 2.], trim_count=0)
    with pytest.raises(ValueError, match="three"):
        mbr_disorder.adjacent_gap_ratios([0., 1.], trim_count=0)


def test_reference_means_are_the_densities_means():
    ratio = np.linspace(0., 1., 200001)
    for pdf, mean in ((mbr_disorder.poisson_pdf, mbr_disorder.POISSON_MEAN),
                      (mbr_disorder.goe_pdf, mbr_disorder.GOE_MEAN)):
        assert np.trapz(pdf(ratio), ratio) == pytest.approx(1., abs=1e-4)
        assert np.trapz(ratio * pdf(ratio), ratio) == pytest.approx(mean, abs=1e-4)


def test_match_levels_is_one_to_one_within_tolerance():
    match = mbr_disorder.match_levels([0.0, 1.02, 5.0], [0.0, 1.0, 2.0], tolerance_MHz=0.1)
    assert match.matched_count == 2 and not match.complete
    np.testing.assert_array_equal(match.missing_theory_MHz, [2.0])
    np.testing.assert_array_equal(match.spurious_poles_MHz, [5.0])
    assert match.mae_MHz == pytest.approx(0.01)
    assert mbr_disorder.match_levels([0., 1.], [0., 1.], 0.1).complete


def test_disorder_direction_is_zero_mean_unit_and_seeded():
    direction = mbr_disorder.disorder_direction(20260816, 4)
    assert abs(direction.sum()) < 1e-12 and np.linalg.norm(direction) == pytest.approx(1.)
    np.testing.assert_array_equal(direction, mbr_disorder.disorder_direction(20260816, 4))


def test_row_selection_covers_every_level():
    weights = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9], [0.4, 0.4, 0.2]])
    rows, floor, coverage = mbr_disorder.select_diagonal_rows(weights, 2)
    assert len(rows) == 2 and floor > 0 and np.all(coverage >= floor)


def test_cycle_grid_respects_nyquist():
    grid = mbr_disorder.cycle_grid(0.1, 0.4, max_cycle=200, min_time_points=100,
                                   nyquist_margin=1.35)
    assert grid.nyquist_MHz >= 1.35 * 0.1
    assert len(grid.cycles) >= 100 and grid.cycles[0] == 0
    with pytest.raises(RuntimeError, match="Nyquist"):
        mbr_disorder.cycle_grid(10., 0.4, max_cycle=200, min_time_points=100,
                                nyquist_margin=1.35)


def test_form_factor_of_a_known_spectrum():
    levels = np.array([0.0, 0.3, 0.7])
    time = np.arange(5) * 0.5
    phases = np.exp(-2j * np.pi * np.outer(levels, time))
    returns = np.stack([phases, phases])          # 2 realizations, diagonal basis
    result = mbr_disorder.trace_and_form_factor(returns)
    np.testing.assert_allclose(result.trace[0], phases.sum(axis=0))
    assert result.sff_normalized[0] == pytest.approx(1.)


def test_hamiltonian_matches_analyze_spectrum():
    probe = AttrDict(dict(occupations=[[3, 0, 0, 0, 0]], cycles=np.array([0, 1]),
                          A=np.ones((1, 2), dtype=complex)))
    args = (3, [0.01, -0.02, 0.005, 0.005], [0.03] * 4, 0.4, -0.04)
    spectrum = analyze_spectrum(probe, *args)
    hamiltonian = fixed_n_hamiltonian(3, 5, *args[1:3], args[4])
    np.testing.assert_array_equal(hamiltonian.energies_MHz, spectrum.energies_MHz)
    assert len(hamiltonian.fock_basis) == comb(3 + 4, 3)


# --------------------------------------------------------------------------
# Assembled class, mock mode
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_parts():
    """Two mock-acquired Spectrum parts with different detunings."""
    st = mock_station(**pinned_config_set("preload_current"))
    assert st.is_mock, "refusing to acquire against real instruments"
    runner = CharacterizationRunner(station=st, ExptClass=MBRTimeTraceExperiment,
                                    default_expt_cfg=mbr_defaults(SWAP_STORS, reps=10),
                                    show=False)
    parts = []
    for detunings in ([0.01, -0.01, 0.02, -0.02], [-0.03, 0.01, 0.01, 0.01]):
        spectrum = MBRSpectrumExperiment([[1, 0, 0, 0, 2], [0, 1, 0, 0, 2]], [0, 2, 4],
                                         SWAP_STORS, detunings=detunings, reps=10)
        spectrum.acquire(runner, batch_size=2)
        parts.append(spectrum)
    return parts


def test_parts_record_their_detunings(mock_parts):
    for part in mock_parts:
        for child in part.children:
            assert list(child.cfg.expt.detunings) == part.detunings


def test_from_parts_checks_the_records(mock_parts):
    records = [dict(realization=0, onsite_MHz=[-d for d in mock_parts[0].detunings]),
               dict(realization=1)]
    ensemble = MBRDisorderEnsembleExperiment.from_parts(mock_parts, realizations=records)
    assert len(ensemble.children) == 2
    assert ensemble.job_ids == mock_parts[0].job_ids + mock_parts[1].job_ids

    with pytest.raises(ValueError, match="onsite_MHz"):
        MBRDisorderEnsembleExperiment.from_parts(
            mock_parts, realizations=[dict(realization=0, onsite_MHz=[0.] * 4),
                                      dict(realization=1)])
    with pytest.raises(ValueError, match="unique"):
        MBRDisorderEnsembleExperiment.from_parts(
            mock_parts, realizations=[dict(realization=0), dict(realization=0)])
    with pytest.raises(ValueError, match="realization records"):
        MBRDisorderEnsembleExperiment.from_parts(mock_parts, realizations=[dict(realization=0)])
    with pytest.raises(NotImplementedError, match="from_parts"):
        ensemble.acquire(None)


# --------------------------------------------------------------------------
# Assembled class, on converted 7-1 data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diagonal_disorder(tmp_path_factory):
    return converted_diagonal_disorder(tmp_path_factory.mktemp("diag71"))


def test_converted_realizations_are_spectra(diagonal_disorder):
    assert len(diagonal_disorder.children) == 2
    assert diagonal_disorder.calibration is not None
    for record, part in zip(diagonal_disorder.realizations, diagonal_disorder.children):
        assert isinstance(part, MBRSpectrumExperiment)
        assert len(part.children) == 10
        assert record["source_keys"] == "diagonal_disorder_*"
        np.testing.assert_allclose(record["onsite_MHz"], -np.asarray(part.detunings))


def test_analyze_display_save(diagonal_disorder, tmp_path):
    data = diagonal_disorder.analyze()
    assert data.dimension == 35 and data.trim_count == 4
    for record in data.realizations:
        assert record.poles_MHz.dtype == float
        assert record.theory_levels_MHz.shape == (35,)
    assert data.form_factor is None, "10 of 35 occupations is not a complete basis"
    assert diagonal_disorder.display() is not None

    arrays = diagonal_disorder.assembled_arrays()
    assert arrays["theory_levels_MHz"].shape == (2, 35)
    assert arrays["detunings"].shape == (2, 4)
    manifest = diagonal_disorder.save(directory=tmp_path)
    reloaded = MBRDisorderEnsembleExperiment.from_manifest(manifest)
    assert reloaded.realizations == diagonal_disorder.realizations
    assert reloaded.calibration is not None
    np.testing.assert_array_equal(reloaded.analyze().realizations[0].poles_MHz,
                                  data.realizations[0].poles_MHz)


def test_migration_refuses_off_diagonal_pair_jobs(tmp_path):
    _, by_realization = disorder_dataset("d72_Sep05_K3p6_g30")
    # The Sep05 files carry no Floquet timing; the archive notes give pi_frac 40.
    timing = dict(floquet_cycle_us=92 / 430.08, m1s_pi_fracs=[40] * 4)
    with pytest.raises(NotImplementedError, match="off-diagonal"):
        migration_tool().migrate_spectrum(by_realization[0][:2], out_root=tmp_path,
                                          load_shots=False, timing=timing)


@pytest.mark.xfail(strict=False, reason="physics audit pending")
def test_baseline_old_preview_analysis_matches(diagonal_disorder):
    """The 7-1 preview cells on the old raw files give the same levels as the ensemble."""
    from experiments.qsim.deprecated import mbr_disorder_preview as preview

    _, by_realization = disorder_dataset("diagonal_disorder_71")
    old, _ = preview.analyze_every_realization({r: by_realization[r] for r in (0, 1)})
    new = diagonal_disorder.analyze(**DISORDER_PREVIEW_ANALYSIS)
    for record in new.realizations:
        np.testing.assert_allclose(record.poles_MHz, np.sort(old[record.realization]["poles_MHz"]),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(record.theory_levels_MHz,
                                   old[record.realization]["theory_levels_MHz"],
                                   rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# 7-1 planning helpers
# --------------------------------------------------------------------------


def test_plan_matches_the_old_selection_code(diagonal_disorder):
    """The planner picks what the old cell-325 code picks on the same inputs."""
    from experiments.qsim.deprecated.mbr_disorder_campaign import _diag_disorder_select_rows
    from experiments.qsim.notebook_helpers.mbr_disorder_campaign import (
        DiagDisorderConfig, plan_diagonal_disorder)

    record = diagonal_disorder.realizations[0]
    config = DiagDisorderConfig(realization_count=2, self_kerr_kHz=record["self_kerr_kHz"])
    calibration = diagonal_disorder.calibration
    plan = plan_diagonal_disorder(calibration, SWAP_STORS, config)
    hardware = calibration.data.hardware
    for new in plan.realizations:
        probe = AttrDict(dict(occupations=[[3, 0, 0, 0, 0]], cycles=np.array([0, 1]),
                              A=np.ones((1, 2), dtype=complex)))
        theory = analyze_spectrum(probe, 3, -np.asarray(new["onsite_MHz"]),
                                  hardware.couplings_MHz, hardware.floquet_cycle_us,
                                  1e-3 * record["self_kerr_kHz"])
        rows, _, _ = _diag_disorder_select_rows(theory.basis_eigenstate_weights, 10)
        assert sorted(tuple(theory.fock_basis[r]) for r in rows) == sorted(
            map(tuple, new["selected_occupations"]))
    np.testing.assert_allclose(plan.realizations[0]["onsite_MHz"], record["onsite_MHz"])


def test_realization_spectrum_plays_the_detunings(diagonal_disorder):
    from experiments.qsim.notebook_helpers.mbr_campaign import MBRCampaign
    from experiments.qsim.notebook_helpers.mbr_disorder_campaign import (
        DiagDisorderConfig, plan_diagonal_disorder, realization_spectrum)

    config = DiagDisorderConfig(realization_count=1)
    calibration = diagonal_disorder.calibration
    plan = plan_diagonal_disorder(calibration, SWAP_STORS, config)
    campaign = MBRCampaign(modes=SWAP_STORS, mode_labels=["M1", "S1", "S2", "S3", "S4"],
                           sync_cycles=10, defaults={})
    record = plan.realizations[0]
    spectrum = realization_spectrum(plan, record, calibration, campaign, config)
    overrides = spectrum.job_overrides()
    assert len(overrides) == config.selected_states
    for override in overrides:
        np.testing.assert_allclose(override["detunings"], -np.asarray(record["onsite_MHz"]))
        assert override["calibration_manifest"] == str(calibration.manifest_path)
