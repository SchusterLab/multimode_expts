"""Matrix Pencil on saved real data: the outputs callers use, pinned.

Function level, so it runs in milliseconds and needs no data mount. The inputs
are two saved reconstructions (the Aug-15 quick-plot set, 4 rows x 200 points,
and the August N=3 complete basis, 35 rows x 150 points), copied out of the
golden baselines. The expected outputs were made by the Matrix Pencil code at
commit f57c202, before its settings were rewritten, for each setting variant
in ``VARIANTS``. The rewrite had to reproduce them to the last bits.

To re-bless after a deliberate change of results:
    MBR_GOLDEN_BLESS=1 pixi run pytest tests/test_matrix_pencil_regression.py
"""
import os
from pathlib import Path

import numpy as np
import pytest

from slab import AttrDict

from fitting.qsim import matrix_pencil
from fitting.qsim.mbr_spectrum import local_spectrum

DATA = Path(__file__).parent / "data" / "matrix_pencil_regression.npz"
DATASETS = ("quickplot", "complete_basis")

# The setting variants the callers use: the defaults, the disorder ensemble's
# defaults, the 7-1 preview's tight tolerances, and the calibration merge.
VARIANTS = {
    "default": dict(),
    "disorder": dict(match_decay=False, track_frequency_tolerance_bins=1.0,
                     merge_frequency_tolerance_bins=0.10,
                     dedup_frequency_tolerance_bins=0.10),
    "tight": dict(track_frequency_tolerance_bins=0.50,
                  merge_frequency_tolerance_bins=0.50,
                  dedup_frequency_tolerance_bins=0.50),
    "calibration": dict(merge_frequency_tolerance_sigma=3.0,
                        merge_frequency_tolerance_floor_MHz=1e-4),
}


def calibration_errors_MHz(n_rows):
    """Made-up per-row calibration standard errors for the calibration merge."""
    return 1e-4 * (1. + np.arange(n_rows) % 3)


def inputs(data, name):
    reconstruction = AttrDict(dict(
        A=data[f"{name}.A"],
        occupations=[tuple(o) for o in data[f"{name}.occupations"]],
        final_occupations=[tuple(o) for o in data[f"{name}.final_occupations"]]))
    spectrum = AttrDict(dict(
        time_us=data[f"{name}.time_us"],
        fft_window="raw", zero_padding=1,
        fft_normalization=data[f"{name}.fft_normalization"]))
    return reconstruction, spectrum, int(data[f"{name}.max_modes"])


def run(reconstruction, spectrum, max_modes, variant):
    options = dict(VARIANTS[variant])
    errors = None
    if variant == "calibration":
        errors = calibration_errors_MHz(len(reconstruction.A))
    settings = matrix_pencil.MatrixPencilSettings(requested_max_modes=max_modes, **options)
    result = matrix_pencil.analyze_matrix_pencil(
        reconstruction, spectrum.time_us, settings,
        row_frequency_standard_errors_MHz=errors)
    return result


def outputs(result, reconstruction, spectrum):
    """-> the pinned numbers, as a flat dict of arrays."""
    modes, fit = result.modes, result.fit
    per_row = result.candidates.per_row
    out = {
        "frequencies_MHz": modes.frequencies_MHz,
        "decay_per_us": modes.decay_per_us,
        "DOS_weights": modes.DOS_weights,
        "local_complex_amplitudes": modes.local_complex_amplitudes,
        "supporting_row_counts": modes.supporting_row_counts,
        "frequency_standard_errors_MHz": modes.frequency_standard_errors_MHz,
        "amplitudes": fit.amplitudes,
        "fitted_return": fit.fitted_return,
        "relative_residual_by_row": fit.relative_residual_by_row,
        "reconstructed_local": local_spectrum(fit.fitted_return, spectrum),
        "row_normalization": result.row_normalization,
        "estimated_signal_rank": np.array([d.estimated_signal_rank for d in result.row_diagnostics]),
        "per_row.row_index": np.array([c.row_index for c in per_row]),
        "per_row.frequency_MHz": np.array([c.frequency_MHz for c in per_row]),
        "per_row.decay_per_us": np.array([c.decay_per_us for c in per_row]),
        "per_row.rank_span": np.array([c.rank_span for c in per_row]),
        "per_row.confidence": np.array([c.confidence for c in per_row]),
    }
    for row in (0, len(reconstruction.A) // 2):
        refit = matrix_pencil.refit_row(result, reconstruction.A[row], row)
        out[f"refit{row}.frequencies_MHz"] = refit.frequencies_MHz
        out[f"refit{row}.normalized_amplitudes"] = refit.normalized_amplitudes
        out[f"refit{row}.fitted_return"] = refit.fitted_return
    return out


def _blessing():
    return os.environ.get("MBR_GOLDEN_BLESS", "").strip() not in ("", "0", "false")


@pytest.fixture(scope="module")
def data():
    with np.load(DATA) as handle:
        return dict(handle)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("name", DATASETS)
def test_matches_saved_outputs(data, name, variant):
    reconstruction, spectrum, max_modes = inputs(data, name)
    got = outputs(run(reconstruction, spectrum, max_modes, variant), reconstruction, spectrum)
    prefix = f"{name}.{variant}."
    if _blessing():
        data.update({prefix + key: np.asarray(value) for key, value in got.items()})
        np.savez_compressed(DATA, **data)
        pytest.skip(f"blessed {prefix}*")
    expected = {key.removeprefix(prefix): value for key, value in data.items()
                if key.startswith(prefix)}
    assert set(got) == set(expected)
    for key, want in expected.items():
        np.testing.assert_allclose(np.asarray(got[key]), want, rtol=1e-12, atol=1e-12,
                                   err_msg=f"{prefix}{key}")
