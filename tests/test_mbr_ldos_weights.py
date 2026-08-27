# -*- coding: utf-8 -*-
"""LDOS weights must not depend on the eigenvector gauge.

These are property tests, not baselines. The bug they cover was introduced on
2026-08-24 (``93c1b20``, "added line to measure off diagonal elements"), which
generalized the spectrum from return amplitudes to full propagator elements:

    -   eigenstate_weights = np.abs(states[basis_rows]) ** 2
    +   spectral_weights   = states[final_rows] * states[basis_rows].conj()
    +   eigenstate_weights = np.abs(spectral_weights)

``theory_A`` was correctly rewired to the signed ``spectral_weights``, but
``eigenstate_weights`` was redefined in place from a probability
|<b|E_k>|^2 to a signed amplitude product |<f|E_k><E_k|b>|, while its consumers
-- the two LDOS displays -- kept treating it as the old probability. Nothing
failed, because for diagonal rows the two definitions are numerically
identical, and every dataset acquired up to then was diagonal.

A pinned baseline cannot catch this: the wrong quantity is perfectly
reproducible on one machine. What distinguishes right from wrong is a symmetry
-- the answer must be invariant under re-mixing eigenvectors inside a degenerate
multiplet -- so that is what is asserted here.
"""
import numpy as np
import pytest
from slab import AttrDict

from fitting.qsim.mbr_spectrum import ldos_weights


def _degenerate_hamiltonian(n_modes=5):
    """A star-coupled Hamiltonian: one central mode, n equal couplings.

    Permutation symmetry among the outer modes forces an (n-1)-fold degenerate
    multiplet at zero energy -- the same mechanism that makes the real MBR
    Hamiltonian degenerate when storage modes share a detuning.
    """
    n = n_modes + 1
    H = np.zeros((n, n))
    for i in range(1, n):
        H[0, i] = H[i, 0] = 1.0
    return H


def _gauge_rotate(energies, states, seed, tol=1e-9):
    """Re-mix eigenvectors within each degenerate multiplet.

    The result is still a valid eigendecomposition of the same Hamiltonian, so
    every physical quantity must be unchanged.
    """
    rng = np.random.default_rng(seed)
    rotated = states.copy()
    start = 0
    while start < len(energies):
        stop = start
        while stop + 1 < len(energies) and abs(energies[stop + 1] - energies[start]) < tol:
            stop += 1
        size = stop - start + 1
        if size > 1:
            q, _ = np.linalg.qr(rng.normal(size=(size, size)))
            rotated[:, start:stop + 1] = states[:, start:stop + 1] @ q
        start = stop + 1
    return rotated


def _spectrum(energies, states, rows):
    """Build the minimal spectrum dict that ldos_weights consumes."""
    final_rows = [f for f, _ in rows]
    basis_rows = [b for _, b in rows]
    spectral_weights = states[final_rows] * states[basis_rows].conj()
    return AttrDict(dict(
        energies_MHz=energies,
        spectral_weights=spectral_weights,
        eigenstate_weights=np.abs(spectral_weights),
    ))


def test_the_test_hamiltonian_is_actually_degenerate():
    """Guard the guard: if this stops being degenerate, the tests below are vacuous."""
    energies, _ = np.linalg.eigh(_degenerate_hamiltonian())
    gaps = np.diff(energies)
    assert np.sum(gaps < 1e-9) >= 3, f"expected a multiplet, got energies {energies}"


def test_gauge_rotation_really_is_a_gauge_choice():
    """The rotated eigenvectors must still diagonalize the same Hamiltonian."""
    H = _degenerate_hamiltonian()
    energies, states = np.linalg.eigh(H)
    rotated = _gauge_rotate(energies, states, seed=0)
    assert np.allclose(H, rotated @ np.diag(energies) @ rotated.T)
    # ...and must genuinely differ, or invariance below proves nothing.
    assert not np.allclose(states, rotated)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("rows,label", [
    ([(1, 1), (2, 2)], "diagonal"),
    ([(1, 2), (2, 3)], "off-diagonal"),
    ([(1, 1), (1, 2)], "mixed"),
])
def test_ldos_weights_are_gauge_invariant(rows, label, seed):
    """The physics cannot depend on which basis eigh picked in a multiplet."""
    H = _degenerate_hamiltonian()
    energies, states = np.linalg.eigh(H)
    rotated = _gauge_rotate(energies, states, seed=seed)

    _, weights = ldos_weights(_spectrum(energies, states, rows))
    _, weights_rotated = ldos_weights(_spectrum(energies, rotated, rows))

    deviation = np.max(np.abs(weights - weights_rotated))
    assert deviation < 1e-12, f"{label} rows moved by {deviation:.3e} under a gauge rotation"


def test_the_old_formula_would_have_failed_off_diagonal():
    """Pin the bug itself, so a revert to abs-before-sum is caught.

    The pre-fix code binned ``np.abs(spectral_weights)``; the fix bins the
    signed weights and takes the modulus of the bin. This asserts the two
    genuinely differ off-diagonal -- i.e. that the fix is not cosmetic.
    """
    H = _degenerate_hamiltonian()
    energies, states = np.linalg.eigh(H)
    rotated = _gauge_rotate(energies, states, seed=7)
    rows = [(1, 2)]

    def old_formula(states_):
        spectrum = _spectrum(energies, states_, rows)
        bins, indices = np.unique(np.round(spectrum.energies_MHz, 10), return_inverse=True)
        return np.bincount(indices, weights=spectrum.eigenstate_weights[0], minlength=len(bins))

    drift = np.max(np.abs(old_formula(states) - old_formula(rotated)))
    assert drift > 1e-3, (
        "the old abs-before-sum formula no longer drifts under a gauge rotation; "
        "this test's Hamiltonian may have lost its degeneracy"
    )


def test_diagonal_rows_are_unchanged_by_the_fix():
    """The fix must be a provable no-op on diagonal data, or baselines move.

    For f == b every term is |<b|E_k>|^2 >= 0, so nothing can cancel and
    sum-then-modulus equals modulus-then-sum exactly.
    """
    H = _degenerate_hamiltonian()
    energies, states = np.linalg.eigh(H)
    rows = [(1, 1), (2, 2), (3, 3)]
    spectrum = _spectrum(energies, states, rows)

    bins, indices = np.unique(np.round(energies, 10), return_inverse=True)
    old = np.asarray([
        np.bincount(indices, weights=w, minlength=len(bins))
        for w in spectrum.eigenstate_weights
    ])
    _, new = ldos_weights(spectrum)
    assert np.allclose(old, new, rtol=0, atol=1e-15)


def test_merged_spectra_without_spectral_weights_are_accepted():
    """merge_spectra emits no spectral_weights; its diagonal probabilities work."""
    energies = np.array([1.0, 1.0, 1.0, 2.0])
    spectrum = AttrDict(dict(
        energies_MHz=energies,
        eigenstate_weights=np.array([[0.1, 0.2, 0.3, 0.4]]),
    ))
    bins, weights = ldos_weights(spectrum)
    assert np.allclose(bins, [1.0, 2.0])
    assert np.allclose(weights, [[0.6, 0.4]])


def test_signed_weights_without_spectral_weights_are_refused():
    """Binning moduli is only valid for non-negative weights; say so loudly."""
    spectrum = AttrDict(dict(
        energies_MHz=np.array([1.0, 1.0]),
        eigenstate_weights=np.array([[0.5, -0.5]]),
    ))
    with pytest.raises(ValueError, match="ambiguous"):
        ldos_weights(spectrum)


def _merge_source(occupations, final_occupations=None):
    """The minimum merge_spectra needs to reach its diagonality check."""
    reconstruction = {"occupations": occupations}
    if final_occupations is not None:
        reconstruction["final_occupations"] = final_occupations
    return AttrDict(dict(
        reconstruction=AttrDict(reconstruction),
        spectrum=AttrDict(dict(energies_MHz=np.array([0.0, 1.0]))),
    ))


def test_merge_spectra_refuses_off_diagonal_rows():
    """merge_spectra rebuilds theory from |<b|E_a>|^2, i.e. the return amplitude.

    That is the pre-2026-08-24 diagonal formula, and it is the wrong quantity for
    a row that measures a different occupation than it prepares -- such a row
    needs <f|E_a><E_a|b>. The generalization updated analyze_spectrum and left
    merge_spectra behind, so it must refuse rather than emit a plausible-looking
    curve that is not the propagator element.
    """
    from fitting.qsim.level_statistics import merge_spectra

    sources = [
        _merge_source([(2, 0, 0)], [(2, 0, 0)]),
        _merge_source([(1, 1, 0)], [(0, 1, 1)]),   # off-diagonal
    ]
    with pytest.raises(NotImplementedError, match="diagonal weights only"):
        merge_spectra(sources)


def test_merge_spectra_accepts_diagonal_rows_past_the_guard():
    """The guard must not reject the diagonal case it was built to allow.

    Only that the guard is passed is asserted here: merge_spectra needs far more
    of a real spectrum to complete, and the golden baselines cover that path.
    """
    from fitting.qsim.level_statistics import merge_spectra

    sources = [
        _merge_source([(2, 0, 0)]),               # final defaults to initial
        _merge_source([(1, 1, 0)], [(1, 1, 0)]),  # explicitly diagonal
    ]
    with pytest.raises(Exception) as excinfo:
        merge_spectra(sources)
    assert not isinstance(excinfo.value, NotImplementedError), (
        f"the diagonality guard rejected diagonal rows: {excinfo.value}")
