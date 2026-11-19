"""Spectral functions of the central-boson model averaged over disorder patterns.

The model is ``theory.qsim.central_boson`` with equal couplings g to all storage modes. A pattern
e is drawn by ``disorder_pattern`` (unit RMS); the storage detunings are W e. Each probe is
diagonal in the Fock basis. Its sticks (Delta_ij > 0, w_ij) come from ``theory.qsim.agp``:
C(t) = dc + 2 sum_k w_k cos(2 pi f_k t).

Probes are given as {name: (mode, kind)}:

- ``(m, "number")``, ``(m, "kerr")``: n or n (n - 1) / 2 of mode m (0 is the hub M1);
- ``(None, "number")``: the single-mode storage number, its sticks pooled over all storage modes
  with weight 1 / (number of modes), so the result is the mean over modes;
- ``(None, "disorder")``: V = sum_s e_s n_s with the pattern e that also sets H (the direction W
  moves along).

Moved from ``analysis_notebooks/guan/agp/understanding/spectral_function.py`` on 2026-11-19.
Energies in cyclic MHz, as everywhere in ``theory/``.
"""
from dataclasses import dataclass

import numpy as np

from theory.qsim.agp import binned_density, positive_sticks, probe_values, spectral_weights
from theory.qsim.central_boson import disorder_pattern, disorder_values, fixed_n_hamiltonian

DEFAULT_PROBES = {
    "hub Kerr": (0, "kerr"),
    "hub number": (0, "number"),
    "storage number": (None, "number"),
    "disorder V": (None, "disorder"),
}


@dataclass(frozen=True)
class StarSetup:
    """The fixed part of the model: photon number, number of storage modes, the common coupling."""
    photon_number: int = 3
    storage_modes: int = 4
    coupling_mhz: float = 0.02

    def model(self, detunings, kerr_mhz):
        """-> the ``fixed_n_hamiltonian`` result for these storage detunings and Kerr."""
        return fixed_n_hamiltonian(self.photon_number, self.storage_modes + 1, detunings,
                                   [self.coupling_mhz] * self.storage_modes, kerr_mhz)

    @property
    def covariance_factor(self):
        """c in E[e e^T] = c (I - J / n) for unit-RMS zero-mean patterns of n modes: c = n / (n - 1)."""
        return self.storage_modes / (self.storage_modes - 1)


def stick_spectrum(model, values):
    """-> (frequencies, weights, dc) for a probe with Fock-basis diagonal ``values``."""
    return positive_sticks(*spectral_weights(model, values))


def mode_spectrum(model, mode_index, probe="number"):
    """-> (frequencies, weights, dc) of ``probe`` (``number`` or ``kerr``) of one mode."""
    return stick_spectrum(model, probe_values(model.fock_basis, mode_index=mode_index, probe=probe))


def probe_sticks(model, unit_pattern, mode, kind, storage_modes):
    """-> (frequencies, weights, dc) of one probe given as (mode, kind), see the module docstring."""
    if kind == "disorder":
        return stick_spectrum(model, disorder_values(model.fock_basis, unit_pattern))
    modes = range(1, storage_modes + 1) if mode is None else [mode]
    parts = [mode_spectrum(model, m, kind) for m in modes]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]) / len(modes),
            sum(p[2] for p in parts) / len(modes))


def ensemble_spectra(setup, kerr_mhz, rms_mhz, samples, edges_MHz, probes=DEFAULT_PROBES, seed=0):
    """-> {probe: dict(f, w, density, sem, dc)} at one (K, W) point, over ``samples`` patterns.

    ``f``, ``w``: all sticks of all patterns pooled, so ``w.sum() / samples`` is the mean
    positive-frequency weight. ``density``, ``sem``: mean and standard error over patterns of the
    weight per MHz in the bins of ``edges_MHz``. ``dc``: mean zero-frequency weight.
    """
    rng = np.random.default_rng(seed)
    bins = len(edges_MHz) - 1
    binned = {name: np.zeros((samples, bins)) for name in probes}
    pooled = {name: ([], []) for name in probes}
    dc = {name: np.zeros(samples) for name in probes}
    for r in range(samples):
        unit = disorder_pattern(rng, setup.storage_modes, 1.0)
        model = setup.model(rms_mhz * unit, kerr_mhz)
        for name, (mode, kind) in probes.items():
            f, w, dc[name][r] = probe_sticks(model, unit, mode, kind, setup.storage_modes)
            pooled[name][0].append(f)
            pooled[name][1].append(w)
            binned[name][r] = binned_density(f, w, edges_MHz)
    return {name: dict(f=np.concatenate(pooled[name][0]), w=np.concatenate(pooled[name][1]),
                       density=binned[name].mean(0), sem=binned[name].std(0, ddof=1) / np.sqrt(samples),
                       dc=dc[name].mean())
            for name in probes}


def _group_lines(frequencies, weights, decimals=9):
    lines, inverse = np.unique(np.round(frequencies, decimals), return_inverse=True)
    return lines, np.bincount(inverse, weights=weights)


def zero_disorder_lines(setup, kerr_mhz, probes=DEFAULT_PROBES):
    """-> {probe: dict(f, w, dc)} at W = 0: one exact spectrum, sticks of equal frequency summed.

    At W = 0 H does not depend on the pattern, so the pattern average of the disorder probe V is
    exact from the covariance c (I - J / n): line weights c sum_s w_s - (c / n) w_hub, because
    sum_s n_s = N - n_0. Its DC is not given by that formula (the constant N), so ``dc`` is NaN.
    """
    model = setup.model(np.zeros(setup.storage_modes), kerr_mhz)
    out = {}
    for name, (mode, kind) in probes.items():
        if kind == "disorder":
            hub = mode_spectrum(model, 0, "number")
            stor = [mode_spectrum(model, m, "number") for m in range(1, setup.storage_modes + 1)]
            c, n = setup.covariance_factor, setup.storage_modes
            f = np.concatenate([p[0] for p in stor] + [hub[0]])
            w = np.concatenate([c * p[1] for p in stor] + [-c / n * hub[1]])
            lines, combined = _group_lines(f, w)
            if combined.min() < -1e-9:
                raise ArithmeticError("negative line weight: the covariance formula failed")
            out[name] = dict(f=lines, w=np.maximum(combined, 0), dc=np.nan)
        else:
            f, w, d = probe_sticks(model, None, mode, kind, setup.storage_modes)
            lines, combined = _group_lines(f, w)
            out[name] = dict(f=lines, w=combined, dc=d)
    return out


def positive_totals(setup, kerr_mhz, rms_mhz, samples, probes=DEFAULT_PROBES, seed=0):
    """-> {probe: (mean, standard error)} over patterns of the positive-frequency weight total."""
    rng = np.random.default_rng(seed)
    totals = {name: np.zeros(samples) for name in probes}
    for r in range(samples):
        unit = disorder_pattern(rng, setup.storage_modes, 1.0)
        model = setup.model(rms_mhz * unit, kerr_mhz)
        for name, (mode, kind) in probes.items():
            totals[name][r] = probe_sticks(model, unit, mode, kind, setup.storage_modes)[1].sum()
    return {name: (t.mean(), t.std(ddof=1) / np.sqrt(samples)) for name, t in totals.items()}


def disorder_proxy_comparison(setup, kerr_mhz, rms_mhz, samples, edges_MHz, seed=0):
    """-> dict(exact, proxy, difference, dc): the disorder probe V against its single-mode proxy.

    exact: the spectrum of V with the pattern that sets H. proxy: c (n S_storage - S_hub / n), the
    pattern average if the pattern and H were independent (valid for f > 0 only). Each entry is
    (mean, standard error) of the weight per MHz over patterns; the same patterns give both, so
    ``difference`` has its own standard error. ``dc``: the mean DC weight of V.
    """
    rng = np.random.default_rng(seed)
    bins = len(edges_MHz) - 1
    c, n = setup.covariance_factor, setup.storage_modes
    exact, proxy = np.zeros((samples, bins)), np.zeros((samples, bins))
    dc = np.zeros(samples)
    for r in range(samples):
        unit = disorder_pattern(rng, n, 1.0)
        model = setup.model(rms_mhz * unit, kerr_mhz)
        f, w, dc[r] = stick_spectrum(model, disorder_values(model.fock_basis, unit))
        exact[r] = binned_density(f, w, edges_MHz)
        hub = mode_spectrum(model, 0, "number")
        stor = [mode_spectrum(model, m, "number") for m in range(1, n + 1)]
        h_stor = np.mean([binned_density(p[0], p[1], edges_MHz) for p in stor], axis=0)
        proxy[r] = c * (n * h_stor - binned_density(hub[0], hub[1], edges_MHz) / n)
    sem = lambda x: x.std(0, ddof=1) / np.sqrt(samples)
    diff = exact - proxy
    return dict(exact=(exact.mean(0), sem(exact)), proxy=(proxy.mean(0), sem(proxy)),
                difference=(diff.mean(0), sem(diff)), dc=dc.mean())
