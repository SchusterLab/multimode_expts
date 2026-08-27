"""Resolve historical Floquet timing from versioned configs, per spec section 2.2.

Why this exists
---------------
``_saved_parameters`` currently gets Floquet timing one of two ways, and both
are wrong for offline analysis:

* from a pickled compiled Program, which is not portable and is absent from
  HDF5; or
* by asking the *current* station, which silently substitutes today's
  calibration for the historical one. Between 2026-08-14 and 2026-08-25 the
  swap dataset moved ``gauss_sigma`` 0.04 -> 0.02 us, so that fallback returns
  roughly half the correct cycle time without any error.

The timing is not a measurement. It is computed from configuration that is
already versioned and immutable, so it can be recomputed exactly. Verified
bit-for-bit against ``JOB-20260815-00009``: this module reproduces
``0.7340315934065934``, the value its pickle held.

Inputs
------
* the versioned Floquet swap CSV -- ``pi_frac``, ``len``, ``freq``,
  ``waveform``, ``gauss_sigma``, ``gauss_n_sigma`` per mode;
* the HDF5's embedded ``expt`` config -- ``swap_stors``,
  ``scramble_sync_cycles``, and any waveform override;
* the embedded ``device.manipulate.ramp_sigma`` (only the ``flat_top`` branch
  uses it); and
* a real ``QickConfig`` for ``us2cycles``/``cycles2us``. These conversions are
  firmware-dependent and are never the identity, so a stub silently corrupts
  every result. The committed ``configs/soccfg_snapshot.json`` is the offline
  source.

This module reads the archive as plain files. It never constructs a station,
never opens the job database, and never writes anything -- see spec section
13.3.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from qick import QickConfig

from experiments.dataset import FloquetStorageSwapDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
SOCCFG_SNAPSHOT = REPO_ROOT / "configs" / "soccfg_snapshot.json"
ARCHIVE_ENV = "MULTIMODE_CONFIG_ARCHIVE"
DEFAULT_ARCHIVE = REPO_ROOT / "configs" / "versions"

# Above this the swap uses the high-frequency flux channel. Mirrors
# QsimBaseProgram.retrieve_swap_parameters.
FLUX_HIGH_THRESHOLD_MHZ = 1800


class TimingResolutionError(RuntimeError):
    """Raised when historical timing cannot be resolved unambiguously."""


def config_archive() -> Path:
    raw = os.environ.get(ARCHIVE_ENV)
    root = Path(raw) if raw else DEFAULT_ARCHIVE
    if not root.is_dir():
        source = f"${ARCHIVE_ENV}={raw!r}" if raw else f"default {DEFAULT_ARCHIVE}"
        raise TimingResolutionError(
            f"Config version archive not found: {root} (from {source}). "
            f"Set {ARCHIVE_ENV} to a copy of configs/versions/."
        )
    return root


@lru_cache(maxsize=1)
def committed_soccfg() -> QickConfig:
    """The committed firmware snapshot, as a real QickConfig."""
    if not SOCCFG_SNAPSHOT.is_file():
        raise TimingResolutionError(
            f"No soccfg snapshot at {SOCCFG_SNAPSHOT}. It is written by a real "
            f"(non-mock) station on the production PC."
        )
    return QickConfig(json.loads(SOCCFG_SNAPSHOT.read_text()))


@lru_cache(maxsize=16)
def floquet_swap_dataset(version_id: str, archive: Path = None):
    """Load one versioned Floquet swap CSV as a dataset object.

    Cached: resolving a whole job set hits the same version repeatedly.
    """
    root = Path(archive) if archive else config_archive()
    path = root / "floquet_storage_swap" / f"{version_id}.csv"
    if not path.is_file():
        raise TimingResolutionError(f"No archived Floquet swap config {version_id} at {path}")
    return FloquetStorageSwapDataset(filename=path.name, parent_path=path.parent)


def resolve_floquet_timing(cfg, floquet_version_id, archive=None, soccfg=None):
    """Recompute the Floquet timing that was compiled at acquisition.

    Args:
        cfg: the experiment configuration embedded in the HDF5 file, with
            ``expt``, ``hw`` and ``device`` sections.
        floquet_version_id: e.g. ``"CFG-FL-20260814-00076"``.
        archive: override the ``configs/versions/`` location.
        soccfg: override the QickConfig (tests pin the committed snapshot).

    Returns:
        dict with ``floquet_cycle_us``, ``m1s_pi_fracs`` (all seven modes),
        ``couplings_MHz`` (one per swapped storage) and ``source``.
    """
    soccfg = soccfg or committed_soccfg()
    swap_ds = floquet_swap_dataset(floquet_version_id, Path(archive) if archive else None)

    ecfg = cfg["expt"]
    qubit = ecfg["qubits"][0]
    dacs = cfg["hw"]["soc"]["dacs"]
    flux_low_ch = dacs["flux_low"]["ch"][qubit]
    flux_high_ch = dacs["flux_high"]["ch"][qubit]

    # --- retrieve_swap_parameters, offline ---
    stor_names = [f"M1-S{n}" for n in range(1, 8)]
    pi_fracs = [swap_ds.get_pi_frac(name) for name in stor_names]
    freqs_MHz = [swap_ds.get_freq(name) for name in stor_names]
    is_low = [freq < FLUX_HIGH_THRESHOLD_MHZ for freq in freqs_MHz]
    channels = [flux_low_ch if low else flux_high_ch for low in is_low]
    lengths = [soccfg.us2cycles(swap_ds.get_len(name), gen_ch=ch)
               for name, ch in zip(stor_names, channels)]

    waveform_override = ecfg.get("floquet_waveform", None)

    def style(name):
        waveform = waveform_override if waveform_override is not None else swap_ds.get_waveform(name)
        return "arb" if waveform in ("gauss", "gaussian", "arb") else "flat_top"

    styles = [style(name) for name in stor_names]

    # --- calculate_floquet_cycle_us, offline ---
    ramp_sigma = cfg["device"]["manipulate"]["ramp_sigma"]
    ramp_cycles_low = soccfg.us2cycles(ramp_sigma, gen_ch=flux_low_ch)
    ramp_cycles_high = soccfg.us2cycles(ramp_sigma, gen_ch=flux_high_ch)

    swap_stors = list(ecfg["swap_stors"])
    sync_cycles = ecfg.get("scramble_sync_cycles", 10)
    cycle_us = len(swap_stors) * soccfg.cycles2us(sync_cycles)
    for stor in swap_stors:
        index = stor - 1
        channel = channels[index]
        if styles[index] == "arb":
            sigma_us = ecfg.get("floquet_gauss_sigma", None)
            if sigma_us is None:
                sigma_us = swap_ds.get_gauss_sigma(f"M1-S{stor}")
            pulse_cycles = (soccfg.us2cycles(sigma_us, gen_ch=channel)
                            * swap_ds.get_gauss_n_sigma(f"M1-S{stor}"))
        else:
            ramp = ramp_cycles_low if is_low[index] else ramp_cycles_high
            pulse_cycles = lengths[index] + 6 * ramp
        cycle_us += soccfg.cycles2us(pulse_cycles, gen_ch=channel)

    if not np.isfinite(cycle_us) or cycle_us <= 0.:
        raise TimingResolutionError(
            f"resolved a non-physical Floquet cycle time {cycle_us!r} "
            f"from {floquet_version_id}"
        )

    swapped_fracs = np.asarray([pi_fracs[stor - 1] for stor in swap_stors], dtype=float)
    couplings_MHz = 1. / (4. * swapped_fracs * cycle_us)

    return dict(
        floquet_cycle_us=float(cycle_us),
        m1s_pi_fracs=[int(value) for value in pi_fracs],
        couplings_MHz=couplings_MHz,
        source=f"versioned config {floquet_version_id}",
    )
