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
from slab import AttrDict

from experiments.dataset import FloquetStorageSwapDataset
from experiments.local_env import load_env

# The archive location is per-machine; read it from the repo-root .env when the
# environment does not already say. See experiments/local_env.py.
load_env()

REPO_ROOT = Path(__file__).resolve().parent.parent
SOCCFG_SNAPSHOT = REPO_ROOT / "configs" / "soccfg_snapshot.json"
ARCHIVE_ENV = "MULTIMODE_CONFIG_ARCHIVE"
DEFAULT_ARCHIVE = REPO_ROOT / "configs" / "versions"

# Above this the swap uses the high-frequency flux channel. Mirrors
# QsimBaseProgram.retrieve_swap_parameters.
FLUX_HIGH_THRESHOLD_MHZ = 1800


def floquet_cycle_us(swap_stors,
                     *,
                     swap_ds,
                     waveform_modes,
                     channels,
                     lengths,
                     ramp_cycles,
                     sync_cycles,
                     us2cycles,
                     cycles2us,
                     clock_ratio,
                     gauss_sigma_override=None):
    """-> scheduled duration of one Floquet cycle, in microseconds.

    One definition, called from two places that cannot share a ``self``: the
    live program (``DarkBaseProgram.calculate_floquet_cycle_us``) and the
    offline resolver below, which has a ``QickConfig`` and a versioned CSV but
    no program. They were line-by-line copies of this arithmetic, which is a
    bad thing to duplicate: the cycle time divides into every coupling rate,
    so a drift between them would show up as a wrong Hamiltonian rather than
    as an error.

    The quantization is the substance. QICK v1's ``sync_all`` emits
    ``synci(int(pulse_end_timestamp + sync_cycles))``, so a cycle is a sum of
    *integer* tProc advances, not of exact pulse durations. Taking the exact
    sum instead ran ~1.2% long on the August configs.

    Args:
        swap_stors: storage numbers in the order they are pulsed.
        swap_ds: the Floquet swap dataset (gauss sigma, n_sigma, ramp sigma).
        waveform_modes, channels, lengths, ramp_cycles: per-storage arrays
            indexed by ``stor - 1``, i.e. seven entries, not one per pulsed
            mode. ``ramp_cycles`` is already resolved to the storage's own
            flux channel.
        sync_cycles: ``scramble_sync_cycles``, the inter-pulse sync.
        us2cycles, cycles2us: the firmware's own conversions. Never the
            identity, so a stub corrupts the result silently.
        clock_ratio: ``channel -> f_time / f_fabric`` for that generator.
        gauss_sigma_override: ``cfg.expt.floquet_gauss_sigma`` when set;
            otherwise each mode's calibrated sigma is used.
    """
    cycle_tproc_cycles = 0
    for stor in swap_stors:
        index = stor - 1
        channel = channels[index]
        mode = waveform_modes[index]
        if mode == "gauss":
            sigma_us = gauss_sigma_override
            if sigma_us is None:
                sigma_us = swap_ds.get_gauss_sigma(f"M1-S{stor}")
            pulse_cycles = (us2cycles(sigma_us, channel)
                            * swap_ds.get_gauss_n_sigma(f"M1-S{stor}"))
        elif mode == "preload_flattop":
            ramp = us2cycles(swap_ds.get_ramp_sigma(f"M1-S{stor}"), channel)
            pulse_cycles = lengths[index] + 6 * ramp
        else:
            # M1-Sx flat tops are six sigma of ramp around the plateau.
            pulse_cycles = lengths[index] + 6 * ramp_cycles[index]
        cycle_tproc_cycles += int(pulse_cycles * clock_ratio(channel)
                                  + sync_cycles)
    return cycles2us(cycle_tproc_cycles)


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

    def waveform_mode(name):
        waveform = waveform_override if waveform_override is not None else swap_ds.get_waveform(name)
        if waveform in ("gauss", "gaussian", "arb"):
            return "gauss"
        if waveform == "preload_flattop":
            return "preload_flattop"
        return "flat_top"

    modes = [waveform_mode(name) for name in stor_names]

    # --- the cycle duration, from the shared definition -----------------
    ramp_sigma = cfg["device"]["manipulate"]["ramp_sigma"]
    ramp_cycles_low = soccfg.us2cycles(ramp_sigma, gen_ch=flux_low_ch)
    ramp_cycles_high = soccfg.us2cycles(ramp_sigma, gen_ch=flux_high_ch)

    swap_stors = list(ecfg["swap_stors"])
    cycle_us = floquet_cycle_us(
        swap_stors,
        swap_ds=swap_ds,
        waveform_modes=modes,
        channels=channels,
        lengths=lengths,
        ramp_cycles=[ramp_cycles_low if low else ramp_cycles_high
                     for low in is_low],
        sync_cycles=int(ecfg.get("scramble_sync_cycles", 10)),
        us2cycles=lambda us, ch: soccfg.us2cycles(us, gen_ch=ch),
        cycles2us=soccfg.cycles2us,
        clock_ratio=lambda ch: (float(soccfg["tprocs"][0]["f_time"])
                                / float(soccfg["gens"][ch]["f_fabric"])),
        gauss_sigma_override=ecfg.get("floquet_gauss_sigma", None),
    )

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


def station_floquet_hardware(station, 
                        swap_stors, 
                        sync_cycles, 
                        floquet_gauss_sigma=None,
                        floquet_waveform=None):
    """
    Moved from ``EncodingHamiltonianSpectroscopyExperiment.hardware_parameters``
    in MBR redesign step 7e (2026-09-24), without changes. Asks the *live*
    station, so it gives today's calibration: for saved jobs use
    :func:`resolve_floquet_timing` or ``experiments.qsim.mbr_saved``.

    Returns hardware related physical paramters such as
        - floquet_cycle_us: time for a single floquet cycle in a microsecond
        - couplings_MHz: an array of effective BS coupling between man and stor
        - physical_kerr_MHz: self Kerr on a central mode (manipulate)
    All the values are calculated from the config/expt_cfg input
    """
    
    if (isinstance(sync_cycles, (bool, np.bool_))
            or not isinstance(sync_cycles, (int, np.integer)) or sync_cycles < 0):
        raise ValueError("sync_cycles must be a nonnegative integer")
    ramp_sigma = station.hardware_cfg.device.manipulate.ramp_sigma
    if isinstance(ramp_sigma, (list, tuple, np.ndarray)):
        ramp_sigma = ramp_sigma[0]
    pulse_us = []
    pi_fracs = []
    cycle_tproc_cycles = 0
    for stor in swap_stors:
        pulse_name = f"M1-S{stor}"
        if station.ds_floquet.get_freq(pulse_name) < FLUX_HIGH_THRESHOLD_MHZ:
            gen_ch = station.hardware_cfg.hw.soc.dacs.flux_low.ch[0]
        else:
            gen_ch = station.hardware_cfg.hw.soc.dacs.flux_high.ch[0]
        waveform = floquet_waveform if floquet_waveform is not None else station.ds_floquet.get_waveform(pulse_name)
        # Match calculate_floquet_cycle_us: round each envelope segment
        # to its generator clock before adding the tProc sync interval.
        if waveform in ("gauss", "gaussian", "arb"):
            sigma = floquet_gauss_sigma
            if sigma is None:
                sigma = station.ds_floquet.get_gauss_sigma(pulse_name)
            sigma_cycles = station.soccfg.us2cycles(sigma, gen_ch=gen_ch)
            pulse_cycles = sigma_cycles * station.ds_floquet.get_gauss_n_sigma(pulse_name)
        elif waveform == "preload_flattop":
            flat_cycles = station.soccfg.us2cycles(station.ds_floquet.get_len(pulse_name), gen_ch=gen_ch)
            ramp_cycles = station.soccfg.us2cycles(station.ds_floquet.get_ramp_sigma(pulse_name), gen_ch=gen_ch)
            pulse_cycles = flat_cycles + 6 * ramp_cycles
        else:
            flat_cycles = station.soccfg.us2cycles(station.ds_floquet.get_len(pulse_name), gen_ch=gen_ch)
            ramp_cycles = station.soccfg.us2cycles(ramp_sigma, gen_ch=gen_ch)
            pulse_cycles = flat_cycles + 6 * ramp_cycles
        pulse_us.append(station.soccfg.cycles2us(pulse_cycles, gen_ch=gen_ch))
        clock_ratio = float(station.soccfg["tprocs"][0]["f_time"]) / float(station.soccfg["gens"][gen_ch]["f_fabric"])
        cycle_tproc_cycles += int(pulse_cycles * clock_ratio + sync_cycles)
        pi_fracs.append(station.ds_floquet.get_pi_frac(pulse_name))

    # Match the integer synci advances, not the unquantized pulse+gap sum.
    floquet_cycle_us = station.soccfg.cycles2us(cycle_tproc_cycles)
    if not np.all(np.isfinite(pulse_us + pi_fracs)) or min(pulse_us + pi_fracs) <= 0.:
        raise ValueError("Floquet pulse lengths and pi fractions must be finite and positive")
    if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0.:
        raise ValueError("Floquet cycle duration must be finite and positive")

    # 2 * pi * g * t_swap = pi / 2 -> g = 1/ 4/ t_swap
    # pi_frac repetitions of each pulse+sync block make a full swap:
    # g_{bare} = 1/ 4 / n_frac / (t_pulse + t_sync)
    # len(swap_stors) -> g_{eff} * T_F = g_{bare} * (t_pulse+t_sync) = 1/4/n_frac
    # So g_{eff} = 1/4/n_frac/T_F
    couplings_MHz = [1. / (4. * pi_frac * floquet_cycle_us) for pi_frac in pi_fracs]
    physical_kerr_MHz = station.hardware_cfg.device.manipulate.kerr
    if isinstance(physical_kerr_MHz, (list, tuple, np.ndarray)):
        physical_kerr_MHz = physical_kerr_MHz[0]
    physical_kerr_MHz = -abs(physical_kerr_MHz)
    if not np.isfinite(physical_kerr_MHz):
        raise ValueError("physical Kerr must be finite")
    return AttrDict(dict(floquet_cycle_us=floquet_cycle_us,
                         couplings_MHz=np.asarray(couplings_MHz),
                         physical_kerr_MHz=physical_kerr_MHz))
