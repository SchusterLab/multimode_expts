# -*- coding: utf-8 -*-
"""Direct M1 local return for the central-boson Floquet model.

Split out of ``floquet_dark_mode_readout.py`` unchanged, config validators
included: they encode this protocol's conventions and nothing else uses them.
``classify_two_parity_readouts`` stays behind because
``DarkBaseExperiment.analyze_multiparity`` calls it.
"""
import numpy as np
from slab import AttrDict

from experiments.MM_base import MMAveragerProgram
from experiments.qsim.floquet_dark_mode_readout import (
    DarkBaseExperiment,
    SidebandScrambleDarkProgramNewNew,
)

CENTRAL_RETURN_PRIMARY_PHOTONS = 3
CENTRAL_RETURN_ALLOWED_TOTAL_PHOTONS = (1, 3)


def validate_central_return_occupations(
        initial_occupations,
        swap_stors,
        expected_total_photons=None,
        require_collision_free=True):
    """Validate the product-Fock convention used by the local-return program.

    ``initial_occupations`` is ordered as ``[n_M1] + [n_S for S in
    swap_stors]``.  The N=3 pilot deliberately defaults to collision-free
    states: they use only the already-calibrated one-photon M1/storage swaps
    and avoid adding a multi-photon state-preparation uncertainty to the
    correlation-hole measurement.
    """
    swap_stors = list(swap_stors)
    occupations = list(initial_occupations)

    if not swap_stors:
        raise ValueError("swap_stors must contain at least one storage mode")
    if any(isinstance(stor, (bool, np.bool_))
           or not isinstance(stor, (int, np.integer))
           for stor in swap_stors):
        raise TypeError("swap_stors entries must be integer storage numbers")
    swap_stors = [int(stor) for stor in swap_stors]
    if len(set(swap_stors)) != len(swap_stors):
        raise ValueError(f"swap_stors entries must be distinct, got {swap_stors}")
    if any(stor < 1 or stor > 7 for stor in swap_stors):
        raise ValueError(f"swap_stors must be in S1..S7, got {swap_stors}")

    expected_len = 1 + len(swap_stors)
    if len(occupations) != expected_len:
        raise ValueError(
            "initial_occupations must be ordered as [n_M1] + occupations "
            f"for swap_stors={swap_stors}; expected {expected_len} entries, "
            f"got {len(occupations)}"
        )
    if any(isinstance(n, (bool, np.bool_))
           or not isinstance(n, (int, np.integer))
           for n in occupations):
        raise TypeError("initial_occupations entries must be non-negative integers")
    occupations = [int(n) for n in occupations]
    if any(n < 0 for n in occupations):
        raise ValueError("initial_occupations entries must be non-negative")
    total_photons = sum(occupations)
    if total_photons not in CENTRAL_RETURN_ALLOWED_TOTAL_PHOTONS:
        raise ValueError(
            "central-return experiment supports the primary N=3 protocol "
            "and the N=1 self-Kerr-inactive control; got occupations "
            f"{occupations} with N={total_photons}"
        )
    if expected_total_photons is not None \
            and total_photons != int(expected_total_photons):
        raise ValueError(
            f"expected total occupation N={int(expected_total_photons)}, "
            f"got occupations {occupations} with N={total_photons}"
        )
    if require_collision_free and any(n > 1 for n in occupations):
        raise ValueError(
            "the decisive pilot uses collision-free product Fock states "
            f"(occupations 0 or 1), got {occupations}"
        )

    return occupations, swap_stors


def configure_central_return_metadata(expt_cfg):
    """Normalize and preserve the invariants for direct M1 local return."""
    if "initial_occupations" not in expt_cfg:
        raise ValueError(
            "initial_occupations is required and must be ordered as "
            "[n_M1] + [n_S for S in swap_stors]"
        )

    expected_total_photons = expt_cfg.get(
        "expected_total_occupation", expt_cfg.get("total_photons", None))
    if not expt_cfg.get("require_collision_free_initial_state", True):
        raise ValueError(
            "CentralBosonLocalReturnProgram currently supports only "
            "collision-free initial occupations; the multi-Fock opt-in does "
            "not share the calibrated compensation path"
        )
    occupations, swap_stors = validate_central_return_occupations(
        expt_cfg.initial_occupations,
        expt_cfg.swap_stors,
        expected_total_photons=expected_total_photons,
        require_collision_free=True,
    )
    total_photons = sum(occupations)

    if int(expt_cfg.get("man_mode_no", 1)) != 1:
        raise ValueError(
            "CentralBosonLocalReturnProgram is hard-wired to M1; "
            "man_mode_no must be 1"
        )
    if not expt_cfg.get("prepulse", False):
        raise ValueError(
            "prepulse must be True to explicitly enable the custom central "
            "product-Fock preparation"
        )
    for custom_flag in ("custom_prepulse", "custom_postpulse"):
        if expt_cfg.get(custom_flag, False):
            raise ValueError(
                f"{custom_flag}=True is incompatible with the fixed "
                "central-return pulse body"
            )
    if expt_cfg.get("map_to_qubit_ge", False):
        raise ValueError(
            "map_to_qubit_ge is not used: central return ends in direct M1 "
            "multiparity readout"
        )

    for legacy_flag in (
            "load_man_dark", "swap_man_dark", "swap_man_large_dark"):
        if expt_cfg.get(legacy_flag, False):
            raise ValueError(
                f"{legacy_flag}=True is incompatible with direct central "
                "return; no dark-mode load or analyzer is used"
            )
    if expt_cfg.get("perform_wigner", False):
        raise ValueError("perform_wigner must be False for direct multiparity readout")
    if expt_cfg.get("parity_check", False):
        raise ValueError(
            "parity_check is not part of the direct central-return pulse body"
        )
    if expt_cfg.get("parity_readout", False):
        raise ValueError(
            "parity_readout must be False; use multiparity_readout for the "
            "four-outcome M1 measurement"
        )
    if "ro_stor" in expt_cfg and int(expt_cfg.ro_stor) != 0:
        raise ValueError("ro_stor must be 0: M1 is read out directly")
    if not expt_cfg.get("multiparity_readout", False):
        raise ValueError("multiparity_readout must be True for central return")
    if not expt_cfg.get("postpulse", False):
        raise ValueError(
            "postpulse must be True to make the final direct M1 readout "
            "explicit in the experiment config"
        )

    cond_sec_phase = float(expt_cfg.get("cond_sec_phase", -90.0))
    if not np.isclose((cond_sec_phase + 90.0) % 360.0, 0.0):
        raise ValueError(
            "cond_sec_phase must be -90 deg (mod 360) so p_mod0..p_mod3 "
            "use the classifier convention"
        )
    second_phase = float(expt_cfg.get("phase_second_pulse", 180.0))
    if not np.isclose(second_phase % 360.0, 180.0):
        raise ValueError(
            "phase_second_pulse must be 180 deg (mod 360) for the direct "
            "multiparity classifier convention"
        )

    target = occupations[0] % 4
    for key, expected in (
            ("initial_central_occupation", occupations[0]),
            ("return_target_nmod4", target)):
        if key in expt_cfg and int(expt_cfg[key]) != expected:
            raise ValueError(
                f"{key}={expt_cfg[key]} conflicts with initial_occupations "
                f"(expected {expected})"
            )

    # These fields live in cfg.expt, so they are saved with every hardware job.
    expt_cfg.initial_occupations = occupations
    expt_cfg.initial_occupation_modes = ["M1"] + [
        f"S{stor}" for stor in swap_stors]
    expt_cfg.total_photons = total_photons
    expt_cfg.expected_total_occupation = total_photons
    expt_cfg.primary_total_photons = CENTRAL_RETURN_PRIMARY_PHOTONS
    expt_cfg.is_primary_n3 = (total_photons == CENTRAL_RETURN_PRIMARY_PHOTONS)
    expt_cfg.is_kerr_inactive_n1_control = (total_photons == 1)
    expt_cfg.initial_central_occupation = occupations[0]
    expt_cfg.return_target_nmod4 = target
    expt_cfg.return_target_key = f"p_mod{target}"
    expt_cfg.return_observable = "P(n_M1(t) mod 4 == return_target_nmod4)"
    expt_cfg.desired_local_return_observable = "P(n_M1(t) == n_M1(0))"
    expt_cfg.mod4_number_mapping_exact_in_fixed_N_le_3 = True
    expt_cfg.mod4_mapping_requires_no_upward_leakage = True
    expt_cfg.readout_mode = "M1_direct_multiparity"
    expt_cfg.initial_state_preparation = "custom_central_product_fock"
    expt_cfg.legacy_base_preparation_bypassed = True
    expt_cfg.require_collision_free_initial_state = True
    expt_cfg.man_mode_no = 1
    expt_cfg.prepulse = True
    expt_cfg.custom_prepulse = False
    expt_cfg.custom_postpulse = False
    expt_cfg.map_to_qubit_ge = False
    for ignored_key in ("init_fock", "init_man_fock_state", "init_stor"):
        expt_cfg.pop(ignored_key, None)
    expt_cfg.ro_stor = 0
    expt_cfg.multiparity_readout = True
    expt_cfg.cond_sec_phase = -90.0
    expt_cfg.phase_second_pulse = 180.0
    expt_cfg.perform_wigner = False
    expt_cfg.parity_check = False
    expt_cfg.parity_readout = False
    expt_cfg.load_man_dark = False
    expt_cfg.swap_man_dark = False
    expt_cfg.swap_man_large_dark = False

    return occupations, swap_stors


class CentralBosonLocalReturnProgram(SidebandScrambleDarkProgramNewNew):
    """Direct M1 local-return protocol for the central-boson Floquet model.

    The pulse body is intentionally separate from ``DarkBaseProgram.body``:
    it prepares one product Fock state, plays only the calibrated sequential
    Floquet scramble, and measures M1 in place.  In particular, it never calls
    a dark-mode load or analyzer.  For the primary fixed-N=3 experiment (and
    the N=1 self-Kerr-inactive control), M1 multiparity resolves n_M1=0..3.
    Thus ``p_mod{n_M1(0)}`` represents local return within the fixed N<=3
    model; upward leakage to n_M1>=4 would alias under modulo-4 readout.
    """

    def initialize(self):
        self.initial_occupations, self.central_return_swap_stors = \
            configure_central_return_metadata(self.cfg.expt)
        super().initialize()

    def _prepare_initial_product_fock(self):
        """Prepare ``[n_M1] + [n_S for S in swap_stors]`` without dark load.

        Each occupied storage is prepared by making its Fock occupation in M1
        and applying the existing calibrated full M1-S swap.  The central
        occupation is prepared last.  The decisive protocol defaults to
        collision-free occupations, for which every nonzero preparation is a
        calibrated one-photon transfer.  Multi-Fock product-state preparation
        is deliberately rejected until it shares the calibrated compensation
        path used by the legacy program.
        """
        occupations, swap_stors = validate_central_return_occupations(
            self.cfg.expt.initial_occupations,
            self.cfg.expt.swap_stors,
            expected_total_photons=self.cfg.expt.total_photons,
            require_collision_free=True,
        )

        prepulse_cfg = []
        for stor, occupation in zip(swap_stors, occupations[1:]):
            if occupation == 0:
                continue
            prepulse_cfg += self.prep_man_fock_state(
                self.cfg.expt.get("man_mode_no", 1),
                str(occupation),
                broadband=False,
            )
            prepulse_cfg.append([
                "storage", f"M1-S{stor}", "pi", 0,
            ])

        central_occupation = occupations[0]
        if central_occupation:
            prepulse_cfg += self.prep_man_fock_state(
                self.cfg.expt.get("man_mode_no", 1),
                str(central_occupation),
                broadband=False,
            )

        if not prepulse_cfg:
            return

        pulse_creator = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            AttrDict(self.cfg),
            pulse_creator.pulse,
            prefix="central_return_pre_",
        )
        self.sync_all()

        if self.cfg.expt.get("debug", False):
            print(
                "[CentralReturn] prepared",
                dict(zip(
                    self.cfg.expt.initial_occupation_modes,
                    occupations,
                )),
            )

    def core_pulses(self):
        """Play the calibrated scramble with no dark-mode load/readout."""
        swap_stors = list(self.central_return_swap_stors)
        phase_offsets = [0.0] * len(swap_stors)
        disorder_phase_offsets = [0.0] * len(swap_stors)

        self._play_scramble_with_phase_offsets(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def body(self):
        """Prepare -> Floquet scramble -> direct M1 multiparity -> measure."""
        self.reset_and_sync()

        if self.cfg.expt.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = self.cfg.expt.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        # This custom body is the only preparation path; it deliberately does
        # not call DarkBaseProgram.body and therefore cannot double-run its
        # legacy init_stor/init_man_fock_state preparation.
        self._prepare_initial_product_fock()
        self.core_pulses()

        # ro_stor is validated to be zero.  Population remains in M1 and is
        # measured directly; no storage-to-M1 or dark-mode analyzer is emitted.
        self.multi_parity_readout(
            man_idx=self.man_mode_idx + 1,
            fast=self.cfg.expt.get("parity_fast", False),
        )
        self.sync_all()
        self.measure_wrapper()


class CentralBosonLocalReturnExperiment(DarkBaseExperiment):
    """Experiment wrapper that stores and extracts the direct local return."""

    def _store_raw_central_return(self):
        """Flatten threshold-classified mod-4 probabilities for safe saving."""
        target = int(self.cfg.expt.return_target_nmod4)
        multiparity = self.analyze_multiparity()
        p_mod4_raw = np.column_stack([
            np.asarray(multiparity[f"p_mod{n}"], dtype=float)
            for n in range(4)
        ])

        # analyze_multiparity() caches a nested dictionary in self.data. Slab's
        # HDF5 writer expects numeric arrays, so never leave that cache in the
        # payload that CharacterizationRunner saves.
        self.data.pop("multiparity", None)
        self.data["p_mod4_raw"] = p_mod4_raw
        self.data["central_return_probability_raw"] = p_mod4_raw[:, target]
        return self.data

    def acquire(self, progress=False, debug=False):
        occupations, _ = configure_central_return_metadata(self.cfg.expt)
        if len(self.cfg.expt.swept_params) != 1:
            raise ValueError(
                "CentralBosonLocalReturnExperiment expects one sweep axis; "
                "run initial states and disorder realizations as separate "
                "jobs so their metadata stay unambiguous"
            )

        data = super().acquire(progress=progress, debug=debug)
        target = int(self.cfg.expt.return_target_nmod4)

        # Store the provenance in the data payload as well as cfg.expt, so it
        # survives both config-aware and data-only aggregation workflows.
        data["initial_occupations"] = np.asarray(occupations, dtype=int)
        data["initial_central_occupation"] = int(occupations[0])
        data["total_photons"] = int(sum(occupations))
        data["return_target_nmod4"] = target

        self.data = data
        return self._store_raw_central_return()

    def analyze(self, data=None, **kwargs):
        if data is not None:
            self.data = data
        return self._store_raw_central_return()
