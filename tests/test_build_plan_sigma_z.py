'''
Offline validation of the closed-loop sigma_z wiring -- no hardware, no queue.

Covers:
  * batch_runner.build_plan (sim mode) threads measure_sigma_z / sigma_z_mode
    from a manifest entry into RunWignerRequest.knobs (and lane_layout resolves
    them to the right tier),
  * validate._check_manifest_entry accepts valid sigma_z fields and rejects a
    bad sigma_z_mode (e.g. the deferred 'measure') and a non-bool measure_sigma_z.

A minimal valid `*_sampled.jld2` is fabricated per the Harmoniqs contract so
build_plan's load + geometry checks pass without touching hardware.
'''

import numpy as np
import pytest
from slab import AttrDict

from job_server.closed_loop import batch_runner, validate
from job_server.closed_loop.validate import PulseValidationError
from experiments.MM_base import MM_base


def _write_sampled_jld2(path):
    '''Minimal valid sampled pulse: 400 ns, tiny amplitudes (well under gain limits).'''
    import h5py
    n = 101
    t = np.linspace(0.0, 400.0, n)            # ns, strictly increasing, > 50 ns floor
    zeros = np.zeros(n)
    small = 1e-3 * np.sin(np.linspace(0, np.pi, n))
    with h5py.File(str(path), "w") as h:
        h["channel_order"] = np.array(
            ["Omega_I_qubit", "Omega_Q_qubit", "epsilon_I_cavity", "epsilon_Q_cavity"],
            dtype="S32")
        h["channel_units"] = np.array(["rad_per_ns"] * 4, dtype="S16")
        h["time_units"] = np.bytes_("ns")
        h["frame"] = np.bytes_("rotating")
        h["times"] = t
        h["Omega_I"] = small
        h["Omega_Q"] = zeros
        h["epsilon_I"] = small
        h["epsilon_Q"] = zeros
        h["target_state"] = np.bytes_("fock1")


def _manifest(group, **entry):
    return {"pulses": [{"dir": group, **entry}]}


# --- build_plan threading ----------------------------------------------------

def test_measure_sigma_z_bool_maps_to_reset_in_isolation():
    # The bool->reset mapping is the lane_layout default when NOTHING sets
    # sigma_z_mode. (In the closed loop, LAB_DEFAULTS sets sigma_z_mode=postselect,
    # which wins -- see test_build_plan_default_is_postselect.)
    cfg = AttrDict({'expt': AttrDict({'measure_sigma_z': True})})
    assert MM_base.lane_layout(cfg)['sigma_z_mode'] == 'reset'


def test_build_plan_explicit_reset_overrides_default(tmp_path):
    # An explicit sigma_z_mode='reset' in the manifest overrides the postselect
    # lab default.
    sp_path = tmp_path / "fock1_sampled.jld2"
    _write_sampled_jld2(sp_path)
    pulse_entry = {"group": "fock1", "sampled_path": str(sp_path)}

    plan = batch_runner.build_plan(
        pulse_entry, manifest=_manifest("fock1", sigma_z_mode="reset"), mode="sim")

    assert plan.validation_error is None
    assert plan.request.knobs.sigma_z_mode == "reset"


def test_build_plan_explicit_postselect(tmp_path):
    sp_path = tmp_path / "fock1_sampled.jld2"
    _write_sampled_jld2(sp_path)
    pulse_entry = {"group": "fock1", "sampled_path": str(sp_path)}

    plan = batch_runner.build_plan(
        pulse_entry,
        manifest=_manifest("fock1", measure_sigma_z=True, sigma_z_mode="postselect"),
        mode="sim")

    assert plan.validation_error is None
    assert plan.request.knobs.sigma_z_mode == "postselect"


def test_build_plan_default_is_measure(tmp_path):
    # No manifest -> LAB_DEFAULTS apply. The closed loop defaults to the
    # non-invasive 'measure' tier and active reset with ef_reset OFF.
    sp_path = tmp_path / "fock1_sampled.jld2"
    _write_sampled_jld2(sp_path)
    pulse_entry = {"group": "fock1", "sampled_path": str(sp_path)}

    plan = batch_runner.build_plan(pulse_entry, manifest=None, mode="sim")

    assert plan.validation_error is None
    assert plan.request.knobs.sigma_z_mode == "measure"
    assert plan.request.knobs.active_reset["active_reset"] is True
    assert plan.request.knobs.active_reset["ef_reset"] is False


def test_build_plan_manifest_overrides_default_mode(tmp_path):
    # A per-pulse manifest sigma_z_mode overrides the LAB_DEFAULTS postselect.
    sp_path = tmp_path / "fock1_sampled.jld2"
    _write_sampled_jld2(sp_path)
    pulse_entry = {"group": "fock1", "sampled_path": str(sp_path)}

    plan = batch_runner.build_plan(
        pulse_entry, manifest=_manifest("fock1", sigma_z_mode="off"), mode="sim")

    assert plan.request.knobs.sigma_z_mode == "off"


# --- validate schema ---------------------------------------------------------

def test_validate_accepts_valid_sigma_z_fields():
    for mode in ("off", "reset", "postselect", "measure"):
        validate._check_manifest_entry({"measure_sigma_z": True, "sigma_z_mode": mode})
    validate._check_manifest_entry({"measure_sigma_z": False})
    validate._check_manifest_entry(None)


def test_validate_accepts_measure_mode():
    # Tier 3 'measure' is supported (non-invasive sigma_z); should NOT raise.
    validate._check_manifest_entry({"sigma_z_mode": "measure"})


def test_validate_rejects_bogus_mode():
    with pytest.raises(PulseValidationError):
        validate._check_manifest_entry({"sigma_z_mode": "bogus"})


def test_validate_rejects_non_bool_measure_sigma_z():
    with pytest.raises(PulseValidationError):
        validate._check_manifest_entry({"measure_sigma_z": "yes"})
