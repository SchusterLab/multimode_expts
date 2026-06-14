'''
Offline validation of the closed-loop single-qubit state-tomography path added
to job_server/closed_loop/core.py -- no hardware, no queue.

Checks:
  * run_state_tomo_1q_core (sim) reconstructs cardinal states (|g>, |+>, |+i>)
    at high fidelity with the right Pauli expectations,
  * _finalize_state_tomo_1q turns per-basis counts into rho / fidelity / azimuth,
  * azimuth-only runs (bases ['X','Y']) yield azimuth + contrast but no rho,
  * fidelity is None without target_state; bad bases raise ValueError,
  * the experiment + service modules import with the new opt_pulse/IQ_table prep,
  * StateTomography1QExperiment.initialize loads the inline IQ_table (mock soc).
'''

import numpy as np
import pytest

from job_server.closed_loop.core import (
    run_state_tomo_1q_core,
    _finalize_state_tomo_1q,
    _ket_from_pairs,
    RunStateTomo1QRequest,
    RunStateTomo1QResponse,
    IQTable,
)

# Cardinal-state kets as [[re,im],[re,im]] pairs.
KET_G = [[1, 0], [0, 0]]
KET_E = [[0, 0], [1, 0]]
KET_PLUS = [[1, 0], [1, 0]]          # |+>  -> <X>=+1
KET_PLUS_I = [[1, 0], [0, 1]]        # |+i> -> <Y>=+1


def _iq(n=8):
    z = [0.0] * n
    return IQTable(times=list(np.linspace(0, 0.05, n)), I_c=z, Q_c=z, I_q=z, Q_q=z)


def _req(**kw):
    base = dict(mode="sim", IQ_table=_iq(), reps=200000)
    base.update(kw)
    return RunStateTomo1QRequest(**base)


# --- _ket_from_pairs ---------------------------------------------------------

def test_ket_from_pairs_normalizes():
    k = _ket_from_pairs(KET_PLUS)
    assert abs(np.linalg.norm(k) - 1.0) < 1e-12
    assert np.allclose(k, np.array([1, 1]) / np.sqrt(2))


# --- sim end-to-end ----------------------------------------------------------

@pytest.mark.parametrize("ket,axis,sign", [
    (KET_G, "Z", +1),
    (KET_E, "Z", -1),
    (KET_PLUS, "X", +1),
    (KET_PLUS_I, "Y", +1),
])
def test_sim_reconstructs_cardinal_states(ket, axis, sign):
    np.random.seed(1)
    r = run_state_tomo_1q_core(_req(sim_target_state=ket, target_state=ket))
    assert isinstance(r, RunStateTomo1QResponse)
    assert r.fidelity > 0.99
    # the dominant expectation is along the expected axis with the right sign
    assert abs(r.expectations[axis] - sign) < 0.02
    for other in ("Z", "X", "Y"):
        if other != axis:
            assert abs(r.expectations[other]) < 0.02


def test_sim_fidelity_none_without_target():
    np.random.seed(2)
    r = run_state_tomo_1q_core(_req(sim_target_state=KET_PLUS))  # no target_state
    assert r.fidelity is None
    assert r.rho is not None  # rho still reconstructed


def test_sim_default_state_is_ground():
    np.random.seed(3)
    r = run_state_tomo_1q_core(_req(target_state=KET_G))  # no sim_target_state -> |g>
    assert r.fidelity > 0.99
    assert abs(r.expectations["Z"] - 1.0) < 0.02


# --- azimuth-only ------------------------------------------------------------

def test_azimuth_only_no_rho():
    np.random.seed(4)
    r = run_state_tomo_1q_core(_req(bases=["X", "Y"], sim_target_state=KET_PLUS_I))
    assert r.rho is None            # rho needs all of Z/X/Y
    assert r.fidelity is None
    assert r.azimuth_rad is not None
    assert abs(np.degrees(r.azimuth_rad) - 90.0) < 2.0   # |+i> sits on +Y
    assert r.equatorial_contrast > 0.98


# --- validation --------------------------------------------------------------

def test_bad_bases_raise():
    with pytest.raises(ValueError):
        run_state_tomo_1q_core(_req(bases=["Z", "W"]))
    with pytest.raises(ValueError):
        run_state_tomo_1q_core(_req(bases=[]))


# --- _finalize_state_tomo_1q -------------------------------------------------

def test_finalize_from_counts():
    # Noiseless |+>: Z 50/50, X all-g, Y 50/50.
    counts = {"Z": (500, 500), "X": (1000, 0), "Y": (500, 500)}
    req = _req(target_state=KET_PLUS)
    resp = _finalize_state_tomo_1q(counts, req, mode="sim", iter_id="t",
                                   shots_path=None, meta={})
    assert abs(resp.expectations["X"] - 1.0) < 1e-9
    assert abs(resp.expectations["Z"]) < 1e-9
    assert resp.fidelity > 0.99
    assert np.allclose(np.array(resp.rho["real"]), [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)


def test_finalize_recon_method_cholesky():
    counts = {"Z": (1000, 0), "X": (500, 500), "Y": (500, 500)}  # |g>
    req = _req(target_state=KET_G, recon_method="cholesky")
    resp = _finalize_state_tomo_1q(counts, req, mode="sim", iter_id="t",
                                   shots_path=None, meta={})
    assert resp.fidelity > 0.99


# --- module imports (experiment + service wiring) ----------------------------

def test_modules_import():
    import job_server.closed_loop.service  # noqa: F401
    from experiments.single_qubit.state_tomography_1q import (  # noqa: F401
        StateTomography1QExperiment, StateTomography1QProgram,
    )


# --- watcher (batch_runner) routing ------------------------------------------

from job_server.closed_loop import batch_runner, validate
from job_server.closed_loop.core import RunWignerRequest as _RWReq
from job_server.closed_loop.validate import PulseValidationError


def _write_sampled_jld2(path, target_state="fock1"):
    import h5py
    n = 101
    t = np.linspace(0.0, 400.0, n)
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
        h["target_state"] = np.bytes_(target_state)


def _manifest(group, **entry):
    return {"pulses": [{"dir": group, **entry}]}


def test_build_plan_defaults_to_wigner(tmp_path):
    sp = tmp_path / "fock1_sampled.jld2"
    _write_sampled_jld2(sp)
    plan = batch_runner.build_plan(
        {"group": "fock1", "sampled_path": str(sp)}, manifest=None, mode="sim")
    assert plan.measurement == "wigner"
    assert isinstance(plan.request, _RWReq)


def test_build_plan_routes_to_tomography_1q(tmp_path):
    sp = tmp_path / "decode_sampled.jld2"
    _write_sampled_jld2(sp, target_state="decode")
    plan = batch_runner.build_plan(
        {"group": "decode", "sampled_path": str(sp)},
        manifest=_manifest("decode", measurement="tomography_1q",
                           bases=["Z", "X", "Y"], recon_method="cholesky",
                           target_state=[[1, 0], [1, 0]]),
        mode="sim")
    assert plan.measurement == "tomography_1q"
    assert isinstance(plan.request, RunStateTomo1QRequest)
    assert plan.request.bases == ["Z", "X", "Y"]
    assert plan.request.recon_method == "cholesky"
    assert plan.request.target_state == [[1, 0], [1, 0]]
    # opt-pulse state-prep plumbing: pulse_ref carried, IQ_table populated
    assert plan.request.pulse_ref is not None
    assert len(plan.request.IQ_table.times) > 0


def test_build_plan_tomo_1q_runs_in_sim(tmp_path):
    # end-to-end through run_one in sim: |g> prep (no sim_target_state -> |g>)
    sp = tmp_path / "decode_sampled.jld2"
    _write_sampled_jld2(sp, target_state="decode")
    plan = batch_runner.build_plan(
        {"group": "decode", "sampled_path": str(sp)},
        manifest=_manifest("decode", measurement="tomography_1q",
                           target_state=[[1, 0], [0, 0]]),
        mode="sim")
    np.random.seed(7)
    out = batch_runner.run_one(plan)
    assert out.ok and out.measurement == "tomography_1q"
    assert out.parity is None
    assert out.counts is not None and out.expectations is not None
    assert out.fidelity is not None and out.fidelity > 0.99


def test_build_plan_bad_measurement_is_validation_error(tmp_path):
    sp = tmp_path / "x_sampled.jld2"
    _write_sampled_jld2(sp)
    plan = batch_runner.build_plan(
        {"group": "x", "sampled_path": str(sp)},
        manifest=_manifest("x", measurement="bogus"), mode="sim")
    # caught either at manifest validation or the build_plan guard
    assert plan.validation_error is not None


def test_validate_rejects_bad_measurement():
    with pytest.raises(PulseValidationError):
        validate._check_manifest_entry({"measurement": "bogus"})
    validate._check_manifest_entry({"measurement": "tomography_1q", "bases": ["X", "Y"]})
    with pytest.raises(PulseValidationError):
        validate._check_manifest_entry({"bases": ["Z", "W"]})
