'''
Offline validation of the closed-loop Wigner reconstruction added to
``job_server/closed_loop/core.py`` -- no hardware, no queue.

Checks:
  * _reconstruct_from_parity round-trips a known Fock state: feeding the ideal
    parity (forward model = WignerAnalysis.extracted_W_single_analytic) back
    through the reconstruction recovers fidelity ~1 and the right populations,
  * fidelity is None when no target_state ket is supplied,
  * _build_target_ket parses fock / ket specs and rejects malformed ones,
  * run_wigner_core in sim mode with knobs.reconstruct=True returns rho /
    populations / fidelity end-to-end (sim path needs no init_core).
'''

import numpy as np
import pytest
import qutip

from fitting.wigner import WignerAnalysis
from job_server.closed_loop.core import (
    _build_target_ket,
    _reconstruct_from_parity,
    run_wigner_core,
    IQTable,
    Knobs,
    RunWignerRequest,
)

FOCK_DIM = 5


def _grid_alphas(extent=2.2, n=11):
    """Square phase-space grid of complex displacements (n*n points)."""
    axis = np.linspace(-extent, extent, n)
    re, im = np.meshgrid(axis, axis)
    return (re + 1j * im).ravel()


def _ideal_parity(rho, alphas_c, fock_dim):
    """Forward model consistent with the reconstruction.

    allocated_readout = 2/pi * parity = W(alpha) = Tr(W_op(alpha) @ rho), so the
    'parity' the pipeline consumes is pi/2 * W(alpha).
    """
    wa = WignerAnalysis(data={"alpha": alphas_c}, threshold=0.0, config=None,
                        mode_state_num=fock_dim, alphas=alphas_c)
    W = wa.extracted_W_single_analytic(rho, alphas_c, fock_dim)
    return np.real(np.pi / 2.0 * np.asarray(W))


# --- _reconstruct_from_parity round-trip -------------------------------------

def test_reconstruct_recovers_fock_one():
    alphas_c = _grid_alphas()
    target = qutip.fock(FOCK_DIM, 1)
    rho_target = qutip.ket2dm(target).full()
    parity = _ideal_parity(rho_target, alphas_c, FOCK_DIM)

    out = _reconstruct_from_parity(parity, alphas_c, FOCK_DIM, target, rotate=False)

    assert out["fidelity"] > 0.999
    pops = out["populations"]
    assert np.argmax(pops) == 1
    assert pops[1] > 0.99
    # rho serialized as JSON-safe real/imag nested lists of the right shape
    assert np.array(out["rho"]["real"]).shape == (FOCK_DIM, FOCK_DIM)
    assert np.array(out["rho"]["imag"]).shape == (FOCK_DIM, FOCK_DIM)


def test_reconstruct_fidelity_none_without_target():
    alphas_c = _grid_alphas()
    rho_target = qutip.ket2dm(qutip.fock(FOCK_DIM, 0)).full()
    parity = _ideal_parity(rho_target, alphas_c, FOCK_DIM)

    out = _reconstruct_from_parity(parity, alphas_c, FOCK_DIM, None, rotate=False)

    assert out["fidelity"] is None
    assert np.argmax(out["populations"]) == 0


# --- _build_target_ket -------------------------------------------------------

def test_build_target_ket_fock():
    ket = _build_target_ket({"type": "fock", "n": 2}, FOCK_DIM)
    assert ket == qutip.fock(FOCK_DIM, 2)


def test_build_target_ket_ket_normalizes():
    amps = [[1.0, 0.0], [1.0, 0.0]] + [[0.0, 0.0]] * (FOCK_DIM - 2)
    ket = _build_target_ket({"type": "ket", "amps": amps}, FOCK_DIM)
    assert abs(ket.norm() - 1.0) < 1e-9


def test_build_target_ket_none():
    assert _build_target_ket(None, FOCK_DIM) is None


@pytest.mark.parametrize("spec", [
    {"type": "fock", "n": FOCK_DIM},          # out of range
    {"type": "fock"},                          # missing n
    {"type": "ket", "amps": [[1.0, 0.0]]},     # wrong length
    {"type": "bogus"},                         # unknown type
    {"n": 1},                                  # missing type
])
def test_build_target_ket_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        _build_target_ket(spec, FOCK_DIM)


# --- run_wigner_core sim end-to-end ------------------------------------------

def _flat_iqtable(n=8):
    z = [0.0] * n
    return IQTable(times=list(np.linspace(0, 0.05, n)), I_c=z, Q_c=z, I_q=z, Q_q=z)


def test_run_wigner_core_sim_reconstructs():
    alphas = [[float(a.real), float(a.imag)] for a in _grid_alphas()]
    req = RunWignerRequest(
        mode="sim",
        IQ_table=_flat_iqtable(),
        alphas=alphas,
        reps=200000,                       # large -> sim parity noise negligible
        sim_target_beta=[0.0, 0.0],        # vacuum: parity = exp(-2|alpha|^2)
        target_state={"type": "fock", "n": 0},
        knobs=Knobs(reconstruct=True, reconstruct_fock_dim=FOCK_DIM),
    )
    resp = run_wigner_core(req)

    assert resp.rho is not None
    assert resp.populations is not None
    assert resp.fidelity is not None and resp.fidelity > 0.97
    assert np.argmax(resp.populations) == 0


def test_run_wigner_core_sim_no_reconstruct_by_default():
    alphas = [[0.0, 0.0], [0.5, 0.0]]
    req = RunWignerRequest(
        mode="sim", IQ_table=_flat_iqtable(), alphas=alphas,
        sim_target_beta=[0.0, 0.0],
    )
    resp = run_wigner_core(req)
    assert resp.rho is None
    assert resp.fidelity is None


# --- watcher (batch_runner) decision path ------------------------------------

from job_server.closed_loop import batch_runner


def _write_sampled_jld2(path, target_state="fock1"):
    '''Minimal valid sampled pulse per the Harmoniqs contract (no hardware).'''
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


@pytest.mark.parametrize("label,expected", [
    ("fock_1", {"type": "fock", "n": 1}),
    ("fock 2", {"type": "fock", "n": 2}),
    ("|3>", {"type": "fock", "n": 3}),
    ("n=0", {"type": "fock", "n": 0}),
    ("4", {"type": "fock", "n": 4}),
    ("cat_alpha2", None),
    (None, None),
])
def test_target_state_from_label(label, expected):
    assert batch_runner._target_state_from_label(label) == expected


def test_build_plan_reconstruct_off_by_default(tmp_path):
    sp = tmp_path / "fock1_sampled.jld2"
    _write_sampled_jld2(sp)
    plan = batch_runner.build_plan(
        {"group": "fock1", "sampled_path": str(sp)}, manifest=None, mode="sim")
    assert plan.validation_error is None
    assert plan.request.knobs.reconstruct is False
    assert plan.request.target_state is None


def test_build_plan_reconstruct_autotarget_from_meta(tmp_path):
    sp = tmp_path / "fock1_sampled.jld2"
    _write_sampled_jld2(sp, target_state="fock1")
    plan = batch_runner.build_plan(
        {"group": "fock1", "sampled_path": str(sp)},
        manifest=_manifest("fock1", reconstruct=True), mode="sim")
    assert plan.request.knobs.reconstruct is True
    assert plan.request.target_state == {"type": "fock", "n": 1}


def test_build_plan_reconstruct_manifest_target_overrides(tmp_path):
    sp = tmp_path / "fock1_sampled.jld2"
    _write_sampled_jld2(sp, target_state="fock1")
    explicit = {"type": "fock", "n": 2}
    plan = batch_runner.build_plan(
        {"group": "fock1", "sampled_path": str(sp)},
        manifest=_manifest("fock1", reconstruct=True, target_state=explicit),
        mode="sim")
    assert plan.request.target_state == explicit


def test_build_plan_reconstruct_noncfock_target_is_none(tmp_path):
    sp = tmp_path / "cat_sampled.jld2"
    _write_sampled_jld2(sp, target_state="cat_alpha2")
    plan = batch_runner.build_plan(
        {"group": "cat", "sampled_path": str(sp)},
        manifest=_manifest("cat", reconstruct=True), mode="sim")
    assert plan.request.knobs.reconstruct is True
    assert plan.request.target_state is None
