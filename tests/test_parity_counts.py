"""bin_ss_data(return_counts=True) — the raw per-alpha shot tallies the
closed-loop results.json exposes so a collaborator can rebuild each parity point
and a binomial error bar.

The invariant under test: the RAW counts (n_excited / n_total) we hand out must,
after the SAME confusion inversion bin_ss_data applies internally, reproduce the
`pe` (and hence parity) the pipeline itself reports. If they ever diverge, the
error bars Harmoniqs computes would describe a different number than `parity`.
"""
import numpy as np
import pytest
from slab import AttrDict

from fitting.fit_display_classes import GeneralFitting


def _make_fitting(excited_per_expt, reps, threshold, conf_mat):
    """A GeneralFitting over the simple (no active-reset, no sigma_z) lane path:
    read_num=1, idx_final=0, no pre-selection. Builds i0 with exactly
    `excited_per_expt[e]` shots above threshold for alpha e.
    """
    expts = len(excited_per_expt)
    rounds = 1
    hi, lo = threshold + 1.0, threshold - 1.0
    # flat ordering bin_ss_data's else-branch expects: (expts, rounds, reps)
    i_3d = np.full((expts, rounds, reps), lo, dtype=float)
    for e, k in enumerate(excited_per_expt):
        i_3d[e, 0, :k] = hi
    i0 = i_3d.reshape(-1)
    q0 = np.zeros_like(i0)

    cfg = AttrDict({
        "expt": {"reps": reps, "expts": expts, "rounds": rounds,
                 "active_reset": False, "post_select_pre_pulse": False,
                 "pre_selection_reset": False, "pre_selection_parity": False,
                 "parity_shot": False, "sigma_z_mode": "off"},
        "device": {"readout": {"threshold": [threshold],
                               "confusion_matrix_without_reset": list(conf_mat)}},
    })
    return GeneralFitting(data={"i0": i0, "q0": q0}, config=cfg, threshold=threshold)


def test_counts_match_raw_fraction_no_confusion():
    excited = [10, 50, 90]
    reps = 100
    gf = _make_fitting(excited, reps, threshold=0.0, conf_mat=[1.0, 0.0, 0.0, 1.0])
    pe, counts = gf.bin_ss_data(return_counts=True)

    assert counts["n_total"] == [reps, reps, reps]
    assert counts["n_excited"] == excited
    assert counts["threshold"] == 0.0
    assert counts["confusion_matrix"] == [1.0, 0.0, 0.0, 1.0]
    # identity confusion -> pe is exactly the raw excited fraction
    np.testing.assert_allclose(pe, np.array(excited) / reps, atol=1e-12)


def test_counts_reconstruct_corrected_pe():
    """With a non-trivial confusion matrix, applying the same inversion to the
    exposed raw counts must reproduce bin_ss_data's confusion-corrected pe."""
    excited = [12, 47, 188, 200]
    reps = 200
    cm = [0.99, 0.01, 0.02, 0.98]  # [Pgg, Pge, Peg, Pee]
    gf = _make_fitting(excited, reps, threshold=0.0, conf_mat=cm)
    pe, counts = gf.bin_ss_data(return_counts=True)

    P = np.array([[cm[0], cm[2]], [cm[1], cm[3]]])
    Pinv = np.linalg.inv(P)
    for i, (ne, nt) in enumerate(zip(counts["n_excited"], counts["n_total"])):
        pe_raw = ne / nt
        pe_corr = (Pinv @ np.array([1 - pe_raw, pe_raw]))[1]
        assert pe[i] == pytest.approx(pe_corr, abs=1e-9)


def test_return_counts_does_not_disturb_default_callers():
    """Default call (no return_counts) still returns just ydata — the new flag
    must not change the legacy single-return contract used across the codebase."""
    gf = _make_fitting([10, 50, 90], 100, threshold=0.0, conf_mat=[1.0, 0.0, 0.0, 1.0])
    out = gf.bin_ss_data()
    assert isinstance(out, np.ndarray)


# --------------------------------------------------------------------------- #
#  parity_from_counts + bootstrap reconstruction                              #
# --------------------------------------------------------------------------- #
from fitting.wigner import WignerAnalysis, parity_from_counts


def _pe_corr(ne, nt, cm):
    P = np.array([[cm[0], cm[2]], [cm[1], cm[3]]])
    pe_raw = ne / nt
    return (np.linalg.inv(P) @ np.array([1 - pe_raw, pe_raw]))[1]


def test_parity_from_counts_reproduces_reported_parity_pulse_correction():
    """Deterministic parity_from_counts must reproduce exactly the parity the
    experiment computed — same confusion + plus/minus combine + alpha_scale."""
    cm = [0.99, 0.01, 0.02, 0.98]
    pc = {
        "pulse_correction": True, "alpha_scale": 0.97, "threshold": 0.0,
        "confusion_matrix": cm,
        "plus":  {"n_total": [200, 200], "n_excited": [20, 180]},
        "minus": {"n_total": [200, 200], "n_excited": [150, 40]},
    }
    got = parity_from_counts(pc, rng=None)
    exp = []
    for i in range(2):
        pe_p = _pe_corr(pc["plus"]["n_excited"][i],  pc["plus"]["n_total"][i],  cm)
        pe_m = _pe_corr(pc["minus"]["n_excited"][i], pc["minus"]["n_total"][i], cm)
        exp.append(((1 - 2 * pe_m) - (1 - 2 * pe_p)) / 2 / pc["alpha_scale"])
    np.testing.assert_allclose(got, exp, atol=1e-12)


def test_parity_from_counts_reproduces_reported_parity_simple():
    cm = [0.97, 0.03, 0.05, 0.95]
    pc = {"pulse_correction": False, "alpha_scale": 1.0, "threshold": 0.0,
          "confusion_matrix": cm,
          "n_total": [300, 300, 300], "n_excited": [30, 150, 270]}
    got = parity_from_counts(pc, rng=None)
    exp = [1 - 2 * _pe_corr(ne, nt, cm)
           for ne, nt in zip(pc["n_excited"], pc["n_total"])]
    np.testing.assert_allclose(got, exp, atol=1e-12)


def test_bootstrap_resample_varies_but_brackets_point():
    """Bootstrap fidelity CI brackets the point estimate, std > 0, and the
    statistical error shrinks as the shot count grows (~1/sqrt(N))."""
    re = im = np.linspace(-1.5, 1.5, 5)
    alphas = np.array([p + 1j * q for q in im for p in re])
    cm = [1.0, 0.0, 0.0, 1.0]

    def counts(reps):
        # a non-trivial but fixed "true" excited fraction per alpha
        rng = np.random.default_rng(1)
        pe_true = rng.uniform(0.2, 0.8, size=len(alphas))
        ne = np.round(pe_true * reps).astype(int).tolist()
        nt = [reps] * len(alphas)
        return {"pulse_correction": False, "alpha_scale": 1.0, "threshold": 0.0,
                "confusion_matrix": cm, "n_total": nt, "n_excited": ne}

    wa = WignerAnalysis(data={"alpha": alphas}, config=None, threshold=0.0,
                        mode_state_num=2, alphas=alphas)
    import qutip
    psi = qutip.fock(2, 0)

    lo = wa.bootstrap_reconstruction(counts(200),  psi, n_boot=200, seed=0)
    hi = wa.bootstrap_reconstruction(counts(5000), psi, n_boot=200, seed=0)

    assert lo["fidelity_std"] > 0
    assert lo["fidelity_ci"][0] <= lo["fidelity_mean"] <= lo["fidelity_ci"][1]
    # 25x more shots -> ~5x tighter; allow generous margin
    assert hi["fidelity_std"] < lo["fidelity_std"]


def test_bootstrap_reports_rho_abs_std_and_plots():
    """Bootstrap exposes per-element |rho| std (m x m, >=0), and the annotated
    plot renders without error when handed the uncertainty dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import qutip

    re = im = np.linspace(-1.5, 1.5, 5)
    alphas = np.array([p + 1j * q for q in im for p in re])
    rng = np.random.default_rng(2)
    ne = np.round(rng.uniform(0.2, 0.8, size=len(alphas)) * 400).astype(int).tolist()
    pc = {"pulse_correction": False, "alpha_scale": 1.0, "threshold": 0.0,
          "confusion_matrix": [1.0, 0.0, 0.0, 1.0],
          "n_total": [400] * len(alphas), "n_excited": ne}

    wa = WignerAnalysis(data={"alpha": alphas}, config=None, threshold=0.0,
                        mode_state_num=3, alphas=alphas)
    boot = wa.bootstrap_reconstruction(pc, qutip.fock(3, 0), n_boot=100, seed=0)

    assert boot["rho_abs_std"].shape == (3, 3)
    assert np.all(boot["rho_abs_std"] >= 0)
    fig = wa.plot_wigner_reconstruction_results(
        boot["point_results"], initial_state=qutip.fock(3, 0), uncertainty=boot)
    plt.close(fig)


def test_bootstrap_requires_counts():
    wa = WignerAnalysis(data={"alpha": np.array([0 + 0j])}, config=None,
                        threshold=0.0, mode_state_num=2, alphas=np.array([0 + 0j]))
    import qutip
    with pytest.raises(ValueError):
        wa.bootstrap_reconstruction(None, qutip.fock(2, 0), n_boot=10)
