"""The three views of fitting.qsim.poles.views draw any fitter on a small synthetic spectrum."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fitting.qsim.poles import joint_pencil, joint_refined
from fitting.qsim.poles.real_benchmarks import RealSpectrum
from fitting.qsim.poles.views import (display_gap_ratio_cdfs, display_gap_ratio_histograms, display_row_heatmaps,
                                      display_sticks, goe_density, poisson_density)

TIME_US = np.arange(200) * 1.0
LEVELS_MHz = np.array([-0.05, -0.02, 0.01, 0.03, 0.06])
WEIGHTS = np.random.default_rng(0).dirichlet(np.ones(5), size=4)


def spectrum():
    rng = np.random.default_rng(1)
    A = WEIGHTS @ np.exp(np.outer(-0.01 - 2j * np.pi * LEVELS_MHz, TIME_US))
    A = A + 0.005 * (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape))
    return RealSpectrum("toy", "toy", False, TIME_US, A, LEVELS_MHz, np.ones(5, dtype=int), WEIGHTS,
                        occupations=((1, 0), (0, 1), (2, 0), (0, 2)))


def test_the_reference_densities_are_normalized():
    r = np.linspace(0, 1, 20001)
    for density in (poisson_density, goe_density):
        assert abs(np.trapz(density(r), r) - 1) < 1e-3


def test_every_view_draws_any_fitter():
    s = spectrum()
    fits = {"B": joint_pencil.fit(s.A, s.time_us), "C": joint_refined.fit(s.A, s.time_us)}
    assert len(display_row_heatmaps(s, fits).axes) == 2
    assert len(display_sticks(s, fits).axes) == 2
    levels = {name: [fit.frequencies_MHz] * 3 for name, fit in fits.items()}
    reference = np.random.default_rng(2).uniform(size=500)
    assert len([ax for ax in display_gap_ratio_histograms(levels, [LEVELS_MHz] * 3, reference).axes if ax.get_visible()]) == 3
    assert len(display_gap_ratio_cdfs(levels, [LEVELS_MHz] * 3, reference).axes) == 1
    plt.close("all")
