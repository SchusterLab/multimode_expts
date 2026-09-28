"""Benchmark 1 (docs/qsim/pole_finding.md 5.1): noise-free returns over the phase diagram.

Fitter B must find every distinct level to 1e-6 bin and every weight to 1e-6 where the
time grid can tell the levels apart at all (Vandermonde conditioning >= 1e-2; below it
no fitter reaches 1e-6 in double precision). The grid is 400 samples: on the measured
100 samples almost no point of the plane is that well conditioned (35 levels in about
25 bins), so this checks the code, not the experiment. Fitters A and E are scored but not
held to it: A's known defects are strict xfails in tests/test_matrix_pencil_synthetic.py,
and E is held to 0.1 bin on separated levels in tests/test_poles.py.
"""
import pytest

from fitting.qsim.poles.benchmarks import FITTERS, ideal_pass, run_ideal_bench
from fitting.qsim.poles.synthetic import Hardware, sample_phase_diagram

WELL_CONDITIONED = 1e-2
POINTS = sample_phase_diagram([0., -1., -3., -10.], [0., 0.3, 1., 3., 10.], draws=2)


@pytest.fixture(scope="module")
def bench():
    return run_ideal_bench(POINTS, Hardware(samples=400), {"B": FITTERS["B"]})


def test_joint_pencil_is_exact_where_the_grid_resolves_the_levels(bench):
    held = [s for s in bench.scores if s.conditioning >= WELL_CONDITIONED]
    assert len(held) >= 20
    failed = [(s.kerr_over_g, s.disorder_over_g, s.draw) for s in held if not ideal_pass(s, 1e-6, 1e-6)]
    assert not failed
