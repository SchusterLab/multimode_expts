"""fitting/plot_style.py: the lab style applies inside the context only, and the named colors are
fixed slots of the categorical palette."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from fitting import plot_style  # noqa: E402


def test_style_applies_inside_the_context_only():
    before = dict(plt.rcParams)
    with plot_style.context():
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        assert to_hex(fig.get_facecolor()) == plot_style.SURFACE
        assert to_hex(line.get_color()) == plot_style.CATEGORICAL[0]
        assert not ax.spines["top"].get_visible()
        assert plt.rcParams["font.size"] == 9
    plt.close(fig)
    assert {k: v for k, v in plt.rcParams.items() if k in plot_style.RC} == \
           {k: v for k, v in before.items() if k in plot_style.RC}


def test_named_colors_are_fixed_slots():
    assert plot_style.STATE_COLORS == dict(g=plot_style.CATEGORICAL[0], e=plot_style.CATEGORICAL[1],
                                           f=plot_style.CATEGORICAL[2])
    assert list(plot_style.MODE_COLORS) == ["M1"] + [f"S{k}" for k in range(1, 8)]
    assert len(set(plot_style.MODE_COLORS.values())) == 8
