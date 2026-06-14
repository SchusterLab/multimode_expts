"""
Unit tests for CharacterizationRunner._render_log_show: the decoupled
show/log routing and display_kwargs forwarding (design B).

We bypass __init__ (object.__new__) and drive the helper directly with a fake
station + fake experiment, so no hardware / job server is needed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.characterization_runner import CharacterizationRunner


class FakeExpt:
    def __init__(self):
        self.calls = []

    # display() accepts only these named kwargs (so unknown ones must be filtered)
    def display(self, initial_state=None, rotate=None, station=None, state_label=""):
        self.calls.append(dict(initial_state=initial_state, rotate=rotate,
                               station=station, state_label=state_label))
        fig = plt.figure()
        return fig


class FakeStation:
    def __init__(self, log_measurements=False, is_mock=False):
        self.log_measurements = log_measurements
        self.is_mock = is_mock
        self.logged = []

    def log_measurement(self, experiment, fig=None):
        self.logged.append((experiment, fig))


def _runner(station, show=True):
    r = object.__new__(CharacterizationRunner)   # skip __init__ (no hardware)
    r.station = station
    r.show = show
    return r


def test_show_and_log_are_independent():
    # log only
    st = FakeStation()
    r = _runner(st)
    e = FakeExpt()
    r._render_log_show(e, show=False, log=True, display_kwargs={})
    assert len(e.calls) == 1            # rendered (needed for the vault fig)
    assert len(st.logged) == 1          # logged
    # show only
    st2 = FakeStation()
    e2 = FakeExpt()
    _runner(st2)._render_log_show(e2, show=True, log=False, display_kwargs={})
    assert len(e2.calls) == 1
    assert len(st2.logged) == 0         # NOT logged
    # neither -> no render at all
    st3 = FakeStation()
    e3 = FakeExpt()
    _runner(st3)._render_log_show(e3, show=False, log=False, display_kwargs={})
    assert len(e3.calls) == 0
    assert len(st3.logged) == 0


def test_show_default_falls_back_to_runner_show():
    st = FakeStation(log_measurements=False)
    e = FakeExpt()
    _runner(st, show=True)._render_log_show(e, show=None, log=False, display_kwargs={})
    assert len(e.calls) == 1            # show=None -> runner default True -> rendered
    e2 = FakeExpt()
    _runner(FakeStation(), show=False)._render_log_show(e2, show=None, log=False, display_kwargs={})
    assert len(e2.calls) == 0           # runner default False -> nothing


def test_log_none_defers_to_station_flag():
    # log=None + log_measurements=True -> logs
    st = FakeStation(log_measurements=True)
    e = FakeExpt()
    _runner(st, show=False)._render_log_show(e, show=False, log=None, display_kwargs={})
    assert len(st.logged) == 1
    # log=None + log_measurements=False -> no log (and show=False -> no render)
    st2 = FakeStation(log_measurements=False)
    e2 = FakeExpt()
    _runner(st2, show=False)._render_log_show(e2, show=False, log=None, display_kwargs={})
    assert len(st2.logged) == 0
    assert len(e2.calls) == 0


def test_display_kwargs_forwarded_and_filtered():
    st = FakeStation()
    e = FakeExpt()
    _runner(st)._render_log_show(
        e, show=True, log=False,
        display_kwargs=dict(initial_state="RHO", rotate="optimal",
                            state_label="|+_L>", bogus=123),  # bogus must be dropped
    )
    assert e.calls[0]["initial_state"] == "RHO"
    assert e.calls[0]["rotate"] == "optimal"
    assert e.calls[0]["state_label"] == "|+_L>"
    assert "bogus" not in e.calls[0]       # filtered to the display signature
    assert e.calls[0]["station"] is st     # station injected (display accepts it)


def test_mock_skips_render_and_log():
    st = FakeStation(log_measurements=True, is_mock=True)
    e = FakeExpt()
    _runner(st, show=True)._render_log_show(e, show=True, log=True, display_kwargs={})
    assert len(e.calls) == 0               # mock: no render
    assert len(st.logged) == 0             # mock: no vault write
