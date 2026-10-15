"""The worker reconnects the coupler yoko once when its socket is dropped (WinError 10053),
and puts it in current mode with the output on before the ramp."""

from types import SimpleNamespace

import pytest

from job_server.worker import JobWorker


class FlakyYoko:
    def __init__(self, failures=0, mode="CURR", empty_replies=0):
        self.failures = failures
        self.mode = mode
        self.empty_replies = empty_replies
        self.output = False
        self.reconnects = 0
        self.ramped_to = None
        self.socket = SimpleNamespace(close=lambda: None)

    def get_mode(self):
        if self.empty_replies > 0:
            self.empty_replies -= 1
            return ""
        return self.mode

    def set_mode(self, mode):
        self.mode = mode

    def set_output(self, state=True):
        self.output = state

    def ramp_current(self, current, sweeprate):
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionAbortedError(10053, "connection aborted")
        assert self.mode == "CURR" and self.output
        self.ramped_to = current

    def on_enable(self):
        self.reconnects += 1


def _worker_with(yoko):
    worker = JobWorker.__new__(JobWorker)
    worker.station = SimpleNamespace(yoko_coupler=yoko)
    return worker


def test_reconnects_once_and_ramps():
    yoko = FlakyYoko(failures=1)
    _worker_with(yoko)._ramp_coupler(3e-4)
    assert yoko.reconnects == 1
    assert yoko.ramped_to == 3e-4


def test_second_failure_raises():
    yoko = FlakyYoko(failures=2)
    with pytest.raises(ConnectionAbortedError):
        _worker_with(yoko)._ramp_coupler(3e-4)
    assert yoko.reconnects == 1


def test_switches_to_current_mode_and_output_on():
    yoko = FlakyYoko(mode="VOLT")
    _worker_with(yoko)._ramp_coupler(3e-4)
    assert yoko.mode == "CURR" and yoko.output
    assert yoko.reconnects == 0
    assert yoko.ramped_to == 3e-4


def test_empty_mode_reply_reconnects_without_mode_switch():
    yoko = FlakyYoko(empty_replies=1)
    _worker_with(yoko)._ramp_coupler(3e-4)
    assert yoko.reconnects == 1
    assert yoko.ramped_to == 3e-4
