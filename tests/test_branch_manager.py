"""Tests for BranchManager (experiments/branch_manager.py).

BranchManager is a local, append-only reflog mapping friendly branch/tag names to
the 4 station config version IDs. Its only real logic is log fold/reduce, the
tag/dirty guards, and the cheap content fingerprint -- none of which need real
hardware. We exercise it against a minimal FakeStation that exposes just the
surface BranchManager touches, keeping these runnable in any env with the repo +
pandas (matching tests/test_mock_mode.py's no-real-station approach).
"""

import pandas as pd
import pytest

from experiments.branch_manager import BranchManager


class FakeStation:
    """Minimal stand-in exposing only what BranchManager touches."""

    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.hardware_cfg = {"device": {"x": 1}}
        self.multimode_cfg = {"mp": 1}
        self.ds_storage = type("DS", (), {"df": pd.DataFrame({"a": [1, 2]})})()
        self.ds_floquet = type("DS", (), {"df": pd.DataFrame({"b": [3]})})()
        self._n = 0

    def _get_sanitized_config_copy(self, cfg):
        return cfg  # nothing non-serializable in the fake

    def update_all_station_snapshots(self, update_main=False):
        # Deterministic IDs that advance each call, so commits are distinguishable.
        self._n += 1
        return {
            "hardware_config": f"CFG-HW-{self._n:05d}",
            "multiphoton_config": f"CFG-MP-{self._n:05d}",
            "man1_storage_swap": f"CFG-M1-{self._n:05d}",
            "floquet_storage_swap": f"CFG-FL-{self._n:05d}",
        }

    def _initialize_configs(self, hw, mp, m1, fl):
        # Simulate a reload by stamping the loaded ids into the in-RAM cfg.
        self.hardware_cfg = {"device": {"loaded": hw}}


@pytest.fixture
def bm(tmp_path):
    # Explicit log_path keeps the test hermetic (the default would write to CWD).
    return BranchManager(FakeStation(tmp_path), log_path=tmp_path / "branches.jsonl")


def test_commit_advances_and_records(bm):
    ids = bm.commit("user1_coupler0.1", note="baseline")
    assert ids["hardware_config"] == "CFG-HW-00001"
    assert bm.show("user1_coupler0.1")["hardware_config"] == "CFG-HW-00001"


def test_dirty_detection_tracks_inram_edits(bm):
    bm.commit("b", note="baseline")
    assert not bm._is_dirty()
    bm.station.hardware_cfg["device"]["x"] = 999
    assert bm._is_dirty()
    bm.commit("b", note="after edit")  # re-snapshot clears dirty
    assert not bm._is_dirty()


def test_history_is_append_only(bm):
    bm.commit("b", note="baseline")
    bm.station.hardware_cfg["device"]["x"] = 2
    bm.commit("b", note="after retune")
    hist = bm.log("b")
    assert [h["note"] for h in hist] == ["baseline", "after retune"]
    assert hist[0]["ids"]["hardware_config"] == "CFG-HW-00001"
    assert hist[1]["ids"]["hardware_config"] == "CFG-HW-00002"


def test_branch_records_lineage(bm):
    bm.commit("parent")
    bm.branch("paper_v1", from_name="parent")
    assert bm.show("paper_v1")["hardware_config"] == "CFG-HW-00001"
    assert bm.log("paper_v1")[0]["from"] == "parent"


def test_branch_defaults_to_loaded_state(bm):
    bm.commit("parent")
    bm.branch("child")  # no from_name -> currently-loaded ids
    assert bm.show("child")["hardware_config"] == bm.show("parent")["hardware_config"]
    assert bm.log("child")[0]["from"] == "parent"


def test_tag_is_frozen_against_commit(bm):
    bm.commit("b")
    bm.tag("frozen_pt")
    with pytest.raises(ValueError):
        bm.commit("frozen_pt")


def test_checkout_dirty_guard_and_reload(bm):
    bm.commit("a")
    bm.branch("paper_v1")
    bm.station.hardware_cfg["device"]["x"] = 7  # make dirty
    with pytest.raises(RuntimeError):
        bm.checkout("paper_v1")
    bm.checkout("paper_v1", force=True)
    assert bm.station.hardware_cfg["device"]["loaded"] == "CFG-HW-00001"


def test_checkout_unknown_name_raises(bm):
    with pytest.raises(KeyError):
        bm.checkout("nope")


def test_delete_tombstones_but_keeps_history(bm):
    bm.commit("b")
    bm.tag("frozen_pt")
    bm.delete("frozen_pt")
    assert "frozen_pt" not in bm.list()
    assert any(e["branch"] == "frozen_pt" for e in bm.log())


def test_state_survives_reopen(bm):
    bm.commit("user1_coupler0.1")
    bm.branch("paper_v1")
    reopened = BranchManager(bm.station, log_path=bm.log_path)  # same log file on disk
    assert set(reopened.list()) == {"user1_coupler0.1", "paper_v1"}


def test_torn_final_line_is_ignored(bm):
    bm.commit("b")
    with bm.log_path.open("a", encoding="utf-8") as f:
        f.write('{"branch": "partial", incomplete')  # no newline, invalid JSON
    assert "b" in bm.list()  # reduce tolerates the torn tail
    assert "partial" not in bm.list()
