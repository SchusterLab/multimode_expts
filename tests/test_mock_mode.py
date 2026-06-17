"""Tests for the mock-instruments refactor (see docs/mock_mode_architecture_plan.md).

These exercise the small surface that's intended to be stable across the
refactor: the shapes/types MockQickSoc returns, the basic semantics of the
no-op stubs, and the MultimodeStation API exposed by the design. We do NOT
instantiate a real MultimodeStation here (that requires the Pyro proxy and
config DB), only introspect its class — keeping these tests runnable in any
env that has the repo on disk + numpy.
"""

import inspect

import numpy as np
import pytest


# ---------- MockQickSoc shape & type contract ----------

@pytest.fixture
def fresh_soc():
    from experiments.mock_hardware import MockQickSoc
    return MockQickSoc()


def test_poll_data_shape_and_dtype(fresh_soc):
    # qick's AcquireMixin.acquire loop asserts:
    #   d[ii].shape[0] == new_points * reads_per_shot[ii]
    # and the buffer it writes into is int64.
    fresh_soc.start_readout(total_count=7, counter_addr=1, ch_list=[0, 1],
                            reads_per_shot=[1, 3])
    result = fresh_soc.poll_data()
    assert isinstance(result, list) and len(result) == 1
    n, (data, stats) = result[0]
    assert n == 7
    assert len(data) == 2
    assert data[0].shape == (7 * 1, 2)
    assert data[1].shape == (7 * 3, 2)
    assert data[0].dtype == np.int64
    assert data[1].dtype == np.int64
    assert np.all(data[0] == 0)
    assert np.all(data[1] == 0)
    assert stats == {}


def test_get_decimated_and_accumulated_shapes(fresh_soc):
    # acquire_decimated does: dec_buf[ii] += get_decimated(...)
    # then get_accumulated(...).reshape((*loop_dims, trigs, 2))
    dec = fresh_soc.get_decimated(ch=0, address=0, length=100)
    assert dec.shape == (100, 2)
    assert np.all(dec == 0)

    acc = fresh_soc.get_accumulated(ch=0, address=0, length=50)
    assert acc.shape == (50, 2)
    assert np.all(acc == 0)


def test_get_tproc_counter_returns_total_count(fresh_soc):
    # Polling loops in acquire_decimated / run_rounds exit when this >= total_count.
    fresh_soc.start_readout(total_count=42, counter_addr=1, ch_list=[0],
                            reads_per_shot=[1])
    assert fresh_soc.get_tproc_counter(addr=1) == 42


def test_all_required_soc_methods_present(fresh_soc):
    # Any qick library version that adds calls beyond this set should make
    # the smoke test fail loudly, not silently no-op.
    required = [
        "start_src", "stop_tproc", "reload_mem",
        "load_pulse_data", "set_nyquist", "set_mixer_freq",
        "config_mux_gen", "configure_readout", "config_mux_readout",
        "load_bin_program", "config_avg", "config_buf",
        "start_readout", "poll_data",
        "start_tproc", "set_tproc_counter", "get_tproc_counter",
        "get_decimated", "get_accumulated",
    ]
    for name in required:
        assert callable(getattr(fresh_soc, name, None)), f"MockQickSoc.{name} missing"


def test_no_op_methods_accept_arbitrary_kwargs(fresh_soc):
    # Future-proofing: qick may add kwargs to these calls. The stubs must not
    # explode if called with extras.
    fresh_soc.start_src("internal", extra_kw=True)
    fresh_soc.set_nyquist(0, 1, 2, mystery=3)
    fresh_soc.configure_readout(0, {"any": "dict"}, extra=None)


# ---------- MockInstrumentManager ----------

def test_instrument_manager_holds_mock_qicksoc():
    from experiments.mock_hardware import MockInstrumentManager, MockQickSoc
    im = MockInstrumentManager(qick_alias="MyAlias")
    assert "MyAlias" in im.keys()
    assert isinstance(im["MyAlias"], MockQickSoc)


def test_instrument_manager_is_dict_subclass():
    # Existing code assumes dict-like (uses .keys(), .__getitem__).
    from experiments.mock_hardware import MockInstrumentManager
    im = MockInstrumentManager()
    assert isinstance(im, dict)


# ---------- MockYokogawa ----------

def test_mock_yokogawa_state_changes():
    from experiments.mock_hardware import MockYokogawa
    yk = MockYokogawa("y1", "fake://addr")
    assert yk.get_voltage() == 0.0
    yk.set_voltage(0.42)
    assert yk.get_voltage() == 0.42
    yk.output_on()
    assert yk._output_enabled is True
    yk.output_off()
    assert yk._output_enabled is False
    yk.ramp_current(0.1)
    assert yk._current == 0.1


def test_mock_yokogawa_prints_actions(capsys):
    # We kept print statements so dry-run sessions still show the "what would
    # have happened" trace, like the original MockYokogawa did.
    from experiments.mock_hardware import MockYokogawa
    yk = MockYokogawa("y1", "fake://addr")
    yk.set_voltage(0.5)
    yk.output_on()
    captured = capsys.readouterr()
    assert "set_voltage" in captured.out
    assert "output_on" in captured.out


# ---------- MultimodeStation API surface (no instantiation) ----------

def test_station_exposes_swap_methods():
    from experiments.station import MultimodeStation
    for name in ("use_mock_instruments", "use_real_instruments",
                 "_install_mock_instruments"):
        assert callable(getattr(MultimodeStation, name, None)), f"missing: {name}"


def test_station_is_mock_property_exists():
    from experiments.station import MultimodeStation
    assert isinstance(getattr(MultimodeStation, "is_mock", None), property)


def test_station_no_longer_exposes_is_mock_instruments_alias():
    # We consolidated on is_mock; this alias was added then removed in the same
    # session. Prevents accidental re-introduction.
    from experiments.station import MultimodeStation
    assert not hasattr(MultimodeStation, "is_mock_instruments")


def test_station_module_no_detect_mock_mode():
    # Renamed to is_production_pc; old name should be gone.
    import experiments.station as s
    assert not hasattr(s, "detect_mock_mode")
    assert callable(s.is_production_pc)


def test_station_soc_attribute_tripwire():
    """station.soc was renamed to .soccfg. A __getattr__ tripwire raises
    an informative AttributeError if anyone tries the old name."""
    from experiments.station import MultimodeStation

    # Minimal instance bypassing __init__ (which would require hardware).
    inst = MultimodeStation.__new__(MultimodeStation)
    inst._is_mock = False
    inst.soccfg = "dummy_qickconfig"

    # new name works
    assert inst.soccfg == "dummy_qickconfig"

    # old name raises with a helpful message
    with pytest.raises(AttributeError, match="renamed to .soccfg"):
        _ = inst.soc

    # unrelated missing attrs still raise normal AttributeError
    with pytest.raises(AttributeError, match="no attribute"):
        _ = inst.this_attr_does_not_exist


def test_swap_methods_preserve_state_per_contract():
    # The state-preservation contract from the plan: only these attrs are
    # touched by the swap methods. Anything outside should be preserved.
    from experiments.station import MultimodeStation
    keys = set(MultimodeStation._MOCK_SWAP_PATH_KEYS)
    # These are the path attrs; im/yokos/_is_mock are handled separately.
    expected = {
        "output_root", "experiment_path", "data_path", "expt_objs_path",
        "plot_path", "log_path", "autocalib_path",
    }
    assert keys == expected, (
        "Swap path keys drifted from the documented contract. "
        "Update docs/mock_mode_architecture_plan.md if intentional."
    )


# ---------- soccfg snapshot (off-prod-PC mock fallback) ----------

def test_snapshot_path_lives_in_configs():
    import experiments.station as s
    assert s.SOCCFG_SNAPSHOT_PATH.parent.name == "configs"
    assert s.SOCCFG_SNAPSHOT_PATH.name == "soccfg_snapshot.json"


def test_committed_snapshot_loads_as_working_qickconfig():
    """The committed snapshot must rebuild a real QickConfig with accurate
    (non-identity) unit conversions — the whole point over a hand-rolled stub."""
    import experiments.station as s
    if not s.SOCCFG_SNAPSHOT_PATH.exists():
        pytest.skip("no committed soccfg snapshot in this checkout")
    cfg = s.read_soccfg_snapshot()
    # cycles2us is a real firmware-dependent conversion, never the identity.
    assert cfg.cycles2us(100) != 100
    assert len(cfg.get_cfg()["gens"]) > 0


def test_read_snapshot_raises_pointed_error_when_missing(tmp_path, monkeypatch):
    import experiments.station as s
    monkeypatch.setattr(s, "SOCCFG_SNAPSHOT_PATH", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="prod PC"):
        s.read_soccfg_snapshot()


def test_write_snapshot_if_changed_semantics(tmp_path, monkeypatch):
    import experiments.station as s

    class FakeCfg:
        def __init__(self, text):
            self._text = text
        def dump_cfg(self):
            return self._text

    target = tmp_path / "soccfg_snapshot.json"
    monkeypatch.setattr(s, "SOCCFG_SNAPSHOT_PATH", target)

    # first write creates the file
    assert s.write_soccfg_snapshot_if_changed(FakeCfg('{"a": 1}')) is True
    assert target.exists()
    # identical content (even reformatted) is a no-op
    assert s.write_soccfg_snapshot_if_changed(FakeCfg('{"a":     1}')) is False
    # changed content rewrites
    assert s.write_soccfg_snapshot_if_changed(FakeCfg('{"a": 2}')) is True


def test_resolve_mock_soccfg_offprod_uses_snapshot_no_proxy(monkeypatch):
    """Off-prod-PC, mock soccfg must come from the snapshot without any proxy
    contact (a proxy attempt off-prod would only hang and time out)."""
    import experiments.station as s
    if not s.SOCCFG_SNAPSHOT_PATH.exists():
        pytest.skip("no committed soccfg snapshot in this checkout")
    monkeypatch.setattr(s, "is_production_pc", lambda: False)
    # Detonate if anything tries to reach the proxy.
    def _boom(*a, **k):
        raise AssertionError("off-prod path must not construct InstrumentManager")
    monkeypatch.setattr(s, "InstrumentManager", _boom)

    inst = s.MultimodeStation.__new__(s.MultimodeStation)
    inst.hardware_cfg = {"aliases": {"soc": "Qick101"}}
    cfg = inst._resolve_mock_soccfg()
    assert cfg.cycles2us(100) != 100


# ---------- Runner mock-mode gating ----------

def test_sweep_runner_execute_gates_on_is_mock():
    from experiments.sweep_runner import SweepRunner
    src = inspect.getsource(SweepRunner.execute)
    # Both branches must be present: the auto-default flip AND the explicit-override warning.
    assert 'is_mock' in src
    assert 'use_queue' in src and 'mock instruments' in src.lower()


def test_characterization_runner_execute_gates_on_is_mock():
    from experiments.characterization_runner import CharacterizationRunner
    src = inspect.getsource(CharacterizationRunner.execute)
    assert 'is_mock' in src
    assert 'use_queue' in src and 'mock instruments' in src.lower()


# ---------- Lab-notebook vault gating in mock mode ----------

def test_log_measurement_skips_when_mock():
    """station.log_measurement is the single source of truth — direct callers
    in notebooks (e.g. coupler_systematic_study_v2.ipynb) all go through it,
    so the mock guard here protects every code path."""
    from experiments.station import MultimodeStation
    src = inspect.getsource(MultimodeStation.log_measurement)
    assert "self._is_mock" in src
    assert "mock mode active" in src


def test_runner_helpers_short_circuit_log_in_mock():
    # Avoid wasted display rendering on zero-data plots.
    from experiments.sweep_runner import SweepRunner
    from experiments.characterization_runner import CharacterizationRunner
    # SweepRunner still uses _maybe_log_measurement; CharacterizationRunner's was
    # generalized into _render_log_show (decoupled show/log). Both keep the guard.
    for cls, meth in ((SweepRunner, "_maybe_log_measurement"),
                      (CharacterizationRunner, "_render_log_show")):
        src = inspect.getsource(getattr(cls, meth))
        assert "is_mock" in src
        assert "mock mode active" in src
