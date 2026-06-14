"""
Offline tests for the opt-in waveform deduplication in
``MM_base.custom_pulse`` (the ``dedupe_waveforms`` flag).

We don't need real hardware or a full QickProgram: ``custom_pulse``'s
gaussian/flat_top branches only call a handful of program methods
(``us2cycles``, ``add_gauss``, ``sync_all``, ``setup_and_pulse``,
``freq2reg``, ``deg2reg``). We pass a lightweight recorder as ``self`` and
call the unbound ``MM_base.custom_pulse`` to count envelope allocations.

Asserts:
  - flag OFF -> one add_gauss per gate, legacy ``temp_gaussian{jj}{prefix}``
    names (regression guard for the ~42 existing call sites);
  - flag ON  -> add_gauss count == number of distinct (channel, sigma, shape)
    envelopes, independent of sequence length, with content-based names;
  - every gate still gets a setup_and_pulse referencing a loaded envelope.
"""

import pytest
from slab import AttrDict

from experiments.MM_base import MM_base


class _Recorder:
    """Minimal stand-in for a QickProgram, recording envelope allocations."""

    def __init__(self):
        self.add_gauss_calls = []   # (ch, name, sigma, length)
        self.setup_waveforms = []   # waveform name passed to setup_and_pulse
        self.cfg = AttrDict({'expt': {}})

    def us2cycles(self, us, gen_ch=None):
        return int(round(us * 1000))

    def freq2reg(self, f, gen_ch=None):
        return f

    def deg2reg(self, ph, gen_ch=None):
        return ph

    def add_gauss(self, ch=None, name=None, sigma=None, length=None):
        self.add_gauss_calls.append((ch, name, sigma, length))

    def sync_all(self, *a, **k):
        pass

    def setup_and_pulse(self, **k):
        self.setup_waveforms.append(k.get('waveform'))


# Three distinct (channel, sigma, shape) envelopes, like the E+D ladder:
# qubit gaussian @0.035, qubit gaussian @0.005, sideband flat_top @0.005.
_TEMPLATES = [
    (2, 0.035, 'gaussian'),
    (2, 0.005, 'gaussian'),
    (0, 0.005, 'flat_top'),
]
_N_DISTINCT = len(_TEMPLATES)


def _pulse_data(n_gates):
    rows = [[], [], [], [], [], [], []]
    for jj in range(n_gates):
        ch, sig, shape = _TEMPLATES[jj % len(_TEMPLATES)]
        rows[0].append(3000.0)   # freq
        rows[1].append(10000)    # gain
        rows[2].append(0.1)      # length (flat-top)
        rows[3].append(0)        # phase
        rows[4].append(ch)       # drive channel
        rows[5].append(shape)    # shape
        rows[6].append(sig)      # ramp sigma
    return rows


def _run(n_gates, dedupe):
    rec = _Recorder()
    cfg = AttrDict({'expt': {'dedupe_waveforms': dedupe}})
    MM_base.custom_pulse(rec, cfg, _pulse_data(n_gates), prefix='state_prep_')
    return rec


def test_off_one_envelope_per_gate_legacy_names():
    n = 49
    rec = _run(n, dedupe=False)
    assert len(rec.add_gauss_calls) == n
    names = [c[1] for c in rec.add_gauss_calls]
    assert len(set(names)) == n  # all unique
    assert all(nm.startswith('temp_gaussian') for nm in names)
    assert rec.setup_waveforms == names  # each gate references its own


def test_on_collapses_to_distinct_envelopes():
    n = 49
    rec = _run(n, dedupe=True)
    assert len(rec.add_gauss_calls) == _N_DISTINCT  # 3, not 49
    names = [c[1] for c in rec.add_gauss_calls]
    assert len(set(names)) == _N_DISTINCT
    # every gate still played, referencing one of the loaded envelopes
    assert len(rec.setup_waveforms) == n
    assert set(rec.setup_waveforms) == set(names)


@pytest.mark.parametrize('n', [12, 49, 120, 240])
def test_on_allocation_independent_of_length(n):
    rec = _run(n, dedupe=True)
    assert len(rec.add_gauss_calls) == _N_DISTINCT


def test_on_uses_content_based_names():
    rec = _run(6, dedupe=True)
    names = set(c[1] for c in rec.add_gauss_calls)
    assert any(nm.startswith('cp_g_') for nm in names)
    assert any(nm.startswith('cp_ft_') for nm in names)
    assert not any(nm.startswith('temp_gaussian') for nm in names)


def test_flat_top_multiplier_in_name():
    # sideband channel (0) uses 6-sigma ramp; key encodes the multiplier
    rec = _run(3, dedupe=True)
    ft_names = [c[1] for c in rec.add_gauss_calls if c[1].startswith('cp_ft_')]
    assert ft_names and all(nm.endswith('_6') for nm in ft_names)
