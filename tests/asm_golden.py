# -*- coding: utf-8 -*-
"""Pulse-level golden: the tProc program every MBR job class compiles to.

Why a separate net
------------------
``test_mbr_acquire_mock`` proves each job class still builds, compiles and
acquires, and it checks one specific envelope property. That is enough to
catch a job class that stops working; it is not enough to certify a *move* of
pulse code, because a pulse bug is silent by construction -- the program
compiles and "acquires" either way, and the only evidence of the error is the
instruction stream itself.

So this module renders the whole observable output of the pulse layer:

* ``prog.asm()`` -- the full tProc listing, the thing that actually plays;
* every registered envelope, by channel and name, as a sample digest, since
  the ASM names a waveform but carries none of its samples (exactly the gap
  the flat-top-vs-gaussian regression lived in).

Both pinned config sets run, for the reason given in the acquire test:
``august_n3`` is all ``gauss``, ``preload_current`` is all
``preload_flattop`` and is the only path that reaches the preloaded
register-bank playback.

Determinism was checked before pinning: two runs of all 12 programs in a set
produce byte-identical listings.

Each golden is the last program a job compiled (last sweep point, Ramsey
phase [180, 90]). Keys are ``{set}__{product}__{job}``, the products of
:func:`experiments.qsim.mbr_campaign.smoke`: ``stark_cal``, ``time_trace``,
``ortho_column_q0`` and ``ortho_column_q4``. They were renamed from the old
per-stage keys in redesign step 6, each checked byte-identical to the old
golden of the same pulses first (see that commit message).

Regenerate with ``pixi run python -m tests.asm_golden``, and read the diff
before committing it -- that diff is the review.
"""
import gzip
import hashlib
from pathlib import Path

import numpy as np

from experiments.qsim.mbr_campaign import (
    mock_station,
    pinned_config_set,
    pinned_sets,
    smoke,
)

GOLDEN_DIR = Path(__file__).parent / "data" / "asm_golden"

# Same shape as the acquire test, deliberately: two occupations of one total
# photon number, so encoder and decoder paths differ.
SWAP_STORS = [1, 2, 3, 4]
OCCUPATIONS = [[0, 0, 0, 0, 3], [1, 0, 0, 0, 2]]
REPS = 10


def _envelope_section(prog):
    """Registered envelopes as ``channel name nsamples digest`` lines.

    A digest, not the samples: a flat top is thousands of points, and what
    matters is that they are the same points. Rounded to 1e-9 first so a
    float-formatting change in the generator cannot read as a pulse change.
    """
    lines = []
    # ``envelopes`` is indexed by generator channel, not keyed by it.
    for ch, per_channel in enumerate(prog.envelopes):
        envs = per_channel.get("envs", {})
        for name in sorted(envs):
            env = envs[name]
            data = env["data"] if isinstance(env, dict) and "data" in env else env
            arr = np.asarray(data, dtype=float)
            digest = hashlib.sha256(
                np.round(arr, 9).tobytes()).hexdigest()[:16]
            lines.append(f"{ch} {name} {arr.shape} {digest}")
    return "\n".join(lines)


def render(prog):
    """The full pinned text for one compiled program."""
    return (f"# envelopes\n{_envelope_section(prog)}\n"
            f"\n# asm\n{prog.asm()}\n")


def programs(set_name):
    """Compile every product's every job for one config set.

    Yields ``(key, prog)`` with ``key`` naming the golden file.
    """
    station = mock_station(**pinned_config_set(set_name))
    products = smoke(station, SWAP_STORS, OCCUPATIONS, reps=REPS)
    for name in sorted(products):
        for index, expt in enumerate(products[name].children):
            yield f"{set_name}__{name}__{index}", expt.prog


def path_for(key):
    return GOLDEN_DIR / f"{key}.txt.gz"


def read(key):
    with gzip.open(path_for(key), "rt", encoding="utf-8") as handle:
        return handle.read()


def write(key, text):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    # mtime=0: a regenerated-but-identical file must not show up as a diff.
    with gzip.GzipFile(path_for(key), "wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))


def regenerate():
    written = []
    for set_name in sorted(pinned_sets()):
        for key, prog in programs(set_name):
            write(key, render(prog))
            written.append(key)
    return written


if __name__ == "__main__":
    for key in regenerate():
        size = path_for(key).stat().st_size
        print(f"wrote {key} ({size/1024:.1f} KiB gz)")
