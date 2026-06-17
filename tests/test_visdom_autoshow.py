"""Verify the visdom_autoshow post_run_cell hook reproduces the inline backend's
'flush figures at cell end' behavior under the Agg backend, for the three display
patterns that appear in real measurement notebooks. Uses a ListSink so no visdom
server is required.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # the overnight-safe target backend

_MOD_PATH = Path(__file__).resolve().parents[1] / "measurement_notebooks" / "visdom_autoshow.py"
_spec = importlib.util.spec_from_file_location("visdom_autoshow", _MOD_PATH)
visdom_autoshow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(visdom_autoshow)


@pytest.fixture
def shell():
    IPython = pytest.importorskip("IPython")
    sh = IPython.core.interactiveshell.InteractiveShell.instance()
    # Make sure plt is fresh / no figures leak across tests.
    sh.run_cell("import matplotlib.pyplot as plt; plt.close('all')")
    yield sh
    sh.run_cell("import matplotlib.pyplot as plt; plt.close('all')")


def test_ad_hoc_pcolormesh_is_captured(shell):
    sink = visdom_autoshow.ListSink()
    hook = visdom_autoshow.enable(sink=sink, ip=shell)
    try:
        # Pattern: ad hoc pcolormesh of an in-memory array, NO show(), NO display().
        res = shell.run_cell(
            "import numpy as np, matplotlib.pyplot as plt\n"
            "plt.pcolormesh(np.random.rand(20, 30))\n"
        )
        assert res.success
        assert len(sink.sent) == 1, "exactly one figure should be flushed"
        title, shape = sink.sent[0]
        assert len(shape) == 3 and shape[2] == 3, "RGB array (H, W, 3)"
        assert shape[0] > 0 and shape[1] > 0
        # Hook must close the figure (mirrors inline backend), no leak.
        import matplotlib.pyplot as plt
        assert plt.get_fignums() == []
    finally:
        shell.events.unregister("post_run_cell", hook)


def test_bare_subplots_cell_no_show(shell):
    sink = visdom_autoshow.ListSink()
    hook = visdom_autoshow.enable(sink=sink, ip=shell)
    try:
        # Pattern from single_qubit_autocalibrate_v2.py:390-399 -- subplots,
        # plot, tight_layout, no show().
        res = shell.run_cell(
            "import numpy as np, matplotlib.pyplot as plt\n"
            "fig, axes = plt.subplots(3, 1, figsize=(6, 6))\n"
            "for ax in axes: ax.plot(np.arange(10), np.random.rand(10))\n"
            "plt.tight_layout()\n"
        )
        assert res.success
        assert len(sink.sent) == 1
    finally:
        shell.events.unregister("post_run_cell", hook)


def test_display_method_style(shell):
    sink = visdom_autoshow.ListSink()
    hook = visdom_autoshow.enable(sink=sink, ip=shell)
    try:
        # Pattern: experiment.display() -- builds a fig and calls plt.show(),
        # which under Agg is an inert no-op. Hook still flushes the open fig.
        res = shell.run_cell(
            "import numpy as np, matplotlib.pyplot as plt\n"
            "class E:\n"
            "    def display(self):\n"
            "        f = plt.figure(); plt.plot(np.random.rand(50)); plt.show()\n"
            "        return f\n"
            "E().display()\n"
        )
        assert res.success
        assert len(sink.sent) == 1
    finally:
        shell.events.unregister("post_run_cell", hook)


def test_multiple_figures_one_cell(shell):
    sink = visdom_autoshow.ListSink()
    hook = visdom_autoshow.enable(sink=sink, ip=shell)
    try:
        res = shell.run_cell(
            "import matplotlib.pyplot as plt\n"
            "plt.figure(); plt.plot([1, 2, 3])\n"
            "plt.figure(); plt.plot([3, 2, 1])\n"
        )
        assert res.success
        assert len(sink.sent) == 2
    finally:
        shell.events.unregister("post_run_cell", hook)


def test_preexisting_figure_not_flushed_when_only_new(shell):
    # A long-lived live-monitor window opened BEFORE enable() must be left alone.
    shell.run_cell("import matplotlib.pyplot as plt; plt.figure(); plt.plot([0, 1])")
    sink = visdom_autoshow.ListSink()
    hook = visdom_autoshow.enable(sink=sink, ip=shell, only_new=True)
    try:
        res = shell.run_cell("import matplotlib.pyplot as plt; plt.figure(); plt.plot([1, 0])")
        assert res.success
        assert len(sink.sent) == 1, "only the new figure, not the pre-existing one"
    finally:
        shell.events.unregister("post_run_cell", hook)


def test_http_sink_serves_png_round_trip():
    """HttpImageSink serves figures over HTTP, advertises them via /state, bumps
    the version on overwrite-in-place, and handles spaced window names."""
    import json
    import urllib.request

    import matplotlib.pyplot as plt
    import numpy as np

    def get(url):
        return urllib.request.urlopen(url, timeout=5).read()

    sink = visdom_autoshow.HttpImageSink(port=0, host="127.0.0.1")  # ephemeral port
    try:
        base = f"http://127.0.0.1:{sink.port}"

        fig = plt.figure()
        plt.pcolormesh(np.random.rand(10, 12))
        sink.send(fig, title="cell")
        plt.close(fig)

        assert get(base + "/fig/cell.png")[:8] == b"\x89PNG\r\n\x1a\n", "PNG served"
        state = json.loads(get(base + "/state"))
        assert state == [{"win": "cell", "ver": 1}]
        assert b"/state" in get(base + "/")  # index is the static poller page

        # overwrite-in-place: same win => one slot, version bumps
        fig2 = plt.figure()
        plt.plot([1, 2, 3])
        sink.send(fig2, title="cell")
        plt.close(fig2)
        assert sink._order == ["cell"]
        assert json.loads(get(base + "/state")) == [{"win": "cell", "ver": 2}]

        # spaced win name (the hook uses "fig 1") must fetch cleanly when encoded
        fig_sp = plt.figure()
        plt.plot([0, 1])
        sink.send(fig_sp, title="fig 1")
        plt.close(fig_sp)
        wins = {s["win"] for s in json.loads(get(base + "/state"))}
        assert "fig 1" in wins
        assert get(base + "/fig/fig%201.png")[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        sink.shutdown()


# --- connection-file gate (the opt-in mechanism on a shared account) -----------

def _run_in_fresh_kernel(conn_name, code):
    """Launch a kernel exactly as guan does (python -m ipykernel -f <name>),
    run `code`, return concatenated stdout. Pure-Python, no hardware."""
    import os
    import subprocess
    import sys
    import tempfile
    import time

    pytest.importorskip("jupyter_client")
    from jupyter_client import BlockingKernelClient

    tmp = tempfile.mkdtemp()
    conn = os.path.join(tmp, conn_name)
    proc = subprocess.Popen([sys.executable, "-m", "ipykernel", "-f", conn],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(150):
            if os.path.exists(conn) and os.path.getsize(conn) > 0:
                break
            time.sleep(0.1)
        kc = BlockingKernelClient()
        kc.load_connection_file(conn)
        kc.start_channels()
        kc.wait_for_ready(timeout=60)
        msg_id = kc.execute(code)
        out = []
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                m = kc.get_iopub_msg(timeout=5)
            except Exception:
                break
            if m["parent_header"].get("msg_id") != msg_id:
                continue
            if m["header"]["msg_type"] == "stream":
                out.append(m["content"]["text"])
            if (m["header"]["msg_type"] == "status"
                    and m["content"]["execution_state"] == "idle"):
                break
        kc.stop_channels()
        return "".join(out)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


_GATE_PROBE = (
    "import sys\n"
    f"sys.path.insert(0, r'{_MOD_PATH.parent}')\n"
    "import visdom_autoshow as v\n"
    "print('ISTARGET=', v.is_target_kernel('guan-meas'))\n"
    "h = v.enable_if_target('guan-meas', set_backend=True, sink=v.ListSink())\n"
    "print('ENABLED=', h is not None)\n"
)


def test_gate_activates_for_matching_connection_file():
    out = _run_in_fresh_kernel("guan-meas.json", _GATE_PROBE)
    assert "ISTARGET= True" in out, out
    assert "ENABLED= True" in out, out


def test_gate_inert_for_other_kernel():
    # mimics a teammate's kernelspec connection file -> must stay dormant
    out = _run_in_fresh_kernel("multimode_direct.json", _GATE_PROBE)
    assert "ISTARGET= False" in out, out
    assert "ENABLED= False" in out, out
