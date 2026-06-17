"""visdom_autoshow.py -- prototype.

Reproduce the IPython *inline backend's* "flush figures at end of cell" behavior,
but ship the figures off-terminal to a sink (visdom) instead of rendering inline.

Why this exists
---------------
Under ``MPLBACKEND=Agg`` (the overnight-safe, non-blocking choice -- ``plt.show()``
becomes an inert no-op), nothing renders: figures created in a cell just sit open
in memory. The inline backend normally auto-captures them via a post-execute hook.
This module registers an equivalent ``post_run_cell`` hook that captures every
figure still open at cell end, renders it, hands it to a ``sink``, and closes it --
so ad hoc ``plt.pcolormesh(arr)``, a bare ``plt.subplots(...)`` cell, and
``some_experiment.display()`` ALL keep working verbatim, with the image appearing
in a browser tab instead of inline.

Opt-in on a SHARED machine/account/CWD/env
-------------------------------------------
git is not the isolation boundary here (everyone runs kernels from the same dir
in the same pixi env on the same Windows account). The boundary is *which kernel
this is*: :func:`enable_if_target` self-gates on the connection-file name, so a
committed startup snippet -- ``visdom_autoshow.enable_if_target("guan-meas")`` --
can live in the shared IPython profile and stay completely inert for every other
kernel (VSCode / JupyterLab / teammates' kernelspecs). Only the kernel launched
with ``-f ~/.molten/kernels/guan-meas.json`` activates the hook + Agg backend.

The sink is pluggable. Default is :class:`HttpImageSink` (stdlib only -- visdom
no longer builds against modern setuptools). :class:`ListSink` lets the capture
mechanism be tested with no server at all.
"""
from __future__ import annotations

import io
import json
import threading
from typing import Protocol
from urllib.parse import unquote


def fig_to_rgb_array(fig):
    """Render a Matplotlib figure to an (H, W, 3) uint8 RGB array via the Agg
    canvas -- no PIL, no PNG round-trip, no GUI backend required."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())  # (H, W, 4) uint8
    return rgba[..., :3].copy()


class Sink(Protocol):
    def send(self, fig, *, title: str) -> None: ...


class ListSink:
    """Test sink: records what *would* be shipped instead of sending it."""

    def __init__(self):
        self.sent = []  # list of (title, (H, W, 3))

    def send(self, fig, *, title: str) -> None:
        arr = fig_to_rgb_array(fig)
        self.sent.append((title, arr.shape))


class VisdomSink:
    """Ships each figure as a static image into a named visdom window.

    Static PNG-equivalent on purpose: identical fidelity to the inline backend
    (which is also a static raster), and robust where ``vis.matplot`` -- which
    converts to plotly -- breaks (pcolormesh / heatmaps / twinx)."""

    def __init__(self, env="main", server="http://localhost", port=8097,
                 win="cell", store_history=True):
        from visdom import Visdom

        self.vis = Visdom(server=server, port=port, env=env,
                          raise_exceptions=False)
        self.win = win
        self.store_history = store_history

    def send(self, fig, *, title: str) -> None:
        arr = fig_to_rgb_array(fig)
        chw = arr.transpose(2, 0, 1)  # visdom.image wants (C, H, W)
        self.vis.image(chw, win=self.win,
                       opts={"caption": title, "store_history": self.store_history})


def fig_to_png_bytes(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


class HttpImageSink:
    """Dependency-free off-terminal display: a tiny background HTTP server that
    serves the figures shipped to it, newest first. Open the URL in any browser
    (workstation-local over RDP, or via ``ssh -L <port>:localhost:<port>``).

    The visdom-free alternative -- visdom no longer installs cleanly against
    modern setuptools (its setup.py imports the removed ``pkg_resources``).
    ``win`` names a slot: re-sending the same ``win`` overwrites it in place (the
    live-monitor idiom); distinct names accumulate as separate panels.
    """

    def __init__(self, port=8099, host="0.0.0.0", history=20):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self._lock = threading.Lock()
        self._slots: dict[str, bytes] = {}   # win -> png bytes (overwrite in place)
        self._ver: dict[str, int] = {}       # win -> version (bumps on each send)
        self._order: list[str] = []          # insertion order, newest last
        self._history = history
        sink = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence access logging
                pass

            def _reply(self, body, content_type):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]  # drop cache-buster query
                if path.startswith("/fig/"):
                    win = unquote(path[len("/fig/"):].rsplit(".png", 1)[0])
                    with sink._lock:
                        png = sink._slots.get(win)
                    if png is None:
                        self.send_error(404)
                        return
                    self._reply(png, "image/png")
                    return
                if path == "/state":
                    # The page polls this and reloads ONLY the images whose
                    # version changed -- no full-page refresh, no blink.
                    with sink._lock:
                        state = [{"win": w, "ver": sink._ver[w]} for w in sink._order]
                    self._reply(json.dumps(state).encode(), "application/json")
                    return
                self._reply(_INDEX_HTML.encode(), "text/html; charset=utf-8")

        self._server = ThreadingHTTPServer((host, port), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def send(self, fig, *, title: str) -> None:
        png = fig_to_png_bytes(fig)
        with self._lock:
            if title not in self._slots:
                self._order.append(title)
            self._slots[title] = png
            self._ver[title] = self._ver.get(title, 0) + 1
            while len(self._order) > self._history:
                old = self._order.pop(0)
                self._slots.pop(old, None)
                self._ver.pop(old, None)

    def shutdown(self):
        self._server.shutdown()


# Static page: polls /state and swaps only the images whose version changed.
# The browser keeps the previous image visible until the new one decodes, so an
# in-place update of a single window is seamless -- no flicker, no page reload.
_INDEX_HTML = """<!doctype html><meta charset=utf-8><title>autoshow</title>
<style>
body{background:#111;color:#ccc;font-family:monospace;margin:16px}
figure{margin:0 0 16px}figcaption{margin-bottom:4px;color:#888}
img{max-width:100%;border:1px solid #333;display:block}
</style>
<div id=c><p id=empty>no figures yet</p></div>
<script>
const c = document.getElementById('c'), els = {};
async function tick(){
  let st;
  try{ st = await (await fetch('/state',{cache:'no-store'})).json(); }
  catch(e){ return; }
  const present = new Set();
  for(const {win, ver} of st){
    present.add(win);
    let rec = els[win];
    if(!rec){
      const e = document.getElementById('empty'); if(e) e.remove();
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption'); cap.textContent = win;
      const img = document.createElement('img');
      fig.appendChild(cap); fig.appendChild(img);
      c.insertBefore(fig, c.firstChild);   // newest window on top
      rec = els[win] = {fig, img, ver: null};
    }
    if(rec.ver !== ver){
      rec.img.src = '/fig/'+encodeURIComponent(win)+'.png?v='+ver;
      rec.ver = ver;
    }
  }
  for(const win in els){
    if(!present.has(win)){ els[win].fig.remove(); delete els[win]; }
  }
}
setInterval(tick, 700); tick();
</script>"""


def _make_hook(sink: Sink, only_new: bool):
    import matplotlib.pyplot as plt

    # Figures already open before we started (e.g. a long-lived live-monitor
    # window) -- skip these so we only flush per-cell output, matching inline.
    baseline = set(plt.get_fignums()) if only_new else set()

    def _flush(result=None):
        for num in plt.get_fignums():
            if num in baseline:
                continue
            fig = plt.figure(num)
            try:
                sink.send(fig, title=f"fig {num}")
            except Exception as exc:  # never let display kill a measurement cell
                print(f"[visdom_autoshow] send failed for fig {num}: {exc}")
            finally:
                plt.close(fig)

    return _flush


def enable(sink: Sink | None = None, *, ip=None, only_new: bool = True, **visdom_kwargs):
    """Register the post-cell flush hook on the current IPython shell.

    Returns the hook callable (so it can be unregistered in tests). ``sink``
    defaults to a :class:`VisdomSink` built from ``**visdom_kwargs``.
    """
    if ip is None:
        from IPython import get_ipython

        ip = get_ipython()
    if ip is None:
        raise RuntimeError("enable() needs an IPython shell (run inside a kernel)")
    if sink is None:
        sink = HttpImageSink(**visdom_kwargs)
    hook = _make_hook(sink, only_new=only_new)
    ip.events.register("post_run_cell", hook)
    return hook


def current_connection_file():
    """Absolute path of *this* kernel's connection file, or None if not in a
    kernel. This is the per-process signal that distinguishes one user's kernel
    from another's on a shared account / CWD / env."""
    try:
        from ipykernel.kernelapp import IPKernelApp
    except Exception:
        return None
    if not IPKernelApp.initialized():
        return None
    try:
        return IPKernelApp.instance().abs_connection_file
    except Exception:
        return None


def is_target_kernel(marker: str) -> bool:
    """True iff this kernel was launched with a connection file whose name
    matches ``marker`` (basename, with or without the .json suffix). e.g. a
    kernel started ``-f ~/.molten/kernels/guan-meas.json`` matches "guan-meas"."""
    import os

    cf = current_connection_file()
    if not cf:
        return False
    stem = os.path.splitext(os.path.basename(cf))[0]
    return stem == os.path.splitext(marker)[0]


def enable_if_target(marker: str, *, set_backend: bool = True, **kwargs):
    """Gated entry point for a *committed, shared* IPython startup file. Does
    nothing unless this kernel's connection file matches ``marker`` -- so it can
    live in the shared profile and stay inert for every other kernel (VSCode /
    JupyterLab / teammates' kernelspecs) on the same account.

    When it matches: optionally force the Agg backend (overnight-safe, non-
    blocking) and register the auto-show hook. Returns the hook, or None."""
    if not is_target_kernel(marker):
        return None
    if set_backend:
        import matplotlib

        matplotlib.use("Agg", force=True)
    return enable(**kwargs)
