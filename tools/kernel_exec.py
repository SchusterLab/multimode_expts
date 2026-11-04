"""Run code in a named kernel of the always-on Jupyter server (spec item A).

The server is the one nvim uses (port 8765, see the nvim config's
lua/config/kernels.lua). Kernels are found by their session name.

    pixi run python tools/kernel_exec.py list
    pixi run python tools/kernel_exec.py start --name analysis --dir C:/python/multimode_expts
    pixi run python tools/kernel_exec.py cells --script measurement_notebooks/guan/x.py
    pixi run python tools/kernel_exec.py run --kernel guan --script x.py --cell "Cell 3"
    pixi run python tools/kernel_exec.py run --kernel guan --file snippet.py
    pixi run python tools/kernel_exec.py run --kernel guan --code "print(station.user)"
    pixi run python tools/kernel_exec.py run --kernel guan --code "long_job()" --no-wait
    pixi run python tools/kernel_exec.py interrupt --kernel guan

Cells are jupytext percent cells (``# %%`` markers). ``--cell`` takes the
index that ``cells`` prints, or a title: "Cell 1" matches "Cell 1: station"
but not "Cell 1b" or "Cell 10". Markdown cells cannot be run.

Output (stdout, stderr, results, tracebacks) goes to the terminal and to the
log file (default ``.cache/kernel_exec/<kernel>.log`` in this checkout); images
are saved next to the log. ``--no-wait`` returns once the code is sent; a
detached process writes the output to the log only.

When ``run`` runs a cell of a script and waits, the kernel has
``__kernel_exec_source__`` ("<script>, cell <title>") while the cell runs, so
``station.log_analysis`` can record where a figure came from.

The server token is read from the server's runtime file and never printed or
logged. Exit codes: 0 done, 1 the code raised, 2 timeout, 3 usage or lookup error.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8765
DEFAULT_SPEC = "pixi-cwd"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MARKER_RE = re.compile(r"^#\s?%%(.*)$")


class KernelExecError(Exception):
    """A lookup or usage error: exit code 3."""


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    index: int
    title: str
    kind: str  # "code" or "markdown"
    code: str
    line: int  # 1-based line of the marker (or 1 for the header)


def split_cells(text: str) -> list:
    """-> the jupytext percent cells of ``text``. Text before the first marker is
    cell 0 ("(header)") if it is not empty."""
    cells = []
    title, kind, start, body = "(header)", "code", 1, []

    def close():
        code = "\n".join(body).strip("\n")
        if cells or code.strip() or title != "(header)":
            cells.append(Cell(len(cells), title, kind, code + "\n" if code else "", start))

    for number, line in enumerate(text.splitlines(), start=1):
        match = MARKER_RE.match(line)
        if not match:
            body.append(line)
            continue
        close()
        rest = match.group(1).strip()
        kind = "code"
        cell_type = re.match(r"^\[(\w+)\]\s*", rest)
        if cell_type:
            kind = "markdown" if cell_type.group(1) in ("markdown", "md") else "code"
            rest = rest[cell_type.end():]
        title, start, body = rest, number, []
    close()
    return cells


def _title_matches(title: str, key: str) -> bool:
    if title == key:
        return True
    return title.startswith(key) and not title[len(key)].isalnum()


def select_cell(cells: list, key: str) -> Cell:
    """-> the cell with index ``key`` (an integer) or with title ``key`` (see module doc)."""
    if re.fullmatch(r"\d+", key.strip()):
        index = int(key)
        if index >= len(cells):
            raise KernelExecError(f"no cell {index}; the script has {len(cells)} cells")
        cell = cells[index]
    else:
        hits = [c for c in cells if _title_matches(c.title, key)]
        if len(hits) != 1:
            listing = "\n".join(f"  {c.index}: {c.title}" for c in (hits or cells))
            raise KernelExecError(f"{len(hits)} cells match {key!r}:\n{listing}")
        cell = hits[0]
    if cell.kind != "code":
        raise KernelExecError(f"cell {cell.index} ({cell.title!r}) is a {cell.kind} cell")
    return cell


# ---------------------------------------------------------------------------
# Server and kernels
# ---------------------------------------------------------------------------

def runtime_dir() -> Path:
    if os.environ.get("JUPYTER_RUNTIME_DIR"):
        return Path(os.environ["JUPYTER_RUNTIME_DIR"])
    from jupyter_core.paths import jupyter_runtime_dir
    return Path(jupyter_runtime_dir())


class Server:
    """The Jupyter server on ``port``.

    The runtime dir keeps the ``jpserver-*.json`` of servers that did not shut
    down cleanly, several of them for the same port. Takes the newest file whose
    token the server on the port accepts.
    """

    def __init__(self, port: int = DEFAULT_PORT):
        files = []
        for path in runtime_dir().glob("jpserver-*.json"):
            try:
                info = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if info.get("port") == port:
                files.append((path.stat().st_mtime, info))
        self.port = port
        for _, info in sorted(files, key=lambda item: item[0], reverse=True):
            self.root = Path(info["root_dir"])
            self.base = f"http://127.0.0.1:{port}{info.get('base_url', '/')}"
            self._token = info.get("token", "")
            try:
                self.api("GET", "api/status")
                return
            except OSError:  # URLError, HTTPError (403 for a stale token), refused
                continue
        raise KernelExecError(f"no Jupyter server answers on port {port} "
                              f"({len(files)} runtime files in {runtime_dir()})")

    def api(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path.lstrip("/"), data=data, method=method)
        request.add_header("Authorization", f"token {self._token}")
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode()
        return json.loads(text) if text else None

    def sessions(self) -> list:
        return self.api("GET", "api/sessions")

    def find(self, key: str) -> dict:
        """-> the session whose name is ``key``, or whose kernel id starts with it."""
        sessions = self.sessions()
        hits = [s for s in sessions if s.get("name") == key]
        if not hits and len(key) >= 4:
            hits = [s for s in sessions if s["kernel"]["id"].startswith(key)]
        if len(hits) != 1:
            names = ", ".join(sorted(s.get("name", "?") for s in sessions))
            raise KernelExecError(f"{len(hits)} kernels match {key!r}; the server has: {names}")
        return hits[0]


def connection_file(kernel_id: str) -> Path:
    path = runtime_dir() / f"kernel-{kernel_id}.json"
    if not path.exists():
        raise KernelExecError(f"no connection file {path}; is the server on this machine?")
    return path


def connect(kernel_id: str, timeout: float = 30):
    """-> a started BlockingKernelClient whose iopub channel gets messages."""
    from jupyter_client import BlockingKernelClient

    client = BlockingKernelClient(connection_file=str(connection_file(kernel_id)))
    client.load_connection_file()
    client.start_channels()
    deadline = time.time() + timeout
    while True:
        try:
            client.wait_for_ready(timeout=5)
            break
        except RuntimeError:
            if time.time() > deadline:
                client.stop_channels()
                raise KernelExecError(f"kernel {kernel_id} did not answer in {timeout:.0f} s")
            time.sleep(1)
    # The iopub subscription can miss the first messages after a connect. Wait
    # until a kernel_info request's status message arrives on iopub.
    while time.time() < deadline:
        msg_id = client.kernel_info()
        end = time.time() + 1
        while time.time() < end:
            try:
                msg = client.get_iopub_msg(timeout=0.2)
            except Exception:
                continue
            if msg["parent_header"].get("msg_id") == msg_id:
                return client
    client.stop_channels()
    raise KernelExecError(f"no iopub messages from kernel {kernel_id}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class Output:
    """Writes the output of a run to the terminal (unless quiet) and the log."""

    def __init__(self, log_path: Path, quiet: bool = False):
        self.log_path = log_path
        self.quiet = quiet
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("a", encoding="utf-8")
        self.images = 0

    def write(self, text: str, stream=None):
        text = ANSI_RE.sub("", text)
        if not self.quiet:
            (stream or sys.stdout).write(text)
            (stream or sys.stdout).flush()
        self.log.write(text)
        self.log.flush()

    def header(self, kernel: str, what: str):
        self.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} kernel {kernel}: {what}\n")

    def data(self, data: dict):
        if "image/png" in data:
            self.images += 1
            stem = f"{self.log_path.stem}_{datetime.now():%Y%m%d-%H%M%S}_{self.images}.png"
            path = self.log_path.parent / stem
            path.write_bytes(base64.b64decode(data["image/png"]))
            self.write(f"[image] {path}\n")
        elif "text/plain" in data:
            self.write(data["text/plain"] + "\n")

    def close(self):
        self.log.close()


def default_log(kernel: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", kernel)
    return REPO / ".cache" / "kernel_exec" / f"{safe}.log"


def collect(client, msg_id: str, out: Output, timeout) -> int:
    """Write the iopub output of request ``msg_id`` until the kernel is idle.
    -> 0 done, 1 the code raised, 2 timeout."""
    deadline = time.time() + timeout if timeout else None
    status = 0
    while True:
        wait = 1.0 if deadline is None else min(1.0, deadline - time.time())
        if wait <= 0:
            out.write(f"\n[kernel_exec] timeout after {timeout} s; the code still runs "
                      f"(use `interrupt` to stop it)\n", sys.stderr)
            return 2
        try:
            msg = client.get_iopub_msg(timeout=wait)
        except Exception:
            continue
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        kind, content = msg["msg_type"], msg["content"]
        if kind == "stream":
            out.write(content["text"], sys.stderr if content["name"] == "stderr" else None)
        elif kind in ("execute_result", "display_data"):
            out.data(content.get("data", {}))
        elif kind == "error":
            out.write("\n".join(content.get("traceback", [])) + "\n", sys.stderr)
            status = 1
        elif kind == "status" and content.get("execution_state") == "idle":
            return status


def set_source(client, source):
    """Set ``__kernel_exec_source__`` in the kernel (None to clear it), silently."""
    client.execute(f"__kernel_exec_source__ = {source!r}", silent=True, store_history=False)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args):
    for s in Server(args.port).sessions():
        kernel = s["kernel"]
        print(f"{s.get('name', ''):<24} {kernel['id'][:8]}  {kernel.get('execution_state', '?'):<9} "
              f"conn={kernel.get('connections', '?')}  {s.get('path', '')}")
    return 0


def cmd_start(args):
    server = Server(args.port)
    directory = Path(args.dir).resolve()
    try:
        rel = directory.relative_to(server.root.resolve()).as_posix()
    except ValueError:
        raise KernelExecError(f"{directory} is outside the server root {server.root}")
    path = f"{rel}/{args.name}" if rel not in ("", ".") else args.name
    known = {s["id"] for s in server.sessions()}
    session = server.api("POST", "api/sessions", {
        "path": path, "name": args.name, "type": "console", "kernel": {"name": args.spec}})
    if session["id"] in known:
        print(f"a kernel named {args.name} already runs in {directory}; not started again")
    client = connect(session["kernel"]["id"], timeout=args.timeout)
    client.stop_channels()
    print(f"kernel {args.name} ({session['kernel']['id'][:8]}) is ready in {directory}")
    return 0


def cmd_cells(args):
    for cell in split_cells(Path(args.script).read_text(encoding="utf-8")):
        mark = "" if cell.kind == "code" else f" [{cell.kind}]"
        print(f"{cell.index:>3}  line {cell.line:<5}{mark} {cell.title}")
    return 0


def code_and_source(args):
    """-> (code, source label) from --script/--cell, --file or --code."""
    if args.script:
        if args.cell is None:
            raise KernelExecError("--script needs --cell")
        script = Path(args.script)
        cell = select_cell(split_cells(script.read_text(encoding="utf-8")), args.cell)
        try:
            label = script.resolve().relative_to(REPO).as_posix()
        except ValueError:
            label = str(script)
        return cell.code, f"{label}, cell {cell.title or cell.index}"
    if args.file:
        return Path(args.file).read_text(encoding="utf-8"), None
    if args.code is not None:
        return args.code, None
    raise KernelExecError("run needs --script/--cell, --file or --code")


def cmd_run(args):
    code, source = code_and_source(args)
    session = Server(args.port).find(args.kernel)
    name = session.get("name") or args.kernel
    log_path = Path(args.log) if args.log else default_log(name)

    if args.no_wait and not args.detached:
        return spawn_detached(args, log_path)

    out = Output(log_path, quiet=args.detached)
    try:
        if session["kernel"].get("execution_state") == "busy" and not args.detached:
            print(f"[kernel_exec] kernel {name} is busy; the code waits in its queue", file=sys.stderr)
        client = connect(session["kernel"]["id"])
        try:
            out.header(name, source or (args.file and f"file {args.file}") or "code")
            if source and not args.no_wait:
                set_source(client, source)
            msg_id = client.execute(code, store_history=True, allow_stdin=False)
            if args.detached:
                print(f"sent {msg_id}", flush=True)
                sys.stdout.close()
            status = collect(client, msg_id, out, None if args.detached else args.timeout)
            if source and not args.no_wait and status != 2:
                set_source(client, None)
        finally:
            client.stop_channels()
    finally:
        out.close()
    if not args.detached:
        print(f"[kernel_exec] log: {log_path}", file=sys.stderr)
    return status


def spawn_detached(args, log_path: Path) -> int:
    """Start this command again in a detached process that sends the code and
    writes the output to the log; return when it has sent the code."""
    argv = [sys.executable, str(Path(__file__).resolve())] + args.argv + [
        "--detached", "--log", str(log_path)]
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, creationflags=flags, text=True)
    line = child.stdout.readline().strip()
    if not line.startswith("sent "):
        child.wait(timeout=10)
        raise KernelExecError(f"the detached run did not send the code (exit {child.returncode}); "
                              f"see {log_path}")
    print(f"[kernel_exec] sent to {args.kernel}; output goes to {log_path}")
    return 0


def cmd_interrupt(args):
    server = Server(args.port)
    session = server.find(args.kernel)
    server.api("POST", f"api/kernels/{session['kernel']['id']}/interrupt", {})
    print(f"interrupted {session.get('name') or args.kernel}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="the server's kernels").set_defaults(func=cmd_list)

    start = sub.add_parser("start", help="start a named kernel")
    start.add_argument("--name", required=True)
    start.add_argument("--dir", default=".", help="checkout dir; the kernel's cwd and pixi env")
    start.add_argument("--spec", default=DEFAULT_SPEC, help="kernelspec")
    start.add_argument("--timeout", type=float, default=120)
    start.set_defaults(func=cmd_start)

    cells = sub.add_parser("cells", help="the cells of a script, with indices")
    cells.add_argument("--script", required=True)
    cells.set_defaults(func=cmd_cells)

    run = sub.add_parser("run", help="run a cell, a file or code in a kernel")
    run.add_argument("--kernel", required=True, help="session name (or kernel id prefix)")
    run.add_argument("--script")
    run.add_argument("--cell", help="index or title of the cell in --script")
    run.add_argument("--file")
    run.add_argument("--code")
    run.add_argument("--timeout", type=float, default=None, help="seconds")
    run.add_argument("--no-wait", action="store_true")
    run.add_argument("--log", help="log file (default .cache/kernel_exec/<kernel>.log)")
    run.add_argument("--detached", action="store_true", help=argparse.SUPPRESS)
    run.set_defaults(func=cmd_run)

    interrupt = sub.add_parser("interrupt", help="interrupt a kernel")
    interrupt.add_argument("--kernel", required=True)
    interrupt.set_defaults(func=cmd_interrupt)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    args.argv = argv
    try:
        return args.func(args)
    except KernelExecError as exc:
        print(f"kernel_exec: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
