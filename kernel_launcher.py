"""
Kernel launcher for the multimode_expts pixi env.

Two responsibilities:

1. Add the pixi env's DLL directories to the Windows DLL search path so C
   extensions (numpy, scipy, h5py, zmq) load without going through pixi.exe.
   Without this the kernel hangs at startup.

2. Make message-mode interrupt actually work in VSCode-Jupyter on Windows.
   ipykernel's default interrupt path on Windows is broken in two ways:
     a) `Kernel._send_interrupt_children` is a no-op (just logs an error).
     b) `interrupt_request` arriving on the shell channel waits for the
        per-cell asyncio lock, so it can't fire until the running cell ends.
   We patch `shell_channel_thread_main` to detect `interrupt_request` *before*
   the lock is taken (it runs on the shell_channel_thread, which is never
   blocked by cell execution), call `_thread.interrupt_main()` to raise SIGINT
   in the main thread (where ipykernel's per-cell SIGINT handler can cancel
   the cell), and reply directly on the shell socket so VSCode doesn't time
   out.

A short diagnostic line is appended to <repo>/_kernel_launcher.log on every
start and on every interrupt, for troubleshooting.
"""
import os
import sys
import threading
import time
import _thread

_repo_dir = os.path.dirname(os.path.abspath(__file__))
pixi_env = os.path.join(_repo_dir, ".pixi", "envs", "default")
_log_path = os.path.join(_repo_dir, "_kernel_launcher.log")


def _log(msg):
    try:
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# 1. DLL search path setup — must happen before any C extension import.
dll_dirs = [
    os.path.join(pixi_env, "Library", "bin"),
    os.path.join(pixi_env, "Library", "lib"),
    os.path.join(pixi_env, "DLLs"),
    pixi_env,
]
_dll_handles = []  # keep alive — GC removes them from search path
for d in dll_dirs:
    if os.path.isdir(d):
        _dll_handles.append(os.add_dll_directory(d))
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
os.environ["CONDA_PREFIX"] = pixi_env

# Diagnostic log (overwrites previous log).
try:
    with open(_log_path, "w", encoding="utf-8") as f:
        f.write(f"start_time = {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"pid = {os.getpid()}\n")
        f.write(f"argv = {sys.argv}\n")
        f.write(f"JPY_INTERRUPT_EVENT = {os.environ.get('JPY_INTERRUPT_EVENT')}\n")
        for k in sorted(os.environ):
            if k.startswith(("JPY", "VSCODE", "IPY", "JUPYTER")):
                f.write(f"  env[{k}] = {os.environ[k]}\n")
except Exception:
    pass


# 2. Patches for ipykernel on Windows.
if os.name == "nt":
    from ipykernel.kernelbase import Kernel as _Kernel

    # 2a. Replace the no-op _send_interrupt_children with one that actually
    # raises SIGINT in the main thread.
    def _send_interrupt_children_windows(self):
        _log("_send_interrupt_children -> _thread.interrupt_main()")
        _thread.interrupt_main()

    _Kernel._send_interrupt_children = _send_interrupt_children_windows

    # 2b. Patch shell_channel_thread_main to short-circuit interrupt_request,
    # bypassing the per-cell asyncio lock that's held during cell execution.
    _orig_shell_channel_thread_main = _Kernel.shell_channel_thread_main

    async def _shell_channel_thread_main_patched(self, msg):
        assert threading.current_thread() is self.shell_channel_thread
        if self.session is None:
            return

        # Peek at message type without taking the lock.
        try:
            idents, msg_for_deser = self.session.feed_identities(msg, copy=False)
            deser = self.session.deserialize(msg_for_deser, content=False, copy=False)
            msg_type = deser["header"].get("msg_type")
        except Exception:
            msg_type = None
            idents = None
            deser = None

        if msg_type == "interrupt_request":
            _log(f"interrupt_request bypass on {threading.current_thread().name}")
            # Raise SIGINT in main thread (where cell execution is awaiting).
            # ipykernel installs _cancel_on_sigint during cell execution, which
            # will cancel the running future when SIGINT is raised.
            _thread.interrupt_main()
            # Send the reply directly on the shell socket so the front-end
            # doesn't hit its interrupt-reply timeout.
            try:
                self.session.send(
                    self.shell_stream,
                    "interrupt_reply",
                    {"status": "ok"},
                    parent=deser,
                    ident=idents,
                )
            except Exception as e:
                _log(f"  send interrupt_reply failed: {e!r}")
            return

        # Default: original behaviour (forward via inproc to subshell).
        return await _orig_shell_channel_thread_main(self, msg)

    _Kernel.shell_channel_thread_main = _shell_channel_thread_main_patched


# 3. Launch ipykernel.
from ipykernel.kernelapp import launch_new_instance
launch_new_instance()
