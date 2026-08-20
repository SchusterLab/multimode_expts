#!/usr/bin/env python
"""Launch a per-user JupyterLab sandbox.

Everyone logs into this machine as the same OS user (slab), so isolation
between people does NOT come from the OS account -- it comes from giving each
person their own server instance: their own port, and their own
workspaces/user-settings dirs. That makes open-tab layout and Lab settings
private per person. The project files under the shared root_dir stay shared
(notebook-level sandboxing is handled by agreement, not by the server).

Auth: a shared password is already configured for the slab user
(C:\\Users\\slab\\.jupyter\\jupyter_server_config.json, set via
`jupyter password`). Every server launched here picks it up automatically.

Usage:
    pixi run lab <name> [extra jupyter flags...]

e.g. pixi run lab guan

The one rule: each person connects ONLY to their own port (their ssh -L tunnel
should forward exactly that port). Two people on one server share a workspace
and collide -- which is the bug this setup exists to avoid.
"""
import subprocess
import sys
from pathlib import Path

# name -> fixed port. Ports are "claimed" here; add new people as needed.
PORTS = {
    "guan": 8765,
    "jonginn": 2405,
    "seb": 8801,
}

# Per-user private dirs live under the slab home, one named folder each.
SANDBOX_ROOT = Path.home() / ".jupyter" / "sandboxes"


def main():
    known = ", ".join(sorted(PORTS))
    if len(sys.argv) < 2:
        sys.exit(f"usage: pixi run lab <name> [extra flags]   (known: {known})")

    name = sys.argv[1].lower()
    if name not in PORTS:
        sys.exit(
            f"unknown user {name!r}. Known: {known}.\n"
            f"Add a line to PORTS in launch_lab.py to claim a port."
        )

    port = PORTS[name]
    workspaces_dir = SANDBOX_ROOT / name / "workspaces"
    user_settings_dir = SANDBOX_ROOT / name / "user-settings"
    workspaces_dir.mkdir(parents=True, exist_ok=True)
    user_settings_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "jupyter", "lab", "--no-browser",
        f"--port={port}",
        # Fail loudly if the port is taken (i.e. someone is already running as
        # this name) instead of silently hopping to a different free port.
        "--ServerApp.port_retries=0",
        f"--LabApp.workspaces_dir={workspaces_dir}",
        f"--LabApp.user_settings_dir={user_settings_dir}",
        *sys.argv[2:],  # pass through any extra flags
    ]

    print(
        f"[launch_lab] {name} -> http://pippin-meas:{port}/  "
        f"(workspaces={workspaces_dir})",
        flush=True,
    )
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
