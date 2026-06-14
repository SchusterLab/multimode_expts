"""Closed-loop experiment service — HTTP facade over `core`.

This service is a thin HTTP front door for the transport-agnostic logic in
`core.py`. It owns NO hardware and NO measurement logic: every `/run_wigner`
and `/calibrate_check` request is delegated to `core.run_wigner_core` /
`core.calibrate_check_core`, which submit a `WignerTomography1ModeExperiment`
job to the queue server. The queue worker is the sole hardware owner.

The same `core` functions are also driven, without any HTTP, by
`batch_runner.py` (the shared-folder "mailbox" runner). Keeping the logic in
`core` means both front doors stay in lock-step.

Lifecycle requirement: the queue server (port 8000) AND its worker must be
running. This service connects to the queue server as a client.

Run it
------
    pixi run python -m job_server.closed_loop.service \
        --hardware-config CFG-HW-20260515-00021 \
        --storage-man-file CFG-M1-20260513-00023

Endpoints
---------
  GET  /                  health, includes pinned config IDs and queue reachability
  POST /echo              JSON round-trip smoke test
  POST /run_wigner        mode="sim" (in-process) or "hw" (queue-routed)
  POST /calibrate_check   known-good fock-N reference, queue-routed
  POST /run_tomography_1q single-qubit Z/X/Y state tomography, sim or queue-routed

Request / response shapes are defined in `core` (RunWignerRequest, etc.) and
re-exported here for back-compat with existing imports.
"""
from __future__ import annotations

import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from job_server.closed_loop import core
# Re-export models/exceptions so existing `from ...service import X` keeps working.
from job_server.closed_loop.core import (  # noqa: F401
    IQTable,
    Knobs,
    RunWignerRequest,
    RunWignerResponse,
    CalibrateCheckRequest,
    CalibrateCheckResponse,
    RunStateTomo1QRequest,
    RunStateTomo1QResponse,
    ServiceNotReady,
)


# ============================== FastAPI app ================================

app = FastAPI(title="closed-loop experiment service (queue-client)", version="0.5.0")
START_TIME = datetime.now().isoformat(timespec="seconds")
HOSTNAME = socket.gethostname()


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "expt_service",
        "version": "0.5.0",
        "mode": "queue_client",
        "hostname": HOSTNAME,
        "started": START_TIME,
        "queue_url": core.queue_url(),
        "queue_reachable": core.queue_reachable(),
        "service_ready": core.is_ready(),
        "pinned": core.pinned_info(),
        "run_root": str(core.run_root()) if core.run_root() else None,
        "endpoints": ["GET /", "POST /echo", "POST /run_wigner",
                      "POST /calibrate_check", "POST /run_tomography_1q"],
    }


@app.post("/echo")
def echo(body: dict):
    return {
        "ok": True,
        "hostname": HOSTNAME,
        "received": body,
        "server_time": datetime.now().isoformat(timespec="seconds"),
    }


@app.post("/run_wigner", response_model=RunWignerResponse)
def run_wigner(req: RunWignerRequest):
    try:
        return core.run_wigner_core(req)
    except ServiceNotReady as e:
        raise HTTPException(503, str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, f"queue execution error: {e}")


@app.post("/calibrate_check", response_model=CalibrateCheckResponse)
def calibrate_check(req: CalibrateCheckRequest):
    try:
        return core.calibrate_check_core(req)
    except ServiceNotReady as e:
        raise HTTPException(503, str(e))
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise HTTPException(400, f"calibrate_check setup error: {e}")
    except RuntimeError as e:
        raise HTTPException(500, f"queue execution error: {e}")


@app.post("/run_tomography_1q", response_model=RunStateTomo1QResponse)
def run_tomography_1q(req: RunStateTomo1QRequest):
    try:
        return core.run_state_tomo_1q_core(req)
    except ServiceNotReady as e:
        raise HTTPException(503, str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, f"queue execution error: {e}")


# ============================== lifecycle ==================================

class ServiceHandle:
    def __init__(self, server, thread):
        self.server = server
        self.thread = thread

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()


def start_service(
    *,
    hardware_config: Optional[str] = None,
    storage_man_file: Optional[str] = None,
    queue_url: str = "http://127.0.0.1:8000",
    port: int = 18765,
    run_root: Optional[str | Path] = None,
    experiment_name: Optional[str] = None,
) -> ServiceHandle:
    """Initialize the core (pin configs, build station template, wire JobClient),
    then start uvicorn in a daemon thread.
    """
    core.init_core(
        hardware_config=hardware_config,
        storage_man_file=storage_man_file,
        queue_url=queue_url,
        run_root=run_root,
        experiment_name=experiment_name,
    )

    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="expt_service")
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)
    return ServiceHandle(server, thread)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="closed_loop service (queue-client edition). "
                    "Forwards /run_wigner and /calibrate_check to the job queue."
    )
    parser.add_argument("--port", type=int, default=18765,
                        help="Port for this service (default 18765).")
    parser.add_argument("--queue-url", default="http://127.0.0.1:8000",
                        help="Job queue server URL (default http://127.0.0.1:8000).")
    parser.add_argument("--experiment-name", default=None,
                        help="MultimodeStation experiment name. Defaults to yymmdd_closed_loop.")
    parser.add_argument("--hardware-config", default=None,
                        help="Pin hardware config: filename or version ID (e.g. CFG-HW-20260515-00021).")
    parser.add_argument("--storage-man-file", default=None,
                        help="Pin man1 storage-swap dataset: filename or version ID (e.g. CFG-M1-20260513-00023).")
    args = parser.parse_args()

    # Make sure repo root is on sys.path (matches the old --hw path)
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    handle = start_service(
        hardware_config=args.hardware_config,
        storage_man_file=args.storage_man_file,
        queue_url=args.queue_url,
        port=args.port,
        experiment_name=args.experiment_name,
    )
    print(f"[expt_service] running on http://127.0.0.1:{args.port} (queue-client mode)")
    print(f"[expt_service]   queue url: {core.queue_url()}")
    print(f"[expt_service]   run root:  {core.run_root()}")
    print("[expt_service] Ctrl-C to stop")
    try:
        while handle.alive:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[expt_service] stopping...")
        handle.stop()
