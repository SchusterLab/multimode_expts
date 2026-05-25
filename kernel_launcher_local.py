"""Minimal kernel launcher for the multimode_expts pixi env, no monkey-patches.

Only responsibility: add the pixi env's DLL directories to the Windows DLL
search path so C extensions (numpy, scipy, h5py, zmq) load. Without this the
kernel hangs at startup. Then launch ipykernel.

Used by the 'multimode-local' kernelspec for a vanilla local kernel that
bypasses the pixi.exe wrapper. Use kernel_launcher.py (with the SSH/Windows
interrupt patches) for the multimode-direct kernel.
"""
import os

_repo_dir = os.path.dirname(os.path.abspath(__file__))
pixi_env = os.path.join(_repo_dir, ".pixi", "envs", "default")

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

from ipykernel.kernelapp import launch_new_instance
launch_new_instance()
