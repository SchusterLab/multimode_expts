"""
Kernel launcher that sets up pixi environment DLL paths before starting ipykernel.

This allows launching the kernel without pixi.exe as a wrapper, which fixes
the interrupt-kills-kernel issue on Windows (pixi.exe dies on SIGINT, taking
the kernel with it). By launching Python directly with proper DLL paths,
VSCode can use Win32 events for interrupt handling.
"""
import os

# Add pixi environment directories to DLL search path
# Without this, C extensions (numpy, scipy, h5py, zmq) can't find their DLLs
pixi_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pixi", "envs", "default")

dll_dirs = [
    os.path.join(pixi_env, "Library", "bin"),
    os.path.join(pixi_env, "Library", "lib"),
    os.path.join(pixi_env, "DLLs"),
    pixi_env,
]

# Keep handles alive — if GC'd, the directories are removed from the search path
_dll_handles = []
for d in dll_dirs:
    if os.path.isdir(d):
        _dll_handles.append(os.add_dll_directory(d))
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

# Set CONDA_PREFIX so conda-aware packages can find their resources
os.environ["CONDA_PREFIX"] = pixi_env

# Launch ipykernel
from ipykernel.kernelapp import launch_new_instance
launch_new_instance()
