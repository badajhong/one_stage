"""Initialize Isaac Sim on Linux ARM64 before Torch/TorchRL are imported."""

import glob
import importlib
import importlib.util
import multiprocessing as mp
import os
import platform
import sys


def bootstrap_isaac_arm64():
    if sys.platform != "linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        return
    # Hydra inspection must work without loading the simulator.
    if any(arg.split("=", 1)[0] in ("--cfg", "-c", "--help", "-h", "--hydra-help", "--info")
           for arg in sys.argv[1:]):
        return

    # Match Isaac Sim's ARM preload check without importing torch. LD_PRELOAD
    # must be set before Python starts, so re-exec once if libraries are missing.
    libraries = sorted(glob.glob("/lib/*/libgomp.so.1"))
    spec = importlib.util.find_spec("torch")
    if spec is not None and spec.submodule_search_locations:
        for path in spec.submodule_search_locations:
            libraries.extend(sorted(glob.glob(os.path.join(path + ".libs", "libgomp*"))))
    preload = os.environ.get("LD_PRELOAD", "")
    loaded = preload.replace(":", " ").split()
    missing = [path for path in libraries if path not in loaded]
    if missing:
        env = os.environ.copy()
        env["LD_PRELOAD"] = ":".join([*loaded, *missing])
        print("[startup] Restarting Python with ARM64 OpenMP libraries preloaded.", flush=True)
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)

    # Isaac Sim 5.1 imports itself in its preload-check child. Under spawn this
    # recursively starts more checks. Use fork only for that early CPU check,
    # before CUDA/worker threads exist; restore the caller's choice afterward.
    # TorchRL can then select spawn normally for training workers.
    previous = mp.get_start_method(allow_none=True)
    try:
        mp.set_start_method("fork", force=True)
        importlib.import_module("isaacsim")
    finally:
        mp.set_start_method(previous, force=True)
