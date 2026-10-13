"""Startup compatibility checks that do not load Isaac Sim or CUDA."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runtime_bootstrap as runtime


@pytest.mark.parametrize("machine", ["aarch64", "arm64", "x86_64"])
def test_early_import_restores_process_start_method(machine):
    with (patch.object(runtime.platform, "machine", return_value=machine),
          patch.object(runtime.sys, "platform", "linux"),
          patch.object(runtime.sys, "argv", ["scripts/train.py"]),
          patch.object(runtime.glob, "glob", return_value=[]),
          patch.object(runtime.importlib.util, "find_spec", return_value=None),
          patch.object(runtime.mp, "get_start_method", return_value="spawn"),
          patch.object(runtime.mp, "set_start_method") as set_method,
          patch.object(runtime.importlib, "import_module") as import_module):
        runtime.bootstrap_isaac_arm64()
        if machine == "x86_64":
            set_method.assert_not_called()
            import_module.assert_not_called()
        else:
            import_module.assert_called_once_with("isaacsim")
            assert [call.args[0] for call in set_method.call_args_list] == ["fork", "spawn"]


def test_preload_reexec_preserves_arguments_and_existing_libraries():
    with (patch.object(runtime.platform, "machine", return_value="aarch64"),
          patch.object(runtime.sys, "platform", "linux"),
          patch.object(runtime.sys, "argv", ["scripts/train.py", "algo=test"]),
          patch.object(runtime.sys, "orig_argv", ["python", "-u", "scripts/train.py", "algo=test"]),
          patch.object(runtime.glob, "glob", return_value=["/lib/aarch64-linux-gnu/libgomp.so.1"]),
          patch.object(runtime.importlib.util, "find_spec", return_value=None),
          patch.dict(runtime.os.environ, {"LD_PRELOAD": "/custom/lib.so"}),
          patch.object(runtime.os, "execve", side_effect=SystemExit) as restart):
        with pytest.raises(SystemExit):
            runtime.bootstrap_isaac_arm64()
        executable, args, env = restart.call_args.args
        assert executable == sys.executable
        assert args == [sys.executable, "-u", "scripts/train.py", "algo=test"]
        assert env["LD_PRELOAD"] == "/custom/lib.so:/lib/aarch64-linux-gnu/libgomp.so.1"


def test_existing_preload_does_not_restart():
    library = "/lib/aarch64-linux-gnu/libgomp.so.1"
    with (patch.object(runtime.platform, "machine", return_value="aarch64"),
          patch.object(runtime.sys, "platform", "linux"),
          patch.object(runtime.sys, "argv", ["scripts/train.py"]),
          patch.object(runtime.glob, "glob", return_value=[library]),
          patch.object(runtime.importlib.util, "find_spec", return_value=None),
          patch.dict(runtime.os.environ, {"LD_PRELOAD": library}),
          patch.object(runtime.os, "execve") as restart,
          patch.object(runtime.mp, "set_start_method"),
          patch.object(runtime.importlib, "import_module", side_effect=RuntimeError("import failed"))):
        with pytest.raises(RuntimeError, match="import failed"):
            runtime.bootstrap_isaac_arm64()
        restart.assert_not_called()


@pytest.mark.parametrize("flag", ["--cfg", "--cfg=job", "--help", "--info"])
def test_config_inspection_does_not_load_simulator(flag):
    with (patch.object(runtime.platform, "machine", return_value="aarch64"),
          patch.object(runtime.sys, "platform", "linux"),
          patch.object(runtime.sys, "argv", ["scripts/train.py", flag]),
          patch.object(runtime.importlib, "import_module") as import_module,
          patch.object(runtime.os, "execve") as restart):
        runtime.bootstrap_isaac_arm64()
        import_module.assert_not_called()
        restart.assert_not_called()
