"""Register FlashSAC algorithms and preserve imports used by old checkpoints."""

import importlib as _importlib
import sys as _sys


__all__ = [
    "flashsac_vel_flat",
    "flashsac_vel",
    "flashsac_vel_finetune",
    "one_stage_flashsac_vel",
    "finetune_schedule",
    "flashsac_upstream",
]

for _name in __all__:
    globals()[_name] = _importlib.import_module(f".{_name}", __name__)


def _register_legacy_imports():
    # Structured configurations in earlier checkpoints were pickled from ppo.
    # Alias modules so those imports preserve the same configuration identities.
    ppo = _importlib.import_module("..ppo", __name__)
    prefix = f"{__name__}."
    for name, module in list(_sys.modules.items()):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            _sys.modules[f"{ppo.__name__}.{suffix}"] = module
            if "." not in suffix:
                setattr(ppo, suffix, module)


_register_legacy_imports()
