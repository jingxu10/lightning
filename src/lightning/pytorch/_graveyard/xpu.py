import sys
from typing import Any

import lightning.pytorch as pl


def _patch_sys_modules() -> None:
    self = sys.modules[__name__]
    sys.modules["lightning.pytorch.accelerators.xpu"] = self


class XPUAccelerator:
    auto_device_count = ...
    get_parallel_devices = ...
    is_available = ...
    parse_devices = ...
    setup_device = ...
    teardown = ...

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "The `XPUAccelerator` class has been moved to an external package."
            " Install the extension package as `pip install lightning-xpu`"
            " and import with `from lightning_xpu import XPUAccelerator`."
            " Please see: https://github.com/Lightning-AI/lightning-XPU for more details."
        )


def _patch_classes() -> None:
    setattr(pl.accelerators, "XPUAccelerator", XPUAccelerator)


_patch_sys_modules()
_patch_classes()
