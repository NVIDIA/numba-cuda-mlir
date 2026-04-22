# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

from cusimt.cuda import *

# Override jit to use cuda.simt defaults (annotations_as_signatures=True)
from cusimt import jit  # noqa: F811

# TODO: Implement redirector or refactor folders
# to allow "from cuda.simt.extending import lowering_registry"
# Import cusimt submodules
from cusimt import (
    tools,
    compiler,
    testing,
    types,
    type_defs,
    errors,
    host,
    linker,
)

# Import libdevice and libdevicefuncs directly to bypass redirector
import cusimt.cuda.libdevice as libdevice
import cusimt.cuda.libdevicefuncs as libdevicefuncs
import cusimt.cuda.misc as misc
import cusimt.cuda.tcgen05_descriptors as tcgen05_descriptors

# Register libdevice submodules in sys.modules so "from cuda.simt.libdevice import X" works
sys.modules["cuda.simt.libdevice"] = libdevice
sys.modules["cuda.simt.libdevicefuncs"] = libdevicefuncs

# Ensure device functions are initialized after cuda.simt is fully loaded
# The module has its own initialization, but we ensure it runs here since
# cusimt.cuda is guaranteed to be loaded at this point
try:
    tcgen05_descriptors._init_device_functions()
except Exception:
    # If initialization fails, device functions will be compiled on first use
    pass


def __getattr__(name):
    """Lazy load modules to avoid circular import issues."""
    if name in ("intrin", "tensor_map"):
        import importlib

        module = importlib.import_module(f"cusimt.cuda.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
