# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
import sys


_MODULE_NAMES = (
    "devicearray",
    "devices",
    "driver",
    "dummyarray",
    "error",
    "libs",
    "linkable_code",
    "mappings",
    "ndarray",
    "nvrtc",
    "nvvm",
    "runtime",
)

for _name in _MODULE_NAMES:
    _module = import_module(f"numba_cuda_mlir.numba_cuda.cudadrv.{_name}")
    globals()[_name] = _module
    sys.modules[f"{__name__}.{_name}"] = _module

__all__ = tuple(_MODULE_NAMES)

del import_module, sys, _module, _name, _MODULE_NAMES
