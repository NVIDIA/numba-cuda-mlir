# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
from cusimt._version import __version__
from cusimt.mlir import make_nanobind_metaclass_inheritable

make_nanobind_metaclass_inheritable()


def jit(*args, **kwargs):
    # Lazy import to avoid import-time cycles when `numba.cuda` is redirected to
    # `cusimt.cuda`.
    from cusimt.mlir_compiler import numba_mlir

    return numba_mlir.jit(*args, **kwargs)


def __getattr__(name: str):
    match name:
        case "numba_mlir":
            from cusimt.mlir_compiler import numba_mlir

            return numba_mlir
        case "typing":
            # `cusimt.typing` is a package/module that re-exports typing helpers.
            import cusimt.typing as _typing

            return _typing
        case "types":
            # `cusimt.types` is a module (mirrors `numba.types` style usage).
            import cusimt.types as _types

            return _types
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["numba_mlir", "jit", "struct", "union", "__version__"]
