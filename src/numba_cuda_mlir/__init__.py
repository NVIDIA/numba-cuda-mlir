# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
from numba_cuda_mlir._version import __version__
from numba_cuda_mlir.mlir import make_nanobind_metaclass_inheritable

make_nanobind_metaclass_inheritable()


def jit(*args, **kwargs):
    # Lazy import to avoid import-time cycles when `numba.cuda` is redirected to
    # `numba_cuda_mlir.cuda`.
    from numba_cuda_mlir.decorators import mlir_jit

    return mlir_jit(*args, **kwargs)


__all__ = ["jit", "struct", "union", "__version__"]
