# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Vendored object-mode lowering (removed).

The llvmlite ``BaseLower``/``Lower``/``CUDALower`` object-mode lowerer that used
to live here has been deleted: numba-cuda-mlir lowers CUDA kernels via
``MLIRLower`` (see ``numba_cuda_mlir.mlir_lowering``), not via the vendored
``NativeLowering``/``CUDANativeLowering`` passes (whose ``lowering_class`` now
returns ``None``). Only the ``Environment`` re-export is retained, for the
(likewise dead) object-mode boxing helpers in ``core.pythonapi``.
"""

from numba_cuda_mlir.numba_cuda.core.environment import Environment  # noqa: F401
