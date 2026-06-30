# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# This module used to register the CUDA math lowerings (math.sqrt/exp/sin/...,
# the libdevice-backed unary/binary/boolean ops, pow/modf/frexp/ldexp/tanh and
# complex pow). They built llvmlite IR and were installed into the target
# context, but they are filtered out on the MLIR path, where math is lowered by
# numba_cuda_mlir.lowering.math. The registry object is retained because the
# target context still installs it.

from numba_cuda_mlir.numba_cuda.core.imputils import Registry

registry = Registry("mathimpl")
