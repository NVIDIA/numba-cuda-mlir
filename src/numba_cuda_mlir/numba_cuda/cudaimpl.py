# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# The built-in CUDA lowerings that used to live here built llvmlite IR and were
# installed into the target context, but MLIRLower filters out every builder
# defined under numba_cuda_mlir.numba_cuda before invoking it (the CUDA
# intrinsics, shared/local arrays and atomics are lowered by
# numba_cuda_mlir.lowering.cuda instead), so none of them ran on the MLIR path.
# Only the registry and its decorator aliases survive: they are the public
# extension point used by out-of-tree code (e.g. ``cuda.cudaimpl.lower``), whose
# lowerings live outside numba_cuda_mlir.numba_cuda and so are not filtered out.

from numba_cuda_mlir.numba_cuda.core.imputils import Registry

registry = Registry("cudaimpl")
lower = registry.lower
lower_attr = registry.lower_getattr
lower_constant = registry.lower_constant
lower_getattr_generic = registry.lower_getattr_generic
lower_setattr = registry.lower_setattr
lower_setattr_generic = registry.lower_setattr_generic
lower_cast = registry.lower_cast
