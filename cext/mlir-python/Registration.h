/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#ifndef NUMBA_CUDA_MLIR_REGISTRATION_H
#define NUMBA_CUDA_MLIR_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers the dialects numba-cuda-mlir uses, their external interface
/// models and conversion-to-LLVM interfaces, and their LLVM IR translations.
MLIR_CAPI_EXPORTED void
numbaCudaMlirRegisterDialects(MlirDialectRegistry registry);

/// Registers the passes named in numba-cuda-mlir's pass pipeline.
MLIR_CAPI_EXPORTED void numbaCudaMlirRegisterPasses(void);

#ifdef __cplusplus
}
#endif

#endif // NUMBA_CUDA_MLIR_REGISTRATION_H
