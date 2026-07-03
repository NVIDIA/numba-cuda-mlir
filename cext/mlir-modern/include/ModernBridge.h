/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#ifndef NUMBA_CUDA_MLIR_MODERN_BRIDGE_H
#define NUMBA_CUDA_MLIR_MODERN_BRIDGE_H

#include <stddef.h>

#ifdef _WIN32
#define MLIR_MODERN_TO_NVVM_EXPORT __declspec(dllexport)
#else
#define MLIR_MODERN_TO_NVVM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

MLIR_MODERN_TO_NVVM_EXPORT int mlir_modern_to_nvvm_translate_for_libnvvm(
    const char *mlir_text, size_t mlir_text_len, int ctk_major, int ctk_minor,
    int nvvm_ir_major, int nvvm_ir_minor, int nvvm_debug_major,
    int nvvm_debug_minor,
    int dump_llvmir, int emit_text_ir, char **out, size_t *out_len,
    char **error_out);

MLIR_MODERN_TO_NVVM_EXPORT void mlir_modern_to_nvvm_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
