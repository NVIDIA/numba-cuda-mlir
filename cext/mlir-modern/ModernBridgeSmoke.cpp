/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "ModernBridge.h"

#include <cstdio>
#include <cstring>

int main() {
    static const char mlir[] = R"MLIR(
gpu.module @kernels attributes {
  llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64-S128",
  llvm.target_triple = "nvptx64-nvidia-cuda"
} {
  llvm.func @simple_kernel() attributes {gpu.kernel} {
    llvm.return
  }
}
)MLIR";

    char *out = nullptr;
    size_t out_len = 0;
    char *error = nullptr;
    int rc = mlir_modern_to_nvvm_translate_for_libnvvm(
        mlir, std::strlen(mlir), 13, 0, 0, &out, &out_len, &error);
    if (rc != 0) {
        std::fprintf(stderr, "bridge smoke failed: %s\n",
                     error ? error : "unknown error");
        mlir_modern_to_nvvm_free(error);
        return 1;
    }
    if (!out || out_len < 2 || out[0] != 'B' || out[1] != 'C') {
        std::fprintf(stderr,
                     "bridge smoke produced non-bitcode output (%zu bytes)\n",
                     out_len);
        mlir_modern_to_nvvm_free(out);
        return 1;
    }
    mlir_modern_to_nvvm_free(out);
    return 0;
}
