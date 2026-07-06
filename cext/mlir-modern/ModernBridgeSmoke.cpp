/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "ModernBridge.h"

#include <cstdio>
#include <cstring>
#include <string>

static int run_case(const char *name, const char *mlir, int ctk_major = 13,
                    int ctk_minor = 0, bool emit_text_ir = false) {
    char *out = nullptr;
    size_t out_len = 0;
    char *error = nullptr;
    int rc = mlir_modern_to_nvvm_translate_for_libnvvm(
        mlir, std::strlen(mlir), ctk_major, ctk_minor, 2, 0, 3, 2, 0,
        emit_text_ir ? 1 : 0, &out, &out_len, &error);
    if (rc != 0) {
        std::fprintf(stderr, "bridge smoke case '%s' failed: %s\n", name,
                     error ? error : "unknown error");
        mlir_modern_to_nvvm_free(error);
        return 1;
    }
    if (!emit_text_ir && (!out || out_len < 2 || out[0] != 'B' || out[1] != 'C')) {
        std::fprintf(stderr,
                     "bridge smoke case '%s' produced non-bitcode output "
                     "(%zu bytes)\n",
                     name, out_len);
        mlir_modern_to_nvvm_free(out);
        return 1;
    }
    if (emit_text_ir) {
        std::string text(out ? out : "", out_len);
        if (text.find("target triple") == std::string::npos) {
            std::fprintf(stderr,
                         "bridge smoke case '%s' produced unexpected text "
                         "output (%zu bytes)\n",
                         name, out_len);
            mlir_modern_to_nvvm_free(out);
            return 1;
        }
    }
    mlir_modern_to_nvvm_free(out);
    return 0;
}

int main() {
    int failures = 0;

    static const char simple_kernel[] = R"MLIR(
gpu.module @kernels attributes {
  llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64-S128",
  llvm.target_triple = "nvptx64-nvidia-cuda"
} {
  llvm.func @simple_kernel() attributes {gpu.kernel} {
    llvm.return
  }
}
)MLIR";
    failures += run_case("simple-kernel", simple_kernel);
    failures += run_case("simple-kernel-text", simple_kernel, 13, 0, true);

    static const char nvvm_intrinsics[] = R"MLIR(
gpu.module @kernels attributes {
  llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64-S128",
  llvm.target_triple = "nvptx64-nvidia-cuda"
} {
  llvm.func @intrinsic_kernel() attributes {gpu.kernel} {
    %ticks = llvm.mlir.constant(32 : i32) : i32
    %pred = llvm.mlir.constant(1 : i32) : i32
    nvvm.nanosleep %ticks
    nvvm.barrier
    %count = nvvm.barrier.reduction #nvvm.reduction<popc> %pred -> i32
    llvm.return
  }
}
)MLIR";
    failures += run_case("nvvm-intrinsics", nvvm_intrinsics);

    static const char atomics[] = R"MLIR(
gpu.module @kernels attributes {
  llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64-S128",
  llvm.target_triple = "nvptx64-nvidia-cuda"
} {
  llvm.func @atomic_kernel(%p32: !llvm.ptr<1>, %p64: !llvm.ptr<1>) attributes {gpu.kernel} {
    %v32 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %v64 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %old32 = llvm.atomicrmw fadd %p32, %v32 monotonic : !llvm.ptr<1>, f32
    %old64 = llvm.atomicrmw fminimum %p64, %v64 monotonic : !llvm.ptr<1>, f64
    llvm.return
  }
}
)MLIR";
    failures += run_case("atomics", atomics);

    static const char downgrade[] = R"MLIR(
gpu.module @kernels attributes {
  llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64-S128",
  llvm.target_triple = "nvptx64-nvidia-cuda"
} {
  llvm.func @downgrade_kernel() attributes {gpu.kernel} {
    %ptr = llvm.mlir.poison : !llvm.ptr
    llvm.intr.lifetime.start %ptr : !llvm.ptr
    %x = llvm.mlir.constant(1.500000e+00 : f64) : f64
    %y = llvm.call @llvm.trunc.f64(%x) : (f64) -> f64
    llvm.return
  }
  llvm.func @llvm.trunc.f64(f64) -> f64
}
)MLIR";
    failures += run_case("downgrade", downgrade, 12, 0);

    return failures == 0 ? 0 : 1;
}
