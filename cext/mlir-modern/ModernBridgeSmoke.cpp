/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "ModernBridge.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

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

static int run_version_case(const char *mlir, int ir_major, int ir_minor,
                            int debug_major, int debug_minor) {
    char *out = nullptr;
    size_t out_len = 0;
    char *error = nullptr;
    int rc = mlir_modern_to_nvvm_translate_for_libnvvm(
        mlir, std::strlen(mlir), 13, 0, ir_major, ir_minor, debug_major,
        debug_minor, 0, 1, &out, &out_len, &error);
    if (rc != 0) {
        std::fprintf(stderr, "concurrent bridge case failed: %s\n",
                     error ? error : "unknown error");
        mlir_modern_to_nvvm_free(out);
        mlir_modern_to_nvvm_free(error);
        return 1;
    }

    std::string text(out ? out : "", out_len);
    char expected[128];
    std::snprintf(expected, sizeof(expected),
                  "!{i32 %d, i32 %d, i32 %d, i32 %d}", ir_major,
                  ir_minor, debug_major, debug_minor);
    bool found = text.find(expected) != std::string::npos;
    if (!found)
        std::fprintf(stderr,
                     "concurrent bridge case expected NVVM version %s\n",
                     expected);
    mlir_modern_to_nvvm_free(out);
    return found ? 0 : 1;
}

static int run_concurrent_version_cases(const char *mlir) {
    constexpr int num_threads = 8;
    constexpr int rounds = 4;
    std::atomic<int> ready{0};
    std::atomic<int> failures{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int worker = 0; worker < num_threads; ++worker) {
        workers.emplace_back([&, worker] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();

            for (int round = 0; round < rounds; ++round) {
                int version = 1000 + worker * rounds + round;
                failures.fetch_add(
                    run_version_case(mlir, version, version + 100,
                                     version + 200, version + 300),
                    std::memory_order_relaxed);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != num_threads)
        std::this_thread::yield();
    start.store(true, std::memory_order_release);

    for (auto &worker : workers)
        worker.join();
    return failures.load(std::memory_order_relaxed) == 0 ? 0 : 1;
}

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

int main() {
    int failures = 0;

    failures += run_case("simple-kernel", simple_kernel);
    failures += run_case("simple-kernel-text", simple_kernel, 13, 0, true);
    failures += run_concurrent_version_cases(simple_kernel);

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
    %count = nvvm.barrier #nvvm.reduction<popc> %pred -> i32
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
