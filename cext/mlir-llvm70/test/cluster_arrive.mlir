// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_90">] {

    llvm.func @cluster_arrive_relaxed() attributes {gpu.kernel} {
      nvvm.cluster.arrive.relaxed
      llvm.return
    }

    llvm.func @cluster_arrive() attributes {gpu.kernel} {
      nvvm.cluster.arrive
      llvm.return
    }

    llvm.func @cluster_wait() attributes {gpu.kernel} {
      nvvm.cluster.wait
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @cluster_arrive_relaxed
// CHECK-IR: call void asm sideeffect "barrier.cluster.arrive.relaxed;"

// CHECK-IR: define ptx_kernel void @cluster_arrive
// CHECK-IR: call void asm sideeffect "barrier.cluster.arrive;"

// CHECK-IR: define ptx_kernel void @cluster_wait
// CHECK-IR: call void asm sideeffect "barrier.cluster.wait;"

// CHECK-PTX: .visible .entry cluster_arrive_relaxed
// CHECK-PTX: barrier.cluster.arrive.relaxed;
// CHECK-PTX: .visible .entry cluster_arrive
// CHECK-PTX: barrier.cluster.arrive;
// CHECK-PTX: .visible .entry cluster_wait
// CHECK-PTX: barrier.cluster.wait;
