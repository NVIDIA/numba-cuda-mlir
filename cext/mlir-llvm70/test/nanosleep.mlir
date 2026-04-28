// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @nanosleep_kernel() attributes {gpu.kernel} {
      %ns = llvm.mlir.constant(1000 : i32) : i32
      nvvm.nanosleep %ns
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @nanosleep_kernel
// CHECK-IR: call void asm sideeffect "nanosleep.u32 $0;", "r"(i32

// CHECK-PTX: .visible .entry nanosleep_kernel
// CHECK-PTX: nanosleep.u32
