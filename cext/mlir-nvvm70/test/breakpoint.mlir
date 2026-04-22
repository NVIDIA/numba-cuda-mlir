// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @breakpoint_kernel() attributes {gpu.kernel} {
      nvvm.breakpoint
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @breakpoint_kernel
// CHECK-IR: call void asm sideeffect "brkpt;"

// CHECK-PTX: .visible .entry breakpoint_kernel
// CHECK-PTX: brkpt;
