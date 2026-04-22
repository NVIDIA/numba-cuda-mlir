// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @membar_cta() attributes {gpu.kernel} {
      nvvm.memory.barrier <cta>
      llvm.return
    }

    llvm.func @membar_gpu() attributes {gpu.kernel} {
      nvvm.memory.barrier <gpu>
      llvm.return
    }

    llvm.func @membar_sys() attributes {gpu.kernel} {
      nvvm.memory.barrier <sys>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @membar_cta
// CHECK-IR: call void @llvm.nvvm.membar.cta()

// CHECK-IR: define ptx_kernel void @membar_gpu
// CHECK-IR: call void @llvm.nvvm.membar.gl()

// CHECK-IR: define ptx_kernel void @membar_sys
// CHECK-IR: call void @llvm.nvvm.membar.sys()

// CHECK-PTX: .visible .entry membar_cta
// CHECK-PTX: membar.cta

// CHECK-PTX: .visible .entry membar_gpu
// CHECK-PTX: membar.gl

// CHECK-PTX: .visible .entry membar_sys
// CHECK-PTX: membar.sys
