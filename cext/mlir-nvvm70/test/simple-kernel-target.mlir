// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s | FileCheck %s
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {
    llvm.func @simple_kernel(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %val = llvm.sitofp %tid : i32 to f32
      llvm.store %val, %ptr : f32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK: gpu.binary @kernels
// CHECK-SAME: #nvvm70.target

// CHECK-IR: define ptx_kernel void @simple_kernel
// CHECK-IR: call i32 asm "mov.u32 $0, %tid.x;", "=r"()

// CHECK-PTX: .visible .entry simple_kernel
// CHECK-PTX: mov.u32
