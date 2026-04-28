// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @ctpop_i64(%val: i64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.ctpop(%val) : (i64) -> i64
      llvm.store %res, %out : i64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @ctpop_i32(%val: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.ctpop(%val) : (i32) -> i32
      llvm.store %res, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @ctpop_i64
// CHECK-IR: call i64 @llvm.ctpop.i64(i64 %{{.*}})

// CHECK-IR: define ptx_kernel void @ctpop_i32
// CHECK-IR: call i32 @llvm.ctpop.i32(i32 %{{.*}})

// CHECK-PTX: .visible .entry ctpop_i64
// CHECK-PTX: popc.b64
// CHECK-PTX: .visible .entry ctpop_i32
// CHECK-PTX: popc.b32
