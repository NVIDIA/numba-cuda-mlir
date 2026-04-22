// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @cttz_i64(%val: i64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = "llvm.intr.cttz"(%val) <{is_zero_poison = false}> : (i64) -> i64
      llvm.store %res, %out : i64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @cttz_i32(%val: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = "llvm.intr.cttz"(%val) <{is_zero_poison = true}> : (i32) -> i32
      llvm.store %res, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @cttz_i64
// CHECK-IR: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)

// CHECK-IR: define ptx_kernel void @cttz_i32
// CHECK-IR: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 true)

// CHECK-PTX: .visible .entry cttz_i64
// CHECK-PTX: .visible .entry cttz_i32
