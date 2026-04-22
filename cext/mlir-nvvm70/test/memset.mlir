// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @memset_kernel(%dst: !llvm.ptr, %val: i8, %len: i64) attributes {gpu.kernel} {
      "llvm.intr.memset"(%dst, %val, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @memset_kernel
// CHECK-IR: call void @llvm.memset.p0i8.i64(i8* %{{.*}}, i8 %{{.*}}, i64 %{{.*}}, i1 false)

// CHECK-PTX: .visible .entry memset_kernel
