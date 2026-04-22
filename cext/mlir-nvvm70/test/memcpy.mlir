// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @memcpy_kernel(%dst: !llvm.ptr, %src: !llvm.ptr, %len: i64) attributes {gpu.kernel} {
      "llvm.intr.memcpy"(%dst, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
      llvm.return
    }

    llvm.func @memcpy_global(%dst: !llvm.ptr<1>, %src: !llvm.ptr<1>, %len: i64) attributes {gpu.kernel} {
      "llvm.intr.memcpy"(%dst, %src, %len) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<1>, i64) -> ()
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @memcpy_kernel
// CHECK-IR: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %{{.*}}, i8* %{{.*}}, i64 %{{.*}}, i1 false)

// CHECK-IR: define ptx_kernel void @memcpy_global
// CHECK-IR: call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %{{.*}}, i8 addrspace(1)* %{{.*}}, i64 %{{.*}}, i1 false)

// CHECK-PTX: .visible .entry memcpy_kernel
// CHECK-PTX: .visible .entry memcpy_global
