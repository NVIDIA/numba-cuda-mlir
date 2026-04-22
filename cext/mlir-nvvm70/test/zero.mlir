// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @zero_i32(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %z = llvm.mlir.zero : i32
      llvm.store %z, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @zero_f32(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %z = llvm.mlir.zero : f32
      llvm.store %z, %out : f32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @zero_ptr(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %z = llvm.mlir.zero : !llvm.ptr<1>
      llvm.store %z, %out : !llvm.ptr<1>, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @zero_i32
// CHECK-IR: store i32 0,

// CHECK-IR: define ptx_kernel void @zero_f32
// CHECK-IR: store float 0.000000e+00,

// CHECK-IR: define ptx_kernel void @zero_ptr
// CHECK-IR: store i8 addrspace(1)* null,

// CHECK-PTX: .visible .entry zero_i32
// CHECK-PTX: .visible .entry zero_f32
// CHECK-PTX: .visible .entry zero_ptr
