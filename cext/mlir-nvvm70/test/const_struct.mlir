// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @const_struct_f64(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %s = llvm.mlir.constant([3.000000e+00, 4.000000e+00]) : !llvm.struct<(f64, f64)>
      %v = llvm.extractvalue %s[0] : !llvm.struct<(f64, f64)>
      llvm.store %v, %out : f64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @const_struct_i32(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %s = llvm.mlir.constant([10 : i32, 20 : i32]) : !llvm.struct<(i32, i32)>
      %v = llvm.extractvalue %s[1] : !llvm.struct<(i32, i32)>
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @const_struct_f64
// CHECK-IR: store double 3.000000e+00,

// CHECK-IR: define ptx_kernel void @const_struct_i32
// CHECK-IR: store i32 20,

// CHECK-PTX: .visible .entry const_struct_f64
// CHECK-PTX: .visible .entry const_struct_i32
