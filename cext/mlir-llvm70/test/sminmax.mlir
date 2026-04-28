// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @smin_i32(%a: i32, %b: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.smin"(%a, %b) : (i32, i32) -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @smax_i32(%a: i32, %b: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.smax"(%a, %b) : (i32, i32) -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @smin_i64(%a: i64, %b: i64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.smin"(%a, %b) : (i64, i64) -> i64
      llvm.store %v, %out : i64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @smax_i64(%a: i64, %b: i64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.smax"(%a, %b) : (i64, i64) -> i64
      llvm.store %v, %out : i64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @umin_i32(%a: i32, %b: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.umin"(%a, %b) : (i32, i32) -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @umax_i32(%a: i32, %b: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.umax"(%a, %b) : (i32, i32) -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: icmp slt i32
// CHECK-IR: select i1
// CHECK-IR: icmp sgt i32
// CHECK-IR: select i1
// CHECK-IR: icmp slt i64
// CHECK-IR: select i1
// CHECK-IR: icmp sgt i64
// CHECK-IR: select i1
// CHECK-IR: icmp ult i32
// CHECK-IR: select i1
// CHECK-IR: icmp ugt i32
// CHECK-IR: select i1

// CHECK-PTX: .visible .entry smin_i32
// CHECK-PTX: .visible .entry smax_i32
// CHECK-PTX: .visible .entry smin_i64
// CHECK-PTX: .visible .entry smax_i64
// CHECK-PTX: .visible .entry umin_i32
// CHECK-PTX: .visible .entry umax_i32
