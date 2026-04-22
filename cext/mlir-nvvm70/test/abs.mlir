// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @abs_i32(%a: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.abs"(%a) <{is_int_min_poison = false}> : (i32) -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @abs_i64(%a: i64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.abs"(%a) <{is_int_min_poison = true}> : (i64) -> i64
      llvm.store %v, %out : i64, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: icmp slt i32 %{{.*}}, 0
// CHECK-IR: select i1
// CHECK-IR: icmp slt i64 %{{.*}}, 0
// CHECK-IR: select i1

// CHECK-PTX: .visible .entry abs_i32
// CHECK-PTX: .visible .entry abs_i64
