// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @switch_kernel(%val: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(10 : i32) : i32
      %c2 = llvm.mlir.constant(20 : i32) : i32
      %c3 = llvm.mlir.constant(30 : i32) : i32
      llvm.switch %val : i32, ^default [
        0: ^case0,
        1: ^case1,
        2: ^case2
      ]
    ^case0:
      llvm.store %c1, %out : i32, !llvm.ptr<1>
      llvm.br ^done
    ^case1:
      llvm.store %c2, %out : i32, !llvm.ptr<1>
      llvm.br ^done
    ^case2:
      llvm.store %c3, %out : i32, !llvm.ptr<1>
      llvm.br ^done
    ^default:
      llvm.store %val, %out : i32, !llvm.ptr<1>
      llvm.br ^done
    ^done:
      llvm.return
    }

    // Multiple cases targeting the same block with block arguments.
    llvm.func @switch_phi_kernel(%val: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c10 = llvm.mlir.constant(10 : i32) : i32
      %c20 = llvm.mlir.constant(20 : i32) : i32
      %c30 = llvm.mlir.constant(30 : i32) : i32
      llvm.switch %val : i32, ^merge(%c30 : i32) [
        0: ^merge(%c10 : i32),
        1: ^merge(%c20 : i32),
        2: ^merge(%c30 : i32)
      ]
    ^merge(%result : i32):
      llvm.store %result, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @switch_kernel
// CHECK-IR: switch i32 %{{.*}}, label %{{.*}} [
// CHECK-IR:   i32 0, label
// CHECK-IR:   i32 1, label
// CHECK-IR:   i32 2, label

// CHECK-IR: define ptx_kernel void @switch_phi_kernel
// CHECK-IR: switch i32 %{{.*}}, label %{{.*}} [
// CHECK-IR:   i32 0, label
// CHECK-IR:   i32 1, label
// CHECK-IR:   i32 2, label
// CHECK-IR: phi i32

// CHECK-PTX: .visible .entry switch_kernel
// CHECK-PTX: .visible .entry switch_phi_kernel
