// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @barrier_id(%id: i32) attributes {gpu.kernel} {
      nvvm.barrier id = %id
      llvm.return
    }

    llvm.func @barrier_id_threads(%id: i32, %n: i32) attributes {gpu.kernel} {
      nvvm.barrier id = %id number_of_threads = %n
      llvm.return
    }

    llvm.func @barrier_no_args() attributes {gpu.kernel} {
      nvvm.barrier
      llvm.return
    }

    llvm.func @barrier_red_popc(%pred: i32) -> i32 attributes {gpu.kernel} {
      %r = nvvm.barrier #nvvm.reduction<popc> %pred -> i32
      llvm.return %r : i32
    }

    llvm.func @barrier_red_and(%pred: i32) -> i32 attributes {gpu.kernel} {
      %r = nvvm.barrier #nvvm.reduction<and> %pred -> i32
      llvm.return %r : i32
    }
  }
}

// CHECK-IR: define ptx_kernel void @barrier_id
// CHECK-IR: call void asm sideeffect "bar.sync $0;", "r"

// CHECK-IR: define ptx_kernel void @barrier_id_threads
// CHECK-IR: call void asm sideeffect "bar.sync $0, $1;", "r,r"

// CHECK-IR: define ptx_kernel void @barrier_no_args
// CHECK-IR: call void asm sideeffect "bar.sync 0;"

// CHECK-IR: define ptx_kernel i32 @barrier_red_popc
// CHECK-IR: call i32 @llvm.nvvm.barrier0.popc(i32 %{{.*}})

// CHECK-IR: define ptx_kernel i32 @barrier_red_and
// CHECK-IR: call i32 @llvm.nvvm.barrier0.and(i32 %{{.*}})

// CHECK-PTX: .visible .entry barrier_id
// CHECK-PTX: bar.sync
// CHECK-PTX: .visible .entry barrier_id_threads
// CHECK-PTX: bar.sync
// CHECK-PTX: .visible .entry barrier_no_args
// CHECK-PTX: bar.sync
// CHECK-PTX: barrier_red_popc
// CHECK-PTX: bar.red.popc
// CHECK-PTX: barrier_red_and
// CHECK-PTX: bar.red.and
