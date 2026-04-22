// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR  --dump-input=fail %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX  --dump-input=fail %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {

    // elect.sync with explicit mask
    llvm.func @elect_with_mask(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %mask = llvm.mlir.constant(0xFFFFFFFF : i32) : i32
      %is_leader = nvvm.elect.sync %mask -> i1
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      llvm.cond_br %is_leader, ^leader, ^done
    ^leader:
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %tid, %ptr : i32, !llvm.ptr<1>
      llvm.br ^done
    ^done:
      llvm.return
    }

    // elect.sync without mask (defaults to 0xFFFFFFFF)
    llvm.func @elect_no_mask() attributes {gpu.kernel} {
      %is_leader = nvvm.elect.sync -> i1
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @elect_with_mask
// CHECK-IR: call i32 asm sideeffect
// CHECK-IR: elect.sync

// CHECK-IR: define ptx_kernel void @elect_no_mask
// CHECK-IR: call i32 asm sideeffect
// CHECK-IR: elect.sync

// CHECK-PTX: .visible .entry elect_with_mask
// CHECK-PTX: elect.sync
// CHECK-PTX: .visible .entry elect_no_mask
// CHECK-PTX: elect.sync
