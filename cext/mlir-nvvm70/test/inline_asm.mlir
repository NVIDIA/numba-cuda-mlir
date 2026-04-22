// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {

    // Inline asm with side effects and a result.
    llvm.func @inline_asm_kernel(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %tid = llvm.inline_asm has_side_effects "mov.u32 $0, %tid.x;", "=r" : () -> i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %val = llvm.sitofp %tid : i32 to f32
      llvm.store %val, %ptr : f32, !llvm.ptr<1>
      llvm.return
    }

    // Inline asm with an input operand.
    llvm.func @inline_asm_with_input(%arg0: !llvm.ptr<1>, %arg1: i32) attributes {gpu.kernel} {
      %result = llvm.inline_asm has_side_effects "add.u32 $0, $1, 1;", "=r,r" %arg1 : (i32) -> i32
      %idx = llvm.sext %result : i32 to i64
      %ptr = llvm.getelementptr %arg0[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %result, %ptr : i32, !llvm.ptr<1>
      llvm.return
    }

    // Void inline asm (no result).
    llvm.func @inline_asm_void(%arg0: i32) attributes {gpu.kernel} {
      llvm.inline_asm has_side_effects "bar.sync $0;", "r" %arg0 : (i32) -> ()
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @inline_asm_kernel
// CHECK-IR: call i32 asm sideeffect "mov.u32 $0, %tid.x;"

// CHECK-IR: define ptx_kernel void @inline_asm_with_input
// CHECK-IR: call i32 asm sideeffect "add.u32 $0, $1, 1;"

// CHECK-IR: define ptx_kernel void @inline_asm_void
// CHECK-IR: call void asm sideeffect "bar.sync $0;"

// CHECK-PTX: .visible .entry inline_asm_kernel
// CHECK-PTX: .visible .entry inline_asm_with_input
// CHECK-PTX: .visible .entry inline_asm_void
