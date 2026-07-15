// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

// Per-instruction fast-math flags cannot be set through the LLVM 7 C API;
// the translator transfers them by naming flagged instructions with a
// __fmf.<mask>_<n> marker and injecting the keywords into the printed IR.
// Check that full and selective flag sets survive translation and that
// unflagged instructions stay clean.

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {
    llvm.func @flags(%a: f32, %b: f32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %full = llvm.fmul %a, %b {fastmathFlags = #llvm.fastmath<fast>} : f32
      %some = llvm.fadd %full, %b {fastmathFlags = #llvm.fastmath<nnan, arcp>} : f32
      %div = llvm.fdiv %some, %a {fastmathFlags = #llvm.fastmath<arcp>} : f32
      %none = llvm.fsub %div, %b : f32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %idx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %out[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %none, %ptr : f32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @flags
// CHECK-IR: fmul fast float
// CHECK-IR: fadd nnan arcp float
// CHECK-IR: fdiv arcp float
// CHECK-IR: fsub float

// CHECK-PTX: .visible .entry flags
