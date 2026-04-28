// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @trunc_f32(%a: f32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.trunc"(%a) : (f32) -> f32
      llvm.store %v, %out : f32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @trunc_f64(%a: f64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = "llvm.intr.trunc"(%a) : (f64) -> f64
      llvm.store %v, %out : f64, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: call float @llvm.trunc.f32(float %{{.*}})
// CHECK-IR: call double @llvm.trunc.f64(double %{{.*}})

// CHECK-PTX: .visible .entry trunc_f32
// CHECK-PTX: .visible .entry trunc_f64
