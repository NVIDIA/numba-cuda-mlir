// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_75">] {

    llvm.func @fma_f32(%a: f32, %b: f32, %c: f32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rn>} : f32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %idx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %out[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %res, %ptr : f32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @fma_f64(%a: f64, %b: f64, %c: f64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.fma %a, %b, %c {rnd = #nvvm.fp_rnd_mode<rn>} : f64
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %idx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %out[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      llvm.store %res, %ptr : f64, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @fma_f32
// CHECK-IR: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})

// CHECK-IR: define ptx_kernel void @fma_f64
// CHECK-IR: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})

// CHECK-PTX: .visible .entry fma_f32
// CHECK-PTX: fma.rn.f32

// CHECK-PTX: .visible .entry fma_f64
// CHECK-PTX: fma.rn.f64
