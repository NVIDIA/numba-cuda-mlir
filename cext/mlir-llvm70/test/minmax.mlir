// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @minimum_f64(%a: f64, %b: f64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.minimum(%a, %b) : (f64, f64) -> f64
      llvm.store %res, %out : f64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @maximum_f64(%a: f64, %b: f64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.maximum(%a, %b) : (f64, f64) -> f64
      llvm.store %res, %out : f64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @minnum_f32(%a: f32, %b: f32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.minnum(%a, %b) : (f32, f32) -> f32
      llvm.store %res, %out : f32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @maxnum_f32(%a: f32, %b: f32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.maxnum(%a, %b) : (f32, f32) -> f32
      llvm.store %res, %out : f32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @minimum_f64
// CHECK-IR: call double @llvm.minnum.f64(double %{{.*}}, double %{{.*}})
// CHECK-IR: fcmp uno double
// CHECK-IR: fadd double
// CHECK-IR: select i1

// CHECK-IR: define ptx_kernel void @maximum_f64
// CHECK-IR: call double @llvm.maxnum.f64(double %{{.*}}, double %{{.*}})
// CHECK-IR: fcmp uno double
// CHECK-IR: fadd double
// CHECK-IR: select i1

// CHECK-IR: define ptx_kernel void @minnum_f32
// CHECK-IR: call float @llvm.minnum.f32(float %{{.*}}, float %{{.*}})

// CHECK-IR: define ptx_kernel void @maxnum_f32
// CHECK-IR: call float @llvm.maxnum.f32(float %{{.*}}, float %{{.*}})

// CHECK-PTX: .visible .entry minimum_f64
// CHECK-PTX: .visible .entry maximum_f64
// CHECK-PTX: .visible .entry minnum_f32
// CHECK-PTX: .visible .entry maxnum_f32
