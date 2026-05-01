// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @ldexp_f64(%val: f64, %exp: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.ldexp(%val, %exp) : (f64, i32) -> f64
      llvm.store %res, %out : f64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @ldexp_f32(%val: f32, %exp: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.ldexp(%val, %exp) : (f32, i32) -> f32
      llvm.store %res, %out : f32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @ldexp_f64
// CHECK-IR: call double @__nv_ldexp(double %{{.*}}, i32 %{{.*}})

// CHECK-IR: define ptx_kernel void @ldexp_f32
// CHECK-IR: call float @__nv_ldexpf(float %{{.*}}, i32 %{{.*}})

// CHECK-PTX: .visible .entry ldexp_f64
// CHECK-PTX: .visible .entry ldexp_f32
