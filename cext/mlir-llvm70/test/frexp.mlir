// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @frexp_f64(%val: f64, %out_m: !llvm.ptr<1>, %out_e: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.frexp(%val) : (f64) -> !llvm.struct<(f64, i32)>
      %m = llvm.extractvalue %res[0] : !llvm.struct<(f64, i32)>
      %e = llvm.extractvalue %res[1] : !llvm.struct<(f64, i32)>
      llvm.store %m, %out_m : f64, !llvm.ptr<1>
      llvm.store %e, %out_e : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @frexp_f32(%val: f32, %out_m: !llvm.ptr<1>, %out_e: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = llvm.intr.frexp(%val) : (f32) -> !llvm.struct<(f32, i32)>
      %m = llvm.extractvalue %res[0] : !llvm.struct<(f32, i32)>
      %e = llvm.extractvalue %res[1] : !llvm.struct<(f32, i32)>
      llvm.store %m, %out_m : f32, !llvm.ptr<1>
      llvm.store %e, %out_e : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @frexp_f64
// CHECK-IR: call double @__nv_frexp(double %{{.*}}, i32*

// CHECK-IR: define ptx_kernel void @frexp_f32
// CHECK-IR: call float @__nv_frexpf(float %{{.*}}, i32*

// CHECK-PTX: .visible .entry frexp_f64
// CHECK-PTX: .visible .entry frexp_f32
