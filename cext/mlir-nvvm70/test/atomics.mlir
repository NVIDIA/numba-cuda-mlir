// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {

    // Kernel using atomicrmw add (i32) — the most common GPU atomic.
    llvm.func @atomic_add_kernel(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %one = llvm.mlir.constant(1 : i32) : i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      %old = llvm.atomicrmw add %ptr, %one monotonic : !llvm.ptr<1>, i32
      llvm.return
    }

    // Kernel using atomicrmw xchg (i32).
    llvm.func @atomic_xchg_kernel(%arg0: !llvm.ptr<1>, %arg1: i32) attributes {gpu.kernel} {
      %old = llvm.atomicrmw xchg %arg0, %arg1 monotonic : !llvm.ptr<1>, i32
      llvm.return
    }

    // Kernel using atomicrmw max (i32, signed).
    llvm.func @atomic_max_kernel(%arg0: !llvm.ptr<1>, %arg1: i32) attributes {gpu.kernel} {
      %old = llvm.atomicrmw max %arg0, %arg1 monotonic : !llvm.ptr<1>, i32
      llvm.return
    }

    // Kernel using cmpxchg (i32).
    llvm.func @atomic_cas_kernel(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32) attributes {gpu.kernel} {
      %result = llvm.cmpxchg %arg0, %arg1, %arg2 seq_cst monotonic : !llvm.ptr<1>, i32
      %val = llvm.extractvalue %result[0] : !llvm.struct<(i32, i1)>
      %success = llvm.extractvalue %result[1] : !llvm.struct<(i32, i1)>
      llvm.return
    }

    // Kernel using atomicrmw add (i64).
    llvm.func @atomic_add_i64_kernel(%arg0: !llvm.ptr<1>, %arg1: i64) attributes {gpu.kernel} {
      %old = llvm.atomicrmw add %arg0, %arg1 monotonic : !llvm.ptr<1>, i64
      llvm.return
    }

    // Kernel using atomicrmw fadd (f32).
    llvm.func @atomic_fadd_f32_kernel(%arg0: !llvm.ptr<1>, %arg1: f32) attributes {gpu.kernel} {
      %old = llvm.atomicrmw fadd %arg0, %arg1 monotonic : !llvm.ptr<1>, f32
      llvm.return
    }

    // Kernel using atomicrmw fadd (f64).
    llvm.func @atomic_fadd_f64_kernel(%arg0: !llvm.ptr<1>, %arg1: f64) attributes {gpu.kernel} {
      %old = llvm.atomicrmw fadd %arg0, %arg1 monotonic : !llvm.ptr<1>, f64
      llvm.return
    }

    // Kernel using atomicrmw fmax (f32) — CAS loop, maxnum semantics.
    llvm.func @atomic_fmax_f32_kernel(%arg0: !llvm.ptr<1>, %arg1: f32) attributes {gpu.kernel} {
      %old = llvm.atomicrmw fmax %arg0, %arg1 monotonic : !llvm.ptr<1>, f32
      llvm.return
    }

    // Kernel using atomicrmw fmin (f64) — CAS loop, minnum semantics.
    llvm.func @atomic_fmin_f64_kernel(%arg0: !llvm.ptr<1>, %arg1: f64) attributes {gpu.kernel} {
      %old = llvm.atomicrmw fmin %arg0, %arg1 monotonic : !llvm.ptr<1>, f64
      llvm.return
    }

    // Kernel using atomicrmw fminimum (f64) — CAS loop, NaN-propagating.
    llvm.func @atomic_fminimum_f64_kernel(%arg0: !llvm.ptr<1>, %arg1: f64) attributes {gpu.kernel} {
      %old = llvm.atomicrmw fminimum %arg0, %arg1 monotonic : !llvm.ptr<1>, f64
      llvm.return
    }

    // Kernel using atomicrmw fmaximum (f32) — CAS loop, NaN-propagating.
    llvm.func @atomic_fmaximum_f32_kernel(%arg0: !llvm.ptr<1>, %arg1: f32) attributes {gpu.kernel} {
      %old = llvm.atomicrmw fmaximum %arg0, %arg1 monotonic : !llvm.ptr<1>, f32
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @atomic_add_kernel
// CHECK-IR: atomicrmw add

// CHECK-IR: define ptx_kernel void @atomic_xchg_kernel
// CHECK-IR: atomicrmw xchg

// CHECK-IR: define ptx_kernel void @atomic_max_kernel
// CHECK-IR: atomicrmw max

// CHECK-IR: define ptx_kernel void @atomic_cas_kernel
// CHECK-IR: cmpxchg

// CHECK-IR: define ptx_kernel void @atomic_add_i64_kernel
// CHECK-IR: atomicrmw add

// CHECK-IR: define ptx_kernel void @atomic_fadd_f32_kernel
// CHECK-IR: call float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)*

// CHECK-IR: define ptx_kernel void @atomic_fadd_f64_kernel
// CHECK-IR: call double @llvm.nvvm.atomic.load.add.f64.p1f64(double addrspace(1)*

// fmax f32: CAS loop with correct NaN handling (maxnum semantics).
// Early-exit when val is NaN. Loop swaps when (dold < val) OR (dold is NaN).
// CHECK-IR: define ptx_kernel void @atomic_fmax_f32_kernel
// CHECK-IR: load float, float addrspace(1)*
// CHECK-IR: fcmp uno float %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-IR: br i1
// CHECK-IR: phi float
// CHECK-IR: fcmp olt float
// CHECK-IR: fcmp uno float
// CHECK-IR: or i1
// CHECK-IR: cmpxchg i32 addrspace(1)*
// Result phi merges early-exit value with CAS loop value.
// CHECK-IR: phi float

// fmin f64: CAS loop with correct NaN handling (minnum semantics).
// Early-exit when val is NaN. Loop swaps when (dold > val) OR (dold is NaN).
// CHECK-IR: define ptx_kernel void @atomic_fmin_f64_kernel
// CHECK-IR: load double, double addrspace(1)*
// CHECK-IR: fcmp uno double %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-IR: br i1
// CHECK-IR: phi double
// CHECK-IR: fcmp ogt double
// CHECK-IR: fcmp uno double
// CHECK-IR: or i1
// CHECK-IR: cmpxchg i64 addrspace(1)*
// CHECK-IR: phi double

// fminimum f64: CAS loop with maxnum/minnum semantics (matches hardware atom.min).
// Early-exit when val is NaN. Loop swaps when (dold > val) OR (dold is NaN).
// CHECK-IR: define ptx_kernel void @atomic_fminimum_f64_kernel
// CHECK-IR: load double, double addrspace(1)*
// CHECK-IR: fcmp uno double %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-IR: br i1
// CHECK-IR: phi double
// CHECK-IR: fcmp ogt double
// CHECK-IR: fcmp uno double
// CHECK-IR: or i1
// CHECK-IR: cmpxchg i64 addrspace(1)*
// CHECK-IR: phi double

// fmaximum f32: CAS loop with maxnum/minnum semantics (matches hardware atom.max).
// Early-exit when val is NaN. Loop swaps when (dold < val) OR (dold is NaN).
// CHECK-IR: define ptx_kernel void @atomic_fmaximum_f32_kernel
// CHECK-IR: load float, float addrspace(1)*
// CHECK-IR: fcmp uno float %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-IR: br i1
// CHECK-IR: phi float
// CHECK-IR: fcmp olt float
// CHECK-IR: fcmp uno float
// CHECK-IR: or i1
// CHECK-IR: cmpxchg i32 addrspace(1)*

// CHECK-PTX: .visible .entry atomic_add_kernel
// CHECK-PTX: {{red.global.add|atom.global.add.u32}}
// CHECK-PTX: .visible .entry atomic_cas_kernel
// CHECK-PTX: {{atom.global.cas|atom.global.add}}
// CHECK-PTX: .visible .entry atomic_fadd_f32_kernel
// CHECK-PTX: {{atom.global.add.f32|atom.global.cas.b32|red.global.add.f32}}
// CHECK-PTX: .visible .entry atomic_fadd_f64_kernel
// CHECK-PTX: {{atom.global.add.f64|atom.global.cas.b64|red.global.add.f64}}
// CHECK-PTX: .visible .entry atomic_fmax_f32_kernel
// CHECK-PTX: atom.global.cas.b32
// CHECK-PTX: .visible .entry atomic_fmin_f64_kernel
// CHECK-PTX: atom.global.cas.b64
// CHECK-PTX: .visible .entry atomic_fminimum_f64_kernel
// CHECK-PTX: atom.global.cas.b64
// CHECK-PTX: .visible .entry atomic_fmaximum_f32_kernel
// CHECK-PTX: atom.global.cas.b32
