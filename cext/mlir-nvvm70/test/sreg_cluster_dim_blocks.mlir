// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_90">] {

    llvm.func @cluster_nctaid_x(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.cluster.nctaid.x : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @cluster_nctaid_y(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.cluster.nctaid.y : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @cluster_nctaid_z(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.cluster.nctaid.z : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: call i32 asm "mov.u32 $0, %cluster_nctaid.x;", "=r"()
// CHECK-IR: call i32 asm "mov.u32 $0, %cluster_nctaid.y;", "=r"()
// CHECK-IR: call i32 asm "mov.u32 $0, %cluster_nctaid.z;", "=r"()

// CHECK-PTX: .visible .entry cluster_nctaid_x
// CHECK-PTX: .visible .entry cluster_nctaid_y
// CHECK-PTX: .visible .entry cluster_nctaid_z
