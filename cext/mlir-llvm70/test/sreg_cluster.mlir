// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_90">] {

    llvm.func @clusterid_x(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.clusterid.x : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @clusterid_y(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.clusterid.y : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @clusterid_z(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.clusterid.z : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @nclusterid_x(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.nclusterid.x : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @nclusterid_y(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.nclusterid.y : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @nclusterid_z(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = nvvm.read.ptx.sreg.nclusterid.z : i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: call i32 asm "mov.u32 $0, %clusterid.x;", "=r"()
// CHECK-IR: call i32 asm "mov.u32 $0, %clusterid.y;", "=r"()
// CHECK-IR: call i32 asm "mov.u32 $0, %clusterid.z;", "=r"()
// CHECK-IR: call i32 asm "mov.u32 $0, %nclusterid.x;", "=r"()
// CHECK-IR: call i32 asm "mov.u32 $0, %nclusterid.y;", "=r"()
// CHECK-IR: call i32 asm "mov.u32 $0, %nclusterid.z;", "=r"()

// CHECK-PTX: .visible .entry clusterid_x
// CHECK-PTX: .visible .entry clusterid_y
// CHECK-PTX: .visible .entry clusterid_z
// CHECK-PTX: .visible .entry nclusterid_x
// CHECK-PTX: .visible .entry nclusterid_y
// CHECK-PTX: .visible .entry nclusterid_z
