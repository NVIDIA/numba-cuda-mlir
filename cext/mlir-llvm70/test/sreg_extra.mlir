// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @read_warpid(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %wid = nvvm.read.ptx.sreg.warpid : i32
      llvm.store %wid, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @read_laneid(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %lid = nvvm.read.ptx.sreg.laneid : i32
      llvm.store %lid, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @read_warpsize(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %ws = nvvm.read.ptx.sreg.warpsize : i32
      llvm.store %ws, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @read_smid(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %sm = nvvm.read.ptx.sreg.smid : i32
      llvm.store %sm, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @read_nwarpid(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %nw = nvvm.read.ptx.sreg.nwarpid : i32
      llvm.store %nw, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @read_warpid
// CHECK-IR: call i32 asm "mov.u32 $0, %warpid;", "=r"()

// CHECK-IR: define ptx_kernel void @read_laneid
// CHECK-IR: call i32 asm "mov.u32 $0, %laneid;", "=r"()

// CHECK-IR: define ptx_kernel void @read_warpsize
// CHECK-IR: call i32 asm "mov.u32 $0, WARP_SZ;", "=r"()

// CHECK-IR: define ptx_kernel void @read_smid
// CHECK-IR: call i32 asm "mov.u32 $0, %smid;", "=r"()

// CHECK-PTX: .visible .entry read_warpid
// CHECK-PTX: mov.u32 %r{{[0-9]+}}, %warpid

// CHECK-PTX: .visible .entry read_laneid
// CHECK-PTX: mov.u32 %r{{[0-9]+}}, %laneid

// CHECK-PTX: .visible .entry read_warpsize
// CHECK-PTX: mov.u32 %r{{[0-9]+}}, WARP_SZ

// CHECK-PTX: .visible .entry read_smid
// CHECK-PTX: mov.u32 %r{{[0-9]+}}, %smid

// CHECK-IR: define ptx_kernel void @read_nwarpid
// CHECK-IR: call i32 asm "mov.u32 $0, %nwarpid;", "=r"()

// CHECK-PTX: .visible .entry read_nwarpid
// CHECK-PTX: mov.u32 %r{{[0-9]+}}, %nwarpid
