// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {

    // shfl.sync bfly i32 (warp-level reduction building block)
    llvm.func @shfl_bfly_i32(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %mask = llvm.mlir.constant(0xFFFFFFFF : i32) : i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %offset = llvm.mlir.constant(16 : i32) : i32
      %clamp = llvm.mlir.constant(31 : i32) : i32
      %result = nvvm.shfl.sync bfly %mask, %tid, %offset, %clamp : i32 -> i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %result, %ptr : i32, !llvm.ptr<1>
      llvm.return
    }

    // shfl.sync bfly f32
    llvm.func @shfl_bfly_f32(%arg0: !llvm.ptr<1>, %arg1: f32) attributes {gpu.kernel} {
      %mask = llvm.mlir.constant(0xFFFFFFFF : i32) : i32
      %offset = llvm.mlir.constant(1 : i32) : i32
      %clamp = llvm.mlir.constant(31 : i32) : i32
      %result = nvvm.shfl.sync bfly %mask, %arg1, %offset, %clamp : f32 -> f32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %result, %ptr : f32, !llvm.ptr<1>
      llvm.return
    }

    // shfl.sync idx i32
    llvm.func @shfl_idx_i32(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %mask = llvm.mlir.constant(0xFFFFFFFF : i32) : i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %src = llvm.mlir.constant(0 : i32) : i32
      %clamp = llvm.mlir.constant(31 : i32) : i32
      %result = nvvm.shfl.sync idx %mask, %tid, %src, %clamp : i32 -> i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %result, %ptr : i32, !llvm.ptr<1>
      llvm.return
    }

    // shfl.sync bfly i32 with predicate return
    llvm.func @shfl_bfly_pred(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %mask = llvm.mlir.constant(0xFFFFFFFF : i32) : i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %offset = llvm.mlir.constant(1 : i32) : i32
      %clamp = llvm.mlir.constant(31 : i32) : i32
      %result = nvvm.shfl.sync bfly %mask, %tid, %offset, %clamp {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
      %val = llvm.extractvalue %result[0] : !llvm.struct<(i32, i1)>
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %val, %ptr : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @shfl_bfly_i32
// CHECK-IR: call i32 @llvm.nvvm.shfl.sync.bfly.i32

// CHECK-IR: define ptx_kernel void @shfl_bfly_f32
// CHECK-IR: call float @llvm.nvvm.shfl.sync.bfly.f32

// CHECK-IR: define ptx_kernel void @shfl_idx_i32
// CHECK-IR: call i32 @llvm.nvvm.shfl.sync.idx.i32

// CHECK-IR: define ptx_kernel void @shfl_bfly_pred
// CHECK-IR: call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p

// CHECK-PTX: .visible .entry shfl_bfly_i32
// CHECK-PTX: shfl.sync.bfly
// CHECK-PTX: .visible .entry shfl_bfly_f32
// CHECK-PTX: shfl.sync.bfly
// CHECK-PTX: .visible .entry shfl_idx_i32
// CHECK-PTX: shfl.sync.idx
// CHECK-PTX: .visible .entry shfl_bfly_pred
// CHECK-PTX: shfl.sync.bfly
