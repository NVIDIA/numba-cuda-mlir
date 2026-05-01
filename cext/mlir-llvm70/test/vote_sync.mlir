// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_80">] {

    // vote.sync ballot — returns i32 bitmask.
    llvm.func @vote_ballot(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %mask = llvm.mlir.constant(0xFFFFFFFF : i32) : i32
      %true = llvm.mlir.constant(true) : i1
      %ballot = nvvm.vote.sync ballot %mask, %true -> i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
      llvm.store %ballot, %ptr : i32, !llvm.ptr<1>
      llvm.return
    }

    // vote.sync any — returns i1.
    llvm.func @vote_any(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %mask = llvm.mlir.constant(0xFFFFFFFF : i32) : i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %zero = llvm.mlir.constant(0 : i32) : i32
      %pred = llvm.icmp "eq" %tid, %zero : i32
      %any = nvvm.vote.sync any %mask, %pred -> i1
      llvm.return
    }

    // vote.sync all — returns i1.
    llvm.func @vote_all(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %mask = llvm.mlir.constant(0xFFFFFFFF : i32) : i32
      %true = llvm.mlir.constant(true) : i1
      %all = nvvm.vote.sync all %mask, %true -> i1
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @vote_ballot
// CHECK-IR: call i32 @llvm.nvvm.vote.ballot.sync

// CHECK-IR: define ptx_kernel void @vote_any
// CHECK-IR: call i1 @llvm.nvvm.vote.any.sync

// CHECK-IR: define ptx_kernel void @vote_all
// CHECK-IR: call i1 @llvm.nvvm.vote.all.sync

// CHECK-PTX: .visible .entry vote_ballot
// CHECK-PTX: vote.sync.ballot
// CHECK-PTX: .visible .entry vote_any
// CHECK-PTX: vote.sync.any
// CHECK-PTX: .visible .entry vote_all
// CHECK-PTX: vote.sync.all
