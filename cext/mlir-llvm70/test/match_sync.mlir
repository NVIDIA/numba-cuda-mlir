// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    // match.sync any with i32
    llvm.func @match_any_i32(%mask: i32, %val: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.match.sync any %mask, %val : i32 -> i32
      llvm.store %res, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    // match.sync any with i64
    llvm.func @match_any_i64(%mask: i32, %val: i64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.match.sync any %mask, %val : i64 -> i32
      llvm.store %res, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    // match.sync all with i32 (returns {i32, i1})
    llvm.func @match_all_i32(%mask: i32, %val: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.match.sync all %mask, %val : i32 -> !llvm.struct<(i32, i1)>
      %matched = llvm.extractvalue %res[0] : !llvm.struct<(i32, i1)>
      llvm.store %matched, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @match_any_i32
// CHECK-IR: call i32 @llvm.nvvm.match.any.sync.i32(i32 %{{.*}}, i32 %{{.*}})

// CHECK-IR: define ptx_kernel void @match_any_i64
// CHECK-IR: call i32 @llvm.nvvm.match.any.sync.i64(i32 %{{.*}}, i64 %{{.*}})

// CHECK-IR: define ptx_kernel void @match_all_i32
// CHECK-IR: call { i32, i1 } @llvm.nvvm.match.all.sync.i32(i32 %{{.*}}, i32 %{{.*}})

// CHECK-PTX: .visible .entry match_any_i32
// CHECK-PTX: match.any.sync.b32

// CHECK-PTX: .visible .entry match_any_i64
// CHECK-PTX: match.any.sync.b64

// CHECK-PTX: .visible .entry match_all_i32
// CHECK-PTX: match.all.sync.b32
