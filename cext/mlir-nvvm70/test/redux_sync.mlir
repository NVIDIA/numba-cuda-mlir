// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {

    llvm.func @redux_add_i32(%val: i32, %mask: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.redux.sync add %val, %mask : i32 -> i32
      llvm.store %res, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @redux_max_i32(%val: i32, %mask: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.redux.sync max %val, %mask : i32 -> i32
      llvm.store %res, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @redux_and_i32(%val: i32, %mask: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.redux.sync and %val, %mask : i32 -> i32
      llvm.store %res, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @redux_umin_i32(%val: i32, %mask: i32, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %res = nvvm.redux.sync umin %val, %mask : i32 -> i32
      llvm.store %res, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @redux_add_i32
// CHECK-IR: call i32 asm sideeffect "redux.sync.add.s32 $0, $1, $2;", "=r,r,r"(i32 %{{.*}}, i32 %{{.*}})

// CHECK-IR: define ptx_kernel void @redux_max_i32
// CHECK-IR: call i32 asm sideeffect "redux.sync.max.s32 $0, $1, $2;", "=r,r,r"(i32 %{{.*}}, i32 %{{.*}})

// CHECK-IR: define ptx_kernel void @redux_and_i32
// CHECK-IR: call i32 asm sideeffect "redux.sync.and.b32 $0, $1, $2;", "=r,r,r"(i32 %{{.*}}, i32 %{{.*}})

// CHECK-IR: define ptx_kernel void @redux_umin_i32
// CHECK-IR: call i32 asm sideeffect "redux.sync.min.u32 $0, $1, $2;", "=r,r,r"(i32 %{{.*}}, i32 %{{.*}})

// CHECK-PTX: .visible .entry redux_add_i32
// CHECK-PTX: redux.sync.add.s32

// CHECK-PTX: .visible .entry redux_max_i32
// CHECK-PTX: redux.sync.max.s32

// CHECK-PTX: .visible .entry redux_and_i32
// CHECK-PTX: redux.sync.and.b32

// CHECK-PTX: .visible .entry redux_umin_i32
// CHECK-PTX: redux.sync.min.u32
