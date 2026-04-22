// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate --mlir-to-llvm-ir %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels <#gpu.select_object<0>> [#nvvm70.target<chip = "sm_80">] {
    llvm.func @simple_kernel(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %tidx = llvm.sext %tid : i32 to i64
      %ptr = llvm.getelementptr %arg0[%tidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %val = llvm.sitofp %tid : i32 to f32
      llvm.store %val, %ptr : f32, !llvm.ptr<1>
      llvm.return
    }
  }

  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}

// CHECK-DAG: @kernels_binary = internal constant
// CHECK-DAG: @llvm.global_ctors = appending global
// CHECK-DAG: @llvm.global_dtors = appending global
// CHECK-DAG: define internal void @kernels_load()
// CHECK-DAG: define internal void @kernels_unload()
// CHECK-DAG: define i32 @main()
