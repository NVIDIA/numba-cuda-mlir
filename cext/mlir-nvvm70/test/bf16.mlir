// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-ptx 2>&1 | FileCheck %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {
    llvm.func @bf16_store(%ptr: !llvm.ptr<1>, %val: bf16) attributes {gpu.kernel} {
      llvm.store %val, %ptr : bf16, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @bf16_add(%ptr: !llvm.ptr<1>, %a: bf16, %b: bf16) attributes {gpu.kernel} {
      %sum = llvm.fadd %a, %b : bf16
      llvm.store %sum, %ptr : bf16, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @bf16_fpext(%ptr: !llvm.ptr<1>, %a: bf16) attributes {gpu.kernel} {
      %f = llvm.fpext %a : bf16 to f32
      llvm.store %f, %ptr : f32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @bf16_fptrunc(%ptr: !llvm.ptr<1>, %a: f32) attributes {gpu.kernel} {
      %bf = llvm.fptrunc %a : f32 to bf16
      llvm.store %bf, %ptr : bf16, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK: .visible .entry bf16_store
// CHECK: .param .u16 bf16_store_param_1
// CHECK: .visible .entry bf16_add
// CHECK: .visible .entry bf16_fpext
// CHECK: .visible .entry bf16_fptrunc
