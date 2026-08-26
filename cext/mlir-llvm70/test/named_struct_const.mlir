// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck %s

// An aggregate constant whose declared type is an identified struct must be
// built as a constant of that named type. Building it as a literal struct
// constant leaves the value's type disagreeing with the type of everything it
// flows into, which NVVM rejects with "Explicit load/store type does not match
// pointee type of pointer operand".

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    // The constant is stored whole, so its type has to match the pointee type.
    llvm.func @named_const_store(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %s = llvm.mlir.constant([1 : i32, 2 : i32]) : !llvm.struct<"Pair", (i32, i32)>
      %p = llvm.alloca %c1 x !llvm.struct<"Pair", (i32, i32)> : (i64) -> !llvm.ptr
      llvm.store %s, %p : !llvm.struct<"Pair", (i32, i32)>, !llvm.ptr
      llvm.return
    }

    // Literal aggregate constants must keep working.
    llvm.func @literal_const_store(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %s = llvm.mlir.constant([3 : i32, 4 : i32]) : !llvm.struct<(i32, i32)>
      %p = llvm.alloca %c1 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
      llvm.store %s, %p : !llvm.struct<(i32, i32)>, !llvm.ptr
      llvm.return
    }
  }
}

// CHECK: %Pair = type { i32, i32 }

// CHECK-LABEL: define ptx_kernel void @named_const_store
// CHECK: %[[P:.*]] = alloca %Pair
// CHECK: store %Pair { i32 1, i32 2 }, %Pair* %[[P]]

// CHECK-LABEL: define ptx_kernel void @literal_const_store
// CHECK: %[[Q:.*]] = alloca { i32, i32 }
// CHECK: store { i32, i32 } { i32 3, i32 4 }, { i32, i32 }* %[[Q]]
