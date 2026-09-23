// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck %s

// Identified (named) MLIR struct types must survive translation as named LLVM
// struct types rather than being flattened into literal structs.

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @named_struct_alloca(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %p = llvm.alloca %c1 x !llvm.struct<"Inner", (i32, i32)> : (i64) -> !llvm.ptr
      %v = llvm.load %p : !llvm.ptr -> !llvm.struct<"Inner", (i32, i32)>
      %e = llvm.extractvalue %v[0] : !llvm.struct<"Inner", (i32, i32)>
      llvm.store %e, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    // "Inner" is referenced again here, and nested inside "Outer". Both must
    // resolve to the same LLVM type rather than a fresh %Inner.0.
    llvm.func @named_struct_nested(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %p = llvm.alloca %c1 x !llvm.struct<"Outer", (struct<"Inner", (i32, i32)>, i1)> : (i64) -> !llvm.ptr
      %v = llvm.load %p : !llvm.ptr -> !llvm.struct<"Outer", (struct<"Inner", (i32, i32)>, i1)>
      %i = llvm.extractvalue %v[0] : !llvm.struct<"Outer", (struct<"Inner", (i32, i32)>, i1)>
      %e = llvm.extractvalue %i[1] : !llvm.struct<"Inner", (i32, i32)>
      llvm.store %e, %out : i32, !llvm.ptr<1>
      llvm.return
    }

    // Literal structs must stay literal.
    llvm.func @literal_struct(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %p = llvm.alloca %c1 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
      %v = llvm.load %p : !llvm.ptr -> !llvm.struct<(i32, i32)>
      %e = llvm.extractvalue %v[0] : !llvm.struct<(i32, i32)>
      llvm.store %e, %out : i32, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK: %Inner = type { i32, i32 }
// CHECK: %Outer = type { %Inner, i1 }
// CHECK-NOT: %Inner.0

// CHECK-LABEL: define ptx_kernel void @named_struct_alloca
// CHECK: alloca %Inner
// CHECK: load %Inner, %Inner*
// CHECK: extractvalue %Inner

// CHECK-LABEL: define ptx_kernel void @named_struct_nested
// CHECK: alloca %Outer
// CHECK: load %Outer, %Outer*
// CHECK: extractvalue %Outer
// CHECK: extractvalue %Inner

// CHECK-LABEL: define ptx_kernel void @literal_struct
// CHECK: alloca { i32, i32 }
// CHECK: load { i32, i32 }, { i32, i32 }*
