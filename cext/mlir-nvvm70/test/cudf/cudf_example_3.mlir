// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 | FileCheck %s

// Tests addressof bitcast from element pointer to i8* for LLVM 7 compat
// (string constant pattern from cuDF UDFs).

module {
  gpu.module @cusimt_gpu_module [#nvvm70.target<O = 3, chip = "sm_75">] attributes {cusimt.link_target} {
    llvm.mlir.global internal constant @__cusimt_str_1("h\00") {addr_space = 0 : i32} : !llvm.array<2 x i8>
    llvm.func @_Z12_3clambda_3e3RowILi831EE(%arg0: !llvm.ptr) -> !llvm.struct<"Masked(bool)", (i1, i1)> attributes {cusimt.arg_attrs = [{}], cusimt.orig_arg_types = [!llvm.ptr]} {
      %0 = llvm.mlir.undef : !llvm.struct<"Masked(bool)", (i1, i1)>
      %1 = llvm.mlir.undef : !llvm.struct<"string_view", (ptr, i32, i32)>
      %2 = llvm.mlir.constant(1 : i32) : i32
      %3 = llvm.mlir.constant(1 : i64) : i64
      %4 = llvm.mlir.addressof @__cusimt_str_1 : !llvm.ptr
      %5 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      %6 = llvm.extractvalue %5[0] : !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      %7 = llvm.extractvalue %5[1] : !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      %8 = llvm.insertvalue %4, %1[0] : !llvm.struct<"string_view", (ptr, i32, i32)>
      %9 = llvm.insertvalue %2, %8[1] : !llvm.struct<"string_view", (ptr, i32, i32)>
      %10 = llvm.insertvalue %2, %9[2] : !llvm.struct<"string_view", (ptr, i32, i32)>
      %11 = llvm.alloca %3 x !llvm.struct<"string_view", (ptr, i32, i32)> : (i64) -> !llvm.ptr
      %12 = llvm.alloca %3 x !llvm.struct<"string_view", (ptr, i32, i32)> : (i64) -> !llvm.ptr
      llvm.store %6, %11 : !llvm.struct<"string_view", (ptr, i32, i32)>, !llvm.ptr
      llvm.store %10, %12 : !llvm.struct<"string_view", (ptr, i32, i32)>, !llvm.ptr
      %13 = llvm.alloca %3 x i1 : (i64) -> !llvm.ptr
      %14 = llvm.call @startswith(%13, %11, %12) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %15 = llvm.load %13 : !llvm.ptr -> i1
      %16 = llvm.insertvalue %15, %0[0] : !llvm.struct<"Masked(bool)", (i1, i1)>
      %17 = llvm.insertvalue %7, %16[1] : !llvm.struct<"Masked(bool)", (i1, i1)>
      llvm.return %17 : !llvm.struct<"Masked(bool)", (i1, i1)>
    }
    llvm.func @startswith(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  }
}

// CHECK: @__cusimt_str_1 = internal global [2 x i8] c"h\00"
// CHECK: define { i1, i1 } @_Z12_3clambda_3e3RowILi831EE
