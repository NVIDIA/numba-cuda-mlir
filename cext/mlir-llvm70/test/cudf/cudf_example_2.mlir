// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 | FileCheck %s

module {
  gpu.module @numba_cuda_mlir_gpu_module [#nvvm_llvm70.target<O = 3, chip = "sm_75">] attributes {numba_cuda_mlir.link_target} {
    llvm.func @_Z7_kernel5TupleI5ArrayIiLi1E1C7mutable7alignedE5ArrayIbLi1E1C7mutable7alignedEEx14string_view_2ax(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: i64) attributes {numba_cuda_mlir.arg_attrs = [{}, {}, {}, {}, {}], numba_cuda_mlir.orig_arg_types = [memref<?xi32, strided<[?], offset: ?>>, memref<?xi1, strided<[?], offset: ?>>, i64, !llvm.ptr, i64], gpu.kernel, llvm.emit_c_interface, nvvm.kernel} {
      %0 = llvm.mlir.constant(1 : i64) : i64
      %1 = llvm.mlir.constant(17 : i64) : i64
      %2 = llvm.mlir.constant(true) : i1
      %3 = llvm.mlir.undef : !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      %4 = nvvm.read.ptx.sreg.tid.x : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = nvvm.read.ptx.sreg.ntid.x : i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = nvvm.read.ptx.sreg.ctaid.x : i32
      %9 = llvm.sext %8 : i32 to i64
      %10 = llvm.mul %9, %7 : i64
      %11 = llvm.add %10, %5 : i64
      %12 = llvm.trunc %11 : i64 to i32
      %13 = llvm.zext %12 : i32 to i64
      %14 = llvm.icmp "slt" %13, %arg10 : i64
      %15 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
      %16 = llvm.alloca %0 x !llvm.struct<"string_view", (ptr, i32, i32)> : (i64) -> !llvm.ptr
      %17 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
      llvm.cond_br %14, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %18 = llvm.add %13, %arg12 : i64
      %19 = llvm.getelementptr %arg11[%18] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"string_view", (ptr, i32, i32)>
      %20 = llvm.load %19 : !llvm.ptr -> !llvm.struct<"string_view", (ptr, i32, i32)>
      %21 = llvm.insertvalue %20, %3[0] : !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      %22 = llvm.insertvalue %2, %21[1] : !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      llvm.store %22, %15 : !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>, !llvm.ptr
      %23 = llvm.load %15 : !llvm.ptr -> !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      %24 = llvm.extractvalue %23[0] : !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      %25 = llvm.extractvalue %23[1] : !llvm.struct<"Masked(string_view)", (struct<"string_view", (ptr, i32, i32)>, i1)>
      llvm.store %24, %16 : !llvm.struct<"string_view", (ptr, i32, i32)>, !llvm.ptr
      %26 = llvm.call @len(%17, %16) : (!llvm.ptr, !llvm.ptr) -> i32
      %27 = llvm.load %17 : !llvm.ptr -> i32
      %28 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %29 = llvm.mul %11, %arg4 overflow<nsw, nuw> : i64
      %30 = llvm.getelementptr inbounds|nuw %28[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %27, %30 : i32, !llvm.ptr
      %31 = llvm.getelementptr %arg6[%arg7] : (!llvm.ptr, i64) -> !llvm.ptr, i1
      %32 = llvm.mul %11, %arg9 overflow<nsw, nuw> : i64
      %33 = llvm.getelementptr inbounds|nuw %31[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i1
      llvm.store %25, %33 : i1, !llvm.ptr
      llvm.return
    ^bb2:  // pred: ^bb0
      llvm.return
    }
    llvm.func @len(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  }
}

// CHECK: define ptx_kernel void @_Z7_kernel5TupleI5ArrayIiLi1E1C7mutable7alignedE5ArrayIbLi1E1C7mutable7alignedEEx14string_view_2ax
