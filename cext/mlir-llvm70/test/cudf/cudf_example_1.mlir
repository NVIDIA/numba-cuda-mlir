// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 | FileCheck %s

module {
  gpu.module @cusimt_gpu_module [#nvvm_llvm70.target<chip = "sm_75">] attributes {cusimt.link_target} {
    llvm.func @_Z7_kernel5TupleI5ArrayIxLi1E1C7mutable7alignedE5ArrayIbLi1E1C7mutable7alignedEEx5TupleI5ArrayIxLi1E1C7mutable7alignedE5ArrayIiLi1E1C7mutable7alignedEE5TupleI5ArrayIxLi1E1C7mutable7alignedE5ArrayIiLi1E1C7mutable7alignedEExx(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr, %arg22: !llvm.ptr, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64) attributes {cusimt.arg_attrs = [{}, {}, {}, {}, {}, {}, {}, {}, {}], cusimt.orig_arg_types = [memref<?xi64, strided<[?], offset: ?>>, memref<?xi1, strided<[?], offset: ?>>, i64, memref<?xi64, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>, memref<?xi64, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>, i64, i64], gpu.kernel, llvm.emit_c_interface, nvvm.kernel} {
      %0 = llvm.mlir.constant(1 : i64) : i64
      %1 = llvm.mlir.constant(25 : i64) : i64
      %2 = llvm.mlir.constant(32 : i64) : i64
      %3 = llvm.mlir.constant(0 : i64) : i64
      %4 = llvm.mlir.undef : !llvm.struct<"Masked(int64)", (i64, i1)>
      %5 = nvvm.read.ptx.sreg.tid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = nvvm.read.ptx.sreg.ntid.x : i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = llvm.mul %10, %8 : i64
      %12 = llvm.add %11, %6 : i64
      %13 = llvm.trunc %12 : i64 to i32
      %14 = llvm.zext %13 : i32 to i64
      %15 = llvm.icmp "slt" %14, %arg10 : i64
      %16 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
      llvm.cond_br %15, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %17 = llvm.add %14, %arg31 : i64
      %18 = llvm.getelementptr %arg12[%arg13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %19 = llvm.mul %17, %arg15 overflow<nsw, nuw> : i64
      %20 = llvm.getelementptr inbounds|nuw %18[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %21 = llvm.load %20 : !llvm.ptr -> i64
      %22 = llvm.udiv %17, %2 : i64
      %23 = llvm.getelementptr %arg17[%arg18] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %24 = llvm.mul %22, %arg20 overflow<nsw, nuw> : i64
      %25 = llvm.getelementptr inbounds|nuw %23[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %26 = llvm.load %25 : !llvm.ptr -> i32
      %27 = llvm.srem %17, %2 : i64
      %28 = llvm.zext %26 : i32 to i64
      %29 = llvm.ashr %28, %27 : i64
      %30 = llvm.and %29, %0 : i64
      %31 = llvm.icmp "ne" %30, %3 : i64
      %32 = llvm.insertvalue %21, %4[0] : !llvm.struct<"Masked(int64)", (i64, i1)> 
      %33 = llvm.insertvalue %31, %32[1] : !llvm.struct<"Masked(int64)", (i64, i1)> 
      %34 = llvm.add %14, %arg32 : i64
      %35 = llvm.getelementptr %arg22[%arg23] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %36 = llvm.mul %34, %arg25 overflow<nsw, nuw> : i64
      %37 = llvm.getelementptr inbounds|nuw %35[%36] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %38 = llvm.load %37 : !llvm.ptr -> i64
      %39 = llvm.udiv %34, %2 : i64
      %40 = llvm.getelementptr %arg27[%arg28] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %41 = llvm.mul %39, %arg30 overflow<nsw, nuw> : i64
      %42 = llvm.getelementptr inbounds|nuw %40[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %43 = llvm.load %42 : !llvm.ptr -> i32
      %44 = llvm.srem %34, %2 : i64
      %45 = llvm.zext %43 : i32 to i64
      %46 = llvm.ashr %45, %44 : i64
      %47 = llvm.and %46, %0 : i64
      %48 = llvm.icmp "ne" %47, %3 : i64
      %49 = llvm.insertvalue %38, %4[0] : !llvm.struct<"Masked(int64)", (i64, i1)> 
      %50 = llvm.insertvalue %48, %49[1] : !llvm.struct<"Masked(int64)", (i64, i1)> 
      llvm.store %33, %16 : !llvm.struct<"Masked(int64)", (i64, i1)>, !llvm.ptr
      %51 = llvm.getelementptr %16[16] : (!llvm.ptr) -> !llvm.ptr, i8
      llvm.store %50, %51 : !llvm.struct<"Masked(int64)", (i64, i1)>, !llvm.ptr
      %52 = llvm.load %16 : !llvm.ptr -> !llvm.struct<"Masked(int64)", (i64, i1)>
      %53 = llvm.load %51 : !llvm.ptr -> !llvm.struct<"Masked(int64)", (i64, i1)>
      %54 = llvm.extractvalue %52[0] : !llvm.struct<"Masked(int64)", (i64, i1)> 
      %55 = llvm.extractvalue %52[1] : !llvm.struct<"Masked(int64)", (i64, i1)> 
      %56 = llvm.extractvalue %53[0] : !llvm.struct<"Masked(int64)", (i64, i1)> 
      %57 = llvm.extractvalue %53[1] : !llvm.struct<"Masked(int64)", (i64, i1)> 
      %58 = llvm.and %55, %57 : i1
      %59 = llvm.add %54, %56 : i64
      %60 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %61 = llvm.mul %12, %arg4 overflow<nsw, nuw> : i64
      %62 = llvm.getelementptr inbounds|nuw %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      llvm.store %59, %62 : i64, !llvm.ptr
      %63 = llvm.getelementptr %arg6[%arg7] : (!llvm.ptr, i64) -> !llvm.ptr, i1
      %64 = llvm.mul %12, %arg9 overflow<nsw, nuw> : i64
      %65 = llvm.getelementptr inbounds|nuw %63[%64] : (!llvm.ptr, i64) -> !llvm.ptr, i1
      llvm.store %58, %65 : i1, !llvm.ptr
      llvm.return
    ^bb2:  // pred: ^bb0
      llvm.return
    }
  }
}

// CHECK: define ptx_kernel void @_Z7_kernel5TupleI5Array
