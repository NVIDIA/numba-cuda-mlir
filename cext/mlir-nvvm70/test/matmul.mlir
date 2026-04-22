// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck %s

// CHECK: define ptx_kernel void @_Z18cusimt_matmul_smem

module {
  gpu.module @cusimt_gpu_module [#nvvm70.target<chip = "sm_80">] attributes {cusimt.link_target} {
    llvm.mlir.global external @__cusimt_error_code(0 : i32) {addr_space = 0 : i32} : i32
    llvm.func @__nv_ceil(f64) -> f64
    llvm.func @_Z18cusimt_matmul_smem5ArrayIfLi2E1C7mutable7alignedE5ArrayIfLi2E1C7mutable7alignedE5ArrayIfLi2E1C7mutable7alignedE(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) attributes {cusimt.arg_attrs = [{}, {}, {}], cusimt.orig_arg_types = [memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>], gpu.kernel, llvm.emit_c_interface, nvvm.kernel} {
      %0 = llvm.mlir.constant(5 : index) : i64
      %1 = llvm.mlir.addressof @static_shared_memory_1 : !llvm.ptr<3>
      %2 = llvm.mlir.addressof @static_shared_memory_0 : !llvm.ptr<3>
      %3 = llvm.mlir.constant(16 : index) : i64
      %4 = llvm.mlir.constant(1.600000e+01 : f64) : f64
      %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %6 = llvm.mlir.constant(2 : index) : i64
      %7 = llvm.mlir.constant(0.000000e+00 : f64) : f64
      %8 = llvm.mlir.constant(0 : i64) : i64
      %9 = llvm.mlir.constant(1 : i64) : i64
      %10 = llvm.mlir.constant(1 : index) : i64
      %11 = llvm.mlir.constant(16 : i64) : i64
      %12 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %arg7, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %14 = llvm.insertvalue %arg8, %13[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %15 = llvm.insertvalue %arg9, %14[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %16 = llvm.insertvalue %arg10, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %17 = llvm.insertvalue %arg12, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %18 = llvm.insertvalue %arg11, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %19 = llvm.insertvalue %arg14, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %20 = llvm.insertvalue %arg15, %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %21 = llvm.insertvalue %arg16, %20[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %22 = llvm.insertvalue %arg17, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %23 = llvm.insertvalue %arg19, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %24 = llvm.insertvalue %arg18, %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %25 = llvm.insertvalue %arg0, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %26 = llvm.insertvalue %arg1, %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %27 = llvm.insertvalue %arg2, %26[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %28 = llvm.insertvalue %arg3, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %29 = llvm.insertvalue %arg5, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %30 = llvm.insertvalue %arg4, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %31 = llvm.alloca %10 x f64 : (i64) -> !llvm.ptr
      %32 = llvm.alloca %10 x f64 : (i64) -> !llvm.ptr
      %33 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<16 x array<16 x f32>>
      %34 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<16 x array<16 x f32>>
      %35 = nvvm.read.ptx.sreg.tid.x : i32
      %36 = llvm.sext %35 : i32 to i64
      %37 = llvm.trunc %36 : i64 to i32
      %38 = nvvm.read.ptx.sreg.tid.y : i32
      %39 = llvm.sext %38 : i32 to i64
      %40 = llvm.trunc %39 : i64 to i32
      %41 = nvvm.read.ptx.sreg.ctaid.x : i32
      %42 = llvm.sext %41 : i32 to i64
      %43 = llvm.trunc %42 : i64 to i32
      %44 = nvvm.read.ptx.sreg.ntid.x : i32
      %45 = llvm.sext %44 : i32 to i64
      %46 = llvm.trunc %45 : i64 to i32
      %47 = llvm.zext %43 : i32 to i64
      %48 = llvm.zext %46 : i32 to i64
      %49 = llvm.mul %47, %48 : i64
      %50 = llvm.zext %37 : i32 to i64
      %51 = llvm.add %49, %50 : i64
      %52 = nvvm.read.ptx.sreg.ctaid.y : i32
      %53 = llvm.sext %52 : i32 to i64
      %54 = llvm.trunc %53 : i64 to i32
      %55 = nvvm.read.ptx.sreg.ntid.y : i32
      %56 = llvm.sext %55 : i32 to i64
      %57 = llvm.trunc %56 : i64 to i32
      %58 = llvm.zext %54 : i32 to i64
      %59 = llvm.zext %57 : i32 to i64
      %60 = llvm.mul %58, %59 : i64
      %61 = llvm.zext %40 : i32 to i64
      %62 = llvm.add %60, %61 : i64
      %63 = llvm.extractvalue %30[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %64 = llvm.alloca %10 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
      llvm.store %63, %64 : !llvm.array<2 x i64>, !llvm.ptr
      %65 = llvm.getelementptr %64[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
      %66 = llvm.load %65 : !llvm.ptr -> i64
      %67 = llvm.sitofp %66 : i64 to f64
      %68 = llvm.fdiv %67, %4 : f64
      %69 = llvm.call @__nv_ceil(%68) : (f64) -> f64
      %70 = llvm.extractvalue %24[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %71 = llvm.alloca %10 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
      llvm.store %70, %71 : !llvm.array<2 x i64>, !llvm.ptr
      %72 = llvm.getelementptr %71[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
      %73 = llvm.load %72 : !llvm.ptr -> i64
      %74 = llvm.icmp "sge" %51, %73 : i64
      %75 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
      %76 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
      llvm.cond_br %74, ^bb2, ^bb1
    ^bb1:  // pred: ^bb0
      %77 = llvm.alloca %10 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
      llvm.store %70, %77 : !llvm.array<2 x i64>, !llvm.ptr
      %78 = llvm.getelementptr %77[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
      %79 = llvm.load %78 : !llvm.ptr -> i64
      %80 = llvm.icmp "sge" %62, %79 : i64
      llvm.cond_br %80, ^bb2, ^bb3
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    ^bb3:  // pred: ^bb1
      llvm.store %7, %31 : f64, !llvm.ptr
      %81 = llvm.fptosi %69 : f64 to i64
      llvm.store %8, %75 : i64, !llvm.ptr
      %82 = llvm.getelementptr inbounds|nuw %75[1] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %81, %82 : i64, !llvm.ptr
      %83 = llvm.getelementptr inbounds|nuw %75[2] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %9, %83 : i64, !llvm.ptr
      %84 = llvm.icmp "sgt" %81, %8 : i64
      %85 = llvm.select %84, %81, %8 : i1, i64
      %86 = llvm.getelementptr inbounds|nuw %75[3] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %85, %86 : i64, !llvm.ptr
      %87 = llvm.getelementptr inbounds|nuw %75[4] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %8, %87 : i64, !llvm.ptr
      llvm.br ^bb4
    ^bb4:  // 2 preds: ^bb3, ^bb16
      %88 = llvm.load %86 : !llvm.ptr -> i64
      %89 = llvm.icmp "sgt" %88, %8 : i64
      %90 = llvm.load %87 : !llvm.ptr -> i64
      %91 = llvm.sub %88, %9 : i64
      %92 = llvm.load %83 : !llvm.ptr -> i64
      %93 = llvm.add %90, %92 : i64
      %94 = llvm.select %89, %91, %88 : i1, i64
      %95 = llvm.select %89, %93, %90 : i1, i64
      llvm.store %94, %86 : i64, !llvm.ptr
      llvm.store %95, %87 : i64, !llvm.ptr
      %96 = llvm.alloca %6 x i64 : (i64) -> !llvm.ptr
      llvm.store %90, %96 : i64, !llvm.ptr
      %97 = llvm.zext %89 : i1 to i64
      %98 = llvm.getelementptr inbounds|nuw %96[1] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %97, %98 : i64, !llvm.ptr
      %99 = llvm.load %96 : !llvm.ptr -> i64
      %100 = llvm.load %98 : !llvm.ptr -> i64
      %101 = llvm.icmp "ne" %100, %8 : i64
      llvm.cond_br %101, ^bb5, ^bb17
    ^bb5:  // pred: ^bb4
      %102 = llvm.mul %99, %11 : i64
      %103 = llvm.add %102, %61 : i64
      %104 = llvm.add %102, %50 : i64
      %105 = llvm.alloca %10 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
      llvm.store %63, %105 : !llvm.array<2 x i64>, !llvm.ptr
      %106 = llvm.getelementptr %105[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
      %107 = llvm.load %106 : !llvm.ptr -> i64
      %108 = llvm.icmp "slt" %51, %107 : i64
      llvm.cond_br %108, ^bb6, ^bb8
    ^bb6:  // pred: ^bb5
      %109 = llvm.icmp "slt" %103, %66 : i64
      llvm.cond_br %109, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %110 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %111 = llvm.mul %51, %arg5 overflow<nsw, nuw> : i64
      %112 = llvm.mul %103, %arg6 overflow<nsw, nuw> : i64
      %113 = llvm.add %111, %112 overflow<nsw, nuw> : i64
      %114 = llvm.getelementptr inbounds|nuw %110[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %115 = llvm.load %114 : !llvm.ptr -> f32
      %116 = llvm.mul %36, %3 overflow<nsw, nuw> : i64
      %117 = llvm.add %116, %39 overflow<nsw, nuw> : i64
      %118 = llvm.getelementptr inbounds|nuw %33[%117] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      llvm.store %115, %118 : f32, !llvm.ptr<3>
      llvm.br ^bb9
    ^bb8:  // 2 preds: ^bb5, ^bb6
      %119 = llvm.mul %36, %3 overflow<nsw, nuw> : i64
      %120 = llvm.add %119, %39 overflow<nsw, nuw> : i64
      %121 = llvm.getelementptr inbounds|nuw %33[%120] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      llvm.store %5, %121 : f32, !llvm.ptr<3>
      llvm.br ^bb9
    ^bb9:  // 2 preds: ^bb7, ^bb8
      %122 = llvm.extractvalue %18[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %123 = llvm.alloca %10 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
      llvm.store %122, %123 : !llvm.array<2 x i64>, !llvm.ptr
      %124 = llvm.getelementptr %123[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
      %125 = llvm.load %124 : !llvm.ptr -> i64
      %126 = llvm.icmp "slt" %104, %125 : i64
      llvm.cond_br %126, ^bb10, ^bb12
    ^bb10:  // pred: ^bb9
      %127 = llvm.alloca %10 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
      llvm.store %122, %127 : !llvm.array<2 x i64>, !llvm.ptr
      %128 = llvm.getelementptr %127[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
      %129 = llvm.load %128 : !llvm.ptr -> i64
      %130 = llvm.icmp "slt" %62, %129 : i64
      llvm.cond_br %130, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %131 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %132 = llvm.mul %104, %arg12 overflow<nsw, nuw> : i64
      %133 = llvm.mul %62, %arg13 overflow<nsw, nuw> : i64
      %134 = llvm.add %132, %133 overflow<nsw, nuw> : i64
      %135 = llvm.getelementptr inbounds|nuw %131[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %136 = llvm.load %135 : !llvm.ptr -> f32
      %137 = llvm.mul %36, %3 overflow<nsw, nuw> : i64
      %138 = llvm.add %137, %39 overflow<nsw, nuw> : i64
      %139 = llvm.getelementptr inbounds|nuw %34[%138] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      llvm.store %136, %139 : f32, !llvm.ptr<3>
      llvm.br ^bb13
    ^bb12:  // 2 preds: ^bb9, ^bb10
      %140 = llvm.mul %36, %3 overflow<nsw, nuw> : i64
      %141 = llvm.add %140, %39 overflow<nsw, nuw> : i64
      %142 = llvm.getelementptr inbounds|nuw %34[%141] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      llvm.store %5, %142 : f32, !llvm.ptr<3>
      llvm.br ^bb13
    ^bb13:  // 2 preds: ^bb11, ^bb12
      %143 = llvm.load %31 : !llvm.ptr -> f64
      llvm.store %143, %32 : f64, !llvm.ptr
      nvvm.barrier0
      llvm.store %8, %76 : i64, !llvm.ptr
      %144 = llvm.getelementptr inbounds|nuw %76[1] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %11, %144 : i64, !llvm.ptr
      %145 = llvm.getelementptr inbounds|nuw %76[2] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %9, %145 : i64, !llvm.ptr
      %146 = llvm.getelementptr inbounds|nuw %76[3] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %11, %146 : i64, !llvm.ptr
      %147 = llvm.getelementptr inbounds|nuw %76[4] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %8, %147 : i64, !llvm.ptr
      llvm.br ^bb14
    ^bb14:  // 2 preds: ^bb13, ^bb15
      %148 = llvm.load %146 : !llvm.ptr -> i64
      %149 = llvm.icmp "sgt" %148, %8 : i64
      %150 = llvm.load %147 : !llvm.ptr -> i64
      %151 = llvm.sub %148, %9 : i64
      %152 = llvm.load %145 : !llvm.ptr -> i64
      %153 = llvm.add %150, %152 : i64
      %154 = llvm.select %149, %151, %148 : i1, i64
      %155 = llvm.select %149, %153, %150 : i1, i64
      llvm.store %154, %146 : i64, !llvm.ptr
      llvm.store %155, %147 : i64, !llvm.ptr
      %156 = llvm.alloca %6 x i64 : (i64) -> !llvm.ptr
      llvm.store %150, %156 : i64, !llvm.ptr
      %157 = llvm.zext %149 : i1 to i64
      %158 = llvm.getelementptr inbounds|nuw %156[1] : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %157, %158 : i64, !llvm.ptr
      %159 = llvm.load %156 : !llvm.ptr -> i64
      %160 = llvm.load %158 : !llvm.ptr -> i64
      %161 = llvm.icmp "ne" %160, %8 : i64
      llvm.cond_br %161, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      %162 = llvm.mul %36, %3 overflow<nsw, nuw> : i64
      %163 = llvm.add %162, %159 overflow<nsw, nuw> : i64
      %164 = llvm.getelementptr inbounds|nuw %33[%163] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %165 = llvm.load %164 : !llvm.ptr<3> -> f32
      %166 = llvm.mul %159, %3 overflow<nsw, nuw> : i64
      %167 = llvm.add %166, %39 overflow<nsw, nuw> : i64
      %168 = llvm.getelementptr inbounds|nuw %34[%167] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %169 = llvm.load %168 : !llvm.ptr<3> -> f32
      %170 = llvm.fmul %165, %169 : f32
      %171 = llvm.load %32 : !llvm.ptr -> f64
      %172 = llvm.fpext %170 : f32 to f64
      %173 = llvm.fadd %171, %172 : f64
      llvm.store %173, %32 : f64, !llvm.ptr
      llvm.br ^bb14
    ^bb16:  // pred: ^bb14
      %174 = llvm.load %32 : !llvm.ptr -> f64
      llvm.store %174, %31 : f64, !llvm.ptr
      nvvm.barrier0
      llvm.br ^bb4
    ^bb17:  // pred: ^bb4
      %175 = llvm.load %31 : !llvm.ptr -> f64
      %176 = llvm.fptrunc %175 : f64 to f32
      %177 = llvm.getelementptr %arg15[%arg16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %178 = llvm.mul %51, %arg19 overflow<nsw, nuw> : i64
      %179 = llvm.mul %62, %arg20 overflow<nsw, nuw> : i64
      %180 = llvm.add %178, %179 overflow<nsw, nuw> : i64
      %181 = llvm.getelementptr inbounds|nuw %177[%180] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %176, %181 : f32, !llvm.ptr
      llvm.return
    }
    llvm.mlir.global private @static_shared_memory_0() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<16 x array<16 x f32>>
    llvm.mlir.global private @static_shared_memory_1() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<16 x array<16 x f32>>
  }
}
