// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

// Test that llvm.intr.dbg.declare and llvm.intr.dbg.value ops are translated
// to @llvm.dbg.declare / @llvm.dbg.value calls with DILocalVariable metadata
// in the LLVM 7 IR, and that libnvvm compiles the result successfully
// (requires FullDebug emission kind).

#di_file = #llvm.di_file<"test.py" in ".">
#di_cu = #llvm.di_compile_unit<id = distinct[0]<>,
    sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false,
    emissionKind = Full>
#di_i32 = #llvm.di_basic_type<name = "int32", sizeInBits = 32,
    encoding = DW_ATE_signed>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>,
    compileUnit = #di_cu, scope = #di_file, name = "add_kernel",
    file = #di_file, line = 5, subprogramFlags = "Definition">
#di_var_a = #llvm.di_local_variable<scope = #di_subprogram,
    name = "a", file = #di_file, line = 5, type = #di_i32>
#di_var_b = #llvm.di_local_variable<scope = #di_subprogram,
    name = "b", file = #di_file, line = 5, type = #di_i32>
#di_var_sum = #llvm.di_local_variable<scope = #di_subprogram,
    name = "sum", file = #di_file, line = 6, type = #di_i32>

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {
    llvm.func @add_kernel(%out: !llvm.ptr<1>, %a: i32, %b: i32)
        attributes {gpu.kernel} {
      %one = llvm.mlir.constant(1 : i64) : i64
      %pa = llvm.alloca %one x i32 : (i64) -> !llvm.ptr loc("test.py":5:0)
      llvm.store %a, %pa : i32, !llvm.ptr loc("test.py":5:0)
      llvm.intr.dbg.declare #di_var_a = %pa : !llvm.ptr loc("test.py":5:0)
      %pb = llvm.alloca %one x i32 : (i64) -> !llvm.ptr loc("test.py":5:0)
      llvm.store %b, %pb : i32, !llvm.ptr loc("test.py":5:0)
      llvm.intr.dbg.declare #di_var_b = %pb : !llvm.ptr loc("test.py":5:0)
      %va = llvm.load %pa : !llvm.ptr -> i32 loc("test.py":6:0)
      %vb = llvm.load %pb : !llvm.ptr -> i32 loc("test.py":6:0)
      %sum = llvm.add %va, %vb : i32 loc("test.py":6:0)
      llvm.intr.dbg.value #di_var_sum = %sum : i32 loc("test.py":6:0)
      llvm.store %sum, %out : i32, !llvm.ptr<1> loc("test.py":6:0)
      llvm.return loc("test.py":7:0)
    } loc("test.py":5:0)
  }
}

// Verify @llvm.dbg.declare calls are emitted with correct metadata.
// CHECK-IR: call void @llvm.dbg.declare(metadata i32* %{{[0-9]+}}, metadata ![[VAR_A:[0-9]+]], metadata !DIExpression())
// CHECK-IR: call void @llvm.dbg.declare(metadata i32* %{{[0-9]+}}, metadata ![[VAR_B:[0-9]+]], metadata !DIExpression())

// Verify @llvm.dbg.value call is emitted for the computed sum.
// CHECK-IR: call void @llvm.dbg.value(metadata i32 %{{[0-9]+}}, metadata ![[VAR_SUM:[0-9]+]], metadata !DIExpression())

// Verify both intrinsics are declared.
// CHECK-IR-DAG: declare void @llvm.dbg.declare(metadata, metadata, metadata)
// CHECK-IR-DAG: declare void @llvm.dbg.value(metadata, metadata, metadata)

// Verify FullDebug emission kind (required by libnvvm for DILocalVariable).
// CHECK-IR: !DICompileUnit(
// CHECK-IR-SAME: emissionKind: FullDebug

// Verify DILocalVariable metadata for all variables with i32 type.
// CHECK-IR-DAG: !DILocalVariable(name: "a",{{.*}}type: ![[TY:[0-9]+]])
// CHECK-IR-DAG: !DILocalVariable(name: "b",{{.*}}type: ![[TY]])
// CHECK-IR-DAG: !DILocalVariable(name: "sum",{{.*}}type: ![[TY]])
// CHECK-IR-DAG: ![[TY]] = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)

// Verify PTX compiles successfully.
// CHECK-PTX: .visible .entry add_kernel
