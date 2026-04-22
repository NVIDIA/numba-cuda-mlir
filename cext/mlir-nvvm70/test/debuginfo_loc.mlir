// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: nvvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: nvvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

// Test that explicit loc() annotations on MLIR ops are propagated to LLVM IR
// debug metadata. This simulates the real-world case where a frontend compiler
// (e.g. numba, triton) attaches source locations to ops.

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_80">] {
    llvm.func @saxpy(%A: !llvm.ptr<1>, %B: !llvm.ptr<1>, %C: !llvm.ptr<1>, %alpha: f32) attributes {gpu.kernel} {
      %tid = nvvm.read.ptx.sreg.tid.x : i32 loc("saxpy.py":10:5)
      %idx = llvm.sext %tid : i32 to i64 loc("saxpy.py":10:5)
      %pA = llvm.getelementptr %A[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32 loc("saxpy.py":11:10)
      %a = llvm.load %pA : !llvm.ptr<1> -> f32 loc("saxpy.py":11:10)
      %pB = llvm.getelementptr %B[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32 loc("saxpy.py":12:10)
      %bv = llvm.load %pB : !llvm.ptr<1> -> f32 loc("saxpy.py":12:10)
      %ax = llvm.fmul %alpha, %a : f32 loc("saxpy.py":13:14)
      %res = llvm.fadd %ax, %bv : f32 loc("saxpy.py":13:10)
      %pC = llvm.getelementptr %C[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32 loc("saxpy.py":14:5)
      llvm.store %res, %pC : f32, !llvm.ptr<1> loc("saxpy.py":14:5)
      llvm.return loc("saxpy.py":15:5)
    } loc("saxpy.py":9:1)
  }
}

// Verify the function gets a DISubprogram pointing to saxpy.py.
// CHECK-IR: define ptx_kernel void @saxpy({{.*}}) !dbg ![[SP:[0-9]+]]

// Verify instructions carry !dbg with correct line numbers from loc().
// CHECK-IR: call i32 asm "mov.u32 $0, %tid.x;", "=r"(){{.*}}!dbg ![[LOC10:[0-9]+]]
// CHECK-IR: fmul float{{.*}}!dbg ![[LOC13a:[0-9]+]]
// CHECK-IR: fadd float{{.*}}!dbg ![[LOC13b:[0-9]+]]
// CHECK-IR: store float{{.*}}!dbg ![[LOC14:[0-9]+]]

// Verify the DISubprogram references saxpy.py with the correct line.
// CHECK-IR: ![[SP]] = distinct !DISubprogram(name: "saxpy",
// CHECK-IR-SAME: file: ![[FILE:[0-9]+]]
// CHECK-IR-SAME: line: 9

// CHECK-IR: ![[FILE]] = !DIFile(filename: "saxpy.py"

// Verify DILocations carry the expected line numbers.
// CHECK-IR: ![[LOC10]] = !DILocation(line: 10,
// CHECK-IR-SAME: scope: ![[SP]])
// CHECK-IR: ![[LOC13a]] = !DILocation(line: 13,
// CHECK-IR: ![[LOC13b]] = !DILocation(line: 13,
// CHECK-IR: ![[LOC14]] = !DILocation(line: 14,

// Verify PTX compiles successfully.
// CHECK-PTX: .visible .entry saxpy
