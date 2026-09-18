// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

// convertType renders an opaque !llvm.ptr as i8*, but an alloca is mapped to
// the elemTy* LLVM 7 gives it, so that debug info still names the real stack
// slot. LLVM 7 has no opaque pointers, so every op that puts a pointer
// somewhere already typed as i8* has to bridge the two, or libNVVM's bitcode
// reader refuses to parse the module.

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @alloca_into_struct(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %p = llvm.alloca %c1 x i64 : (i64) -> !llvm.ptr
      %u = llvm.mlir.poison : !llvm.struct<(ptr, i64)>
      %s = llvm.insertvalue %p, %u[0] : !llvm.struct<(ptr, i64)>
      %f = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>
      %v = llvm.load %f : !llvm.ptr -> i64
      llvm.store %v, %out : i64, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @alloca_stored_as_ptr(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %p = llvm.alloca %c1 x i64 : (i64) -> !llvm.ptr
      %pp = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
      llvm.store %p, %pp : !llvm.ptr, !llvm.ptr
      %q = llvm.load %pp : !llvm.ptr -> !llvm.ptr
      %v = llvm.load %q : !llvm.ptr -> i64
      llvm.store %v, %out : i64, !llvm.ptr<1>
      llvm.return
    }

    // The two arms have different element types, so a select over the typed
    // pointers would not even agree with itself.
    llvm.func @alloca_selected(%out: !llvm.ptr<1>, %c: i1) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %a = llvm.alloca %c1 x i64 : (i64) -> !llvm.ptr
      %b = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
      %s = llvm.select %c, %a, %b : i1, !llvm.ptr
      %v = llvm.load %s : !llvm.ptr -> i64
      llvm.store %v, %out : i64, !llvm.ptr<1>
      llvm.return
    }

    // Block arguments are already converted to i8*, so the incoming alloca has
    // to match or the phi is ill typed.
    llvm.func @alloca_through_block_arg(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %a = llvm.alloca %c1 x i64 : (i64) -> !llvm.ptr
      llvm.br ^bb1(%a : !llvm.ptr)
    ^bb1(%q: !llvm.ptr):
      %v = llvm.load %q : !llvm.ptr -> i64
      llvm.store %v, %out : i64, !llvm.ptr<1>
      llvm.return
    }
  }
}

// CHECK-LABEL: define ptx_kernel void @alloca_into_struct
// CHECK: %[[P:.*]] = alloca i64
// CHECK: %[[C:.*]] = bitcast i64* %[[P]] to i8*
// CHECK: insertvalue { i8*, i64 } undef, i8* %[[C]], 0

// CHECK-LABEL: define ptx_kernel void @alloca_stored_as_ptr
// CHECK: store i8* %{{.*}}, i8** %

// CHECK-LABEL: define ptx_kernel void @alloca_selected
// CHECK: select i1 %{{.*}}, i8* %{{.*}}, i8* %

// CHECK-LABEL: define ptx_kernel void @alloca_through_block_arg
// CHECK: phi i8* [ %

// CHECK-PTX: .visible .entry alloca_into_struct
// CHECK-PTX: .visible .entry alloca_stored_as_ptr
// CHECK-PTX: .visible .entry alloca_selected
// CHECK-PTX: .visible .entry alloca_through_block_arg
