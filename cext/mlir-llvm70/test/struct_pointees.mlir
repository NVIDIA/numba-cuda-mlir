// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck %s

// `llvm70.struct_pointees` carries the pointee type of pointer members of
// identified structs, so that a struct built here matches the same struct
// coming from a typed producer instead of spelling every pointer `i8*`. The
// Python side derives it from the Numba types the structs were built from.
//
// The refinement is confined to the struct layout: values of MLIR `!llvm.ptr`
// type stay `i8*` everywhere else, so insertvalue casts in and extractvalue
// casts back out.

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] attributes {
      llvm70.struct_pointees = {
        // Member 0 is refined, member 1 is a pointer left at the default,
        // member 2 is not a pointer at all and the entry is ignored.
        Arr = [f32, unit, i64],
        // Only the tail member of this one is described.
        Partial = [unit, i32],
        // Names with no matching struct are ignored.
        Absent = [f64]
      }} {

    // CHECK-DAG: %Arr = type { float addrspace(1)*, i8*, i64 }
    // CHECK-DAG: %Partial = type { i8*, i32 addrspace(1)* }
    // CHECK-DAG: %Plain = type { i8 addrspace(1)*, i64 }
    // A nested struct picks up the refinement in its own layout too.
    // CHECK-DAG: %Nest = type { %Partial, i64 }
    // CHECK-NOT: %Absent

    // Building the struct: the incoming i8* is cast into the member type.
    // CHECK-LABEL: define ptx_kernel void @build_and_read
    // CHECK: %[[IN:.*]] = bitcast i8 addrspace(1)* %0 to float addrspace(1)*
    // CHECK: %[[AGG:.*]] = insertvalue %Arr undef, float addrspace(1)* %[[IN]], 0
    // CHECK: %[[OUT:.*]] = extractvalue %Arr %[[AGG]], 0
    // CHECK: bitcast float addrspace(1)* %[[OUT]] to i8 addrspace(1)*
    // CHECK: load float
    llvm.func @build_and_read(%p: !llvm.ptr<1>, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %u = llvm.mlir.undef : !llvm.struct<"Arr", (ptr<1>, ptr, i64)>
      %s = llvm.insertvalue %p, %u[0] : !llvm.struct<"Arr", (ptr<1>, ptr, i64)>
      %q = llvm.extractvalue %s[0] : !llvm.struct<"Arr", (ptr<1>, ptr, i64)>
      %v = llvm.load %q : !llvm.ptr<1> -> f32
      llvm.store %v, %out : f32, !llvm.ptr<1>
      llvm.return
    }

    // An extracted member flows on as an ordinary opaque pointer: stored into a
    // pointer slot and passed to a call, both of which expect i8*.
    // CHECK-LABEL: define ptx_kernel void @extracted_ptr_is_opaque
    // CHECK: %[[E:.*]] = extractvalue %Arr
    // CHECK: %[[C:.*]] = bitcast float addrspace(1)* %[[E]] to i8 addrspace(1)*
    // CHECK: store i8 addrspace(1)* %[[C]], i8 addrspace(1)**
    // CHECK: call void @sink(i8 addrspace(1)* %[[C]])
    llvm.func @sink(!llvm.ptr<1>)
    llvm.func @extracted_ptr_is_opaque(%p: !llvm.ptr<1>, %slot: !llvm.ptr) attributes {gpu.kernel} {
      %u = llvm.mlir.undef : !llvm.struct<"Arr", (ptr<1>, ptr, i64)>
      %s = llvm.insertvalue %p, %u[0] : !llvm.struct<"Arr", (ptr<1>, ptr, i64)>
      %q = llvm.extractvalue %s[0] : !llvm.struct<"Arr", (ptr<1>, ptr, i64)>
      llvm.store %q, %slot : !llvm.ptr<1>, !llvm.ptr
      llvm.call @sink(%q) : (!llvm.ptr<1>) -> ()
      llvm.return
    }

    // A struct with no entry keeps the old behaviour: every pointer is i8* and
    // insertvalue needs no cast.
    // CHECK-LABEL: define ptx_kernel void @undeclared_struct
    // CHECK: insertvalue %Plain undef, i8 addrspace(1)* %0, 0
    // CHECK-NOT: bitcast i8 addrspace(1)* %0
    llvm.func @undeclared_struct(%p: !llvm.ptr<1>) attributes {gpu.kernel} {
      %u = llvm.mlir.undef : !llvm.struct<"Plain", (ptr<1>, i64)>
      %s = llvm.insertvalue %p, %u[0] : !llvm.struct<"Plain", (ptr<1>, i64)>
      %q = llvm.extractvalue %s[0] : !llvm.struct<"Plain", (ptr<1>, i64)>
      llvm.call @sink(%q) : (!llvm.ptr<1>) -> ()
      llvm.return
    }

    // A refined member reached through a nested position list.
    // CHECK-LABEL: define ptx_kernel void @nested
    // CHECK: %[[N:.*]] = bitcast i8 addrspace(1)* %0 to i32 addrspace(1)*
    // CHECK: insertvalue %Partial {{.*}} i32 addrspace(1)* %[[N]], 1
    // CHECK: insertvalue %Nest
    llvm.func @nested(%p: !llvm.ptr<1>) attributes {gpu.kernel} {
      %u = llvm.mlir.undef : !llvm.struct<"Nest", (struct<"Partial", (ptr, ptr<1>)>, i64)>
      %s = llvm.insertvalue %p, %u[0, 1] : !llvm.struct<"Nest", (struct<"Partial", (ptr, ptr<1>)>, i64)>
      %q = llvm.extractvalue %s[0, 1] : !llvm.struct<"Nest", (struct<"Partial", (ptr, ptr<1>)>, i64)>
      llvm.call @sink(%q) : (!llvm.ptr<1>) -> ()
      llvm.return
    }
  }
}
