// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    llvm.func @lifetime_kernel(%ptr: !llvm.ptr) attributes {gpu.kernel} {
      "llvm.intr.lifetime.start"(%ptr) : (!llvm.ptr) -> ()
      "llvm.intr.lifetime.end"(%ptr) : (!llvm.ptr) -> ()
      llvm.return
    }
  }
}

// CHECK-IR: define ptx_kernel void @lifetime_kernel
// CHECK-IR: call void @llvm.lifetime.start.p0i8(i64 -1, i8* %{{.*}})
// CHECK-IR: call void @llvm.lifetime.end.p0i8(i64 -1, i8* %{{.*}})

// CHECK-PTX: .visible .entry lifetime_kernel
