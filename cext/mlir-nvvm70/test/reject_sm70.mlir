// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// UNSUPPORTED: chip-override
// RUN: not nvvm70-translate %s --dump-ptx 2>&1 | FileCheck %s

module {
  gpu.module @kernels [#nvvm70.target<chip = "sm_70">] {
    llvm.func @simple_kernel() attributes {gpu.kernel} {
      llvm.return
    }
  }
}

// CHECK: NVVM70 does not support sm_70
// CHECK: Minimum supported architecture is sm_75
