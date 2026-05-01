// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Some numba-cuda intrinsics are just wrappers around libdevice functions.
// We declare them here so we can reuse their declarations in lowering.
module attributes {numba_cuda_mlir.link_target} {
    func.func private @__nv_ffs(%value: i32) -> i32
    func.func private @__nv_ffsll(%value: i64) -> i32
    func.func private @ffs(%value: i32) -> i32 attributes {always_inline} {
        %result = call @__nv_ffs(%value) : (i32) -> i32
        return %result : i32
    }
    func.func private @ffsll(%value: i64) -> i32 attributes {always_inline} {
        %result = call @__nv_ffsll(%value) : (i64) -> i32
        return %result : i32
    }
}
