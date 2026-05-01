// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
module attributes {numba_cuda_mlir.link_target} {
    func.func private @breakpoint() attributes {always_inline} {
        nvvm.breakpoint
        return
    }
    func.func private @nanosleep(%ticks: i32) attributes {always_inline} {
        nvvm.nanosleep %ticks
        return
    }
}
