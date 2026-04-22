// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
module {
    func.func @pass_by_ptr(%a: !llvm.ptr, %i: i64) attributes {always_inline} {
        %deadbeef = arith.constant 3735928559 : i64
        %ptr = llvm.getelementptr %a[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %deadbeef, %ptr : i64, !llvm.ptr
        func.return
    }
}
