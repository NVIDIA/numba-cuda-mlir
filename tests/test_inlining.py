# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cuda
from cuda.simt import compiler
from cuda.simt import types, testing


def test_inline_always():
    @cuda.jit(device=True, inline="always")
    def device_func(x: types.f64) -> types.f64:
        return x * 2.0

    mlir = compiler.compile_mlir(device_func, types.f64(types.f64))
    testing.filecheck(
        """
        CHECK: func.func @{{.*}}device_func{{.*}} attributes {always_inline
        """,
        mlir,
    )


def test_inline_never():
    @cuda.jit(device=True, inline="never")
    def device_func(x: types.f64) -> types.f64:
        return x * 2.0

    mlir = compiler.compile_mlir(device_func, types.f64(types.f64))
    testing.filecheck(
        """
        CHECK: func.func @{{.*}}device_func{{.*}} attributes {
        CHECK-NOT: always_inline
        CHECK-SAME: no_inline
        """,
        mlir,
    )


def test_inline_auto():
    @cuda.jit(device=True, inline="auto")
    def device_func(x: types.f64) -> types.f64:
        return x * 2.0

    mlir = compiler.compile_mlir(device_func, types.f64(types.f64))
    testing.filecheck(
        """
        CHECK: func.func @{{.*}}device_func{{.*}} attributes {
        CHECK-NOT: always_inline
        CHECK-NOT: no_inline
        """,
        mlir,
    )
