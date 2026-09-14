# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import compiler
from numba_cuda_mlir import types, testing
import pytest


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


@pytest.mark.parametrize("device", [True, False])
def test_inline_auto_rejected_at_decoration(device):
    with pytest.raises(ValueError, match="Expected inline to be one of .*always.*never.*got auto"):
        cuda.jit(device=device, inline="auto")(lambda x: x * 2.0)


@pytest.mark.parametrize("inline", ["always", "never", True, False, lambda *args: True])
def test_supported_inline_options(inline):
    func = cuda.jit(device=True, inline=inline)(lambda x: x * 2.0)
    expected = {True: "always", False: "never"}.get(inline, inline)
    assert func.targetoptions["inline"] == expected
