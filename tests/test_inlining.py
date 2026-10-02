# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import compiler
from numba_cuda_mlir import decorators
from numba_cuda_mlir import types, testing
import pytest
from types import SimpleNamespace


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


def test_inline_callable():
    # A cost-model callable defers the decision to the Numba IR inliner, so the
    # MLIR function carries neither inline attribute.
    @cuda.jit(device=True, inline=lambda expr, caller_ir, callee_ir: True)
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


@pytest.mark.parametrize(
    ("block_sizes", "expected"),
    [
        ([9], True),
        ([11], True),
        ([64], True),
        ([65], False),
        ([32, 32], True),
        ([33, 32], False),
    ],
)
def test_default_inline_cost_model(block_sizes, expected):
    func_ir = SimpleNamespace(
        blocks={
            index: SimpleNamespace(body=[None] * size) for index, size in enumerate(block_sizes)
        }
    )
    assert decorators._default_inline(None, None, func_ir) is expected
    assert decorators._default_inline(None, None, SimpleNamespace(func_ir=func_ir)) is expected


def test_default_inline_policy():
    func = cuda.jit(device=True)(lambda x: x * 2.0)
    assert func.targetoptions["inline"] is decorators._default_inline


def test_default_inline_has_no_mlir_attribute():
    @cuda.jit(device=True)
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


@pytest.mark.parametrize(
    "inline",
    ["always", "never", lambda expr, caller_ir, callee_ir: True],
    ids=["always", "never", "callable"],
)
def test_kernel_compiles_with_supported_inline_policies(inline):
    # Every policy accepted at decoration must also be accepted by the
    # caller's Numba IR inlining pass. inline="auto" passed decoration but
    # failed there, so compile a kernel that calls the device function.
    @cuda.jit(device=True, inline=inline)
    def device_func(x: types.f64) -> types.f64:
        return x * 2.0

    @cuda.jit
    def kernel(out):
        out[0] = device_func(out[0])

    compiler.compile_mlir(kernel, types.void(types.f64[::1]))
