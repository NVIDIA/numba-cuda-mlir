# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import compiler
from numba_cuda_mlir import decorators
from numba_cuda_mlir import types, testing
import pytest
from types import SimpleNamespace
import numpy as np
from numba_cuda_mlir.numba_cuda.compiler import run_frontend


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


@pytest.mark.parametrize("inline", [None, "always"], ids=["default", "always"])
def test_large_higher_order_device_function(inline):
    @cuda.jit(device=True)
    def add(a, b):
        return a + b

    def accumulate(x, op):
        acc = 0.0
        acc = op(acc, x + 1.0)
        acc = op(acc, x + 2.0)
        acc = op(acc, x + 3.0)
        acc = op(acc, x + 4.0)
        acc = op(acc, x + 5.0)
        acc = op(acc, x + 6.0)
        acc = op(acc, x + 7.0)
        acc = op(acc, x + 8.0)
        acc = op(acc, x + 9.0)
        acc = op(acc, x + 10.0)
        acc = op(acc, x + 11.0)
        acc = op(acc, x + 12.0)
        acc = op(acc, x + 13.0)
        acc = op(acc, x + 14.0)
        acc = op(acc, x + 15.0)
        acc = op(acc, x + 16.0)
        acc = op(acc, x + 17.0)
        acc = op(acc, x + 18.0)
        acc = op(acc, x + 19.0)
        acc = op(acc, x + 20.0)
        return acc

    func_ir = run_frontend(accumulate)
    assert sum(len(block.body) for block in func_ir.blocks.values()) > 64
    options = {} if inline is None else {"inline": inline}
    device_func = cuda.jit(device=True, **options)(accumulate)

    @cuda.jit
    def kernel(out, x):
        out[0] = device_func(x, add)

    compiler.compile_mlir(kernel, types.void(types.f64[::1], types.f64))
    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](out, 1.0)
    assert out[0] == 230.0


def test_default_inline_aliased_argument():
    def helper(x, op):
        fn = op
        return fn(x, x)

    func_ir = run_frontend(helper)
    # Padding isolates the mandatory-inline decision from the size heuristic.
    next(iter(func_ir.blocks.values())).body.extend([None] * 65)
    assert decorators._default_inline(None, None, func_ir)
    assert decorators._default_inline(None, None, SimpleNamespace(func_ir=func_ir))


def test_default_inline_ambiguous_call_target():
    def helper(x, first, second):
        if x:
            op = first
        else:
            op = second
        return op(x, x)

    func_ir = run_frontend(helper)
    next(iter(func_ir.blocks.values())).body.extend([None] * 65)
    assert decorators._default_inline(None, None, func_ir)


def test_default_inline_large_ordinary_call():
    def helper(x):
        return abs(x)

    func_ir = run_frontend(helper)
    next(iter(func_ir.blocks.values())).body.extend([None] * 65)
    assert not decorators._default_inline(None, None, func_ir)
    assert not decorators._default_inline(None, None, SimpleNamespace(func_ir=func_ir))
