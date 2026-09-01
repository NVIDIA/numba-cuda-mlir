# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from numba_cuda_mlir import cuda, mlir_compiler
from numba_cuda_mlir.ast_transforms import ConstevalError
from numba_cuda_mlir.cuda.experimental import consteval, current_target_options
from numba_cuda_mlir.errors import TypingError


def _kernel_chip(kernel) -> str:
    (cres,) = kernel.overloads.values()
    return cres.metadata["targetoptions"]["chip"]


def test_inlined_callee_consteval_loop():
    n = 4

    @cuda.jit(device=True, inline=True)
    def fill(out, base):
        for i in consteval(range(n)):
            out[i] = base + i

    @cuda.jit
    def kernel(out):
        fill(out, 10)

    out = cuda.to_device(np.zeros(8, dtype=np.int32))
    kernel[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [10, 11, 12, 13, 0, 0, 0, 0])


def test_nested_inlined_callees_transform():
    n = 3

    @cuda.jit(device=True, inline=True)
    def inner(out, base):
        for i in consteval(range(n)):
            out[i] = base * consteval(10**i)

    @cuda.jit(device=True, inline=True)
    def outer(out):
        inner(out, 2)
        out[n] = consteval(n * 100)

    @cuda.jit
    def kernel(out):
        outer(out)

    out = cuda.to_device(np.zeros(4, dtype=np.int32))
    kernel[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [2, 20, 200, 300])


def test_inlined_callee_sees_caller_target_options():
    @cuda.jit(device=True, inline=True)
    def inner(out):
        chip = consteval(current_target_options()["chip"])
        out[1] = consteval(int(chip[3:]))

    @cuda.jit(device=True, inline=True)
    def outer(out):
        chip = consteval(current_target_options()["chip"])
        out[0] = consteval(int(chip[3:]))
        inner(out)

    @cuda.jit
    def kernel(out):
        outer(out)

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    kernel[1, 1](out)
    expected = int(_kernel_chip(kernel)[3:])
    np.testing.assert_array_equal(out.copy_to_host(), [expected, expected])


def test_inlined_callee_parameter_types_unavailable():
    @cuda.jit(device=True, inline=True)
    def callee(out):
        out[0] = consteval(out.ndim)

    @cuda.jit
    def kernel(out):
        callee(out)

    with pytest.raises(ConstevalError, match="Cannot evaluate consteval argument"):
        kernel.compile("void(int32[:])")


_shadowed = 3


def test_inlined_callee_parameter_shadows_global():
    @cuda.jit(device=True, inline=True)
    def callee(out, _shadowed):
        for i in consteval(range(_shadowed)):
            out[i] = 1

    @cuda.jit
    def kernel(out):
        callee(out, 5)

    with pytest.raises(ConstevalError, match="Cannot evaluate consteval argument"):
        kernel.compile("void(int32[:])")


def test_callee_option_controls_transform():
    n = 2

    @cuda.jit(device=True, inline=True, experimental_ast_transforms=False)
    def callee(out):
        for i in consteval(range(n)):
            out[i] = 1

    @cuda.jit(experimental_ast_transforms=True)
    def kernel(out):
        callee(out)

    with pytest.raises(TypingError, match="consteval"):
        kernel.compile("void(int32[:])")


def test_rejected_cost_model_does_not_transform(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("rejected inlinee was transformed")

    monkeypatch.setattr(mlir_compiler, "transform_inline_callee", fail_if_called)

    def never_inline(expr, caller_info, callee_info):
        return False

    @cuda.jit(device=True, inline=never_inline)
    def callee(out):
        out[0] = 1

    @cuda.jit
    def kernel(out):
        callee(out)

    kernel.compile("void(int32[:])")
