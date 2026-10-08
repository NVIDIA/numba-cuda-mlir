# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import ModuleType

from numba_cuda_mlir import cuda, mlir_compiler
from numba_cuda_mlir.ast_transforms import ConstevalError
from numba_cuda_mlir.cuda.experimental import consteval, current_target_options
from numba_cuda_mlir.errors import TypingError
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.misc.special import literally
import numpy as np
import pytest


def _kernel_chip(kernel) -> str:
    (cres,) = kernel.overloads.values()
    return cres.metadata["targetoptions"]["chip"]


def test_inlined_callee_consteval_loop():
    """Test that a consteval loop in an inlined callee unrolls and runs."""
    n = 4

    @cuda.jit(device=True, inline=True)
    def fill(out, base):
        for i in consteval(range(n)):
            out[i] = base + i

    @cuda.jit
    def kernel(out):
        fill(out, 10)

    out = np.zeros(8, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, [10, 11, 12, 13, 0, 0, 0, 0])


@pytest.mark.xfail(strict=True, reason="fold_arguments cannot bind keyword-only arguments")
def test_inlined_callee_keyword_only_default():
    """Test that an omitted keyword-only default reaches an inlined callee."""

    @cuda.jit(device=True, inline=True)
    def fill(out, *, base=7):
        for i in consteval(range(2)):
            out[i] = base + i

    @cuda.jit
    def kernel(out):
        fill(out)

    out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, [7, 8])


def test_nested_inlined_callees_transform():
    """Test that a callee inlined inside another inlined callee is transformed."""
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

    out = np.zeros(4, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, [2, 20, 200, 300])


def test_inlined_callee_sees_caller_target_options():
    """Test that current_target_options() in nested inlined callees is the kernel's."""

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

    out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](out)
    expected = int(_kernel_chip(kernel)[3:])
    np.testing.assert_array_equal(out, [expected, expected])


def test_inlined_callee_parameter_resolves_to_type():
    """Test that a parameter in an inlined callee's consteval resolves to its type."""

    @cuda.jit(device=True, inline=True)
    def callee(out):
        out[0] = consteval(out.ndim)

    @cuda.jit
    def kernel(out):
        callee(out)

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out)
    assert out[0] == out.ndim


def _inlined_and_called(body):
    """Return the same device function compiled to be inlined and to be called."""
    return cuda.jit(device=True, inline=True)(body), cuda.jit(device=True)(body)


def _assert_inlined_matches_called(kernel, *args):
    """Run a kernel that writes the inlined result to out[0] and the called one to out[1]."""
    out = np.zeros(2, dtype=np.int64)
    kernel[1, 1](out, *args)
    assert out[0] == out[1]


def test_inlined_callee_computed_argument_types_match_call():
    """Test that a computed argument resolves as it does in a call."""

    def body(out, i, v):
        out[i] = consteval(v.bitwidth)

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, a):
        inlined(out, 0, a * 2)
        called(out, 1, a * 2)

    _assert_inlined_matches_called(kernel, np.float32(1.5))


def test_inlined_callee_keyword_argument_types_match_call():
    """Test that a keyword argument resolves as it does in a call."""

    def body(out, i, v):
        out[i] = consteval(v.bitwidth)

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, a):
        inlined(out, 0, v=a * 2)
        called(out, 1, v=a * 2)

    _assert_inlined_matches_called(kernel, np.float32(1.5))


def test_inlined_callee_consteval_block_reads_parameter():
    """Test that a with consteval() block reads a parameter as it does in a call."""

    def body(out, i, v):
        with consteval():
            width = v.bitwidth
        out[i] = consteval(width)

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, a):
        inlined(out, 0, a * 2)
        called(out, 1, a * 2)

    _assert_inlined_matches_called(kernel, np.float32(1.5))


def test_inlined_callee_argument_from_inlined_call_matches_call():
    """Test that an argument returned by another inlined call resolves as in a call."""

    @cuda.jit(device=True, inline=True)
    def widen(a):
        return a * 2.0

    def body(out, i, v):
        out[i] = consteval(v.bitwidth)

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, a):
        v = widen(a)
        inlined(out, 0, v)
        called(out, 1, v)

    _assert_inlined_matches_called(kernel, np.float32(1.5))


def test_inlined_callee_argument_from_overload_matches_call():
    """Test that an argument returned by an overload resolves as it does in a call."""

    def double(x):
        raise NotImplementedError

    @overload(double, strict=False, typing_registry=typing_registry)
    def _ol_double(x):
        return lambda x: x * 2

    def body(out, i, v):
        out[i] = consteval(v.bitwidth)

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, a):
        v = double(a)
        inlined(out, 0, v)
        called(out, 1, v)

    _assert_inlined_matches_called(kernel, np.float32(1.5))


def test_inlined_callee_omitted_default_matches_call():
    """Test that an omitted default resolves as Omitted, as it does in a call."""

    def body(out, i, n=4):
        out[i] = consteval(isinstance(n, types.Omitted))

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out):
        inlined(out, 0)
        called(out, 1)

    _assert_inlined_matches_called(kernel)


def test_inlined_callee_call_sites_resolve_separately():
    """Test that calls to one callee with different argument types each resolve their own."""

    def body(out, i, v):
        out[i] = consteval(v.bitwidth)

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, a):
        inlined(out, 0, a)
        inlined(out, 1, a * 2)
        called(out, 2, a)
        called(out, 3, a * 2)

    out = np.zeros(4, dtype=np.int64)
    kernel[1, 1](out, np.float32(1.5))
    np.testing.assert_array_equal(out[:2], out[2:])
    assert out[0] != out[1]


def test_nested_inlined_callee_parameter_types_match_call():
    """Test that a callee inlined inside another inlined callee resolves as in a call."""

    def body(out, i, v):
        out[i] = consteval(v.bitwidth)

    inner, called = _inlined_and_called(body)

    @cuda.jit(device=True, inline=True)
    def outer(out, i, a):
        inner(out, i, a * 2)

    @cuda.jit
    def kernel(out, a):
        outer(out, 0, a)
        called(out, 1, a * 2)

    _assert_inlined_matches_called(kernel, np.float32(1.5))


def test_inlined_callee_literally_matches_call():
    """Test that literally() on a parameter gives the same result inlined and called."""

    def body(out, i, n):
        literally(n)
        out[i] = consteval(isinstance(n, types.IntegerLiteral))

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, n):
        inlined(out, 0, n)
        called(out, 1, n)

    _assert_inlined_matches_called(kernel, 3)


def test_inlined_callee_untyped_argument_raises_typing_error():
    """Test that an argument that fails to type raises the caller's typing error."""

    @cuda.jit(device=True, inline=True)
    def fill(out, v):
        out[0] = consteval(v.bitwidth)

    @cuda.jit
    def kernel(out):
        fill(out, out.no_such_attribute)

    with pytest.raises(TypingError, match="no_such_attribute"):
        kernel.compile("void(int64[:])")


def test_recursive_callee_compiles_under_transforms():
    """Test that a self-recursive inline callee compiles and runs with the transforms on."""

    @cuda.jit(device=True, inline=True)
    def triangle(n):
        if n <= 0:
            return 0
        return n + triangle(n - 1)

    @cuda.jit
    def kernel(out, n):
        out[0] = triangle(n)

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](out, 4)
    assert out[0] == 4 + 3 + 2 + 1


_shadowed = 3


def test_inlined_callee_parameter_shadows_global():
    """Test that a parameter named like a global resolves to its type, as in a kernel."""

    @cuda.jit(device=True, inline=True)
    def callee(out, _shadowed):
        for i in consteval(range(_shadowed)):
            out[i] = 1

    @cuda.jit
    def kernel(out):
        callee(out, 5)

    with pytest.raises(ConstevalError, match="Cannot evaluate consteval argument"):
        kernel.compile("void(int32[:])")


def test_caller_option_controls_transform():
    """Test that the kernel's experimental_ast_transforms decides whether a callee is transformed."""
    n = 2

    @cuda.jit(device=True, inline=True, experimental_ast_transforms=False)
    def callee_off(out):
        for i in consteval(range(n)):
            out[i] = 1

    @cuda.jit(experimental_ast_transforms=True)
    def kernel_on(out):
        callee_off(out)

    out = np.zeros(2, dtype=np.int32)
    kernel_on[1, 1](out)
    np.testing.assert_array_equal(out, [1, 1])

    @cuda.jit(device=True, inline=True, experimental_ast_transforms=True)
    def callee_on(out):
        for i in consteval(range(n)):
            out[i] = 1

    @cuda.jit(experimental_ast_transforms=False)
    def kernel_off(out):
        callee_on(out)

    with pytest.raises(TypingError, match="consteval"):
        kernel_off.compile("void(int32[:])")


def test_rejected_cost_model_does_not_transform(monkeypatch):
    """Test that a callee whose cost model declines inlining is never transformed."""

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


def test_typed_inlinee_is_not_compiled():
    """Test that a callee typed for inlining is typed from its IR and never compiled."""

    @cuda.jit(device=True, inline=True)
    def fill(out, v):
        out[0] = consteval(v.bitwidth)

    @cuda.jit
    def kernel(out, a):
        fill(out, a * 2)

    kernel.compile("void(int64[:], float32)")
    assert not fill.overloads


_helpers = ModuleType("_helpers")


def test_typed_inlinee_through_module_attribute_is_not_compiled():
    """Test that a callee reached through a module attribute is also never compiled."""

    @cuda.jit(device=True, inline=True)
    def fill(out, v):
        out[0] = consteval(v.bitwidth)

    _helpers.fill = fill

    @cuda.jit
    def kernel(out, a):
        _helpers.fill(out, a * 2)

    kernel.compile("void(int64[:], float32)")
    assert not fill.overloads
