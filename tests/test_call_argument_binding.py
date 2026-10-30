# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A device-function call whose arguments do not bind raises a TypeError.

The error is also a TypingError, carries Python's message prefixed with the
callee's name, and points at the call, whether or not the callee is inlined.
Inlined calls used to raise a plain TypingError reading "Cannot bind 'args=[Var(...)]
kws=...' to signature ...", with no location.
"""

import inspect

import numpy as np
import pytest

from numba_cuda_mlir import cuda, compiler, types
from numba_cuda_mlir.numba_cuda.core.errors import ArgumentBindingError, TypingError

INLINE = pytest.mark.parametrize(
    "inline", ["always", "never", None], ids=["always", "never", "default"]
)


def _device(inline, func):
    options = {} if inline is None else {"inline": inline}
    return cuda.jit(device=True, **options)(func)


def _compile(kernel):
    compiler.compile_mlir(kernel, types.void(types.int64[::1]))


def _call_line(func, text):
    lines, first = inspect.getsourcelines(func)
    return first + next(i for i, line in enumerate(lines) if text in line)


def _assert_binding_error(excinfo, message, body, call):
    exc = excinfo.value
    assert isinstance(exc, ArgumentBindingError)
    assert isinstance(exc, TypingError)
    text = str(exc)
    assert message in text
    assert "Cannot bind" not in text
    assert f"line {_call_line(body, call)}:" in text
    assert call in text


@INLINE
def test_unknown_keyword(inline):
    def f(a, *, b=None):
        return a

    f = _device(inline, f)

    def body(out):
        out[0] = f(out[0], c=1)

    with pytest.raises(TypeError) as excinfo:
        _compile(cuda.jit(body))
    _assert_binding_error(
        excinfo, "f() got an unexpected keyword argument 'c'", body, "f(out[0], c=1)"
    )


@INLINE
def test_too_many_positional_arguments(inline):
    def g(a, b):
        return a + b

    g = _device(inline, g)

    def body(out):
        out[0] = g(out[0], 1, 2)

    with pytest.raises(TypeError) as excinfo:
        _compile(cuda.jit(body))
    _assert_binding_error(excinfo, "g() too many positional arguments", body, "g(out[0], 1, 2)")


@INLINE
def test_keyword_only_passed_positionally(inline):
    def f(a, *, b=None):
        return a

    f = _device(inline, f)

    def body(out):
        out[0] = f(out[0], 1)

    with pytest.raises(TypeError) as excinfo:
        _compile(cuda.jit(body))
    _assert_binding_error(excinfo, "f() too many positional arguments", body, "f(out[0], 1)")


@INLINE
def test_positional_only_passed_by_keyword(inline):
    def f(a, b, /):
        return a + b

    f = _device(inline, f)

    def body(out):
        out[0] = f(out[0], b=1)

    with pytest.raises(TypeError) as excinfo:
        _compile(cuda.jit(body))
    _assert_binding_error(
        excinfo,
        "f() missing a required positional-only argument: 'b'",
        body,
        "f(out[0], b=1)",
    )


@INLINE
def test_missing_required_argument(inline):
    def g(a, b):
        return a + b

    g = _device(inline, g)

    def body(out):
        out[0] = g(out[0])

    with pytest.raises(TypeError) as excinfo:
        _compile(cuda.jit(body))
    _assert_binding_error(excinfo, "g() missing a required argument: 'b'", body, "g(out[0])")


@INLINE
def test_argument_given_twice(inline):
    def g(a, b):
        return a + b

    g = _device(inline, g)

    def body(out):
        out[0] = g(out[0], a=1)

    with pytest.raises(TypeError) as excinfo:
        _compile(cuda.jit(body))
    _assert_binding_error(excinfo, "g() multiple values for argument 'a'", body, "g(out[0], a=1)")


@INLINE
def test_valid_calls_bind_as_in_python(inline):
    def h(a, /, b, *, c=7, d=0):
        return a * 1000 + b * 100 + c * 10 + d

    h = _device(inline, h)

    @cuda.jit
    def kernel(out):
        out[0] = h(1, 2)
        out[1] = h(1, b=2, d=3)
        out[2] = h(1, 2, d=4, c=5)

    out = np.zeros(3, dtype=np.int64)
    kernel[1, 1](out)
    np.testing.assert_equal(
        out, [h.py_func(1, 2), h.py_func(1, b=2, d=3), h.py_func(1, 2, d=4, c=5)]
    )
