# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ZeroDivisionError`` raised by division and modulo in kernels.

``@cuda.jit(debug=True)`` selects the Python error model, under which dividing
by zero raises just as it does in CPython; ``debug=False`` selects the NumPy
error model, under which it does not. This mirrors numba-cuda, and the pair of
vendored tests in ``numba_cuda_tests/cudapy/test_exception.py``
(``test_zero_division_error_in_debug`` and ``test_no_zero_division_error``).

Only scalar operands are covered here. Array expressions keep the NumPy error
model regardless of ``debug``, matching numba's ufunc kernels, which wrap their
body in ``force_error_model(context, "numpy")``.
"""

import numpy as np
import pytest

from numba_cuda_mlir import cuda

SCALAR_DTYPES = [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64]
INT_DTYPES = [np.int32, np.int64, np.uint32, np.uint64]
FLOAT_DTYPES = [np.float32, np.float64]


def _floordiv(out, a, b):
    out[0] = a[0] // b[0]


def _mod(out, a, b):
    out[0] = a[0] % b[0]


def _truediv(out, a, b):
    out[0] = a[0] / b[0]


OPS = {"floordiv": _floordiv, "mod": _mod, "truediv": _truediv}


def _launch(body, dtype, lhs, rhs, *, debug, fastmath=False, out_dtype=None):
    """Compile *body* and run it on one element; return the device result."""
    kwargs = {"debug": True, "opt": False} if debug else {}
    if fastmath:
        kwargs["fastmath"] = True
    kernel = cuda.jit(**kwargs)(body)

    out = cuda.device_array(1, dtype=out_dtype or dtype)
    a = cuda.to_device(np.array([lhs], dtype=dtype))
    b = cuda.to_device(np.array([rhs], dtype=dtype))
    kernel[1, 1](out, a, b)
    cuda.synchronize()
    return out.copy_to_host()[0]


# --------------------------------------------------------------------------
# debug=True: the Python error model raises, as CPython does.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_int_floordiv_by_zero_raises_with_debug(dtype):
    with pytest.raises(ZeroDivisionError):
        _launch(_floordiv, dtype, 1, 0, debug=True)


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_int_mod_by_zero_raises_with_debug(dtype):
    with pytest.raises(ZeroDivisionError):
        _launch(_mod, dtype, 1, 0, debug=True)


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_int_truediv_by_zero_raises_with_debug(dtype):
    # ``int / int`` is computed in floating point, but CPython still raises.
    with pytest.raises(ZeroDivisionError):
        _launch(_truediv, dtype, 1, 0, debug=True, out_dtype=np.float64)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("op", ["floordiv", "mod", "truediv"])
def test_float_div_by_zero_raises_with_debug(dtype, op):
    # CPython raises for floats too: 1.0 / 0.0, 1.0 // 0.0 and 1.0 % 0.0 are
    # all ZeroDivisionError. Only the NumPy error model returns inf/nan.
    with pytest.raises(ZeroDivisionError):
        _launch(OPS[op], dtype, 1.0, 0.0, debug=True)


@pytest.mark.parametrize("lhs", [-3, 3])
@pytest.mark.parametrize("op", ["floordiv", "mod"])
def test_negative_operand_by_zero_raises_with_debug(lhs, op):
    with pytest.raises(ZeroDivisionError):
        _launch(OPS[op], np.int64, lhs, 0, debug=True)


def test_negative_zero_divisor_raises_with_debug():
    # -0.0 == 0.0, so it must raise like CPython's 1.0 / -0.0.
    with pytest.raises(ZeroDivisionError):
        _launch(_truediv, np.float64, 1.0, -0.0, debug=True)


def test_nan_divisor_does_not_raise_with_debug():
    # Only a zero divisor raises; NaN propagates, as in CPython.
    got = _launch(_truediv, np.float64, 1.0, np.nan, debug=True)
    assert np.isnan(got)


def test_zero_division_error_message():
    with pytest.raises(ZeroDivisionError, match="Division by zero in kernel"):
        _launch(_floordiv, np.int64, 1, 0, debug=True)


def test_variable_divisor_only_raises_when_zero():
    """The guard is a runtime check, not a constant fold on a literal zero."""

    def body(out, a, b):
        out[0] = a[0] // b[0]

    assert _launch(body, np.int64, 7, 2, debug=True) == 3
    with pytest.raises(ZeroDivisionError):
        _launch(body, np.int64, 7, 0, debug=True)


def test_raise_happens_before_the_store():
    """A faulting lane must leave the output untouched.

    This is what the vendored ``test_zero_division_error_in_debug`` asserts
    with ``"Expected result to be left unset"``.
    """

    def body(r, x, y):
        r[0] = y[0] / x[0]
        r[1] = y[0]

    kernel = cuda.jit(debug=True, opt=False)(body)
    r = np.zeros(2)
    with pytest.raises(ZeroDivisionError):
        kernel[1, 1](r, np.zeros(1), np.ones(1))
    assert r[0] == 0
    assert r[1] == 0


# --------------------------------------------------------------------------
# debug=False: the NumPy error model never raises.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("op", ["floordiv", "mod"])
def test_int_div_by_zero_does_not_raise_without_debug(dtype, op):
    _launch(OPS[op], dtype, 1, 0, debug=False)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_float_truediv_by_zero_gives_inf_without_debug(dtype):
    assert np.isposinf(_launch(_truediv, dtype, 1.0, 0.0, debug=False))
    assert np.isneginf(_launch(_truediv, dtype, -1.0, 0.0, debug=False))


def test_execution_continues_past_zero_division_without_debug():
    """``debug=False`` must not halt the kernel at the division."""

    def body(r, x, y):
        r[0] = y[0] / x[0]
        r[1] = y[0]

    kernel = cuda.jit(body)
    r = np.zeros(2)
    kernel[1, 1](r, np.zeros(1), np.ones(1))
    cuda.synchronize()
    assert np.isinf(r[0])
    assert r[1] == 1.0


def test_lineinfo_alone_does_not_raise():
    """``lineinfo`` turns on debug info but must not change the error model."""

    @cuda.jit(lineinfo=True)
    def body(out, a, b):
        out[0] = a[0] // b[0]

    out = cuda.device_array(1, dtype=np.int64)
    body[1, 1](out, cuda.to_device(np.array([1])), cuda.to_device(np.array([0])))
    cuda.synchronize()


# --------------------------------------------------------------------------
# fastmath relaxes the floating-point contract, so the float guard is dropped.
# --------------------------------------------------------------------------


def test_fastmath_suppresses_float_zero_division():
    got = _launch(_truediv, np.float32, 10.0, 0.0, debug=True, fastmath=True)
    assert np.isinf(got)


def test_fastmath_keeps_integer_zero_division():
    # fastmath relaxes floating-point semantics only.
    with pytest.raises(ZeroDivisionError):
        _launch(_floordiv, np.int64, 10, 0, debug=True, fastmath=True)


# --------------------------------------------------------------------------
# Non-zero divisors keep CPython's sign semantics.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rhs", [3, -3])
@pytest.mark.parametrize("lhs", [7, -7, 6, -6, 0])
@pytest.mark.parametrize("op", ["floordiv", "mod"])
def test_signed_int_semantics_match_cpython(op, lhs, rhs):
    got = _launch(OPS[op], np.int64, lhs, rhs, debug=True)
    expected = lhs // rhs if op == "floordiv" else lhs % rhs
    assert got == expected


@pytest.mark.parametrize("rhs", [3.0, -3.0])
@pytest.mark.parametrize("lhs", [7.0, -7.0, 6.0, -6.0])
@pytest.mark.parametrize("op", ["floordiv", "mod", "truediv"])
def test_float_semantics_match_cpython(op, lhs, rhs):
    got = _launch(OPS[op], np.float64, lhs, rhs, debug=True)
    if op == "floordiv":
        expected = lhs // rhs
    elif op == "mod":
        expected = lhs % rhs
    else:
        expected = lhs / rhs
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("dtype", SCALAR_DTYPES)
@pytest.mark.parametrize("debug", [True, False])
def test_nonzero_divisor_unaffected(dtype, debug):
    assert _launch(_floordiv, dtype, 8, 2, debug=debug) == 4
    assert _launch(_mod, dtype, 8, 3, debug=debug) == 2


# --------------------------------------------------------------------------
# The guard is emitted at compile time, not merely gated on the host readback.
# --------------------------------------------------------------------------


def _ptx(debug, fastmath=False):
    from numba_cuda_mlir.compiler import compile_ptx
    from numba_cuda_mlir.numba_cuda import types

    def body(out, a, b):
        out[0] = a[0] // b[0]

    sig = types.void(types.int64[:], types.int64[:], types.int64[:])
    ptx, _ = compile_ptx(body, sig, debug=debug, opt=False, fastmath=fastmath, cc=(8, 0))
    return ptx


def test_guard_is_absent_from_non_debug_ptx():
    """Without ``debug`` the error global is declared but never written, so
    the divide costs no comparison and no branch."""
    ptx = _ptx(debug=False)
    assert "__numba_cuda_mlir_error_code" in ptx
    assert "atom.cas" not in ptx


def test_guard_is_present_in_debug_ptx():
    assert "atom.cas" in _ptx(debug=True)


def test_guard_is_absent_from_fastmath_float_ptx():
    from numba_cuda_mlir.compiler import compile_ptx
    from numba_cuda_mlir.numba_cuda import types

    def body(out, a, b):
        out[0] = a[0] / b[0]

    sig = types.void(types.float32[:], types.float32[:], types.float32[:])
    checked, _ = compile_ptx(body, sig, debug=True, opt=False, cc=(8, 0))
    relaxed, _ = compile_ptx(body, sig, debug=True, opt=False, fastmath=True, cc=(8, 0))
    assert "atom.cas" in checked
    assert "atom.cas" not in relaxed
