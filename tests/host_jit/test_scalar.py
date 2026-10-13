# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from numba_cuda_mlir import types
from numba_cuda_mlir.experimental import host
from numba_cuda_mlir.numba_cuda.core.errors import TypingError


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.bool_, [False, True]),
        (np.int8, [-128, 127]),
        (np.uint8, [0, 255]),
        (np.int16, [-32768, 32767]),
        (np.uint16, [0, 65535]),
        (np.int32, [-(2**31), 2**31 - 1]),
        (np.uint32, [0, 2**32 - 1]),
        (np.int64, [-(2**63), 2**63 - 1]),
        (np.uint64, [0, 2**64 - 1]),
        (np.float32, [-1.25, 3.5]),
        (np.float64, [-1.25, 3.5]),
    ],
)
def test_scalar_roundtrip(dtype, values):
    @host.jit
    def identity(value):
        return value

    for value in values:
        assert identity(dtype(value)) == dtype(value)


def test_arithmetic_branching_and_loops():
    @host.jit
    def compute(n):
        result = 0
        for i in range(n):
            if i % 2 == 0:
                result += i * 3
            else:
                result -= i
        while result < 0:
            result += 2
        return result

    for n in (0, 1, 2, 7, 20):
        assert compute(n) == compute.py_func(n)


def test_dynamic_range_step_and_failure_recovery():
    @host.jit
    def sum_range(start, stop, step):
        total = 0
        for i in range(start, stop, step):
            total += i
        return total

    for args in ((0, 10, 2), (10, -3, -2), (3, 1, 1)):
        assert sum_range(*args) == sum(range(*args))
    with pytest.raises(ValueError, match="must not be zero"):
        sum_range(0, 10, 0)
    assert sum_range(0, 10, 2) == 20


@pytest.mark.parametrize("a,b", [(7, 3), (-7, 3), (7, -3), (-7, -3)])
def test_integer_division_and_remainder(a, b):
    @host.jit
    def quotient(a, b):
        return a // b

    @host.jit
    def remainder(a, b):
        return a % b

    assert quotient(a, b) == a // b
    assert remainder(a, b) == a % b
    with pytest.raises(ZeroDivisionError):
        quotient(a, 0)
    with pytest.raises(ZeroDivisionError):
        remainder(a, 0)
    assert quotient(a, b) == a // b


def test_float_division_and_failure_recovery():
    @host.jit
    def divide(a, b):
        return a / b

    assert divide(7.5, 2.5) == 3.0
    with pytest.raises(ZeroDivisionError):
        divide(7.5, 0.0)
    assert divide(7.5, 2.5) == 3.0


def test_none_and_no_argument_returns():
    @host.jit
    def no_result(value):
        if value:
            return

    @host.jit
    def answer():
        return 42

    assert no_result(True) is None
    assert no_result(False) is None
    assert answer() == 42


def test_explicit_signature_and_annotations():
    @host.jit(types.int32(types.int32))
    def narrow(value):
        return value + 1

    @host.jit
    def annotated(value: types.float32) -> types.float32:
        return value * 2

    assert narrow(np.int32(41)) == 42
    assert annotated(np.float32(1.25)) == 2.5
    with pytest.raises(TypeError, match="no matching"):
        narrow(41)


@pytest.mark.parametrize("value", [np.arange(3), (1, 2), [1, 2], 1 + 2j, "text"])
def test_reject_non_scalar_arguments(value):
    @host.jit
    def identity(value):
        return value

    with pytest.raises((TypeError, ValueError), match="scalar|type"):
        identity(value)


def test_reject_non_scalar_return():
    @host.jit
    def pair(value):
        return value, value

    with pytest.raises(TypingError, match="scalar"):
        pair(1)


def test_argument_count_and_keywords():
    @host.jit
    def add(a, b):
        return a + b

    with pytest.raises(TypeError, match="expected 2"):
        add(1)
    with pytest.raises(TypeError, match="keyword"):
        add(a=1, b=2)


def test_signed_division_overflow():
    @host.jit
    def quotient(a, b):
        return a // b

    with pytest.raises(OverflowError):
        quotient(-(2**63), -1)
    assert quotient(-(2**63), 1) == -(2**63)


def test_range_endpoint_does_not_wrap():
    @host.jit
    def count(start, stop, step):
        total = 0
        for _ in range(start, stop, step):
            total += 1
        return total

    assert count(2**63 - 2, 2**63 - 1, 2) == 1
    assert count(-(2**63) + 1, -(2**63), -2) == 1


@pytest.mark.parametrize(
    "a,b",
    [
        (7.5, 2.0),
        (-7.5, 2.0),
        (7.5, -2.0),
        (-7.5, -2.0),
        (0.0, -2.0),
        (-0.0, 2.0),
        (1.0, 0.1),
    ],
)
def test_float_floor_division_and_remainder(a, b):
    @host.jit
    def quotient(a, b):
        return a // b

    @host.jit
    def remainder(a, b):
        return a % b

    for compiled, expected in ((quotient, a // b), (remainder, a % b)):
        actual = compiled(a, b)
        assert actual == expected
        assert np.signbit(actual) == np.signbit(expected)


def test_integer_true_division():
    @host.jit
    def divide(a, b):
        return a / b

    assert divide(-7, 2) == -3.5
    with pytest.raises(ZeroDivisionError):
        divide(7, 0)


@pytest.mark.parametrize("value", [np.arange(3), (1, 2), 1 + 2j])
def test_annotated_identity_rejects_non_scalar_actuals(value):
    @host.jit
    def identity(value: types.int64):
        return value

    with pytest.raises((TypeError, ValueError), match="scalar|type"):
        identity(value)
    assert identity(41) == 41
    with pytest.raises((TypeError, ValueError), match="scalar|type"):
        identity(value)
    assert identity(42) == 42


def test_concurrent_first_calls_publish_one_specialization():
    @host.jit
    def add_one(value):
        return value + 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(add_one, range(1, 9)))
    assert results == list(range(2, 10))
    assert len(add_one.overloads) == 1


@pytest.mark.parametrize("compile_name", ["compile_host_to_llvm_dialect", "compile_host"])
def test_direct_compile_boundaries(compile_name):
    from numba_cuda_mlir.experimental.host import compiler

    def add_one(value):
        return value + 1

    dispatcher = host.jit(add_one)
    compile_function = getattr(compiler, compile_name)
    result = compile_function(add_one, None, (types.int64,), dispatcher.targetoptions)
    if compile_name == "compile_host_to_llvm_dialect":
        result, module = result
        assert module.operation.verify()
    else:
        assert result.entry_point(41) == 42
    assert result.signature.args == (types.int64,)
    assert result.signature.return_type == types.int64

    def pair(value):
        return value, value

    with pytest.raises(TypingError, match="scalar"):
        compile_function(pair, None, (types.int64,), dispatcher.targetoptions)


def _inplace_add(a, b):
    a += b
    return a


def _inplace_subtract(a, b):
    a -= b
    return a


def _inplace_multiply(a, b):
    a *= b
    return a


def _inplace_bit_and(a, b):
    a &= b
    return a


def _inplace_bit_or(a, b):
    a |= b
    return a


def _inplace_bit_xor(a, b):
    a ^= b
    return a


@pytest.mark.parametrize(
    "function",
    [
        _inplace_add,
        _inplace_subtract,
        _inplace_multiply,
        _inplace_bit_and,
        _inplace_bit_or,
        _inplace_bit_xor,
    ],
    ids=["add", "subtract", "multiply", "and", "or", "xor"],
)
def test_inplace_integer_operations(function):
    compiled = host.jit(function)
    for args in ((7, 3), (-7, 3), (7, -3), (-7, -3)):
        assert compiled(*args) == function(*args)


def _inplace_divide(a, b):
    a /= b
    return a


def _inplace_quotient(a, b):
    a //= b
    return a


def _inplace_remainder(a, b):
    a %= b
    return a


@pytest.mark.parametrize(
    "function",
    [_inplace_divide, _inplace_quotient, _inplace_remainder],
    ids=["divide", "quotient", "remainder"],
)
def test_inplace_division_and_failure_recovery(function):
    compiled = host.jit(function)
    for args in ((7, 3), (-7, 3), (7, -3), (-7, -3)):
        assert compiled(*args) == function(*args)
    with pytest.raises(ZeroDivisionError):
        compiled(7, 0)
    assert compiled(-7, 3) == function(-7, 3)
