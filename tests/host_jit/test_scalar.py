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


def test_bitwise_unary_and_comparisons():
    @host.jit
    def compute(a, b):
        value = ~a
        value &= b
        value |= a
        value ^= b
        if a < b and a != b:
            return -value
        if a >= b or a == b:
            return +value
        return 0

    for args in ((5, 3), (-5, 3), (3, 3)):
        assert compute(*args) == compute.py_func(*args)


@pytest.mark.parametrize(
    "args",
    [
        (0, 10, 2),
        (10, -3, -2),
        (3, 1, 1),
        (2**63 - 2, 2**63 - 1, 2),
        (-(2**63) + 1, -(2**63), -2),
        (-(2**63), 2**63 - 1, 2**63 - 1),
        (2**63 - 1, -(2**63), -(2**63)),
        (np.int8(-128), np.int8(127), np.int8(1)),
        (np.int8(127), np.int8(-128), np.int8(-1)),
        (np.uint64(0), np.uint64(2**64 - 1), np.uint64(2**63)),
        (np.uint64(2**64 - 1), np.uint64(0), np.uint64(1)),
    ],
)
def test_range_iteration(args):
    @host.jit
    def count(start, stop, step):
        total = 0
        for _ in range(start, stop, step):
            total += 1
        return total

    assert count(*args) == len(range(*(int(value) for value in args)))


def test_range_values_and_errors():
    @host.jit
    def sum_range(start, stop, step):
        total = 0
        for value in range(start, stop, step):
            total += value
        return total

    assert sum_range(np.int8(-128), np.int8(127), np.int8(1)) == -255
    with pytest.raises(ValueError):
        sum_range(0, 10, 0)
    assert sum_range(0, 10, 2) == 20


@pytest.mark.parametrize("args", [(0, 4, 1), (4, -1, -2), (3, 1, 1)])
def test_range_reuse(args):
    @host.jit
    def compute(start, stop, step):
        values = range(start, stop, step)
        total = 0
        for value in values:
            total += value
        for value in values:
            total += value
        for outer in values:
            for inner in values:
                total += outer * inner
        return total

    assert compute(*args) == compute.py_func(*args)


@pytest.mark.parametrize(
    "args",
    [(7, 3), (-7, 3), (7, -3), (-7, -3), (7.5, 2.5)],
)
def test_division_and_remainder(args):
    @host.jit
    def divide(a, b):
        return a / b

    @host.jit
    def quotient(a, b):
        return a // b

    @host.jit
    def remainder(a, b):
        return a % b

    for compiled in (divide, quotient, remainder):
        expected = compiled.py_func(*args)
        assert compiled(*args) == expected
        with pytest.raises(ZeroDivisionError):
            compiled(args[0], 0)
        assert compiled(*args) == expected


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
    with pytest.raises(TypeError):
        narrow(41)


def test_signature_annotation_precedence():
    def identity(value: types.int32):
        return value

    with pytest.raises(TypeError):
        host.jit(types.int64(types.int64))(identity)
    with pytest.raises(TypeError):
        host.jit([types.int32(types.int32), types.int64(types.int64)])(identity)

    compiled = host.jit(types.int32(types.int32))(identity)
    assert compiled(np.int32(41)) == 41


@pytest.mark.parametrize("value", [0, 2, -2, 0.0, -0.0, 2.0, float("nan")])
def test_boolean_conversions(value):
    @host.jit
    def captured():
        return bool(value)

    @host.jit
    def runtime(value):
        return bool(value)

    assert captured() is bool(value)
    assert runtime(value) is bool(value)


def test_scalar_constructors():
    @host.jit
    def construct(value):
        result = float(np.int32(abs(int(value))))
        return result + 1.0 if bool(2) else result

    for value in (-3.75, 0.0, 2.5):
        assert construct(value) == construct.py_func(value)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
)
@pytest.mark.parametrize("source_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("opt_level", [0, 3])
def test_float_to_integer_boundaries(dtype, source_dtype, opt_level):
    @host.jit(opt_level=opt_level)
    def convert(value):
        return dtype(value)

    limits = np.iinfo(dtype)
    lower = source_dtype(int(limits.min))
    upper = source_dtype(int(limits.max) + 1)
    last = np.nextafter(upper, source_dtype(-np.inf))
    for value in (source_dtype(-0.5), source_dtype(0.5), lower, last):
        assert convert(value) == int(value)
    for value in (
        upper,
        source_dtype(int(limits.min) - 1),
        np.nextafter(lower, source_dtype(-np.inf)),
        source_dtype(np.inf),
        source_dtype(-np.inf),
    ):
        # Fractions just below the minimum still truncate to the minimum.
        if np.isfinite(value) and int(limits.min) <= int(value) <= int(limits.max):
            assert convert(value) == int(value)
        else:
            with pytest.raises(OverflowError):
                convert(value)
    with pytest.raises(ValueError):
        convert(source_dtype(np.nan))
    assert convert(source_dtype(1.5)) == 1


@pytest.mark.parametrize(
    "value,error",
    [
        (float("nan"), ValueError),
        (float("inf"), OverflowError),
        (-float("inf"), OverflowError),
        (1e30, OverflowError),
    ],
)
@pytest.mark.parametrize("opt_level", [0, 3])
def test_builtin_int_conversion_errors(value, error, opt_level):
    @host.jit(opt_level=opt_level)
    def runtime(value):
        return int(value)

    @host.jit(opt_level=opt_level)
    def captured():
        return int(value)

    @host.jit(types.int64(types.float64), opt_level=opt_level)
    def return_cast(value):
        return value

    with pytest.raises(error):
        runtime(value)
    with pytest.raises(error):
        captured()
    with pytest.raises(error):
        return_cast(value)
    assert runtime(2.75) == 2
    assert return_cast(-2.75) == -2


@pytest.mark.parametrize("value", [np.arange(3), (1, 2), [1, 2], 1 + 2j, "text"])
def test_reject_non_scalar_arguments(value):
    @host.jit
    def identity(value):
        return value

    with pytest.raises((TypeError, ValueError)):
        identity(value)


def test_reject_non_scalar_return():
    @host.jit
    def pair(value):
        return value, value

    with pytest.raises(TypingError):
        pair(1)


def test_argument_count_and_keywords():
    @host.jit
    def add(a, b):
        return a + b

    with pytest.raises(TypeError):
        add(1)
    with pytest.raises(TypeError):
        add(a=1, b=2)


@pytest.mark.parametrize("opt_level", [0, 3])
@pytest.mark.parametrize("inplace", [False, True])
def test_signed_remainder_boundary(opt_level, inplace):
    def ordinary(a, b):
        return a % b

    def mutate(a, b):
        a %= b
        return a

    remainder = host.jit(opt_level=opt_level)(mutate if inplace else ordinary)
    minimum = -(2**63)
    assert remainder(minimum, -1) == 0


@pytest.mark.parametrize("opt_level", [0, 3])
def test_mixed_integer_division_and_remainder(opt_level):
    @host.jit(opt_level=opt_level)
    def quotient(a, b):
        return a // b

    @host.jit(opt_level=opt_level)
    def remainder(a, b):
        return a % b

    @host.jit(opt_level=opt_level)
    def divide(a, b):
        return a / b

    # Widening a narrow -1 must preserve its sign before checking overflow.
    with pytest.raises(OverflowError):
        quotient(-(2**63), np.int8(-1))
    assert remainder(-(2**63), np.int8(-1)) == 0
    for compiled, expected in ((quotient, -3), (remainder, 2), (divide, -7 / 3)):
        assert compiled(np.int8(-7), np.uint32(3)) == expected


def test_signed_division_overflow():
    @host.jit
    def quotient(a, b):
        return a // b

    with pytest.raises(OverflowError):
        quotient(-(2**63), -1)
    assert quotient(-(2**63), 1) == -(2**63)


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


@pytest.mark.parametrize("value", [np.arange(3), (1, 2), 1 + 2j])
def test_annotated_identity_rejects_non_scalar_actuals(value):
    @host.jit
    def identity(value: types.int64):
        return value

    with pytest.raises((TypeError, ValueError)):
        identity(value)
    assert identity(41) == 41
    with pytest.raises((TypeError, ValueError)):
        identity(value)


def test_concurrent_first_calls_publish_one_specialization():
    @host.jit
    def add_one(value):
        return value + 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(add_one, range(1, 9)))
    assert results == list(range(2, 10))
    assert len(add_one.overloads) == 1


@pytest.mark.parametrize("compile_name", ["compile_to_llvm_dialect", "compile"])
def test_direct_compile_boundaries(compile_name):
    from numba_cuda_mlir.experimental.host import compiler

    def add_one(value):
        return value + 1

    dispatcher = host.jit(add_one)
    compile_function = getattr(compiler, compile_name)
    result = compile_function(add_one, None, (types.int64,), dispatcher.targetoptions)
    if compile_name == "compile_to_llvm_dialect":
        result, module = result
        assert module.operation.verify()
    else:
        assert result.entry_point(41) == 42

    def pair(value):
        return value, value

    with pytest.raises(TypingError):
        compile_function(pair, None, (types.int64,), dispatcher.targetoptions)


@pytest.mark.parametrize("args", [(7, 3), (-7, 3), (7.5, 2.5)])
def test_inplace_arithmetic(args):
    @host.jit
    def compute(a, b):
        a += b
        a -= 1
        a *= 2
        a /= b
        a //= 2
        a %= 3
        return a

    assert compute(*args) == compute.py_func(*args)
    with pytest.raises(ZeroDivisionError):
        compute(args[0], 0)
    assert compute(*args) == compute.py_func(*args)


@pytest.mark.parametrize(
    "options", [{"opt_level": -1}, {"opt_level": 4}, {"dump": 1}, {"cache": True}]
)
def test_reject_unsupported_options(options):
    with pytest.raises((TypeError, ValueError)):
        host.jit(**options)


@pytest.mark.parametrize("opt_level", [0, 3])
def test_optimization_levels(opt_level):
    @host.jit(opt_level=opt_level)
    def compute(value):
        return value * 3 - 1

    assert compute(7) == 20


def test_signature_list():
    @host.jit([types.int32(types.int32), types.float64(types.float64)])
    def identity(value):
        return value

    assert identity(np.int32(41)) == 41
    assert identity(1.25) == 1.25
    with pytest.raises(TypeError):
        identity(41)


def test_reject_unsupported_scalar_operations():
    def power(value):
        return value**2

    def shift(value):
        return value << 1

    def attribute(value):
        return value.real

    def helper(value):
        return abs(value)

    def call(value):
        return helper(value)

    for function in (power, shift, attribute, call):
        with pytest.raises(TypingError):
            host.jit(function)(1)


def test_reject_unsupported_parameter_forms():
    def variadic(*args):
        return 1

    def keyword_only(*, value):
        return value

    for function in (variadic, keyword_only):
        with pytest.raises(TypeError):
            host.jit(function)


def test_none_return_annotation():
    @host.jit
    def no_result(value: types.int64) -> None:
        if value:
            return

    assert no_result(0) is None
    assert no_result(1) is None


@pytest.mark.parametrize(
    "typ,cases",
    [
        (types.int8, [(128, -128), (-129, 127)]),
        (types.uint8, [(-1, 255), (256, 0)]),
        (types.float32, [(16777217, 16777216.0), (7, 7.0)]),
    ],
)
def test_annotated_argument_coercion(typ, cases):
    @host.jit
    def identity(value: typ):
        return value

    for value, expected in cases:
        assert identity(value) == expected


def test_integer_argument_rejection_and_recovery():
    @host.jit
    def identity(value: types.int8):
        return value

    # Reject before first compilation and after a specialization exists.
    with pytest.raises(TypeError):
        identity(1.5)
    assert identity(7) == 7
    with pytest.raises(TypeError):
        identity(1.0)
    assert identity(9) == 9

    # Direct entry points bypass dispatch, but retain the slot conversion policy.
    entry_point = identity.compile((types.int8,)).entry_point
    assert entry_point(128) == -128
    with pytest.raises(TypeError):
        entry_point(1.5)
    assert entry_point(7) == 7


def test_inferred_annotated_and_explicit_argument_specialization():
    @host.jit
    def inferred(value):
        return value

    @host.jit
    def annotated(value: types.int8):
        return value

    @host.jit(types.int8(types.int8))
    def explicit(value):
        return value

    assert inferred(128) == 128
    assert inferred(1.5) == 1.5
    assert annotated(128) == -128
    with pytest.raises(TypeError):
        explicit(128)
    assert explicit(np.int8(7)) == 7
