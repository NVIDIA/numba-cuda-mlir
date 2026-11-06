# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consumer-visible behavior of escaping read-only runtime device closures."""

import dis

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir.numba_cuda.core.errors import UnsupportedError
from numba_cuda_mlir.runtime_closures import _explicit_capture_function

GLOBAL_SCALE = 7


@cuda.jit(device=True)
def consume(fn, value):
    return fn(value)


def extension_consume(fn, value):
    raise RuntimeError("device-only")


@extending.overload(extension_consume, typing_registry=extending.typing_registry)
def overload_extension_consume(fn, value):
    specialization = extending.get_callable_specialization(fn, (value,))
    if specialization is None:
        return None

    def impl(fn, value):
        return fn(value)

    return impl


def test_runtime_capture_passed_to_device_function_and_globals():
    @cuda.jit
    def kernel(bias, output):
        fn = lambda x: x * GLOBAL_SCALE + bias
        output[0] = consume(fn, 4)

    output = np.zeros(1, dtype=np.int64)
    for bias in (3, 19, -8):
        kernel[1, 1](bias, output)
        assert output[0] == 4 * GLOBAL_SCALE + bias


def test_runtime_capture_extension_and_independent_state():
    @cuda.jit
    def kernel(bias, other, output):
        first = lambda x: x + bias
        second = lambda x: x + other
        output[0] = extension_consume(first, 4)
        output[1] = extension_consume(second, 4)
        output[2] = extension_consume(first, 8)

    output = np.zeros(3, dtype=np.int64)
    kernel[1, 1](11, -5, output)
    np.testing.assert_array_equal(output, [15, -1, 19])
    kernel[1, 1](-9, 20, output)
    np.testing.assert_array_equal(output, [-5, 24, -1])


def test_array_and_boolean_captures():
    @cuda.jit
    def kernel(values, enabled, output):
        fn = lambda x: values[x] if enabled else -values[x]
        output[0] = consume(fn, 1)

    values = np.array([5, 13], dtype=np.int64)
    output = np.zeros(1, dtype=np.int64)
    for enabled in (True, False):
        kernel[1, 1](values, enabled, output)
        assert output[0] == (13 if enabled else -13)


def test_same_code_multiple_runtime_environments():
    @cuda.jit(device=True)
    def make(bias):
        return lambda x: x + bias

    @cuda.jit
    def kernel(bias, other, output):
        first = make(bias)
        second = make(other)
        output[0] = consume(first, 6)
        output[1] = consume(second, 6)

    output = np.zeros(2, dtype=np.int64)
    kernel[1, 1](15, -7, output)
    np.testing.assert_array_equal(output, [21, -1])


def test_device_dispatcher_passed_through_extension():
    @cuda.jit(device=True)
    def transform(value):
        return value * 3 - 2

    @cuda.jit
    def kernel(value, output):
        output[0] = extension_consume(transform, value)

    output = np.zeros(1, dtype=np.int64)
    for value in (7, -4):
        kernel[1, 1](value, output)
        assert output[0] == value * 3 - 2


def test_static_function_capture_preserves_runtime_fields():
    helper = abs

    @cuda.jit
    def kernel(bias, output):
        fn = lambda x: helper(x) + bias
        output[0] = consume(fn, -4)

    output = np.zeros(1, dtype=np.int64)
    for bias in (3, -9):
        kernel[1, 1](bias, output)
        assert output[0] == 4 + bias


def test_explicit_capture_function_preserves_globals_and_local_variables():
    def outer(bias):
        def inner(x):
            local = x * GLOBAL_SCALE
            return local + bias

        return inner

    function = outer(1000)
    unsupported = [
        i.opname
        for i in dis.get_instructions(function)
        if "FAST" in i.opname
        and i.opname
        not in {"LOAD_FAST", "STORE_FAST", "DELETE_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_AND_CLEAR"}
    ]
    if unsupported:
        with pytest.raises(UnsupportedError, match=f"opcode {unsupported[0]} .*unsupported"):
            _explicit_capture_function(function.__code__, function.__globals__)
    else:
        body = _explicit_capture_function(function.__code__, function.__globals__)
        assert body(3, 4) == 31
        assert body(-8, 2) == 6


def test_explicit_capture_function_rejects_unhandled_fast_local_opcodes():
    def outer(bias):
        def inner(x):
            local = x * GLOBAL_SCALE
            return x + local + bias

        return inner

    function = outer(1000)
    supported = {"LOAD_FAST", "STORE_FAST", "DELETE_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_AND_CLEAR"}
    unhandled = [
        i.opname
        for i in dis.get_instructions(function)
        if "FAST" in i.opname and i.opname not in supported
    ]
    if not unhandled:
        pytest.skip("This Python version does not emit fused or borrowed fast-local opcodes")
    with pytest.raises(UnsupportedError, match=f"opcode {unhandled[0]} .*unsupported"):
        _explicit_capture_function(function.__code__, function.__globals__)


def test_nonlocal_mutation_rejected():
    def outer(bias):
        def inner(x):
            nonlocal bias
            bias += x
            return bias

        return inner

    function = outer(1)
    with pytest.raises(UnsupportedError, match="nonlocal mutation"):
        _explicit_capture_function(function.__code__, function.__globals__)


def test_reassigned_shared_cell_rejected():
    @cuda.jit
    def kernel(bias, output):
        fn = lambda x: x + bias
        bias = bias + 1
        output[0] = consume(fn, 4)

    with pytest.raises(UnsupportedError, match="reassigned shared cell 'bias'"):
        kernel[1, 1](2, np.zeros(1, dtype=np.int64))


def test_unrelated_reassignment_allowed():
    @cuda.jit
    def kernel(bias, output):
        fn = lambda x: x + bias
        temporary = 1
        temporary = temporary + 2
        output[0] = consume(fn, temporary)

    output = np.zeros(1, dtype=np.int64)
    kernel[1, 1](9, output)
    assert output[0] == 12


@pytest.mark.parametrize("use_extension", [False, True])
def test_void_runtime_closure_side_effects(use_extension):
    @cuda.jit
    def kernel(bias, output):
        def write(value):
            output[0] = value + bias

        if use_extension:
            result = extension_consume(write, 4)
        else:
            result = consume(write, 4)
        output[1] = 1 if result is None else 0

    output = np.zeros(2, dtype=np.int64)
    for bias in (3, 19):
        kernel[1, 1](bias, output)
        np.testing.assert_array_equal(output, [4 + bias, 1])


def test_void_external_lto_call_returned_by_runtime_closure():
    from numba_cuda_mlir import compiler

    def native_write(pointer, value):
        pointer[0] = value

    native = compiler.compile(
        native_write,
        types.void(types.CPointer(types.int64), types.int64),
        device=True,
        abi="c",
        output="ltoir",
        abi_info={"abi_name": "runtime_closure_native_write"},
    )
    write = cuda.declare_device(
        "runtime_closure_native_write",
        types.void(types.CPointer(types.int64), types.int64),
        link=cuda.LTOIR(native[0]),
        abi="c",
    )

    namespace = {"write": write, "consume": consume}
    exec(
        "def kernel(pointer, bias):\n"
        "    fn = lambda value: write(pointer, value + bias)\n"
        "    consume(fn, 4)\n",
        namespace,
    )

    output = cuda.to_device(np.zeros(1, dtype=np.int64))
    pointer = output.__cuda_array_interface__["data"][0]
    kernel = cuda.jit(types.void(types.CPointer(types.int64), types.int64))(namespace["kernel"])
    for bias in (3, 19):
        kernel[1, 1](pointer, bias)
        np.testing.assert_array_equal(output.copy_to_host(), [4 + bias])


def test_unsupported_dispatcher_specialization_declines():
    from numba_cuda_mlir.numba_cuda.types import Dispatcher

    @cuda.jit(types.int64(types.int64), device=True)
    def scalar_only(value):
        return value + 1

    scalar_only.disable_compile()
    specialization = extending.get_callable_specialization(
        Dispatcher(scalar_only), (types.float32[::1],)
    )
    assert specialization is None
