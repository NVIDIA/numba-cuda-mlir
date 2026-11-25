# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consumer-visible behavior of escaping read-only runtime device closures."""

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir.numba_cuda.core.errors import UnsupportedError
from numba_cuda_mlir.runtime_closures import _make_closure_body

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


def test_repeated_closure_conversion_does_not_grow_registries():
    from types import SimpleNamespace

    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.runtime_closures import ConvertRuntimeClosures

    def factory(bias):
        return lambda value: value + bias

    registries = (
        extending.typing_registry.functions,
        extending.typing_registry.globals,
        extending.lowering_registry.functions,
    )
    previous = None
    for _ in range(3):
        state = SimpleNamespace(func_ir=run_frontend(factory))
        ConvertRuntimeClosures().run_pass(state)
        sizes = tuple(len(registry) for registry in registries)
        if previous is not None:
            assert sizes == previous
        previous = sizes


@pytest.mark.parametrize("nested", [False, True])
def test_captured_dtype_through_noninlined_device_call(nested):
    @cuda.jit(device=True, inline="never")
    def forward(fn, value):
        return fn(value)

    if nested:

        @cuda.jit
        def kernel(bias, output):
            environment = ((np.int64, None), bias)
            fn = lambda value: environment[0][0](value) + environment[1]
            output[0] = forward(fn, 3.5)
    else:

        @cuda.jit
        def kernel(bias, output):
            cast = np.int64
            fn = lambda value: cast(value) + bias
            output[0] = forward(fn, 3.5)

    output = np.zeros(1, dtype=np.int64)
    for bias in (7, -3):
        kernel[1, 1](bias, output)
        assert output[0] == np.int64(3.5) + bias


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


@pytest.mark.parametrize("nested", [False, True])
def test_shared_memory_array_capture_across_device_call(nested):
    @cuda.jit(device=True, inline="never")
    def forward(fn, index):
        return fn(index)

    @cuda.jit
    def kernel(bias, output):
        values = cuda.shared.array(2, types.int64)
        values[0] = bias
        values[1] = bias + 1
        if nested:
            environment = ((values, None), bias)
            fn = lambda index: environment[0][0][index] + environment[1]
        else:
            fn = lambda index: values[index]
        output[0] = forward(fn, 1)

    output = np.zeros(1, dtype=np.int64)
    for bias in (7, -3):
        kernel[1, 1](bias, output)
        assert output[0] == (2 * bias + 1 if nested else bias + 1)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("capture_kind", ["scalar", "tuple", "closure"])
def test_complex_capture_preserves_real_and_imaginary_parts(dtype, capture_kind):
    if capture_kind == "scalar":

        @cuda.jit(device=True)
        def make(bias):
            return lambda value: value * bias + bias
    elif capture_kind == "tuple":

        @cuda.jit(device=True)
        def make(bias):
            environment = ((None, bias),)
            return lambda value: value * environment[0][1] + environment[0][1]
    else:

        @cuda.jit(device=True)
        def make(bias):
            inner = lambda value: value * bias
            return lambda value: inner(value) + bias

    @cuda.jit
    def kernel(bias, value, output):
        output[0] = consume(make(bias), value)

    output = np.zeros(1, dtype=dtype)
    value = dtype(2 - 3j)
    for bias in (dtype(3 + 2j), dtype(-7 + 5j)):
        kernel[1, 1](bias, value, output)
        np.testing.assert_array_equal(output, [value * bias + bias])


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


@pytest.mark.parametrize("mode", ["never", "always", "lto"])
@pytest.mark.parametrize("source_available", [True, False])
def test_closure_body_preserves_globals_locals_and_runtime_captures(source_available, mode):
    def outer(bias, scale):
        def inner(x):
            __capture_bias = x * GLOBAL_SCALE
            signature = __capture_bias * scale
            if x > 0:
                return x + signature + bias
            return __capture_bias - bias

        return inner

    if not source_available:
        namespace = {"GLOBAL_SCALE": GLOBAL_SCALE}
        exec(
            "def outer(bias, scale):\n"
            "    def inner(x):\n"
            "        __capture_bias = x * GLOBAL_SCALE\n"
            "        signature = __capture_bias * scale\n"
            "        if x > 0:\n"
            "            return x + signature + bias\n"
            "        return __capture_bias - bias\n"
            "    return inner\n",
            namespace,
        )
        outer = namespace["outer"]
    function = outer(1000, 2000)
    body = _make_closure_body(function.__code__, function.__globals__)
    if mode == "lto":
        from numba_cuda_mlir import compiler

        symbol = "explicit_ir_capture_body"
        sig = types.int64(types.int64, types.int64, types.int64)
        native, _ = compiler.compile(
            body, sig, device=True, abi="c", output="ltoir", abi_info={"abi_name": symbol}
        )
        device_body = cuda.declare_device(symbol, sig, abi="c", link=cuda.LTOIR(native))
    else:
        device_body = cuda.jit(device=True, inline=mode)(body)

    @cuda.jit
    def kernel(bias, scale, x, output):
        output[0] = device_body(bias, scale, x)

    output = np.zeros(1, dtype=np.int64)
    for bias, scale, x in ((3, 2, 4), (-8, 5, 2), (9, 3, -2)):
        expected = outer(bias, scale)(x)
        assert body(bias, scale, x) == expected
        kernel[1, 1](bias, scale, x, output)
        assert output[0] == expected


def test_nonlocal_mutation_rejected():
    def outer(bias):
        def inner(x):
            nonlocal bias
            bias += x
            return bias

        return inner

    function = outer(1)
    with pytest.raises(UnsupportedError, match="nonlocal mutation"):
        _make_closure_body(function.__code__, function.__globals__)


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


@pytest.mark.parametrize("runtime_capture", [False, True])
def test_unspecializable_callable_allows_overload_fallback(runtime_capture):
    def invoke_or_negate(fn, value):
        raise RuntimeError("device-only")

    @extending.overload(invoke_or_negate, typing_registry=extending.typing_registry)
    def overload_invoke_or_negate(fn, value):
        if extending.get_callable_specialization(fn, (value,)) is None:

            def fallback(fn, value):
                return -value

            return fallback

        def invoke(fn, value):
            return fn(value)

        return invoke

    @cuda.jit(device=True)
    def array_only(value):
        return value[0]

    @cuda.jit
    def kernel(bias, output):
        if runtime_capture:
            fn = lambda value: value[0] + bias
            output[0] = invoke_or_negate(fn, bias)
        else:
            output[0] = invoke_or_negate(array_only, bias)

    output = np.zeros(1, dtype=np.int64)
    for bias in (7, -3):
        kernel[1, 1](bias, output)
        assert output[0] == -bias


def test_loop_carried_shared_cell_rejected():
    @cuda.jit
    def kernel(output):
        for bias in range(3):
            fn = lambda x: x + bias
            if bias == 0:
                first = fn
        output[0] = consume(first, 10)

    # A single STORE_DEREF executes repeatedly. Snapshotting the first closure
    # would return 10, whereas Python's final shared-cell value produces 12.
    with pytest.raises(UnsupportedError, match="loop-carried shared cell 'bias'"):
        kernel[1, 1](np.zeros(1, dtype=np.int64))


def test_sibling_mutation_rejected_before_local_inlining():
    @cuda.jit
    def kernel(bias, output):
        def mutate():
            nonlocal bias
            bias += 1

        fn = lambda x: x + bias
        mutate()
        output[0] = consume(fn, 4)

    with pytest.raises(UnsupportedError, match="mutable nonlocal cell 'bias'"):
        kernel[1, 1](3, np.zeros(1, dtype=np.int64))


def test_directly_inlined_mutable_closure_remains_supported():
    @cuda.jit
    def kernel(bias, output):
        def update(value):
            nonlocal bias
            bias += value
            return bias

        output[0] = update(4)
        output[1] = update(5)

    output = np.zeros(2, dtype=np.int64)
    kernel[1, 1](3, output)
    np.testing.assert_array_equal(output, [7, 12])


@pytest.mark.parametrize("inline", ["never", "always"])
@pytest.mark.parametrize("capture_kind", ["tuple", "closure"])
def test_composed_environments_across_nested_device_calls(inline, capture_kind):
    @cuda.jit(device=True)
    def transform(value):
        return value * 3

    if capture_kind == "tuple":

        @cuda.jit(device=True, inline=inline)
        def make(bias, enabled, values):
            environment = ((bias, enabled), (values, transform, None))
            return lambda index: (
                environment[1][1](environment[1][0][index]) + environment[0][0]
                if environment[0][1]
                else -environment[1][0][index]
            )
    else:

        @cuda.jit(device=True, inline=inline)
        def make(bias, enabled, values):
            inner = lambda index: values[index] if enabled else -values[index]
            return lambda index: inner(index) + bias

    @cuda.jit(device=True, inline=inline)
    def forward(bias, enabled, values):
        return make(bias, enabled, values)

    @cuda.jit
    def kernel(bias, enabled, values, output):
        output[0] = consume(forward(bias, enabled, values), 1)

    values = np.array([5, 13], dtype=np.int64)
    output = np.zeros(1, dtype=np.int64)
    for bias, enabled in ((7, True), (-2, False)):
        kernel[1, 1](bias, enabled, values, output)
        if capture_kind == "tuple":
            expected = 3 * values[1] + bias if enabled else -values[1]
        else:
            expected = (values[1] if enabled else -values[1]) + bias
        assert output[0] == expected


def test_closure_with_only_static_environment():
    helper = abs

    @cuda.jit
    def kernel(value, output):
        fn = lambda x: helper(x)
        output[0] = consume(fn, value)

    output = np.zeros(1, dtype=np.int64)
    for value in (-4, 7):
        kernel[1, 1](value, output)
        assert output[0] == abs(value)


def test_local_inlining_preserves_nested_escaping_capture_analysis():
    @cuda.jit
    def kernel(bias, output):
        def make(value):
            return lambda x: x + value

        output[0] = consume(make(bias), 4)

    output = np.zeros(1, dtype=np.int64)
    kernel[1, 1](7, output)
    assert output[0] == 11


def test_capture_requires_dominating_initialization():
    @cuda.jit
    def kernel(initialize, bias, output):
        if initialize:
            local = bias
        fn = lambda x: x + local
        output[0] = consume(fn, 4)

    with pytest.raises(UnsupportedError, match="uninitialized shared cell 'local'"):
        kernel[1, 1](False, 7, np.zeros(1, dtype=np.int64))


def test_immutable_capture_constructed_in_loop():
    @cuda.jit
    def kernel(bias, output):
        for index in range(output.size):
            fn = lambda x: x + bias
            output[index] = consume(fn, index)

    output = np.zeros(3, dtype=np.int64)
    kernel[1, 1](7, output)
    np.testing.assert_array_equal(output, [7, 8, 9])


@pytest.mark.parametrize("inline", ["never", "always"])
def test_repeated_calls_through_same_runtime_closure(inline):
    @cuda.jit(device=True, inline=inline)
    def twice(fn, value):
        return fn(value) + fn(value + 1)

    @cuda.jit
    def kernel(bias, output):
        fn = lambda value: value + bias
        output[0] = twice(fn, 4)

    output = np.zeros(1, dtype=np.int64)
    for bias in (7, -3):
        kernel[1, 1](bias, output)
        assert output[0] == 9 + 2 * bias


@pytest.mark.parametrize("nested", [False, True])
def test_static_dispatcher_through_noninlined_device_call(nested):
    @cuda.jit(device=True)
    def transform(value):
        return value * 3 - 2

    if nested:

        @cuda.jit(device=True, inline="never")
        def forward(environment, value):
            return environment[0][0](value) + environment[1]

        @cuda.jit
        def kernel(value, bias, output):
            output[0] = forward(((transform, None), bias), value)
    else:

        @cuda.jit(device=True, inline="never")
        def forward(fn, value, bias):
            return fn(value) + bias

        @cuda.jit
        def kernel(value, bias, output):
            output[0] = forward(transform, value=value, bias=bias)

    output = np.zeros(1, dtype=np.int64)
    for value, bias in ((7, 5), (-4, -9)):
        kernel[1, 1](value, bias, output)
        assert output[0] == value * 3 - 2 + bias


def test_captured_module_constant_through_device_call():
    import math

    @cuda.jit(device=True, inline="never")
    def forward(fn, value):
        return fn(value)

    @cuda.jit
    def kernel(bias, output):
        alias = math
        fn = lambda value: value + bias + alias.pi
        output[0] = forward(fn, 4)

    output = np.zeros(1, dtype=np.float64)
    for bias in (7, -3):
        kernel[1, 1](bias, output)
        assert output[0] == 4 + bias + math.pi


def test_variadic_runtime_closure_across_device_call():
    @cuda.jit(device=True, inline="never")
    def forward(fn, left, right):
        return fn(left, right)

    @cuda.jit
    def kernel(bias, left, right, output):
        def fn(*values):
            return values[0] * 100 + values[1] + bias

        output[0] = forward(fn, left, right)

    output = np.zeros(1, dtype=np.int64)
    for bias, left, right in ((7, 2, 3), (-3, 5, -2)):
        kernel[1, 1](bias, left, right, output)
        assert output[0] == left * 100 + right + bias


@pytest.mark.parametrize("with_value", [False, True])
def test_static_callable_tuple_return_with_zero_or_one_runtime_leaf(with_value):
    @cuda.jit(device=True)
    def transform(value):
        return value * 3

    if with_value:

        @cuda.jit(device=True, inline="never")
        def package(fn, value):
            return ((fn, None), value)

        @cuda.jit
        def kernel(value, output):
            environment = package(transform, value)
            output[0] = environment[0][0](environment[1])
    else:

        @cuda.jit(device=True, inline="never")
        def package(fn):
            return ((fn,), None)

        @cuda.jit
        def kernel(value, output):
            environment = package(transform)
            output[0] = environment[0][0](value) if environment[1] is None else -1

    output = np.zeros(1, dtype=np.int64)
    for value in (7, -3):
        kernel[1, 1](value, output)
        assert output[0] == value * 3
