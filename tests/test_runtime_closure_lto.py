# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Callable-boundary elimination, not byte-identical C++ code generation.

The isolated integer kernels need no runtime helpers, so every SASS CALL is a
retained callable boundary, including register-target calls. A no-inline C++
bridge control proves that the check detects it. No nvmath dependency is required.
"""

import re
import subprocess

import numpy as np
import pytest
from cuda import pathfinder

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm
from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.runtime_closures import RuntimeClosure


def _cross_language_consumer(inline):
    def consume(fn, value):
        raise RuntimeError("device-only C++ callable consumer")

    @extending.overload(consume, inline="always", typing_registry=extending.typing_registry)
    def overload_consume(fn, value):
        if not isinstance(fn, RuntimeClosure):
            return
        symbol = "closure_callback_" + inline
        bridge_symbol = "closure_bridge_" + inline

        def state_pointer(fn):
            raise RuntimeError("device-only closure state")

        @extending.type_callable(state_pointer)
        def type_state_pointer(context):
            return lambda fn: types.CPointer(types.uint8)

        @lowering_registry.lower(state_pointer, fn)
        def lower_state_pointer(ctx, target, args, kwargs):
            state = ctx.load_var(args[0])
            pointer = llvm.alloca(
                llvm.PointerType.get(),
                arith.constant(ir.IntegerType.get_signless(32), 1),
                elem_type=state.type,
            )
            llvm.store(state, pointer)
            ctx.store_var(target, pointer)

        def load_state(pointer):
            raise RuntimeError("device-only closure state")

        @extending.type_callable(load_state)
        def type_load_state(context):
            return lambda pointer: fn

        @lowering_registry.lower(load_state, types.CPointer(types.uint8))
        def lower_load_state(ctx, target, args, kwargs):
            ctx.store_var(target, llvm.load(ctx.get_mlir_type(fn), ctx.load_var(args[0])))

        extending.refresh_registries(typing=False, target=True)

        def callback(pointer, value):
            return load_state(pointer)(value)

        callback_dispatcher = cuda.jit(device=True, abi="c", lto=True, inline="always")(callback)
        result = callback_dispatcher.compile(
            types.int64(types.CPointer(types.uint8), types.int64),
            abi_info={"abi_name": symbol},
            output="ltoir",
        )
        links = [cuda.LTOIR(result.metadata["ltoir"])]
        links.extend(cuda.LTOIR(code) for code in result.metadata["linker"]._ltoirs.values())
        links.extend(result.metadata.get("linked_external_link_items", ()))
        qualifier = (
            "__forceinline__ __attribute__((used))" if inline == "always" else "__noinline__"
        )
        links.append(
            cuda.CUSource(f"""
extern "C" __device__ long long {symbol}(void*, long long);
struct PythonFunctor {{
    void* state;
    __device__ __forceinline__ long long operator()(long long value) const {{
        return {symbol}(state, value);
    }}
}};
extern "C" __device__ {qualifier}
long long {bridge_symbol}(void* state, long long value) {{
    return PythonFunctor{{state}}(value);
}}
""")
        )
        bridge = cuda.declare_device(
            bridge_symbol,
            types.int64(types.CPointer(types.uint8), types.int64),
            abi="c",
            link=links,
        )

        def impl(fn, value):
            return bridge(state_pointer(fn), value)

        return impl

    return consume


@pytest.mark.parametrize("bridge_inline", ["always", "never"])
def test_python_callable_inlining_matches_cpp_boundary(bridge_inline, tmp_path):
    nvdisasm = pathfinder.find_nvidia_binary_utility("nvdisasm")
    if nvdisasm is None:
        pytest.skip("nvdisasm required for final callable-boundary inspection")
    consume = _cross_language_consumer(bridge_inline)

    @cuda.jit(lto=True)
    def python_kernel(scale, bias, values, output):
        index = cuda.grid(1)
        if index < values.size:
            fn = lambda value: value * scale + bias
            output[index] = consume(fn, values[index])

    cpp_source = cuda.CUSource("""
struct CppFunctor {
    long long scale;
    long long bias;
    __device__ __forceinline__ long long operator()(long long value) const {
        return value * scale + bias;
    }
};
extern "C" __device__ __forceinline__ __attribute__((used))
long long cpp_callable_bridge(long long scale, long long bias, long long value) {
    return CppFunctor{scale, bias}(value);
}
""")
    cpp_bridge = cuda.declare_device(
        "cpp_callable_bridge",
        types.int64(types.int64, types.int64, types.int64),
        abi="c",
        link=cpp_source,
    )

    @cuda.jit(lto=True)
    def cpp_kernel(scale, bias, values, output):
        index = cuda.grid(1)
        if index < values.size:
            output[index] = cpp_bridge(scale, bias, values[index])

    values = np.array([-17, -1, 0, 4, 29], dtype=np.int64)
    device_values = cuda.to_device(values)
    python_output = cuda.device_array_like(values)
    cpp_output = cuda.device_array_like(values)
    for scale, bias in ((3, 7), (-2, 19), (3, 7)):
        python_kernel[1, 32](scale, bias, device_values, python_output)
        cpp_kernel[1, 32](scale, bias, device_values, cpp_output)
        expected = values * scale + bias
        np.testing.assert_array_equal(python_output.copy_to_host(), expected)
        np.testing.assert_array_equal(cpp_output.copy_to_host(), expected)

    calls = {}
    for name, kernel in (("python", python_kernel), ("cpp", cpp_kernel)):
        cubin = tmp_path / f"{name}.cubin"
        cubin.write_bytes(next(iter(kernel.get_metadata().values()))["cubin"])
        sass = subprocess.check_output([nvdisasm, "-c", str(cubin)], text=True)
        (tmp_path / f"{name}.sass").write_text(sass)
        calls[name] = [
            line.strip() for line in sass.splitlines() if re.search(r"\bCALL(?:\.|\s)", line)
        ]
    assert not calls["cpp"], calls["cpp"]
    if bridge_inline == "always":
        assert not calls["python"], calls["python"]
    else:
        assert calls["python"], "No-inline control failed to retain its C++ bridge boundary"


def test_composed_environment_through_cpp_lto_bridge():
    consume = _cross_language_consumer("always")

    @cuda.jit(lto=True)
    def kernel(values, bias, enabled, output):
        index = cuda.grid(1)
        if index < values.size:
            environment = (values, bias, enabled)
            inner = lambda i: (
                environment[0][i] + environment[1] if environment[2] else -environment[0][i]
            )
            fn = lambda i: inner(i) * 2
            output[index] = consume(fn, index)

    values = np.array([-17, -1, 0, 4, 29], dtype=np.int64)
    device_values = cuda.to_device(values)
    output = cuda.device_array_like(values)
    for bias, enabled in ((7, True), (-3, False), (-9, True)):
        kernel[1, 32](device_values, bias, enabled, output)
        expected = (values + bias if enabled else -values) * 2
        np.testing.assert_array_equal(output.copy_to_host(), expected)


@pytest.mark.parametrize("nested", [False, True])
def test_inferred_specialized_body_static_captures_through_declared_lto_call(nested):
    from numba_cuda_mlir import compiler

    @cuda.jit(device=True)
    def transform(value):
        return value * 3

    def consume(fn, value):
        raise RuntimeError("device-only specialized callable consumer")

    @extending.overload(consume, typing_registry=extending.typing_registry)
    def overload_consume(fn, value):
        specialization = extending.get_callable_specialization(fn, (value,))
        if specialization is None:
            return
        symbol = "static_capture_body_" + str(nested)
        native, _ = compiler.compile(
            specialization.body_function,
            specialization.body_signature,
            device=True,
            abi="c",
            output="ltoir",
            abi_info={"abi_name": symbol},
        )
        external = cuda.declare_device(
            symbol, specialization.body_signature, abi="c", link=cuda.LTOIR(native)
        )

        def captures(fn):
            raise RuntimeError("device-only capture extraction")

        @extending.type_callable(captures)
        def type_captures(context):
            return lambda fn: types.Tuple(specialization.capture_types)

        @lowering_registry.lower(captures, fn)
        def lower_captures(ctx, target, args, kwargs):
            model = ctx.context.data_model_manager.lookup(fn)
            ctx.store_var(target, model.unpack_captures(ctx, ctx.load_var(args[0])))

        extending.refresh_registries(typing=False, target=True)

        if nested:

            def impl(fn, value):
                environment = captures(fn)
                return external(environment[0], value)
        else:

            def impl(fn, value):
                environment = captures(fn)
                return external(environment[0], environment[1], value)

        return impl

    @cuda.jit
    def kernel(bias, value, output):
        if nested:
            environment = (bias, transform, None)
            fn = lambda value: environment[1](value) + environment[0]
            output[0] = consume(fn, value)
        else:
            helper = transform
            fn = lambda value: helper(value) + bias
            output[0] = consume(fn, value)

    output = np.zeros(1, dtype=np.int64)
    for bias, value in ((7, 4), (-3, -2)):
        kernel[1, 1](bias, value, output)
        assert output[0] == value * 3 + bias


def test_external_numba_aggregate_return_captured_by_closure():
    pair_type = types.Tuple((types.int64, types.boolean))
    source = cuda.CUSource("""
struct Pair { long long value; signed char positive; };
extern "C" __device__ int closure_external_pair(Pair* out, long long value) {
    out->value = value * 3;
    out->positive = value >= 0;
    return 0;
}
""")
    pair = cuda.declare_device(
        "closure_external_pair", pair_type(types.int64), abi="numba", link=source
    )

    @cuda.jit(device=True, inline="never")
    def forward(fn, adjustment):
        return fn(adjustment)

    @cuda.jit
    def kernel(value, adjustment, output):
        environment = pair(value)
        fn = lambda delta: (environment[0] if environment[1] else -environment[0]) + delta
        output[0] = forward(fn, adjustment)

    output = np.zeros(1, dtype=np.int64)
    for value, adjustment in ((7, 2), (-3, -1)):
        kernel[1, 1](value, adjustment, output)
        assert output[0] == abs(value * 3) + adjustment
