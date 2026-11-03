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
