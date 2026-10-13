# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir import descriptor as descriptor_mod
from numba_cuda_mlir.models import StructModel, register_model
from numba_cuda_mlir.numba_cuda.extending import typeof_impl


class RuntimePairType(types.Type):
    def __init__(self):
        super().__init__(name="RuntimePair")


runtime_pair_type = RuntimePairType()


@register_model(RuntimePairType)
class RuntimePairModel(StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, (("left", types.int32), ("right", types.int32)))


class RuntimePair:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def prepare_args(self, ty, val, stream=None, retr=None):
        if val is not self:
            return ty, val
        assert ty == runtime_pair_type
        return ty, (np.int32(self.left), np.int32(self.right))


@typeof_impl.register(RuntimePair)
def typeof_runtime_pair(val, c):
    return runtime_pair_type


extending.make_attribute_wrapper(RuntimePairType, "left", "left")
extending.make_attribute_wrapper(RuntimePairType, "right", "right")
extending.refresh_registries()


@cuda.jit
def _runtime_pair_kernel(out, pair):
    out[0] = pair.left
    out[1] = pair.right


@pytest.mark.parametrize("left,right", [(3, 7), (-11, 29)])
def test_value_owned_struct_argument_matches_flattened_launcher_abi(left, right):
    out = np.zeros(2, dtype=np.int32)

    _runtime_pair_kernel[1, 1](out, RuntimePair(left, right))

    np.testing.assert_array_equal(out, np.array([left, right], dtype=np.int32))


def test_flattened_argument_disables_misaligned_constant_flags():
    @cuda.jit
    def kernel(out, pair, scalar):
        out[0] = pair.right + scalar

    native_compile_calls = []

    def compile_with_count(args):
        native_compile_calls.append(args)
        return kernel._compile(args)

    # The flags describe the three logical arguments. RuntimePair expands to
    # two native leaves, so indexing these flags against the four flattened
    # arguments would incorrectly mark pair.right, rather than scalar,
    # constant and create a second native cache entry when pair.right changes.
    kernel._c = descriptor_mod._cext.KernelDispatcher(
        compile_with_count,
        (False, False, True),
        descriptor_mod._ensure_numba_cuda_context,
    )

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out, RuntimePair(1, 2), np.int32(5))
    assert out[0] == 7

    kernel[1, 1](out, RuntimePair(1, 3), np.int32(5))
    assert out[0] == 8
    assert len(native_compile_calls) == 1


@pytest.mark.parametrize("struct_first", [True, False], ids=["struct-first", "scalar-first"])
def test_literal_scalar_beside_struct_preserves_native_dispatch(monkeypatch, struct_first):
    from numba_cuda_mlir import mlir_compiler
    from numba_cuda_mlir._whole_function_planners import _planner_registry
    from numba_cuda_mlir.cuda.experimental import consteval
    from numba_cuda_mlir.errors import ForceLiteralArg
    from numba_cuda_mlir.numba_cuda.types import literal

    # Discover the literal request inside the native compile callback, without
    # a registered planner causing Python-side precompilation before launch.
    monkeypatch.setattr(_planner_registry, "_planners", [])

    if struct_first:

        @cuda.jit(experimental_ast_transforms=True)
        def kernel(out, pair, selector):
            if cuda.threadIdx.x == 0:
                if consteval(isinstance(selector, types.Literal)):
                    value = consteval(
                        selector.literal_value if isinstance(selector, types.Literal) else 0
                    )
                else:
                    value = selector
                bias = consteval(1000 if isinstance(selector, types.Boolean) else 0)
                out[0] = pair.left * 10 + pair.right + value * 100 + bias

        scalar_position = 2
    else:

        @cuda.jit(experimental_ast_transforms=True)
        def kernel(out, selector, pair):
            if cuda.threadIdx.x == 0:
                if consteval(isinstance(selector, types.Literal)):
                    value = consteval(
                        selector.literal_value if isinstance(selector, types.Literal) else 0
                    )
                else:
                    value = selector
                bias = consteval(1000 if isinstance(selector, types.Boolean) else 0)
                out[0] = pair.left * 10 + pair.right + value * 100 + bias

        scalar_position = 1

    compiler_types = []
    native_compile_calls = []
    original_compiler_entry = mlir_compiler.mlir_compiler_entry
    original_compile = kernel._compile

    def request_scalar_literal(pyfunc, func_args, targetoptions, override_argtypes):
        assert native_compile_calls, "first discovery must enter through the native callback"
        selector_type = override_argtypes[scalar_position]
        compiler_types.append(selector_type)
        if isinstance(selector_type, (types.Integer, types.Boolean)) and not isinstance(
            selector_type, types.Literal
        ):
            raise ForceLiteralArg({scalar_position})
        return original_compiler_entry(
            pyfunc, func_args, targetoptions, override_argtypes=override_argtypes
        )

    def counting_compile(args):
        native_compile_calls.append(tuple(args))
        return original_compile(args)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", request_scalar_literal)
    monkeypatch.setattr(kernel, "_compile", counting_compile)
    kernel._c = kernel._new_kernel_dispatcher()
    retained_launcher = kernel[1, 1]
    out = np.zeros(1, dtype=np.float64)

    def launch(launcher, selector, seed):
        pair = RuntimePair(seed, seed + 3)
        args = (out, pair, selector) if struct_first else (out, selector, pair)
        launcher(*args)
        bias = 1000 if isinstance(selector, bool) else 0
        assert out[0] == pair.left * 10 + pair.right + selector * 100 + bias

    for index, selector in enumerate((1, 4, 1, True, 1, True, 1.5, 2.5, 4)):
        launcher = retained_launcher if index % 2 == 0 else kernel[1, 2]
        launch(launcher, selector, index)

    fresh_launcher = kernel[1, 2]
    assert fresh_launcher is not retained_launcher
    assert kernel._literal_arg_positions == frozenset({scalar_position})
    assert not isinstance(compiler_types[0], types.Literal)
    assert compiler_types[1:] == [
        literal(1),
        literal(4),
        literal(True),
        types.float64,
    ]
    assert {argtypes[scalar_position] for argtypes in kernel.overloads} == {
        literal(1),
        literal(4),
        literal(True),
        types.float64,
    }
    assert all(len(args) == 4 for args in native_compile_calls)

    # Changes to runtime struct fields or generic float values must reuse each
    # warmed native specialization through both retained and fresh launchers.
    compile_count = len(native_compile_calls)
    for launcher in (retained_launcher, fresh_launcher):
        for index, selector in enumerate((1, 4, True, 3.5)):
            launch(launcher, selector, index + 20)
    assert len(native_compile_calls) == compile_count
