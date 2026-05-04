# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda, extending, types, testing
from numba_cuda_mlir.lowering_registry import LoweringRegistry
from numba_cuda_mlir.numba_cuda.typing.templates import Registry
import numpy as np
import pytest


def test_extending_intrinsic():
    def codegen(builder, target, args, kwargs):
        from numba_cuda_mlir._mlir.dialects import cf, arith
        from numba_cuda_mlir._mlir.extras import types as T

        true = arith.constant(result=T.bool(), value=1)
        cf.assert_(true, "This should not be executed")
        builder.store_var(target, builder.load_var(args[0]))

    @extending.intrinsic
    def do_nothing(typingctx, x):
        return x(x), codegen

    @cuda.jit
    def k(x):
        x[0] = do_nothing(x[0])

    x = np.array([1])

    k[1, 1](x)
    assert x[0] == 1

    mlir = next(iter(k.inspect_mlir().values()))
    # CHECK: %true = arith.constant true
    # CHECK-NEXT: cf.assert %true, "This should not be executed"
    testing.filecheck_with_comments(mlir)


def test_overload_requires_typing_registry():
    def my_func(x):
        return x

    with pytest.raises(ValueError, match="overload.*typing_registry"):
        extending.overload(my_func)


def test_overload_method_requires_typing_registry():
    with pytest.raises(ValueError, match="overload_method.*typing_registry"):
        extending.overload_method(types.Array, "missing_registry")


def test_overload_attribute_requires_typing_registry():
    with pytest.raises(ValueError, match="overload_attribute.*typing_registry"):
        extending.overload_attribute(
            types.Array,
            "missing_typing_registry",
            lowering_registry=extending.lowering_registry,
        )


def test_overload_attribute_requires_lowering_registry():
    with pytest.raises(ValueError, match="overload_attribute.*lowering_registry"):
        extending.overload_attribute(
            types.Array,
            "missing_lowering_registry",
            typing_registry=extending.typing_registry,
        )


def test_register_jitable_requires_typing_registry():
    def helper(x):
        return x

    with pytest.raises(ValueError, match="register_jitable.*typing_registry"):
        extending.register_jitable(helper)


def test_register_jitable_with_options_requires_typing_registry():
    with pytest.raises(ValueError, match="register_jitable.*typing_registry"):
        extending.register_jitable(inline="always")


def test_overload_uses_supplied_typing_registry():
    custom_typing_registry = Registry()

    def my_func(x):
        return x

    @extending.overload(my_func, typing_registry=custom_typing_registry)
    def my_func_overload(x):
        def impl(x):
            return x

        return impl

    assert custom_typing_registry.functions[-1]._typing_registry is custom_typing_registry
    assert custom_typing_registry.globals[-1][0] is my_func


def test_overload_attribute_uses_supplied_registries():
    custom_typing_registry = Registry()
    custom_lowering_registry = LoweringRegistry()

    @extending.overload_attribute(
        types.Array,
        "custom_attr",
        typing_registry=custom_typing_registry,
        lowering_registry=custom_lowering_registry,
    )
    def array_custom_attr(arr):
        def get(arr):
            return arr.size

        return get

    template = custom_typing_registry.attributes[-1]
    assert template._attribute_lowering_registry is custom_lowering_registry


def test_extending_overload_with_lowering():
    from numba_cuda_mlir.extending import lowering_registry

    @lowering_registry.lower(np.sum, types.Any, types.Any, types.Any)
    def codegen(builder, target, args, kwargs):
        """
        TODO: we shouldn't need this...
        Numba should be able to type _and lower_ the overload as it is
        without defining a new lowering. Likely due to a misconfiguration in our
        lowering registry.
        """
        x, _, _ = builder.load_vars(args)
        builder.store_var(target, x)

    @extending.overload(np.sum, typing_registry=extending.typing_registry)
    def sum_overload(a, b, c):
        """
        Silly overload of np.sum that takes three arguments and returns the first.
        """

        def ol(a, b, c):
            return a

        return ol

    @cuda.jit
    def k(x):
        x[0] = np.sum(x, x, x)[2]

    input = x = np.array([1, 2, 3])
    k[1, 1](x)
    assert x[0] == 3


def test_extending_overload_without_lowering():
    import logging

    logging.basicConfig(level=logging.DEBUG)

    @extending.overload(np.sum, typing_registry=extending.typing_registry)
    def sum_overload(a, c):
        def ol(a, c):
            return a

        return ol

    @cuda.jit
    def k(x):
        np.sum(x, x)

    x = np.array([1, 2, 3])
    k[1, 1](x)


def test_extending_overload_method():
    """User-defined @overload_method dispatches through BoundFunction."""

    @extending.overload_method(
        types.Array,
        "doubled_first",
        typing_registry=extending.typing_registry,
    )
    def array_doubled_first(arr):
        def impl(arr):
            return arr[0] * 2

        return impl

    @cuda.jit
    def kernel(arr, out):
        out[0] = arr.doubled_first()

    arr = np.array([21.0], dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](arr, out)
    assert out[0] == 42.0


def test_register_jitable():
    """register_jitable makes a plain Python function callable from device code."""

    @extending.register_jitable(typing_registry=extending.typing_registry)
    def triple(x):
        return x * 3

    @cuda.jit
    def kernel(arr):
        arr[0] = triple(arr[0])

    arr = np.array([7], dtype=np.int64)
    kernel[1, 1](arr)
    assert arr[0] == 21


def test_register_jitable_calls_register_jitable():
    """Chained register_jitable: one jitable function calls another."""

    @extending.register_jitable(typing_registry=extending.typing_registry)
    def add_one(x):
        return x + 1

    @extending.register_jitable(typing_registry=extending.typing_registry)
    def add_two(x):
        return add_one(add_one(x))

    @cuda.jit
    def kernel(arr):
        arr[0] = add_two(arr[0])

    arr = np.array([10], dtype=np.int64)
    kernel[1, 1](arr)
    assert arr[0] == 12


def test_overload_attribute():
    """overload_attribute exposes a read-only property on a Numba type."""

    @extending.overload_attribute(
        types.Array,
        "doubled_size",
        typing_registry=extending.typing_registry,
        lowering_registry=extending.lowering_registry,
    )
    def array_doubled_size(arr):
        def get(arr):
            return arr.size * 2

        return get

    @cuda.jit
    def kernel(arr, out):
        out[0] = arr.doubled_size

    arr = np.zeros(10, dtype=np.float64)
    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](arr, out)
    assert out[0] == 20


def test_overload_method_with_args():
    """overload_method with arguments beyond self."""

    @extending.overload_method(
        types.Array,
        "elem_plus",
        typing_registry=extending.typing_registry,
    )
    def array_elem_plus(arr, idx, val):
        def impl(arr, idx, val):
            return arr[idx] + val

        return impl

    @cuda.jit
    def kernel(arr, out):
        out[0] = arr.elem_plus(1, 100)

    arr = np.array([10, 20, 30], dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](arr, out)
    assert out[0] == 120


def test_overload_dispatches_on_type():
    """overload can return different implementations based on argument types."""

    def my_func(x):
        raise NotImplementedError

    @extending.overload(my_func, typing_registry=extending.typing_registry)
    def my_func_overload(x):
        if isinstance(x, types.Integer):

            def impl(x):
                return x + 1

            return impl
        elif isinstance(x, types.Float):

            def impl(x):
                return x * 2.0

            return impl

    @cuda.jit
    def kernel(int_out, float_out):
        int_out[0] = my_func(int_out[0])
        float_out[0] = my_func(float_out[0])

    int_out = np.array([10], dtype=np.int64)
    float_out = np.array([3.0], dtype=np.float64)
    kernel[1, 1](int_out, float_out)
    assert int_out[0] == 11
    assert float_out[0] == 6.0


def test_struct_field_model():
    """UnicodeTypeModel exposes field names and positions for struct access."""
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir.models import UnicodeTypeModel

    with ir.Context(), ir.Location.unknown():
        from numba_cuda_mlir.descriptor import MLIRTargetContext, MLIRTypingContext

        tc = MLIRTypingContext()
        tc.refresh()
        ctx = MLIRTargetContext(tc, target="numba_cuda_mlir")
        ctx.refresh()

        model = ctx.data_model_manager.lookup(types.UnicodeType("unicode_type"))
        assert isinstance(model, UnicodeTypeModel)
        assert model._fields == (
            "data",
            "length",
            "kind",
            "is_ascii",
            "hash",
            "meminfo",
            "parent",
        )
        assert model.get_field_position("data") == 0
        assert model.get_field_position("length") == 1
        assert model.get_field_position("kind") == 2
        assert model.get_field_position("is_ascii") == 3
        assert model.get_field_position("hash") == 4
        assert model.get_field_position("meminfo") == 5
        assert model.get_field_position("parent") == 6


def test_aggregate_type_field_model():
    """AggregateTypeModel exposes field names and positions for struct access."""
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir.models import AggregateTypeModel
    from numba_cuda_mlir.type_defs.aggregate_types import AggregateType

    with ir.Context(), ir.Location.unknown():
        from numba_cuda_mlir.descriptor import MLIRTargetContext, MLIRTypingContext

        tc = MLIRTypingContext()
        tc.refresh()
        ctx = MLIRTargetContext(tc, target="numba_cuda_mlir")
        ctx.refresh()

        my_type = AggregateType(
            "TestPoint",
            [("x", types.float32), ("y", types.float64), ("z", types.int32)],
        )
        model = ctx.data_model_manager.lookup(my_type)
        assert isinstance(model, AggregateTypeModel)
        assert model._fields == ("x", "y", "z")
        assert model.get_field_position("x") == 0
        assert model.get_field_position("y") == 1
        assert model.get_field_position("z") == 2
