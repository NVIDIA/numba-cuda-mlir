# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for argument handler extensions.

prepare_args transforms both types and values. Original types (from typeof() on
the original args) are used for compilation. Transformed types/values are used
at launch time for argument marshalling.
"""

import operator
import numpy as np
from numba_cuda_mlir._mlir.ir import IntegerType
import pytest

from numba_cuda_mlir.numba_cuda.datamodel.models import PrimitiveModel
from numba_cuda_mlir.numba_cuda.typing.templates import AbstractTemplate, signature
from numba_cuda_mlir.numba_cuda.extending import typeof_impl

from numba_cuda_mlir.errors import ExtensionError
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types


class Uint64Pointer(types.Type):
    def __init__(self):
        super().__init__(name="Uint64Pointer")


uint64_pointer = Uint64Pointer()


@register_model(Uint64Pointer)
class Uint64PointerMLIRModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = IntegerType.get_signless(64)
        super().__init__(dmm, fe_type, be_type)


class PointerWrapper:
    def __init__(self, value):
        self.ptr = value


@typeof_impl.register(PointerWrapper)
def typeof_pointer_wrapper(val, c):
    return uint64_pointer


class PointerHandler:
    """Extension handler that transforms PointerWrapper (type and value) to uint64."""

    def prepare_args(self, ty, val, stream=None, retr=None):
        if isinstance(val, PointerWrapper):
            assert ty == uint64_pointer
            return types.uint64, val.ptr
        return ty, val


def _register_uint64_pointer_setitem():
    from numba_cuda_mlir.extending import typing_registry, lowering_registry
    from numba_cuda_mlir import lowering_utilities
    from numba_cuda_mlir.mlir_lowering import MLIRLower
    from numba_cuda_mlir.mlir.dialect_exts import memref

    @typing_registry.register_global(operator.setitem)
    class Uint64PointerSetitemTemplate(AbstractTemplate):
        def generic(self, args, kws):
            if len(args) != 3:
                return None
            array, idx, value = args
            if (
                isinstance(array, types.Array)
                and array.dtype == types.uint64
                and isinstance(idx, types.Integer)
                and value is uint64_pointer
            ):
                return signature(types.none, array, idx, value)
            return None

    def lower_array_setitem_uint64_pointer(builder: MLIRLower, target, args, kwargs):
        array = builder.load_var(args[0])
        index = builder.load_var(args[1])
        index = lowering_utilities.index_of(index)
        value = builder.load_var(args[2])
        value = lowering_utilities.convert(value, array.type.element_type)
        memref.store(value=value, memref=array, indices=[index])

    lowering_registry.lower(operator.setitem, types.Array, types.Integer, Uint64Pointer)(
        lower_array_setitem_uint64_pointer
    )


_register_uint64_pointer_setitem()


def test_pointer_handler_transforms_pointerwrapper():
    ptr_value = 0x12345678
    my_wrapper = PointerWrapper(ptr_value)

    @cuda.jit(extensions=[PointerHandler()])
    def kernel(out, x):
        out[0] = x

    out = np.zeros(1, dtype=np.uint64)
    kernel[1, 1](out, my_wrapper)
    assert out[0] == my_wrapper.ptr


def test_multiple_extension_handlers():
    class ValueModifier:
        def __init__(self, offset):
            self.offset = offset

        def prepare_args(self, ty, val, stream=None, retr=None):
            if isinstance(ty, types.Integer) and isinstance(val, int):
                return ty, val + self.offset
            return ty, val

    @cuda.jit(extensions=[ValueModifier(10), ValueModifier(5)])
    def kernel(out, x):
        out[0] = x

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out, 0)

    assert out[0] == 15


def test_passthrough():
    class NoOpHandler:
        def prepare_args(self, ty, val, stream=None, retr=None):
            return ty, val

    @cuda.jit(extensions=[NoOpHandler()])
    def kernel(out, a, b):
        out[0] = a + b

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, 10.0, 20.0)

    assert out[0] == 30.0


def test_ordering():
    transformations = []

    class TrackedHandler:
        def __init__(self, name):
            self.name = name

        def prepare_args(self, ty, val, stream=None, retr=None):
            if isinstance(val, int):
                transformations.append(self.name)
            return ty, val

    @cuda.jit(extensions=[TrackedHandler("first"), TrackedHandler("second")])
    def kernel(out, x):
        out[0] = x

    out = np.zeros(1, dtype=np.int32)
    transformations.clear()
    kernel[1, 1](out, 42)

    assert transformations == ["second", "first"]
    assert out[0] == 42


def test_prepare_args_receives_argument_types():
    """Test that prepare_args is called with correct (ty, val) for each argument."""
    received_types = []

    class TypeTracker:
        def prepare_args(self, ty, val, stream=None, retr=None):
            received_types.append((type(val).__name__, ty))
            return ty, val

    @cuda.jit(extensions=[TypeTracker()])
    def kernel(out, x):
        out[0] = x

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, 1.0)

    assert any(name == "float" and isinstance(ty, types.Float) for name, ty in received_types)


def test_transforms_value():
    """Test that extension transforms the runtime value (and may transform the type)."""
    transformed_values = []

    class ValueTransformer:
        def prepare_args(self, ty, val, stream=None, retr=None):
            if isinstance(val, int) and val == 42:
                transformed_values.append(val)
                return ty, val * 2
            return ty, val

    @cuda.jit(extensions=[ValueTransformer()])
    def kernel(out, x):
        out[0] = x

    out = np.zeros(1, dtype=np.int32)
    transformed_values.clear()
    kernel[1, 1](out, 42)

    assert transformed_values == [42]
    assert out[0] == 84


def test_error_on_unsupported_type():
    class UnsupportedType:
        def __init__(self, value):
            self.value = value

    class UnsupportedTypeHandler:
        def prepare_args(self, ty, val, stream=None, retr=None):
            return ty, val

    @cuda.jit(extensions=[UnsupportedTypeHandler()])
    def kernel(out, x):
        out[0] = x

    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(ExtensionError):
        kernel[1, 1](out, UnsupportedType(10))


def test_transformed_value_not_cached_across_calls():
    class Doubler:
        def prepare_args(self, ty, val, stream=None, retr=None):
            if isinstance(val, int):
                return ty, val * 2
            return ty, val

    @cuda.jit(extensions=[Doubler()])
    def kernel(out, x):
        out[0] = x

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out, 3)
    assert out[0] == 6

    kernel[1, 1](out, 7)
    assert out[0] == 14


def test_prepare_args_exception_propagates():
    class BrokenHandler:
        def prepare_args(self, ty, val, stream=None, retr=None):
            raise RuntimeError("extension failure")

    @cuda.jit(extensions=[BrokenHandler()])
    def kernel(out, x):
        out[0] = x

    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(RuntimeError, match="extension failure"):
        kernel[1, 1](out, 1.0)
