# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda, models, types
from numba_cuda_mlir.extending import overload, refresh_registries, typing_registry
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir.numba_cuda.extending import typeof_impl


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tuple_arg", [False, True])
@pytest.mark.parametrize("launch_config", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        (np.float32(1.5), np.float64(2.5)),
        (np.int32(7), np.int64(2**40 + 3)),
    ],
)
def test_exact_argument_types(values, reverse, tuple_arg, launch_config):
    class Handler:
        uses_launch_config = True

        def prepare_args(self, ty, val, **kwargs):
            return ty, val

    @cuda.jit(extensions=[Handler()] if launch_config else [])
    def kernel(out, value):
        if tuple_arg:
            out[0] = value[0]
        else:
            out[0] = value

    out = cuda.device_array(1, dtype=values[1].dtype)
    launch = kernel[1, 1]
    if reverse:
        values = values[::-1]
    for value in values + values[::-1]:
        launch(out, (value,) if tuple_arg else value)
        assert out.copy_to_host()[0] == value


def test_extension_type_identity_survives_packing_and_recompile():
    class TaggedType(types.Type):
        def __init__(self, tag):
            self.tag = tag
            super().__init__(name=f"TaggedType({tag})")

    class Value:
        def __init__(self, tag):
            self.tag = tag

    @typeof_impl.register(Value)
    def typeof_value(value, context):
        return TaggedType(value.tag)

    register_model(TaggedType)(models.CPointerModel)

    def tag_of(value):
        pass

    @overload(tag_of, inline="always", typing_registry=typing_registry)
    def overload_tag_of(value):
        tag = value.tag
        return lambda value: tag

    refresh_registries(include_uninitialized_cuda=False)

    class Handler:
        def prepare_args(self, ty, val, **kwargs):
            return (ty, 0) if isinstance(val, Value) else (ty, val)

    @cuda.jit(extensions=[Handler()])
    def kernel(out, value):
        out[0] = tag_of(value)

    out = cuda.device_array(1, dtype=np.int64)
    launch = kernel[1, 1]
    for tag in (1, 2, 1):
        launch(out, Value(tag))
        assert out.copy_to_host()[0] == tag
    kernel.recompile()
    for tag in (2, 1, 2):
        launch(out, Value(tag))
        assert out.copy_to_host()[0] == tag


def test_scalar_only_native_fast_path():
    @cuda.jit(debug=True, opt=False)
    def kernel(value):
        if value != 1.5:
            raise ValueError("wrong scalar specialization")

    launch = kernel[1, 1]
    launch(np.float32(1.5))
    launch(np.float64(1.5))


def test_record_layout_changes_with_same_python_class():
    @cuda.jit
    def kernel(out, value):
        out[0] = value.x

    out = cuda.device_array(1, dtype=np.int64)
    launch = kernel[1, 1]
    for dtype in (np.dtype([("x", "i8")]), np.dtype([("pad", "i8"), ("x", "i8")])):
        value = np.zeros(1, dtype=dtype)[0]
        value["x"] = 17
        launch(out, value)
        assert out.copy_to_host()[0] == 17
