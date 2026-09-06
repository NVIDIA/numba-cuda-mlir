# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir.numba_cuda.extending import typeof_impl


class CompileTimeValueType(types.Type):
    def __init__(self, value):
        self.value = value
        super().__init__(name=f"CompileTimeValue({value})")

    @property
    def key(self):
        return self.value


@register_model(CompileTimeValueType)
class CompileTimeValueModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntegerType.get_signless(32))


class CompileTimeValue:
    def __init__(self, value):
        self.value = value

    def prepare_args(self, ty, val, stream=None, retr=None):
        if val is not self:
            return ty, val
        return ty, np.int32(0)


@typeof_impl.register(CompileTimeValue)
def typeof_compile_time_value(val, c):
    return CompileTimeValueType(val.value)


@extending.overload_attribute(
    CompileTimeValueType,
    "constant",
    inline="always",
    typing_registry=extending.typing_registry,
    lowering_registry=extending.lowering_registry,
)
def compile_time_value_constant(value):
    constant = value.value

    def get(value):
        return constant

    return get


@pytest.mark.parametrize("first,second", [(2, 3), (3, 2)])
def test_value_owned_semantic_type_participates_in_native_dispatch(first, second):
    @cuda.jit
    def kernel(out, value):
        out[0] = value.constant

    out = np.zeros(1, dtype=np.int32)

    kernel[1, 1](out, CompileTimeValue(first))
    assert out[0] == first

    kernel[1, 1](out, CompileTimeValue(second))
    assert out[0] == second
    assert len(kernel.overloads) == 2


@pytest.mark.parametrize(
    "values",
    [
        (np.float32(2.5), 1.5),
        (1.5, np.float32(2.5)),
    ],
)
def test_native_dispatch_distinguishes_compiled_scalar_widths(values):
    @cuda.jit
    def kernel(out, value):
        out[0] = value

    out = np.zeros(1, dtype=np.float64)
    for value in values:
        kernel[1, 1](out, value)
        assert out[0] == value
    assert len(kernel.overloads) == 2
