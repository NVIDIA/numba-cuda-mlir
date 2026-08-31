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
