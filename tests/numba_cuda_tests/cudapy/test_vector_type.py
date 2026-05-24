# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
CUDA vector type tests. Note that this test file imports
`cuda.vector_type` module to programmatically test all the
vector types. However, `vector_type` module is internal
and should not be imported by user, user should only import the
corresponding vector type from `cuda` module in kernel to use them.
"""

import sys

import numpy as np
import pytest

from numba_cuda_mlir.testing import NumbaCUDATestCase

import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types

from numba_cuda_mlir.cuda.vector_types import _vector_types as _cuda_vector_types


class _VectorTypeTestInfo:
    def __init__(self, vec_type):
        self.name = vec_type.name
        self.base_type = vec_type.dtype
        self.num_elements = vec_type.length
        self.user_facing_object = vec_type


def _vector_types():
    return [_VectorTypeTestInfo(vec_type) for vec_type in _cuda_vector_types]


def make_kernel(vtype):
    """
    Returns a jit compiled kernel that constructs a vector types of
    the given type, using the exact number of primitive types to
    construct the vector type.
    """
    vobj = vtype.user_facing_object
    base_type = vtype.base_type

    def kernel_1elem(res):
        v = vobj(base_type(0))
        res[0] = v.x

    def kernel_2elem(res):
        v = vobj(base_type(0), base_type(1))
        res[0] = v.x
        res[1] = v.y

    def kernel_3elem(res):
        v = vobj(base_type(0), base_type(1), base_type(2))
        res[0] = v.x
        res[1] = v.y
        res[2] = v.z

    def kernel_4elem(res):
        v = vobj(base_type(0), base_type(1), base_type(2), base_type(3))
        res[0] = v.x
        res[1] = v.y
        res[2] = v.z
        res[3] = v.w

    host_function = {
        1: kernel_1elem,
        2: kernel_2elem,
        3: kernel_3elem,
        4: kernel_4elem,
    }[vtype.num_elements]
    return numba_cuda_mlir.cuda.jit(host_function)


def make_fancy_creation_kernel(vtype):
    """
    Returns a jit compiled kernel that constructs a vector type using the
    "fancy" construction, that is, with arbitrary combinations of primitive
    types and vector types, as long as the total element of the construction
    is the same as the number of elements of the vector type.
    """
    base_type = vtype.base_type
    v1 = getattr(cuda, f"{vtype.name[:-1]}1")
    v2 = getattr(cuda, f"{vtype.name[:-1]}2")
    v3 = getattr(cuda, f"{vtype.name[:-1]}3")
    v4 = getattr(cuda, f"{vtype.name[:-1]}4")

    def kernel(res):
        one = base_type(1.0)
        two = base_type(2.0)
        three = base_type(3.0)
        four = base_type(4.0)

        j = 0  # index of the result array

        # Construct a 1-component vector type, possible combination includes:
        # 2C1 = 2 combinations.

        f1_1 = v1(one)  # 1
        f1_2 = v1(f1_1)  # 1

        res[0] = f1_1.x
        res[1] = f1_2.x
        j += 2

        # Construct a 2-component vector type, possible combination includes:
        # 1 + 2C1 * 2 = 5 combinations

        f2_1 = v2(two, three)  # 2 3
        f2_2 = v2(f1_1, three)  # 1 3
        f2_3 = v2(two, f1_1)  # 2 1
        f2_4 = v2(f1_1, f1_1)  # 1 1
        f2_5 = v2(f2_1)  # 2 3

        for v in (f2_1, f2_2, f2_3, f2_4, f2_5):
            res[j] = v.x
            res[j + 1] = v.y
            j += 2

        # Construct a 3-component vector type, possible combination includes:
        # 1 + 2C1 * 2 + 2^3 = 13 combinations

        f3_1 = v3(f2_1, one)  # 2 3 1
        f3_2 = v3(f2_1, f1_1)  # 2 3 1
        f3_3 = v3(one, f2_1)  # 1 2 3
        f3_4 = v3(f1_1, f2_1)  # 1 2 3

        f3_5 = v3(one, two, three)  # 1 2 3
        f3_6 = v3(f1_1, two, three)  # 1 2 3
        f3_7 = v3(one, f1_1, three)  # 1 1 3
        f3_8 = v3(one, two, f1_1)  # 1 2 1
        f3_9 = v3(f1_1, f1_1, three)  # 1 1 3
        f3_10 = v3(one, f1_1, f1_1)  # 1 1 1
        f3_11 = v3(f1_1, two, f1_1)  # 1 2 1
        f3_12 = v3(f1_1, f1_1, f1_1)  # 1 1 1

        f3_13 = v3(f3_1)  # 2 3 1

        for v in (
            f3_1,
            f3_2,
            f3_3,
            f3_4,
            f3_5,
            f3_6,
            f3_7,
            f3_8,
            f3_9,
            f3_10,
            f3_11,
            f3_12,
            f3_13,
        ):
            res[j] = v.x
            res[j + 1] = v.y
            res[j + 2] = v.z
            j += 3

        # Construct a 4-component vector type, possible combination includes:
        # 1 + (2C1 * 2 + 1) + 3C1 * 2^2 + 2^4 = 34 combinations

        f4_1 = v4(one, two, three, four)  # 1 2 3 4
        f4_2 = v4(f1_1, two, three, four)  # 1 2 3 4
        f4_3 = v4(one, f1_1, three, four)  # 1 1 3 4
        f4_4 = v4(one, two, f1_1, four)  # 1 2 1 4
        f4_5 = v4(one, two, three, f1_1)  # 1 2 3 1
        f4_6 = v4(f1_1, f1_1, three, four)  # 1 1 3 4
        f4_7 = v4(f1_1, two, f1_1, four)  # 1 2 1 4
        f4_8 = v4(f1_1, two, three, f1_1)  # 1 2 3 1
        f4_9 = v4(one, f1_1, f1_1, four)  # 1 1 1 4
        f4_10 = v4(one, f1_1, three, f1_1)  # 1 1 3 1
        f4_11 = v4(one, two, f1_1, f1_1)  # 1 2 1 1
        f4_12 = v4(f1_1, f1_1, f1_1, four)  # 1 1 1 4
        f4_13 = v4(f1_1, f1_1, three, f1_1)  # 1 1 3 1
        f4_14 = v4(f1_1, two, f1_1, f1_1)  # 1 2 1 1
        f4_15 = v4(one, f1_1, f1_1, f1_1)  # 1 1 1 1
        f4_16 = v4(f1_1, f1_1, f1_1, f1_1)  # 1 1 1 1

        f4_17 = v4(f2_1, two, three)  # 2 3 2 3
        f4_18 = v4(f2_1, f1_1, three)  # 2 3 1 3
        f4_19 = v4(f2_1, two, f1_1)  # 2 3 2 1
        f4_20 = v4(f2_1, f1_1, f1_1)  # 2 3 1 1
        f4_21 = v4(one, f2_1, three)  # 1 2 3 3
        f4_22 = v4(f1_1, f2_1, three)  # 1 2 3 3
        f4_23 = v4(one, f2_1, f1_1)  # 1 2 3 1
        f4_24 = v4(f1_1, f2_1, f1_1)  # 1 2 3 1
        f4_25 = v4(one, four, f2_1)  # 1 4 2 3
        f4_26 = v4(f1_1, four, f2_1)  # 1 4 2 3
        f4_27 = v4(one, f1_1, f2_1)  # 1 1 2 3
        f4_28 = v4(f1_1, f1_1, f2_1)  # 1 1 2 3

        f4_29 = v4(f2_1, f2_1)  # 2 3 2 3
        f4_30 = v4(f3_1, four)  # 2 3 1 4
        f4_31 = v4(f3_1, f1_1)  # 2 3 1 1
        f4_32 = v4(four, f3_1)  # 4 2 3 1
        f4_33 = v4(f1_1, f3_1)  # 1 2 3 1

        f4_34 = v4(f4_1)  # 1 2 3 4

        for v in tuple(
            (
                f4_1,
                f4_2,
                f4_3,
                f4_4,
                f4_5,
                f4_6,
                f4_7,
                f4_8,
                f4_9,
                f4_10,
                f4_11,
                f4_12,
                f4_13,
                f4_14,
                f4_15,
                f4_16,
                f4_17,
                f4_18,
                f4_19,
                f4_20,
                f4_21,
                f4_22,
                f4_23,
                f4_24,
                f4_25,
                f4_26,
                f4_27,
                f4_28,
                f4_29,
                f4_30,
                f4_31,
                f4_32,
                f4_33,
                f4_34,
            )
        ):
            res[j] = v.x
            res[j + 1] = v.y
            res[j + 2] = v.z
            res[j + 3] = v.w
            j += 4

    return numba_cuda_mlir.cuda.jit(kernel)


class TestCudaVectorType(NumbaCUDATestCase):
    def test_basic(self):
        """Basic test that makes sure that vector type and aliases
        are available within the cuda module from both device and
        simulator mode. This is an important sanity check, since other
        tests below tests the vector type objects programmatically.
        """

        @numba_cuda_mlir.cuda.jit("void(float64[:])")
        def kernel(arr):
            v1 = cuda.float64x4(1.0, 3.0, 5.0, 7.0)
            v2 = cuda.short2(10, 11)
            arr[0] = v1.x
            arr[1] = v1.y
            arr[2] = v1.z
            arr[3] = v1.w
            arr[4] = v2.x
            arr[5] = v2.y

        res = np.zeros(6, dtype=np.float64)
        kernel[1, 1](res)
        self.assertTrue(np.allclose(res, [1.0, 3.0, 5.0, 7.0, 10, 11]))

    def test_creation_readout(self):
        for vty in _vector_types():
            arr = np.zeros((vty.num_elements,))
            kernel = make_kernel(vty)
            kernel[1, 1](arr)
            np.testing.assert_almost_equal(arr, np.array(range(vty.num_elements)))

    @pytest.mark.xfail(
        sys.platform == "win32",
        reason="Fancy vector construction currently only fails on Windows",
    )
    def test_fancy_creation_readout(self):
        for vty in _vector_types():
            kernel = make_fancy_creation_kernel(vty)

            expected = np.array(
                [
                    # 1-component vectors
                    1,
                    1,
                    # 2-component vectors
                    2,
                    3,
                    1,
                    3,
                    2,
                    1,
                    1,
                    1,
                    2,
                    3,
                    # 3-component vectors
                    2,
                    3,
                    1,
                    2,
                    3,
                    1,
                    1,
                    2,
                    3,
                    1,
                    2,
                    3,
                    1,
                    2,
                    3,
                    1,
                    2,
                    3,
                    1,
                    1,
                    3,
                    1,
                    2,
                    1,
                    1,
                    1,
                    3,
                    1,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    2,
                    3,
                    1,
                    # 4-component vectors
                    1,
                    2,
                    3,
                    4,
                    1,
                    2,
                    3,
                    4,
                    1,
                    1,
                    3,
                    4,
                    1,
                    2,
                    1,
                    4,
                    1,
                    2,
                    3,
                    1,
                    1,
                    1,
                    3,
                    4,
                    1,
                    2,
                    1,
                    4,
                    1,
                    2,
                    3,
                    1,
                    1,
                    1,
                    1,
                    4,
                    1,
                    1,
                    3,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    4,
                    1,
                    1,
                    3,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    1,
                    3,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    1,
                    1,
                    1,
                    2,
                    3,
                    3,
                    1,
                    2,
                    3,
                    3,
                    1,
                    2,
                    3,
                    1,
                    1,
                    2,
                    3,
                    1,
                    1,
                    4,
                    2,
                    3,
                    1,
                    4,
                    2,
                    3,
                    1,
                    1,
                    2,
                    3,
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    1,
                    4,
                    2,
                    3,
                    1,
                    1,
                    4,
                    2,
                    3,
                    1,
                    1,
                    2,
                    3,
                    1,
                    1,
                    2,
                    3,
                    4,
                ]
            )
            arr = np.zeros(expected.shape)
            kernel[1, 1](arr)
            np.testing.assert_almost_equal(arr, expected)

    def test_vector_type_alias(self):
        """Tests that `cuda.<vector_type.alias>` are importable and
        that is the same as `cuda.<vector_type.name>`.

        `test_fancy_creation_readout` only test vector types imported
        with its name. This test makes sure that construction with
        objects imported with alias should work the same.
        """
        from numba_cuda_mlir.cuda.vector_types import vector_types_by_alias

        for alias, vec_type in vector_types_by_alias.items():
            self.assertEqual(id(getattr(cuda, vec_type.name)), id(getattr(cuda, alias)))

    @pytest.mark.xfail(
        sys.platform == "win32",
        reason="Vector local array readback currently fails on Windows",
    )
    def test_vector_local_array(self):
        """Tests that vector types can be used as dtype for cuda.local.array"""

        @cuda.jit
        def kernel(out):
            arr = cuda.local.array(1, dtype=cuda.float32x4)
            arr[0] = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
            out[0] = arr[0].x
            out[1] = arr[0].y
            out[2] = arr[0].z
            out[3] = arr[0].w

        out = np.zeros(4, dtype=np.float32)
        kernel[1, 1](out)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    @pytest.mark.xfail(
        sys.platform == "win32",
        reason="Vector shared array readback currently fails on Windows",
    )
    def test_vector_shared_array(self):
        """Tests that vector types can be used as dtype for cuda.shared.array"""

        @cuda.jit
        def kernel(out):
            arr = cuda.shared.array(1, dtype=cuda.float32x4)
            arr[0] = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
            out[0] = arr[0].x
            out[1] = arr[0].y
            out[2] = arr[0].z
            out[3] = arr[0].w

        out = np.zeros(4, dtype=np.float32)
        kernel[1, 1](out)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
