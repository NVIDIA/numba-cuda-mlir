# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir.testing import filecheck, filecheck_with_comments


def test_vector_load_store_1d():
    @cuda.jit
    def kernel(arr_in, arr_out):
        i = cuda.threadIdx.x * 4
        vec = cuda.vector.load(arr_in, i, 4)
        cuda.vector.store(arr_out, i, vec)

    arr_in = np.arange(32, dtype=np.float32)
    arr_out = np.zeros(32, dtype=np.float32)
    kernel[1, 8](arr_in, arr_out)
    np.testing.assert_array_equal(arr_in, arr_out)

    (mlir,) = kernel.inspect_mlir().values()
    # CHECK:             %{{.*}} = vector.transfer_read %{{.*}}{{\[}}%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<?xf32, strided<[?], offset: ?>>, vector<4xf32>
    # CHECK:             vector.transfer_write %{{.*}}, %{{.*}}{{\[}}%{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<?xf32, strided<[?], offset: ?>>
    filecheck_with_comments(mlir)


def test_vector_load_store_1d_float16_unaligned():
    @cuda.jit
    def kernel(arr_in, arr_out):
        i = cuda.threadIdx.x * 4
        vec = cuda.vector.load(arr_in, i, 4)
        cuda.vector.store(arr_out, i, vec)

    arr_in = np.arange(32, dtype=np.float16)
    arr_out = np.zeros(32, dtype=np.float16)
    kernel[1, 8](arr_in, arr_out)
    np.testing.assert_array_equal(arr_in, arr_out)


def test_vector_load_store_1d_bool_unaligned():
    @cuda.jit
    def kernel(arr_in, arr_out):
        i = cuda.threadIdx.x * 4
        vec = cuda.vector.load(arr_in, i, 4)
        cuda.vector.store(arr_out, i, vec)

    arr_in = np.array(
        [True, False, True, True, False, False, True, False],
        dtype=np.bool_,
    )
    arr_out = np.zeros(8, dtype=np.bool_)
    kernel[1, 2](arr_in, arr_out)
    np.testing.assert_array_equal(arr_in, arr_out)


def test_vector_load_store_aligned():
    """Test aligned vector load/store generates vectorized PTX."""

    @cuda.jit(dump_ptx=True)
    def kernel(arr_in, arr_out):
        i = cuda.threadIdx.x
        if i < 8:
            idx = i * 4
            # 4 x fp16 = 8 bytes alignment
            alignment = 8
            vec = cuda.vector.load(arr_in, idx, 4, alignment=alignment)
            cuda.vector.store(arr_out, idx, vec, alignment)

    arr_in = np.arange(32, dtype=np.float16)
    arr_out = np.zeros(32, dtype=np.float16)
    kernel[1, 32](arr_in, arr_out)
    np.testing.assert_array_equal(arr_in, arr_out)

    # Check PTX contains vectorized load/store instructions
    (ptx,) = kernel.inspect_ptx().values()
    filecheck(
        r"""
        CHECK: ld.global.{{(v[24]\.([bf][0-9]+|u16|u32)|b32)}}
        CHECK: st.global.v{{[24]}}.{{([bf][0-9]+|u16|u32)}}
        """,
        ptx,
    )


def test_vector_load_store_2d_array():
    @cuda.jit
    def kernel(arr_in, arr_out):
        row = cuda.threadIdx.x
        vec = cuda.vector.load(arr_in, (row, 0), 4)
        cuda.vector.store(arr_out, (row, 0), vec)

    arr_in = np.arange(32, dtype=np.float32).reshape(8, 4)
    arr_out = np.zeros((8, 4), dtype=np.float32)
    kernel[1, 8](arr_in, arr_out)
    np.testing.assert_array_equal(arr_in, arr_out)


def test_vector_2d_shape():
    """Test multi-dimensional vector (4x4 = 16 elements)."""

    @cuda.jit
    def kernel(arr_in, arr_out):
        row = cuda.threadIdx.x
        vec = cuda.vector.load(arr_in, (row * 4, 0), (4, 4))
        cuda.vector.store(arr_out, (row * 4, 0), vec)

    arr_in = np.arange(64, dtype=np.float32).reshape(16, 4)
    arr_out = np.zeros((16, 4), dtype=np.float32)
    kernel[1, 4](arr_in, arr_out)
    np.testing.assert_array_equal(arr_in, arr_out)


def test_vector_int32():
    """Test vector with int32 elements."""

    @cuda.jit
    def kernel(arr_in, arr_out):
        i = cuda.threadIdx.x * 4
        vec = cuda.vector.load(arr_in, i, 4)
        cuda.vector.store(arr_out, i, vec)

    arr_in = np.arange(32, dtype=np.int32)
    arr_out = np.zeros(32, dtype=np.int32)
    kernel[1, 8](arr_in, arr_out)
    np.testing.assert_array_equal(arr_in, arr_out)


def test_vector_float64():
    """Test vector with float64 elements."""

    @cuda.jit
    def kernel(arr_in, arr_out):
        i = cuda.threadIdx.x * 4
        vec = cuda.vector.load(arr_in, i, 4)
        cuda.vector.store(arr_out, i, vec)

    arr_in = np.arange(32, dtype=np.float64)
    arr_out = np.zeros(32, dtype=np.float64)
    kernel[1, 8](arr_in, arr_out)
    np.testing.assert_array_equal(arr_in, arr_out)


def test_cuda_vector_float64x4_basic():
    """Test basic float64x4 construction and attribute access."""

    @cuda.jit
    def kernel(arr):
        v = cuda.float64x4(1.0, 3.0, 5.0, 7.0)
        arr[0] = v.x
        arr[1] = v.y
        arr[2] = v.z
        arr[3] = v.w

    arr = np.zeros(4, dtype=np.float64)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1.0, 3.0, 5.0, 7.0])


def test_cuda_vector_float32x4_basic():
    """Test basic float32x4 construction and attribute access."""

    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(2.0, 4.0, 6.0, 8.0)
        arr[0] = v.x
        arr[1] = v.y
        arr[2] = v.z
        arr[3] = v.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [2.0, 4.0, 6.0, 8.0])


def test_cuda_vector_constructor_from_multidimensional_vector():
    @cuda.jit
    def kernel(arr_in, arr_out):
        vec = cuda.vector.load(arr_in, (0, 0), (2, 2))
        constructed = cuda.float32x4(vec)
        arr_out[0] = constructed.x
        arr_out[1] = constructed.y
        arr_out[2] = constructed.z
        arr_out[3] = constructed.w

    arr_in = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr_out = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr_in, arr_out)
    np.testing.assert_allclose(arr_out, [1.0, 2.0, 3.0, 4.0])


def test_cuda_vector_int32x4_basic():
    """Test basic int32x4 construction and attribute access."""

    @cuda.jit
    def kernel(arr):
        v = cuda.int32x4(10, 20, 30, 40)
        arr[0] = v.x
        arr[1] = v.y
        arr[2] = v.z
        arr[3] = v.w

    arr = np.zeros(4, dtype=np.int32)
    kernel[1, 1](arr)
    np.testing.assert_array_equal(arr, [10, 20, 30, 40])


def test_cuda_vector_float64x2():
    """Test float64x2 (2-element vector)."""

    @cuda.jit
    def kernel(arr):
        v = cuda.float64x2(1.5, 2.5)
        arr[0] = v.x
        arr[1] = v.y

    arr = np.zeros(2, dtype=np.float64)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1.5, 2.5])


def test_cuda_vector_float32x3():
    """Test float32x3 (3-element vector)."""

    @cuda.jit
    def kernel(arr):
        v = cuda.float32x3(1.0, 2.0, 3.0)
        arr[0] = v.x
        arr[1] = v.y
        arr[2] = v.z

    arr = np.zeros(3, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


def test_cuda_vector_alias_double4():
    """Test C-style alias double4 -> float64x4."""

    @cuda.jit
    def kernel(arr):
        v = cuda.double4(1.0, 2.0, 3.0, 4.0)
        arr[0] = v.x
        arr[1] = v.y
        arr[2] = v.z
        arr[3] = v.w

    arr = np.zeros(4, dtype=np.float64)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0, 4.0])


def test_cuda_vector_alias_float4():
    """Test C-style alias float4 -> float32x4."""

    @cuda.jit
    def kernel(arr):
        v = cuda.float4(5.0, 6.0, 7.0, 8.0)
        arr[0] = v.x
        arr[1] = v.y
        arr[2] = v.z
        arr[3] = v.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [5.0, 6.0, 7.0, 8.0])


def test_cuda_vector_alias_half2():
    """Test C-style alias half2 -> float16x2."""

    @cuda.jit
    def kernel(arr):
        v = cuda.half2(1.5, 2.5)
        arr[0] = v.x
        arr[1] = v.y

    arr = np.zeros(2, dtype=np.float16)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1.5, 2.5])


def test_cuda_vector_alias_int2():
    """Test C-style alias int2 -> int32x2."""

    @cuda.jit
    def kernel(arr):
        v = cuda.int2(100, 200)
        arr[0] = v.x
        arr[1] = v.y

    arr = np.zeros(2, dtype=np.int32)
    kernel[1, 1](arr)
    np.testing.assert_array_equal(arr, [100, 200])


def test_cuda_vector_alias_short2():
    """Test C-style alias short2 -> int16x2."""

    @cuda.jit
    def kernel(arr):
        v = cuda.short2(10, 11)
        arr[0] = v.x
        arr[1] = v.y

    arr = np.zeros(2, dtype=np.int16)
    kernel[1, 1](arr)
    np.testing.assert_array_equal(arr, [10, 11])


def test_cuda_vector_closure_capture():
    """Test vector type used as closure variable."""
    vobj = cuda.float64x4

    @cuda.jit
    def kernel(arr):
        v = vobj(1.0, 2.0, 3.0, 4.0)
        arr[0] = v.x
        arr[1] = v.y
        arr[2] = v.z
        arr[3] = v.w

    arr = np.zeros(4, dtype=np.float64)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "vec_type,num_elements,dtype",
    [
        (cuda.float16x1, 1, np.float16),
        (cuda.float16x2, 2, np.float16),
        (cuda.float16x3, 3, np.float16),
        (cuda.float16x4, 4, np.float16),
        (cuda.float32x1, 1, np.float32),
        (cuda.float32x2, 2, np.float32),
        (cuda.float32x3, 3, np.float32),
        (cuda.float32x4, 4, np.float32),
        (cuda.float64x1, 1, np.float64),
        (cuda.float64x2, 2, np.float64),
        (cuda.float64x3, 3, np.float64),
        (cuda.float64x4, 4, np.float64),
        (cuda.int32x1, 1, np.int32),
        (cuda.int32x2, 2, np.int32),
        (cuda.int32x3, 3, np.int32),
        (cuda.int32x4, 4, np.int32),
    ],
)
def test_cuda_vector_type_variants(vec_type, num_elements, dtype):
    """Parametrized test for various vector type variants."""
    attrs = ["x", "y", "z", "w"][:num_elements]
    values = list(range(num_elements))

    # Dynamically build kernel based on num_elements
    if num_elements == 1:

        @cuda.jit
        def kernel(arr):
            v = vec_type(dtype(0))
            arr[0] = v.x

    elif num_elements == 2:

        @cuda.jit
        def kernel(arr):
            v = vec_type(dtype(0), dtype(1))
            arr[0] = v.x
            arr[1] = v.y

    elif num_elements == 3:

        @cuda.jit
        def kernel(arr):
            v = vec_type(dtype(0), dtype(1), dtype(2))
            arr[0] = v.x
            arr[1] = v.y
            arr[2] = v.z

    elif num_elements == 4:

        @cuda.jit
        def kernel(arr):
            v = vec_type(dtype(0), dtype(1), dtype(2), dtype(3))
            arr[0] = v.x
            arr[1] = v.y
            arr[2] = v.z
            arr[3] = v.w

    arr = np.zeros(num_elements, dtype=dtype)
    kernel[1, 1](arr)
    np.testing.assert_array_equal(arr, values)


def test_cuda_vector_fancy_creation():
    """Test fancy vector creation from combinations of scalars and vectors."""

    @cuda.jit
    def kernel(res):
        one = 1.0
        two = 2.0
        three = 3.0
        four = 4.0

        j = 0

        # 1-element vector from scalar
        f1_1 = cuda.float64x1(one)
        # 1-element vector from another 1-element vector (copy)
        f1_2 = cuda.float64x1(f1_1)

        res[j] = f1_1.x
        res[j + 1] = f1_2.x
        j += 2

        # 2-element vectors from various combinations
        f2_1 = cuda.float64x2(two, three)  # two scalars
        f2_2 = cuda.float64x2(f1_1, three)  # vec1 + scalar
        f2_3 = cuda.float64x2(two, f1_1)  # scalar + vec1
        f2_4 = cuda.float64x2(f1_1, f1_1)  # vec1 + vec1
        f2_5 = cuda.float64x2(f2_1)  # copy from vec2

        res[j] = f2_1.x
        res[j + 1] = f2_1.y
        j += 2
        res[j] = f2_2.x
        res[j + 1] = f2_2.y
        j += 2
        res[j] = f2_3.x
        res[j + 1] = f2_3.y
        j += 2
        res[j] = f2_4.x
        res[j + 1] = f2_4.y
        j += 2
        res[j] = f2_5.x
        res[j + 1] = f2_5.y
        j += 2

        # 3-element vectors from various combinations
        f3_1 = cuda.float64x3(one, two, three)  # three scalars
        f3_2 = cuda.float64x3(f2_1, one)  # vec2 + scalar
        f3_3 = cuda.float64x3(one, f2_1)  # scalar + vec2
        f3_4 = cuda.float64x3(f1_1, f1_1, f1_1)  # vec1 + vec1 + vec1

        res[j] = f3_1.x
        res[j + 1] = f3_1.y
        res[j + 2] = f3_1.z
        j += 3
        res[j] = f3_2.x
        res[j + 1] = f3_2.y
        res[j + 2] = f3_2.z
        j += 3
        res[j] = f3_3.x
        res[j + 1] = f3_3.y
        res[j + 2] = f3_3.z
        j += 3
        res[j] = f3_4.x
        res[j + 1] = f3_4.y
        res[j + 2] = f3_4.z
        j += 3

        # 4-element vectors from various combinations
        f4_1 = cuda.float64x4(one, two, three, four)  # four scalars
        f4_2 = cuda.float64x4(f2_1, f2_1)  # vec2 + vec2
        f4_3 = cuda.float64x4(f2_1, three, four)  # vec2 + scalar + scalar
        f4_4 = cuda.float64x4(one, f2_1, four)  # scalar + vec2 + scalar
        f4_5 = cuda.float64x4(one, two, f2_1)  # scalar + scalar + vec2
        f4_6 = cuda.float64x4(f1_1, f1_1, f1_1, f1_1)  # four vec1s

        res[j] = f4_1.x
        res[j + 1] = f4_1.y
        res[j + 2] = f4_1.z
        res[j + 3] = f4_1.w
        j += 4
        res[j] = f4_2.x
        res[j + 1] = f4_2.y
        res[j + 2] = f4_2.z
        res[j + 3] = f4_2.w
        j += 4
        res[j] = f4_3.x
        res[j + 1] = f4_3.y
        res[j + 2] = f4_3.z
        res[j + 3] = f4_3.w
        j += 4
        res[j] = f4_4.x
        res[j + 1] = f4_4.y
        res[j + 2] = f4_4.z
        res[j + 3] = f4_4.w
        j += 4
        res[j] = f4_5.x
        res[j + 1] = f4_5.y
        res[j + 2] = f4_5.z
        res[j + 3] = f4_5.w
        j += 4
        res[j] = f4_6.x
        res[j + 1] = f4_6.y
        res[j + 2] = f4_6.z
        res[j + 3] = f4_6.w

    # Total elements: 2 + 10 + 12 + 24 = 48
    res = np.zeros(48, dtype=np.float64)
    kernel[1, 1](res)

    expected = [
        # f1_1, f1_2
        1.0,
        1.0,
        # f2_1 through f2_5
        2.0,
        3.0,  # f2_1: (2, 3)
        1.0,
        3.0,  # f2_2: (f1_1=1, 3)
        2.0,
        1.0,  # f2_3: (2, f1_1=1)
        1.0,
        1.0,  # f2_4: (f1_1=1, f1_1=1)
        2.0,
        3.0,  # f2_5: copy of f2_1
        # f3_1 through f3_4
        1.0,
        2.0,
        3.0,  # f3_1: (1, 2, 3)
        2.0,
        3.0,
        1.0,  # f3_2: (f2_1=(2,3), 1)
        1.0,
        2.0,
        3.0,  # f3_3: (1, f2_1=(2,3))
        1.0,
        1.0,
        1.0,  # f3_4: (f1_1, f1_1, f1_1)
        # f4_1 through f4_6
        1.0,
        2.0,
        3.0,
        4.0,  # f4_1: (1, 2, 3, 4)
        2.0,
        3.0,
        2.0,
        3.0,  # f4_2: (f2_1, f2_1)
        2.0,
        3.0,
        3.0,
        4.0,  # f4_3: (f2_1, 3, 4)
        1.0,
        2.0,
        3.0,
        4.0,  # f4_4: (1, f2_1, 4)
        1.0,
        2.0,
        2.0,
        3.0,  # f4_5: (1, 2, f2_1)
        1.0,
        1.0,
        1.0,
        1.0,  # f4_6: (f1_1, f1_1, f1_1, f1_1)
    ]
    np.testing.assert_allclose(res, expected)
