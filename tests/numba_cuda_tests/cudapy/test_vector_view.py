import numpy as np
import pytest
from numba_cuda_mlir import cuda

vector_types_to_test = [
    (cuda.float16x1, np.float16, 1),
    (cuda.float16x2, np.float16, 2),
    (cuda.float16x3, np.float16, 3),
    (cuda.float16x4, np.float16, 4),
    (cuda.float32x1, np.float32, 1),
    (cuda.float32x2, np.float32, 2),
    (cuda.float32x3, np.float32, 3),
    (cuda.float32x4, np.float32, 4),
    (cuda.float64x1, np.float64, 1),
    (cuda.float64x2, np.float64, 2),
    (cuda.float64x3, np.float64, 3),
    (cuda.float64x4, np.float64, 4),
    (cuda.int32x1, np.int32, 1),
    (cuda.int32x2, np.int32, 2),
    (cuda.int32x3, np.int32, 3),
    (cuda.int32x4, np.int32, 4),
    (cuda.int64x1, np.int64, 1),
    (cuda.int64x2, np.int64, 2),
    (cuda.int64x3, np.int64, 3),
    (cuda.int64x4, np.int64, 4),
]


@pytest.mark.parametrize("vec_type, scalar_type, vec_len", vector_types_to_test)
def test_scalar_to_vector_view(vec_type, scalar_type, vec_len):
    @cuda.jit
    def kernel(arr, out):
        arr_view = arr.view(vec_type)
        out[0] = arr_view[0].x
        if vec_len > 1:
            out[1] = arr_view[0].y
        if vec_len > 2:
            out[2] = arr_view[0].z
        if vec_len > 3:
            out[3] = arr_view[0].w

    arr = np.arange(vec_len, dtype=scalar_type)
    out = np.zeros(vec_len, dtype=scalar_type)
    kernel[1, 1](arr, out)
    np.testing.assert_allclose(out, arr)


@pytest.mark.parametrize("vec_type, scalar_type, vec_len", vector_types_to_test)
def test_vector_to_scalar_view(vec_type, scalar_type, vec_len):
    @cuda.jit
    def kernel_wrapper(arr, out):
        vec_arr = arr.view(vec_type)
        scalar_arr = vec_arr.view(scalar_type)
        for i in range(vec_len):
            out[i] = scalar_arr[i]

    arr = np.arange(vec_len, dtype=scalar_type)
    out = np.zeros(vec_len, dtype=scalar_type)
    kernel_wrapper[1, 1](arr, out)
    np.testing.assert_allclose(out, arr)


@pytest.mark.parametrize("vec_type, scalar_type, vec_len", vector_types_to_test)
def test_vector_view_len(vec_type, scalar_type, vec_len):
    @cuda.jit
    def kernel(arr, out_len):
        vec_arr = arr.view(vec_type)
        out_len[0] = len(vec_arr)

        scalar_arr = vec_arr.view(scalar_type)
        out_len[1] = len(scalar_arr)

    arr = np.arange(vec_len * 4, dtype=scalar_type)
    out_len = np.zeros(2, dtype=np.int64)
    kernel[1, 1](arr, out_len)

    assert out_len[0] == 4
    assert out_len[1] == vec_len * 4


def test_vector_to_vector_view():
    @cuda.jit
    def kernel(arr, out):
        # arr is float32 array of size 8
        vec4_arr = arr.view(cuda.float32x4)  # size 2
        vec2_arr = vec4_arr.view(cuda.float32x2)  # size 4

        # Read from vec2_arr
        out[0] = vec2_arr[0].x
        out[1] = vec2_arr[0].y
        out[2] = vec2_arr[1].x
        out[3] = vec2_arr[1].y
        out[4] = vec2_arr[2].x
        out[5] = vec2_arr[2].y
        out[6] = vec2_arr[3].x
        out[7] = vec2_arr[3].y

    arr = np.arange(8, dtype=np.float32)
    out = np.zeros(8, dtype=np.float32)
    kernel[1, 1](arr, out)
    np.testing.assert_allclose(out, arr)


def test_vector_to_vector_view_len():
    @cuda.jit
    def kernel(arr, out_len):
        # arr is float32 array of size 12
        vec4_arr = arr.view(cuda.float32x4)  # size 3
        vec2_arr = vec4_arr.view(cuda.float32x2)  # size 6
        vec3_arr = vec2_arr.view(cuda.float32x3)  # size 4

        out_len[0] = len(vec4_arr)
        out_len[1] = len(vec2_arr)
        out_len[2] = len(vec3_arr)

    arr = np.arange(12, dtype=np.float32)
    out_len = np.zeros(3, dtype=np.int64)
    kernel[1, 1](arr, out_len)

    assert out_len[0] == 3
    assert out_len[1] == 6
    assert out_len[2] == 4
