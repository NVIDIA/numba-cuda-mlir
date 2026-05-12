import numpy as np
import pytest
from numba_cuda_mlir import cuda

vector_types_to_test = [
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
