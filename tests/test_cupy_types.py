# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda import types
import pytest
import numpy as np


@pytest.mark.parametrize(
    "dtype, numba_type, test_value",
    [
        (np.bool_, types.boolean, True),
        (np.int8, types.int8, 42),
        (np.int16, types.int16, 42),
        (np.int32, types.int32, 42),
        (np.int64, types.int64, 42),
        (np.uint8, types.uint8, 42),
        (np.uint16, types.uint16, 42),
        (np.uint32, types.uint32, 42),
        (np.uint64, types.uint64, 42),
        (np.float16, types.float16, 5.0),
        (np.float32, types.float32, 5.0),
        (np.float64, types.float64, 5.0),
        (np.complex64, types.complex64, 5.0 + 1.0j),
        (np.complex128, types.complex128, 5.0 + 1.0j),
    ],
)
def test_cupy_arrays(dtype, numba_type, test_value):
    cupy = pytest.importorskip("cupy")
    import numba_cuda_mlir.type_defs.cupy_types  # noqa: F401

    if dtype in (np.complex64, np.complex128):
        x = cupy.array([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]], dtype=dtype)
    elif dtype == np.bool_:
        x = cupy.array([[True, False], [False, True]], dtype=dtype)
    else:
        x = cupy.array([[1, 2], [3, 4]], dtype=dtype)

    @cuda.jit(types.void(numba_type[:, :]))
    def k(x):
        x[0, 0] = test_value

    k[1, 1](x)

    x_cpu = x.get()
    if numba_type in (types.complex64, types.complex128):
        assert x_cpu[0, 0] == test_value
    elif numba_type in (types.float16, types.float32, types.float64):
        np.testing.assert_allclose(x_cpu[0, 0], test_value)
    else:
        assert x_cpu[0, 0] == test_value


if __name__ == "__main__":
    test_cupy_arrays()
