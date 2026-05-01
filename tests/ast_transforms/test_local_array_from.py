# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numba_cuda_mlir
from numba_cuda_mlir import cuda
import numpy as np
import pytest


def test_local_array_from_transform():
    """Test that local_array_from is transformed to local_array + loop."""

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def kernel(out, indices):
        arr = numba_cuda_mlir.cuda.local_array_from(
            (i + 1 for i in indices), dtype=np.int64
        )
        out[0] = arr[0]
        out[1] = arr[1]
        out[2] = arr[2]

    cres = kernel.compile("void(int64[:], UniTuple(int64, 3))")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "local_array_from" not in source
    assert "local_array(len(indices)" in source
    assert "for __laf_i_0 in range(len(indices))" in source


def test_local_array_from_runs_correctly():
    """Test that local_array_from produces correct runtime behavior."""

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def kernel(out, indices):
        arr = numba_cuda_mlir.cuda.local_array_from(
            (i + 1 for i in indices), dtype=np.int64
        )
        out[0] = arr[0]
        out[1] = arr[1]
        out[2] = arr[2]

    out = np.zeros(3, dtype=np.int64)
    d_out = cuda.to_device(out)
    kernel[1, 1](d_out, (0, 1, 2))
    result = d_out.copy_to_host()
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_local_array_from_expression():
    """Test local_array_from with a more complex expression."""

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def kernel(out, values):
        arr = numba_cuda_mlir.cuda.local_array_from(
            (v * 2 + 1 for v in values), dtype=np.float32
        )
        for i in range(3):
            out[i] = arr[i]

    out = np.zeros(3, dtype=np.float32)
    d_out = cuda.to_device(out)
    kernel[1, 1](d_out, (1.0, 2.0, 3.0))
    result = d_out.copy_to_host()
    np.testing.assert_array_equal(result, [3.0, 5.0, 7.0])


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_local_array_from_dtypes(dtype):
    """Test local_array_from with various dtypes."""

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def kernel(out, indices):
        arr = numba_cuda_mlir.cuda.local_array_from((i for i in indices), dtype=dtype)
        out[0] = arr[0]

    out = np.zeros(1, dtype=dtype)
    d_out = cuda.to_device(out)
    kernel[1, 1](d_out, (42,))
    result = d_out.copy_to_host()
    assert result[0] == 42
