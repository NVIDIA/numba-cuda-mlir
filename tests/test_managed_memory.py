# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CUDA managed memory support.

Managed memory (unified memory) is accessible from both the host and the device
without explicit copies. These tests verify that numba_cuda_mlir correctly handles
managed arrays passed to kernels.
"""

import warnings

import numpy as np
import pytest
from numba_cuda_mlir import cuda


def skip_if_managed_memory_unsupported():
    """Skip test if managed memory is not supported on this device."""
    ctx = cuda.current_context()
    cc_major = ctx.device.compute_capability[0]
    if cc_major < 3:
        pytest.skip("Managed memory unsupported prior to CC 3.0")


class TestManagedMemory:
    """Tests for managed memory arrays."""

    def test_managed_array_kernel_write(self):
        """Test that a kernel can write to a managed array."""
        skip_if_managed_memory_unsupported()

        @cuda.jit("void(double[:])")
        def kernel(x):
            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            if i < x.shape[0]:
                x[i] = float(i)

        ary = cuda.managed_array(100, dtype=np.double)
        ary.fill(0.0)

        kernel[10, 10](ary)
        cuda.current_context().synchronize()

        expected = np.arange(100, dtype=np.double)
        np.testing.assert_array_equal(ary, expected)

    def test_managed_array_kernel_read(self):
        """Test that a kernel can read from a managed array."""
        skip_if_managed_memory_unsupported()

        @cuda.jit("void(double[:], double[:])")
        def kernel(src, dst):
            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            if i < src.shape[0]:
                dst[i] = src[i] * 2.0

        src = cuda.managed_array(100, dtype=np.double)
        dst = cuda.managed_array(100, dtype=np.double)

        src[:] = np.arange(100, dtype=np.double)
        dst.fill(0.0)

        kernel[10, 10](src, dst)
        cuda.current_context().synchronize()

        expected = np.arange(100, dtype=np.double) * 2.0
        np.testing.assert_array_equal(dst, expected)

    def test_managed_array_no_copy_warning(self):
        """Test that managed arrays don't trigger host copy warning."""
        skip_if_managed_memory_unsupported()

        @cuda.jit("void(double[:])")
        def kernel(x):
            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            if i < x.shape[0]:
                x[i] = 1.0

        ary = cuda.managed_array(10, dtype=np.double)

        # This should NOT produce a warning about host array copy
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            kernel[1, 10](ary)
            cuda.current_context().synchronize()

        # Filter for our specific warning
        copy_warnings = [
            w for w in record if "Host array used in CUDA kernel" in str(w.message)
        ]
        assert len(copy_warnings) == 0, "Managed arrays should not trigger copy warning"

    def test_managed_array_mixed_with_device_array(self):
        """Test using both managed and device arrays in the same kernel."""
        skip_if_managed_memory_unsupported()

        @cuda.jit("void(double[:], double[:])")
        def kernel(managed, device):
            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            if i < managed.shape[0]:
                device[i] = managed[i] + 1.0

        managed = cuda.managed_array(100, dtype=np.double)
        device = cuda.to_device(np.zeros(100, dtype=np.double))

        managed[:] = np.arange(100, dtype=np.double)

        kernel[10, 10](managed, device)
        cuda.current_context().synchronize()

        result = device.copy_to_host()
        expected = np.arange(100, dtype=np.double) + 1.0
        np.testing.assert_array_equal(result, expected)

    def test_managed_array_cpu_access_after_kernel(self):
        """Test that managed array is accessible from CPU after kernel execution."""
        skip_if_managed_memory_unsupported()

        @cuda.jit("void(double[:])")
        def kernel(x):
            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            if i < x.shape[0]:
                x[i] = 42.0

        ary = cuda.managed_array(100, dtype=np.double)
        ary.fill(0.0)

        kernel[10, 10](ary)
        cuda.current_context().synchronize()

        # Access from CPU should work without explicit copy
        assert all(ary == 42.0)
        assert ary.sum() == 42.0 * 100

    def test_managed_array_2d(self):
        """Test managed array with 2D shape."""
        skip_if_managed_memory_unsupported()

        @cuda.jit("void(double[:,:])")
        def kernel(x):
            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
            if i < x.shape[0] and j < x.shape[1]:
                x[i, j] = float(i * 10 + j)

        ary = cuda.managed_array((8, 8), dtype=np.double)
        ary.fill(0.0)

        kernel[(1, 1), (8, 8)](ary)
        cuda.current_context().synchronize()

        for i in range(8):
            for j in range(8):
                assert ary[i, j] == i * 10 + j


class TestDeviceArrayInterface:
    """Tests for arrays with __cuda_array_interface__."""

    def test_device_array_not_copied(self):
        """Test that device arrays are not unnecessarily copied."""

        @cuda.jit("void(double[:])")
        def kernel(x):
            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            if i < x.shape[0]:
                x[i] = 1.0

        device_ary = cuda.to_device(np.zeros(10, dtype=np.double))

        # This should NOT produce a warning
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            kernel[1, 10](device_ary)
            cuda.current_context().synchronize()

        copy_warnings = [
            w for w in record if "Host array used in CUDA kernel" in str(w.message)
        ]
        assert len(copy_warnings) == 0, "Device arrays should not trigger copy warning"

    def test_numpy_array_triggers_warning(self):
        """Test that plain numpy arrays trigger a copy warning."""
        from numba_cuda_mlir.numba_cuda.core.errors import NumbaPerformanceWarning

        @cuda.jit("void(double[:])")
        def kernel(x):
            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            if i < x.shape[0]:
                x[i] = 1.0

        numpy_ary = np.zeros(10, dtype=np.double)

        # This SHOULD produce a warning about host array copy
        with pytest.warns(
            NumbaPerformanceWarning, match="Host array used in CUDA kernel"
        ):
            kernel[1, 10](numpy_ary)
            cuda.current_context().synchronize()
