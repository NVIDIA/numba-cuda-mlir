# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, tools


@pytest.mark.skipif(
    tools.get_gpu_compute_capability(tuple) != (10, 0),
    reason=f"Expected compute capability 10.0, got {tools.get_gpu_compute_capability(tuple)}",
)
def test_shared_memory_96kb():
    """Test allocation and usage of 96KB shared memory."""

    SHMEM_SIZE = 98304  # 96KB dynamic shared memory
    SMEM_ELEMENTS = 12288  # Base size: 24KB of float16 data
    BLOCK_SIZE = 128

    @cuda.jit(chip="sm_100")
    def test_kernel(array, modifier):
        tid = cuda.threadIdx.x

        # Force dynamic shared memory allocation
        smem_size = SMEM_ELEMENTS * modifier  # Will be 49152 elements for modifier=4

        # Allocate shared memory with dynamically calculated size
        smem = cuda.shared_array(shape=(smem_size,), dtype=types.float16)

        # Write to two locations per thread
        idx1 = tid
        idx2 = tid + BLOCK_SIZE
        smem[idx1] = types.float16(tid)
        smem[idx2] = types.float16(tid + 100)

        cuda.syncthreads()

        # Read back and sum
        val1 = types.int32(smem[idx1])
        val2 = types.int32(smem[idx2])
        array[tid] = val1 + val2

    # Allocate output array and move to device
    array = np.zeros(BLOCK_SIZE, dtype=np.int32)
    array_d = cuda.to_device(array)

    # Launch kernel with large dynamic shared memory size > 48KB
    # This tests that the dispatcher calls cuFuncSetAttribute for larger than default smem allocations
    test_kernel[1, BLOCK_SIZE, 0, SHMEM_SIZE](array_d, 4)

    # Copy back to host
    result = array_d.copy_to_host()

    # Each thread writes tid and (tid+100), so sum is 2*tid + 100
    expected = np.arange(BLOCK_SIZE, dtype=np.int32) * 2 + 100

    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    test_shared_memory_96kb()
