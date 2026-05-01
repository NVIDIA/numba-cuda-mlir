# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda


def test_early_return_and_break():
    """
    Test early returns and break statements in kernel control flow.
    """
    BLOCK_SIZE = 256

    @cuda.jit
    def test_kernel(output, threshold):
        """
        Kernel that tests early return and break in for loop.
        - Threads with tid >= threshold return early
        - For loops break when counter reaches tid
        """
        tid = cuda.threadIdx.x

        # Early return if tid is past threshold
        if tid >= threshold:
            return

        # Accumulate sum with break condition
        sum_val = 0
        for i in range(100):
            sum_val += i
            # Each thread breaks at its own tid value
            if i >= tid:
                break

        output[tid] = sum_val

    # Allocate arrays
    output_host = np.zeros(BLOCK_SIZE, dtype=np.int32)
    output_device = cuda.to_device(output_host)

    # Only first 128 threads should execute
    threshold = 128

    # Launch kernel
    test_kernel[(1, 1, 1), (BLOCK_SIZE, 1, 1)](output_device, threshold)

    # Copy back
    result = output_device.copy_to_host()

    # Verify results
    expected = np.zeros(BLOCK_SIZE, dtype=np.int32)
    for tid in range(threshold):
        # Loop: for i in range(100): sum += i; if i >= tid: break
        # When i == tid, we add i first, then break
        # So we compute sum(0, 1, 2, ..., tid) = tid*(tid+1)/2
        # But if tid >= 100, the loop completes without breaking
        if tid < 100:
            expected[tid] = tid * (tid + 1) // 2
        else:
            expected[tid] = sum(range(100))

    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    test_early_return_and_break()
