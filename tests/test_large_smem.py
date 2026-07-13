# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from cuda.bindings import driver
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


def test_shared_memory_exactly_48kb_with_static_smem():
    """Regression test for issue #143.

    A kernel that requests exactly 48 KiB of dynamic shared memory while also
    using a small amount of static shared memory must still launch. The default
    per-function dynamic limit is 48 KiB minus the static usage, so the launcher
    has to opt in to the larger limit. Previously the launcher only opted in for
    ``sharedmem > 48 * 1024``, leaving the exactly-48-KiB case unreachable and
    failing with CUDA_ERROR_INVALID_VALUE.
    """

    DYN_BYTES = 48 * 1024  # exactly 48 KiB
    BLOCK_SIZE = 128

    @cuda.jit
    def k(out):
        # 1 byte of static shared memory + the dynamic shared array. The static
        # array must be written and read so the compiler cannot optimize it
        # away, otherwise the kernel reports zero static shared memory.
        flag = cuda.shared_array((1,), dtype=types.uint8)
        smem = cuda.shared_array(0, dtype=types.uint8)
        i = cuda.threadIdx.x
        if i == 0:
            flag[0] = types.uint8(7)
        if i < smem.size:
            smem[i] = types.uint8(i)
        cuda.syncthreads()
        if i == 0:
            out[0] = types.int32(flag[0])

    out = cuda.to_device(np.zeros(1, dtype=np.int32))

    # Previously failed with CUDA_ERROR_INVALID_VALUE (invalid argument).
    k[1, BLOCK_SIZE, 0, DYN_BYTES](out)
    cuda.synchronize()

    np.testing.assert_array_equal(out.copy_to_host(), np.array([7], dtype=np.int32))


def test_shared_memory_exactly_48kb_no_static_smem():
    """Exactly 48 KiB of dynamic shared memory with no static shared memory.

    Without static shared memory the per-function default dynamic limit is the
    full 48 KiB, so requesting exactly 48 KiB sits right at the boundary. This
    guards the boundary behaviour of the launcher's opt-in logic.
    """

    DYN_BYTES = 48 * 1024  # exactly 48 KiB
    BLOCK_SIZE = 128

    @cuda.jit
    def k(out):
        smem = cuda.shared_array(0, dtype=types.uint8)
        i = cuda.threadIdx.x
        if i < smem.size:
            smem[i] = types.uint8(i)
        cuda.syncthreads()
        if i == 0:
            out[0] = 1

    out = cuda.to_device(np.zeros(1, dtype=np.int32))

    k[1, BLOCK_SIZE, 0, DYN_BYTES](out)
    cuda.synchronize()

    np.testing.assert_array_equal(out.copy_to_host(), np.array([1], dtype=np.int32))


def _max_optin_shared_memory(device_id=0):
    """Return the device's maximum opt-in dynamic shared memory per block."""
    err, device = driver.cuDeviceGet(device_id)
    assert err == driver.CUresult.CUDA_SUCCESS, err
    err, max_optin = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device,
    )
    assert err == driver.CUresult.CUDA_SUCCESS, err
    return max_optin


def test_shared_memory_exceeds_device_limit():
    """Requesting more shared memory than the device allows must raise a clear error.

    The launcher opts in to the larger dynamic shared memory limit, but clamps
    the request to the device's maximum opt-in size. When the request exceeds
    that maximum we should surface a descriptive ValueError instead of an opaque
    CUDA_ERROR_INVALID_VALUE from the driver.
    """

    max_optin = _max_optin_shared_memory()
    too_much = max_optin + 1024  # 1 KiB over what the device supports
    BLOCK_SIZE = 128

    @cuda.jit
    def k(out):
        smem = cuda.shared_array(0, dtype=types.uint8)
        i = cuda.threadIdx.x
        if i < smem.size:
            smem[i] = types.uint8(i)
        cuda.syncthreads()
        if i == 0:
            out[0] = 1

    out = cuda.to_device(np.zeros(1, dtype=np.int32))

    with pytest.raises(ValueError, match="exceeds the device maximum opt-in shared memory"):
        k[1, BLOCK_SIZE, 0, too_much](out)
        cuda.synchronize()


def test_shared_memory_dynamic_plus_static_exceeds_device_limit():
    """Dynamic + static shared memory must be checked against the device limit.

    The device opt-in limit covers the total per-block shared memory (static +
    dynamic). A request for exactly ``max_optin`` bytes of dynamic shared memory
    fits on its own, but once the kernel's static shared memory is added the
    block exceeds the device limit. The launcher must account for the static
    usage and raise a descriptive ValueError rather than over-requesting and
    failing at launch with an opaque CUDA_ERROR_INVALID_VALUE.
    """

    max_optin = _max_optin_shared_memory()
    # Fits as dynamic-only (<= max_optin) but overflows once static is added.
    dyn_bytes = max_optin
    BLOCK_SIZE = 128

    @cuda.jit
    def k(out):
        # Static shared memory consumes part of the per-block budget. It must be
        # written and read so the compiler retains it (and reports a nonzero
        # static shared size).
        flag = cuda.shared_array((16,), dtype=types.uint8)
        smem = cuda.shared_array(0, dtype=types.uint8)
        i = cuda.threadIdx.x
        if i < 16:
            flag[i] = types.uint8(i)
        cuda.syncthreads()
        if i < smem.size:
            smem[i] = flag[i % 16]
        cuda.syncthreads()
        if i == 0:
            out[0] = types.int32(flag[0])

    out = cuda.to_device(np.zeros(1, dtype=np.int32))

    with pytest.raises(ValueError, match="static shared memory"):
        k[1, BLOCK_SIZE, 0, dyn_bytes](out)
        cuda.synchronize()


if __name__ == "__main__":
    test_shared_memory_96kb()
    test_shared_memory_exactly_48kb_with_static_smem()
    test_shared_memory_exactly_48kb_no_static_smem()
    test_shared_memory_exceeds_device_limit()
    test_shared_memory_dynamic_plus_static_exceeds_device_limit()
