# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
import cuda.simt as cuda


def test_slicing():
    @cuda.jit(dump=False)
    def slicing(
        byte_arr: cuda.DeviceNDArray, start: int, stop: int, output: cuda.DeviceNDArray
    ):
        val = byte_arr[start:stop, start:stop]
        output[0] = val[0, 0]

    byte_arr_host = np.array(range(9), dtype=np.uint8).reshape(3, 3)
    byte_arr = cuda.to_device(byte_arr_host)
    output = cuda.to_device(np.zeros(1, dtype=np.int32))
    slicing[1, 1, 0, 0](byte_arr, 1, 2, output)
    output = output.copy_to_host()
    expect = byte_arr_host[1, 1]
    assert output[0] == expect, f"output: {output} != {expect}"


@pytest.mark.xfail
def tests_array_view():
    @cuda.jit(dump=False)
    def reinterpret_array_type(
        byte_arr: cuda.DeviceNDArray, start: int, stop: int, output: cuda.DeviceNDArray
    ):
        # Tested with just one thread
        val = byte_arr[start:stop].view(np.int32)[0]
        output[0] = val

    h = np.array(range(10), dtype=np.uint64)
    d = cuda.to_device(h)
    output = cuda.to_device(np.zeros(1, dtype=np.int32))
    reinterpret_array_type[1, 1](d, 1, 3, output)
    output = output.copy_to_host()
    expect = h[1:2].view(np.int32)[0]
    assert output[0] == expect, f"output: {output} != {expect}"


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    # test_slicing()
    tests_array_view()
