# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import cuda.simt as cuda


def shfl_sync_test(ary, mask, source_lane):
    lane_id = cuda.threadIdx.x % 32
    result = cuda.shfl_sync(mask, lane_id, source_lane)
    ary[lane_id] = result


def shfl_up_sync_test(ary, mask, offset):
    lane_id = cuda.threadIdx.x % 32
    result = cuda.shfl_up_sync(mask, lane_id, offset)
    ary[lane_id] = result


def shfl_down_sync_test(ary, mask, offset):
    lane_id = cuda.threadIdx.x % 32
    result = cuda.shfl_down_sync(mask, lane_id, offset)
    ary[lane_id] = result


def shfl_xor_sync_test(ary, mask, lane_mask):
    lane_id = cuda.threadIdx.x % 32
    result = cuda.shfl_xor_sync(mask, lane_id, lane_mask)
    ary[lane_id] = result


class TestCudaIntrinsic(unittest.TestCase):
    def test_shfl_sync(self):
        compiled = cuda.jit("void(int32[:], int32, int32)")(shfl_sync_test)
        ary = np.zeros(32, dtype=np.int32)
        mask = 0xFFFFFFFF
        source_lane = 5
        compiled[1, 32](ary, mask, source_lane)
        expected = np.full(32, 5, dtype=np.int32)
        np.testing.assert_array_equal(ary, expected)

    def test_shfl_up_sync(self):
        compiled = cuda.jit("void(int32[:], int32, int32)")(shfl_up_sync_test)
        ary = np.zeros(32, dtype=np.int32)
        mask = 0xFFFFFFFF
        offset = 2
        compiled[1, 32](ary, mask, offset)
        expected = np.array([i - offset if i >= offset else i for i in range(32)], dtype=np.int32)
        np.testing.assert_array_equal(ary, expected)

    def test_shfl_down_sync(self):
        compiled = cuda.jit("void(int32[:], int32, int32)")(shfl_down_sync_test)
        ary = np.zeros(32, dtype=np.int32)
        mask = 0xFFFFFFFF
        offset = 2
        compiled[1, 32](ary, mask, offset)
        expected = np.array(
            [i + offset if i + offset < 32 else i for i in range(32)], dtype=np.int32
        )
        np.testing.assert_array_equal(ary, expected)

    def test_shfl_xor_sync(self):
        compiled = cuda.jit("void(int32[:], int32, int32)")(shfl_xor_sync_test)
        ary = np.zeros(32, dtype=np.int32)
        mask = 0xFFFFFFFF
        lane_mask = 1
        compiled[1, 32](ary, mask, lane_mask)
        expected = np.array([i ^ lane_mask for i in range(32)], dtype=np.int32)
        np.testing.assert_array_equal(ary, expected)
