# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.testing import NumbaCUDATestCase

# Basic tests that stream APIs execute on the hardware and in the simulator.
#
# Correctness of semantics is exercised elsewhere in the test suite (though we
# could improve the comprehensiveness of testing by adding more correctness
# tests here in future).


class TestStreamAPI(NumbaCUDATestCase):
    def test_stream_create_and_sync(self):
        s = cuda.stream()
        s.synchronize()

    def test_default_stream_create_and_sync(self):
        s = cuda.default_stream()
        s.synchronize()

    def test_legacy_default_stream_create_and_sync(self):
        s = cuda.legacy_default_stream()
        s.synchronize()

    def test_ptd_stream_create_and_sync(self):
        s = cuda.per_thread_default_stream()
        s.synchronize()

    def test_external_stream_create(self):
        #  A dummy pointer value
        ptr = 0x12345678
        s = cuda.external_stream(ptr)
        # We don't test synchronization on the stream because it's not a real
        # stream - we used a dummy pointer for testing the API, so we just
        # ensure that the stream handle matches the external stream pointer.
        value = int(s.handle)
        self.assertEqual(ptr, value)
