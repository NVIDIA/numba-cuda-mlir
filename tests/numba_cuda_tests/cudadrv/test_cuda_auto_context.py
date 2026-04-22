# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np

import cuda.simt as cuda
from cusimt.testing import NumbaCUDATestCase


class TestCudaAutoContext(NumbaCUDATestCase):
    def test_auto_context(self):
        """A problem was revealed by a customer that the use cuda.to_device
        does not create a CUDA context.
        This tests the problem
        """
        A = np.arange(10, dtype=np.float32)
        newA = np.empty_like(A)
        dA = cuda.to_device(A)

        dA.copy_to_host(newA)
        self.assertTrue(np.allclose(A, newA))


if __name__ == "__main__":
    unittest.main()
