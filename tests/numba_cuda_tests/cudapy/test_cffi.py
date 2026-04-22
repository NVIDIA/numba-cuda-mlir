# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import cusimt
import cuda.simt as cuda
import pathlib
from numba import types
from cusimt.testing import NumbaCUDATestCase
from cusimt.numba_cuda.testing import test_data_dir

test_data_dir = pathlib.Path(__file__).parent.parent / "data"


class TestCFFI(NumbaCUDATestCase):
    def test_from_buffer(self):
        import cffi

        ffi = cffi.FFI()

        link = str(test_data_dir / "jitlink.ptx")
        sig = types.void(types.CPointer(types.int32))
        array_mutator = cuda.declare_device("array_mutator", sig)

        @cusimt.jit(link=[link])
        def mutate_array(x):
            x_ptr = ffi.from_buffer(x)
            array_mutator(x_ptr)

        x = np.arange(2).astype(np.int32)
        mutate_array[1, 1](x)

        # The foreign function should have copied element 1 to element 0
        self.assertEqual(x[0], x[1])
