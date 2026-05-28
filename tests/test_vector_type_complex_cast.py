# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir.testing import NumbaCUDATestCase
from numba_cuda_mlir.errors import TypingError

from numba_cuda_mlir import cuda
from numba_cuda_mlir import types

from numba_cuda_mlir.cuda.vector_types import float32x2, float16x2, float16x4, float64x2


class TestVectorTypeComplexCast(NumbaCUDATestCase):
    def test_vector_to_complex(self):
        @cuda.jit("void(complex64[:], complex128[:])")
        def kernel(arr64, arr128):
            v1 = cuda.float2(1.5, 2.5)
            v2 = cuda.double2(3.5, 4.5)

            c1 = complex(v1)
            c2 = complex(v2)

            arr64[0] = c1
            arr128[0] = c2

        res64 = np.zeros(1, dtype=np.complex64)
        res128 = np.zeros(1, dtype=np.complex128)

        kernel[1, 1](res64, res128)

        self.assertEqual(res64[0], complex(1.5, 2.5))
        self.assertEqual(res128[0], complex(3.5, 4.5))

    def test_int_vector_to_complex(self):
        @cuda.jit("void(complex128[:])")
        def kernel(arr):
            v = cuda.int2(1, 2)
            arr[0] = complex(v)

        res = np.zeros(1, dtype=np.complex128)
        kernel[1, 1](res)

        self.assertEqual(res[0], complex(1.0, 2.0))

    def test_invalid_complex_cast(self):
        with self.assertRaises(TypingError):

            @cuda.jit("void(complex128[:])")
            def kernel(arr):
                v = cuda.float4(1, 2, 3, 4)
                arr[0] = complex(v)

    def test_complex_to_vector(self):
        @cuda.jit("void(float32[:], float64[:])")
        def kernel(arr32, arr64):
            c1 = complex(1.5, 2.5)
            c2 = complex(3.5, 4.5)

            # Need to cast to specific complex type to ensure it's complex64/128
            # but Numba will type complex(1.5, 2.5) as complex128 by default.
            # Let's rely on the input complex type or just use complex128 for both.

            v1 = cuda.float2(c1)
            v2 = cuda.double2(c2)

            arr32[0] = v1.x
            arr32[1] = v1.y
            arr64[0] = v2.x
            arr64[1] = v2.y

        res32 = np.zeros(2, dtype=np.float32)
        res64 = np.zeros(2, dtype=np.float64)

        kernel[1, 1](res32, res64)

        self.assertEqual(res32[0], 1.5)
        self.assertEqual(res32[1], 2.5)
        self.assertEqual(res64[0], 3.5)
        self.assertEqual(res64[1], 4.5)

    def test_complex_arg_to_vector(self):
        @cuda.jit("void(complex64, complex128, float32[:], float64[:])")
        def kernel(c64, c128, arr32, arr64):
            v1 = cuda.float2(c64)
            v2 = cuda.double2(c128)

            arr32[0] = v1.x
            arr32[1] = v1.y
            arr64[0] = v2.x
            arr64[1] = v2.y

        res32 = np.zeros(2, dtype=np.float32)
        res64 = np.zeros(2, dtype=np.float64)

        c64 = np.complex64(1.5 + 2.5j)
        c128 = np.complex128(3.5 + 4.5j)

        kernel[1, 1](c64, c128, res32, res64)

        self.assertEqual(res32[0], 1.5)
        self.assertEqual(res32[1], 2.5)
        self.assertEqual(res64[0], 3.5)
        self.assertEqual(res64[1], 4.5)

    def test_float64x2_setitem_complex128_array(self):
        @cuda.jit
        def kernel(matrix, out):
            smem = cuda.shared.array(shape=(1,), dtype=float64x2)
            smem[0] = matrix[0]
            out[0] = smem[0]

        matrix = np.array([1 + 2j], dtype=np.complex128)
        out = np.zeros(1, dtype=np.complex128)
        kernel[1, 1](matrix, out)
        np.testing.assert_equal(out[0], matrix[0])

    def test_np_complex64_from_float32x2(self):
        @cuda.jit("void(complex64[:])")
        def kernel(out):
            v = float32x2(1.0, 2.0)
            out[0] = np.complex64(v) / 4.0

        res = np.zeros(1, dtype=np.complex64)
        kernel[1, 1](res)
        self.assertEqual(res[0], np.complex64(0.25 + 0.5j))

    def test_np_complex64_from_float16x2_widens(self):
        @cuda.jit("void(complex64[:])")
        def kernel(out):
            v = float16x2(np.float16(1.0), np.float16(2.0))
            out[0] = np.complex64(v)

        res = np.zeros(1, dtype=np.complex64)
        kernel[1, 1](res)
        self.assertEqual(res[0], np.complex64(1.0 + 2.0j))

    def test_np_complex128_from_float64x2(self):
        @cuda.jit("void(complex128[:])")
        def kernel(out):
            v = float64x2(1.0, 2.0)
            out[0] = np.complex128(v) / 4.0

        res = np.zeros(1, dtype=np.complex128)
        kernel[1, 1](res)
        self.assertEqual(res[0], np.complex128(0.25 + 0.5j))

    def test_np_complex128_from_float32x2_widens(self):
        @cuda.jit("void(complex128[:])")
        def kernel(out):
            v = float32x2(1.0, 2.0)
            out[0] = np.complex128(v)

        res = np.zeros(1, dtype=np.complex128)
        kernel[1, 1](res)
        self.assertEqual(res[0], np.complex128(1.0 + 2.0j))

    def test_np_complex64_from_float64x2_rejected(self):
        with self.assertRaises(TypingError):

            @cuda.jit("void(complex64[:])")
            def kernel(out):
                v = float64x2(1.0, 2.0)
                out[0] = np.complex64(v)

    def test_np_complex64_from_float16x4_rejected(self):
        with self.assertRaises(TypingError):

            @cuda.jit("void(complex64[:])")
            def kernel(out):
                v = float16x4(np.float16(1.0), np.float16(2.0), np.float16(3.0), np.float16(4.0))
                out[0] = np.complex64(v)
