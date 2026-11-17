# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import itertools
import numpy as np
import pytest

import numba_cuda_mlir
from numba_cuda_mlir import cuda

from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.testing import NumbaCUDATestCase


SIMPLE_DTYPES = [np.uint8, np.uint32, np.uint64]

# Record dtypes with and without alignment.
RECORD_DTYPES = []
for _align in (True, False):
    RECORD_DTYPES += [
        np.dtype(
            [("a", np.uint8), ("b", np.int32), ("c", np.float64)],
            align=_align,
        ),
        np.dtype(
            [("a", np.uint32), ("b", np.uint8)],
            align=_align,
        ),
        np.dtype(
            [
                ("a", np.uint8),
                ("b", np.int32),
                ("c", np.float64),
                ("d", np.complex64),
                ("e", (np.uint8, 5)),
            ],
            align=_align,
        ),
    ]

# N.B. We name the test class TestArrayAddressAlignment to avoid name conflict
#      with the test_alignment.TestArrayAlignment class.


class TestArrayAddressAlignment(NumbaCUDATestCase):
    """
    Test cuda.local.array and cuda.shared.array support for an alignment
    keyword argument.
    """

    def test_array_alignment_1d(self):
        shapes = (1, 8, 50)
        alignments = (None, 16, 64, 256)
        array_types = [(0, "local"), (1, "shared")]
        self._do_test(array_types, shapes, SIMPLE_DTYPES, alignments)

    def test_array_alignment_2d(self):
        shapes = ((2, 3),)
        alignments = (None, 16, 64, 256)
        array_types = [(0, "local"), (1, "shared")]
        self._do_test(array_types, shapes, SIMPLE_DTYPES, alignments)

    def test_array_alignment_3d(self):
        shapes = ((2, 3, 4), (1, 4, 5), (4, 5, 6))
        alignments = (None, 16, 64, 256)
        array_types = [(0, "local"), (1, "shared")]
        self._do_test(array_types, shapes, SIMPLE_DTYPES, alignments)

    @pytest.mark.xfail(reason="Record dtypes not yet supported")
    def test_array_alignment_1d_record_dtypes(self):
        shapes = (1, 8, 50)
        alignments = (None, 16, 64, 256)
        array_types = [(0, "local"), (1, "shared")]
        self._do_test(array_types, shapes, RECORD_DTYPES, alignments)

    @pytest.mark.xfail(reason="Record dtypes not yet supported")
    def test_array_alignment_2d_record_dtypes(self):
        shapes = ((2, 3),)
        alignments = (None, 16, 64, 256)
        array_types = [(0, "local"), (1, "shared")]
        self._do_test(array_types, shapes, RECORD_DTYPES, alignments)

    @pytest.mark.xfail(reason="Record dtypes not yet supported")
    def test_array_alignment_3d_record_dtypes(self):
        shapes = ((2, 3, 4), (1, 4, 5), (4, 5, 6))
        alignments = (None, 16, 64, 256)
        array_types = [(0, "local"), (1, "shared")]
        self._do_test(array_types, shapes, RECORD_DTYPES, alignments)

    def _do_test(self, array_types, shapes, dtypes, alignments):
        items = itertools.product(array_types, shapes, dtypes, alignments)

        for (which, array_type), shape, dtype, alignment in items:

            @numba_cuda_mlir.cuda.jit
            def f(loc, shrd, which):
                i = cuda.grid(1)
                if which == 0:
                    local_array = cuda.local.array(
                        shape=shape,
                        dtype=dtype,
                        alignment=alignment,
                    )
                    if i == 0:
                        loc[0] = local_array.ctypes.data
                else:
                    shared_array = cuda.shared.array(
                        shape=shape,
                        dtype=dtype,
                        alignment=alignment,
                    )
                    if i == 0:
                        shrd[0] = shared_array.ctypes.data

            loc = np.zeros(1, dtype=np.uint64)
            shrd = np.zeros(1, dtype=np.uint64)
            f[1, 1](loc, shrd, which)

            required_alignment = 8 if alignment is None else alignment
            address = loc[0] if which == 0 else shrd[0]
            self.assertEqual(int(address % required_alignment), 0)

    def test_default_alignment_local(self):
        @numba_cuda_mlir.cuda.jit
        def f(dest):
            local_array = cuda.local.array(shape=16, dtype=np.uint8)
            i = cuda.grid(1)
            if i == 0:
                dest[0] = local_array.ctypes.data

        dest = np.zeros(1, dtype=np.uint64)
        f[1, 1](dest)
        self.assertEqual(int(dest[0] % 8), 0)

    def test_default_alignment_shared(self):
        @numba_cuda_mlir.cuda.jit
        def f(dest):
            shared_array = cuda.shared.array(shape=16, dtype=np.uint8)
            i = cuda.grid(1)
            if i == 0:
                dest[0] = shared_array.ctypes.data

        dest = np.zeros(1, dtype=np.uint64)
        f[1, 1](dest)
        self.assertEqual(int(dest[0] % 8), 0)

    def test_invalid_alignments(self):
        shapes = (1, 50)
        dtypes = (np.uint8, np.uint64)
        invalid_alignment_values = (-1, 0, 3, 17, 33)
        invalid_alignment_types = ("1.0", "1", "foo", 1.0, 1.5, 3.2)
        alignments = invalid_alignment_values + invalid_alignment_types
        array_types = [(0, "local"), (1, "shared")]

        # Use regex pattern to match error message, handling potential ANSI
        # color codes which appear on CI.
        expected_invalid_type_error_regex = (
            r"RequireLiteralValue:.*alignment must be a constant integer"
        )

        items = itertools.product(array_types, shapes, dtypes, alignments)

        for (which, array_type), shape, dtype, alignment in items:
            if which == 0:

                @numba_cuda_mlir.cuda.jit
                def f(dest_array):
                    i = cuda.grid(1)
                    local_array = cuda.local.array(
                        shape=shape,
                        dtype=dtype,
                        alignment=alignment,
                    )
                    if i == 0:
                        dest_array[0] = local_array.ctypes.data

            else:

                @numba_cuda_mlir.cuda.jit
                def f(dest_array):
                    i = cuda.grid(1)
                    shared_array = cuda.shared.array(
                        shape=shape,
                        dtype=dtype,
                        alignment=alignment,
                    )
                    if i == 0:
                        dest_array[0] = shared_array.ctypes.data

            array = np.zeros(1, dtype=np.uint64)

            # The type of error we expect differs between an invalid value
            # that is still an int, and an invalid type.
            if isinstance(alignment, int):
                self.assertRaisesRegex(ValueError, r"Alignment must be.*", f[1, 1], array)
            else:
                self.assertRaisesRegex(
                    TypingError,
                    expected_invalid_type_error_regex,
                    f[1, 1],
                    array,
                )
