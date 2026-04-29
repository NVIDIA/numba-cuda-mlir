# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
import itertools
import numpy as np
import pytest

import cusimt
import cuda.simt as cuda

from cusimt.numba_cuda.core.errors import TypingError
from cusimt.testing import NumbaCUDATestCase


# In order to verify the alignment of the local and shared memory arrays, we
# inspect the MLIR of the generated kernel using the following regexes.

# Shared memory example:
# memref.global @smem : memref<16xi8, 3> {alignment = 16}
SMEM_MLIR_PATTERN = re.compile(r"memref\.global.*alignment\s*=\s*(\d+)")

# Local memory example:
# %0 = memref.alloca() {alignment = 64} : memref<16xi8>
LMEM_MLIR_PATTERN = re.compile(r"memref\.alloca.*alignment\s*=\s*(\d+)")

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

            @cusimt.jit
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

            mlir = f.inspect_mlir(f.signatures[0])

            if alignment is None:
                if which == 0:
                    # Local memory shouldn't have any alignment information
                    # when no alignment is specified.
                    match = LMEM_MLIR_PATTERN.findall(mlir)
                    self.assertEqual(len(match), 0)
                else:
                    # Shared memory should at least have a power-of-two
                    # alignment when no alignment is specified.
                    match = SMEM_MLIR_PATTERN.findall(mlir)
                    self.assertEqual(len(match), 1)
                    actual = int(match[0])
                    self.assertTrue(actual & (actual - 1) == 0)
            else:
                # Verify alignment is in the MLIR.
                if which == 0:
                    match = LMEM_MLIR_PATTERN.findall(mlir)
                    self.assertEqual(len(match), 1)
                    self.assertEqual(alignment, int(match[0]))
                else:
                    match = SMEM_MLIR_PATTERN.findall(mlir)
                    self.assertEqual(len(match), 1)
                    self.assertEqual(alignment, int(match[0]))

                # Also verify that the address of the array is aligned.
                # If this fails, there is likely a problem with NVVM.
                address = loc[0] if which == 0 else shrd[0]
                self.assertEqual(int(address % alignment), 0)

    def test_default_alignment_local(self):
        @cusimt.jit
        def f(dest):
            local_array = cuda.local.array(shape=16, dtype=np.uint8)
            i = cuda.grid(1)
            if i == 0:
                dest[0] = local_array.ctypes.data

        dest = np.zeros(1, dtype=np.uint64)
        f[1, 1](dest)
        self.assertEqual(int(dest[0] % 8), 0)

    def test_default_alignment_shared(self):
        @cusimt.jit
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

                @cusimt.jit
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

                @cusimt.jit
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
                self.assertRaisesRegex(
                    ValueError, r"Alignment must be.*", f[1, 1], array
                )
            else:
                self.assertRaisesRegex(
                    TypingError,
                    expected_invalid_type_error_regex,
                    f[1, 1],
                    array,
                )
