# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg, TypingError
import pytest


def test_shape_from_kernel_argument():
    @cuda.jit
    def kernel(out, n):
        a = cuda.local.array(n, np.int32)
        for i in range(n):
            a[i] = i + 1
        s = 0
        for i in range(n):
            s += a[i]
        out[0] = s

    for n, expected in ((4, 10), (8, 36), (4, 10)):
        out = np.zeros(1, dtype=np.int32)
        kernel[1, 1](out, n)
        assert out[0] == expected

    assert len(kernel.overloads) == 2


def test_shape_keyword_from_kernel_argument():
    @cuda.jit
    def kernel(out, n):
        a = cuda.local.array(shape=n, dtype=np.float64)
        for i in range(n):
            a[i] = 0.5
        s = 0.0
        for i in range(n):
            s += a[i]
        out[0] = s

    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](out, 3)
    assert out[0] == 1.5


def test_shape_from_kernel_argument_with_alignment():
    @cuda.jit
    def kernel(out, n):
        a = cuda.local.array(n, np.float64, alignment=16)
        a[0] = 2.0
        out[0] = a[0]

    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](out, 2)
    assert out[0] == 2.0


def _alloca_lines(kernel):
    (mlir,) = kernel.inspect_mlir().values()
    return [line for line in mlir.splitlines() if "memref.alloca(" in line]


def test_shape_from_kernel_argument_with_alignas():
    @cuda.jit
    def kernel(out, n):
        a = cuda.local.array(n, np.float64, alignas=16)
        a[0] = 2.0
        out[0] = a[0]

    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](out, 2)
    assert out[0] == 2.0
    (alloca,) = _alloca_lines(kernel)
    assert "alignment = 16" in alloca


def test_literal_shape_with_alignas():
    @cuda.jit
    def kernel(out):
        a = cuda.local.array(2, np.float64, alignas=16)
        a[0] = 2.0
        out[0] = a[0]

    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](out)
    assert out[0] == 2.0
    (alloca,) = _alloca_lines(kernel)
    assert "alignment = 16" in alloca


def test_alignment_and_alignas_together_are_refused():
    @cuda.jit
    def kernel(out, n):
        a = cuda.local.array(n, np.float64, alignment=32, alignas=16)
        a[0] = 2.0
        out[0] = a[0]

    with pytest.raises(TypingError, match="pass either alignment or alignas, not both"):
        kernel[1, 1](np.zeros(1, dtype=np.float64), 2)


def test_shared_array_with_alignas():
    @cuda.jit
    def kernel(out):
        a = cuda.shared.array(4, np.float64, alignas=16)
        a[0] = 2.0
        out[0] = a[0]

    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](out)
    assert out[0] == 2.0
    (mlir,) = kernel.inspect_mlir().values()
    assert any("memref.global" in line and "alignment = 16" in line for line in mlir.splitlines())


def test_shape_through_device_function():
    @cuda.jit(device=True)
    def device_sum(n):
        a = cuda.local.array(n, np.int32)
        for i in range(n):
            a[i] = i
        s = 0
        for i in range(n):
            s += a[i]
        return s

    @cuda.jit
    def kernel(out, n):
        out[0] = device_sum(n)

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out, 5)
    assert out[0] == 10


def test_computed_shape_is_refused():
    @cuda.jit
    def kernel(out, n):
        a = cuda.local.array(n + 1, np.int32)
        a[0] = 1
        out[0] = a[0]

    with pytest.raises(TypingError, match="Cannot request literal type"):
        kernel[1, 1](np.zeros(1, dtype=np.int32), 4)


def test_tuple_shape_from_kernel_argument_is_refused():
    @cuda.jit
    def kernel(out, n):
        a = cuda.local.array((n, 2), np.int32)
        a[0, 0] = 1
        out[0] = a[0, 0]

    with pytest.raises(TypingError, match="No implementation of function"):
        kernel[1, 1](np.zeros(1, dtype=np.int32), 4)


def test_invalid_dtype_with_argument_shape_reports_dtype():
    @cuda.jit
    def kernel(out, n):
        a = cuda.local.array(n, "not_a_dtype")
        out[0] = a[0]

    with pytest.raises(TypingError, match="Invalid NumPy dtype specified"):
        kernel[1, 1](np.zeros(1, dtype=np.int32), 4)


def test_force_literal_arg_bind_fold_arguments():
    def fold_arguments(args, kws):
        return tuple(args)

    original = ForceLiteralArg({0})
    bound = original.bind_fold_arguments(fold_arguments)

    assert isinstance(bound, ForceLiteralArg)
    assert bound.requested_args == frozenset({0})
    assert bound.fold_arguments is fold_arguments
