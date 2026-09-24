# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np

from numba_cuda_mlir import cuda


def _compiled_mlir(kernel):
    return next(iter(kernel.inspect_mlir().values()))


def _assert_memref_slot(mlir, slot_type):
    escaped_type = re.escape(slot_type)
    assert re.search(rf"memref\.alloca\(\) : {escaped_type}", mlir)
    assert re.search(rf"memref\.store .* : {escaped_type}", mlir)
    assert re.search(rf"memref\.load .* : {escaped_type}", mlir)


def _assert_llvm_slot(mlir, slot_type):
    escaped_type = re.escape(slot_type)
    assert re.search(rf"llvm\.alloca .* x {escaped_type}", mlir)
    assert re.search(rf"llvm\.store .* : {escaped_type}, !llvm\.ptr", mlir)
    assert re.search(rf"llvm\.load .* -> {escaped_type}", mlir)


def test_multiply_assigned_boolean_uses_value_type_slot():
    @cuda.jit
    def kernel(inp, out):
        flag = inp[0] > 0
        if inp[1] > 0:
            flag = inp[2] > 0
        else:
            flag = inp[3] > 0
        out[0] = flag

    out = np.zeros(1, dtype=np.bool_)
    for inp, expected in (
        (np.array([1, 1, 0, 1], dtype=np.int32), False),
        (np.array([0, 0, 0, 1], dtype=np.int32), True),
    ):
        kernel[1, 1](inp, out)
        assert out[0] == expected

    mlir = _compiled_mlir(kernel)
    _assert_memref_slot(mlir, "memref<1xi1>")
    assert "memref<?xi8" in mlir


def test_multiply_assigned_unituple_uses_value_type_slots():
    @cuda.jit
    def kernel(inp, out):
        flags = (inp[0] > 0, inp[1] > 0)
        if inp[2] > 0:
            flags = (inp[3] > 0, inp[4] > 0)
        else:
            flags = (inp[5] > 0, inp[6] > 0)
        out[0] = flags[0]
        out[1] = flags[1]

    out = np.zeros(2, dtype=np.bool_)
    for inp, expected in (
        (np.array([0, 0, 1, 1, 0, 0, 1], dtype=np.int32), [True, False]),
        (np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int32), [False, True]),
    ):
        kernel[1, 1](inp, out)
        np.testing.assert_array_equal(out, expected)

    _assert_memref_slot(_compiled_mlir(kernel), "memref<2xi1>")


def test_multiply_assigned_heterogeneous_tuple_uses_value_type_slots():
    @cuda.jit
    def kernel(inp, bool_out, int_out):
        pair = (inp[0] > 0, inp[1])
        if inp[2] > 0:
            pair = (inp[3] > 0, inp[4])
        else:
            pair = (inp[5] > 0, inp[6])
        bool_out[0] = pair[0]
        int_out[0] = pair[1]

    bool_out = np.zeros(1, dtype=np.bool_)
    int_out = np.zeros(1, dtype=np.int32)
    for inp, expected in (
        (np.array([0, 0, 1, 1, 8, 0, 9], dtype=np.int32), (True, 8)),
        (np.array([1, 1, 0, 0, 8, 0, 9], dtype=np.int32), (False, 9)),
    ):
        kernel[1, 1](inp, bool_out, int_out)
        assert (bool_out[0], int_out[0]) == expected

    mlir = _compiled_mlir(kernel)
    _assert_memref_slot(mlir, "memref<1xi1>")
    _assert_memref_slot(mlir, "memref<1xi32>")


def test_multiply_assigned_optional_uses_value_type_llvm_slot():
    @cuda.jit
    def kernel(inp, out):
        value = inp[0]
        if inp[1] > 0:
            value = None
        else:
            value = inp[2]
        if value is None:
            out[0] = np.int32(-1)
        else:
            out[0] = value

    out = np.zeros(1, dtype=np.int32)
    for inp, expected in (
        (np.array([7, 1, 9], dtype=np.int32), -1),
        (np.array([7, 0, 9], dtype=np.int32), 9),
    ):
        kernel[1, 1](inp, out)
        assert out[0] == expected

    mlir = _compiled_mlir(kernel)
    _assert_llvm_slot(mlir, "!llvm.struct<(i32, i1)>")
    assert "!llvm.struct<(i32, i8)>" not in mlir
