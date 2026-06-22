# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.lowering_utilities import get_type_size_bytes


def test_get_type_size_bytes_bf16_and_i1():
    with ir.Context():
        assert get_type_size_bytes(T.bf16()) == 2
        assert get_type_size_bytes(ir.IntegerType.get_signless(1)) == 1
