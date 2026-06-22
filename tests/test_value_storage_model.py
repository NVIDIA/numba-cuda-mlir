# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir import types
from numba_cuda_mlir.models import mlir_data_manager, register_fp8_models
from numba_cuda_mlir.numba_cuda.types.ext_types import bfloat16
from numba_cuda_mlir.numba_cuda.np import numpy_support


def _model_pair(numba_type):
    model = mlir_data_manager.lookup(numba_type)
    return str(model.get_value_type()), str(model.get_data_type())


def test_scalar_value_and_storage_types_are_distinct_where_required():
    with ir.Context(), ir.Location.unknown():
        register_fp8_models()
        assert _model_pair(types.bool) == ("i1", "i8")
        assert _model_pair(types.float16) == ("f16", "i16")
        assert _model_pair(bfloat16) == ("bf16", "i16")
        assert _model_pair(types.f4E2M1FN) == ("f4E2M1FN", "i8")
        assert _model_pair(types.f6E2M3FN) == ("f6E2M3FN", "i8")
        assert _model_pair(types.f8E4M3FN) == ("f8E4M3FN", "i8")
        assert _model_pair(types.tf32) == ("tf32", "i32")
        assert _model_pair(types.float32) == ("f32", "f32")
        assert _model_pair(types._type_fp8_e5m2) == ("f8E5M2", "i8")
        assert _model_pair(types._type_fp8_e4m3) == ("f8E4M3FN", "i8")
        assert _model_pair(types._type_fp8_e8m0) == ("f8E8M0FNU", "i8")


def test_array_memrefs_use_dtype_storage_type():
    with ir.Context(), ir.Location.unknown():
        assert "memref<?xi8" in str(mlir_data_manager.lookup(types.bool[:]).get_value_type())
        assert "memref<?xi16" in str(mlir_data_manager.lookup(types.float16[:]).get_value_type())
        assert "memref<?xi8" in str(mlir_data_manager.lookup(types.f4E2M1FN[:]).get_value_type())
        assert "memref<?xi8" in str(mlir_data_manager.lookup(types.f8E4M3FN[:]).get_value_type())
        assert "memref<?xi32" in str(mlir_data_manager.lookup(types.tf32[:]).get_value_type())


def test_storage_itemsize_uses_byte_addressable_storage():
    from numba_cuda_mlir.lowering_utilities import storage_itemsize_bytes

    with ir.Context(), ir.Location.unknown():
        assert storage_itemsize_bytes(types.bool) == 1
        assert storage_itemsize_bytes(types.float16) == 2
        assert storage_itemsize_bytes(bfloat16) == 2
        assert storage_itemsize_bytes(types.f4E2M1FN) == 1
        assert storage_itemsize_bytes(types.f6E2M3FN) == 1
        assert storage_itemsize_bytes(types.f8E4M3FN) == 1
        assert storage_itemsize_bytes(types.tf32) == 4
        assert storage_itemsize_bytes(types.CharSeq(3)) == 3
        assert storage_itemsize_bytes(types.UnicodeCharSeq(3)) == 12

        record_type = numpy_support.from_dtype(
            np.dtype([("x", np.int32), ("y", np.float16)], align=True)
        )
        assert storage_itemsize_bytes(record_type) == record_type.size
