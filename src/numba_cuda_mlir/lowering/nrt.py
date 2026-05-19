# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR-based NRT (Numba Runtime) lowering for numba_cuda_mlir.

Provides MLIR LLVM dialect equivalents of numba-cuda's NRT intrinsics,
enabling on-device heap allocation for strings, arrays, etc.
"""

import sys

from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.cpython.unicode import (
    _malloc_string,
    deref_uint8,
    deref_uint16,
    deref_uint32,
    set_uint8,
    set_uint16,
    set_uint32,
)
from numba_cuda_mlir._mlir.dialects import func
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.mlir.dialect_exts import llvm

from numba_cuda_mlir.lowering_utilities import (
    get_or_insert_function,
    convert,
    GEP_DYNAMIC_INDEX,
    int_of,
)
from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
lower = registry.lower

_HASH_WIDTH = 64 if sys.maxsize > 2**32 else 32


def _get_or_declare(module, name, result_types, arg_types):
    func_type = ir.FunctionType.get(arg_types, result_types)
    return get_or_insert_function(name, func_type, module)


def nrt_meminfo_alloc(module, size, align=None):
    """Allocate a MemInfo. Uses NRT_MemInfo_alloc_aligned (the only CUDA variant)."""
    if align is None:
        align = int_of(8, T.i32())
    callee = _get_or_declare(module, "NRT_MemInfo_alloc_aligned", [llvm.ptr()], [T.i64(), T.i32()])
    return func.call(result=[llvm.ptr()], callee=callee.name.value, operands_=[size, align])


def nrt_meminfo_data(module, meminfo):
    callee = _get_or_declare(module, "NRT_MemInfo_data_fast", [llvm.ptr()], [llvm.ptr()])
    return func.call(result=[llvm.ptr()], callee=callee.name.value, operands_=[meminfo])


def nrt_incref(module, meminfo):
    callee = _get_or_declare(module, "NRT_incref", [], [llvm.ptr()])
    func.call(result=[], callee=callee.name.value, operands_=[meminfo])


def nrt_decref(module, meminfo):
    callee = _get_or_declare(module, "NRT_decref", [], [llvm.ptr()])
    func.call(result=[], callee=callee.name.value, operands_=[meminfo])


def _lower_malloc_string(builder, target, args, kwargs):
    """MLIR equivalent of numba-cuda's _malloc_string intrinsic."""
    kind_val, char_bytes_val, length_val, is_ascii_val = builder.load_vars(args)
    module = builder.mlir_gpu_module

    kind_val = convert(kind_val, T.i32())
    char_bytes_val = convert(char_bytes_val, T.i64())
    length_val = convert(length_val, T.i64())
    is_ascii_val = convert(is_ascii_val, T.i32())

    nbytes = char_bytes_val * (length_val + 1)

    meminfo = nrt_meminfo_alloc(module, nbytes)
    data = nrt_meminfo_data(module, meminfo)

    hash_type = ir.IntegerType.get_signless(_HASH_WIDTH)
    neg_one = int_of(-1, hash_type)
    null_ptr = llvm.mlir_zero(res=llvm.ptr())

    struct_type = llvm.StructType.get_literal(
        [llvm.ptr(), T.i64(), T.i32(), T.i32(), hash_type, llvm.ptr(), llvm.ptr()]
    )
    desc = llvm.mlir_undef(res=struct_type)
    for i, value in enumerate(
        [data, length_val, kind_val, is_ascii_val, neg_one, meminfo, null_ptr]
    ):
        desc = llvm.insertvalue(desc, value, i)

    builder.store_var(target, desc)


def _make_deref_lowering(bitsize):
    """MLIR lowering for deref_uint{8,16,32}: read a character from string data."""

    def _lower_deref(builder, target, args, kwargs):
        data, idx = builder.load_vars(args)
        data = convert(data, llvm.ptr())
        idx = convert(idx, T.i64())
        int_type = ir.IntegerType.get_signless(bitsize)
        addr = llvm.getelementptr(llvm.ptr(), data, [idx], [GEP_DYNAMIC_INDEX], int_type, None)
        loaded = llvm.load(int_type, addr)
        result = convert(loaded, T.i32())
        builder.store_var(target, result)

    return _lower_deref


def _make_set_lowering(bitsize):
    """MLIR lowering for set_uint{8,16,32}: write a character to string data."""

    def _lower_set(builder, target, args, kwargs):
        data, idx, ch = builder.load_vars(args)
        data = convert(data, llvm.ptr())
        idx = convert(idx, T.i64())
        ch = convert(ch, T.i32())
        int_type = ir.IntegerType.get_signless(bitsize)
        ch = convert(ch, int_type)
        addr = llvm.getelementptr(llvm.ptr(), data, [idx], [GEP_DYNAMIC_INDEX], int_type, None)
        llvm.store(ch, addr)

    return _lower_set


lower(_malloc_string, types.Any, types.Any, types.Any, types.Any)(_lower_malloc_string)

lower(deref_uint8, types.Any, types.Any)(_make_deref_lowering(8))
lower(deref_uint16, types.Any, types.Any)(_make_deref_lowering(16))
lower(deref_uint32, types.Any, types.Any)(_make_deref_lowering(32))

lower(set_uint8, types.Any, types.Any, types.Any)(_make_set_lowering(8))
lower(set_uint16, types.Any, types.Any, types.Any)(_make_set_lowering(16))
lower(set_uint32, types.Any, types.Any, types.Any)(_make_set_lowering(32))
