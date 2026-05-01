# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for materializing string constants as MLIR values.

String literals are kept as Python ``str`` throughout lowering so that
compile-time operations (e.g. constant-folded equality) can work on
the raw value.  Call ``materialize_string_constant_if_needed`` at the
boundary where an ``ir.Value`` is actually required (e.g. lowerings
that accept both ``UnicodeType`` and ``StringLiteral`` operands).
"""

import sys
import itertools

from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.mlir.dialect_exts import llvm
from numba_cuda_mlir.lowering_utilities import constant

_HASH_WIDTH = 64 if sys.maxsize > 2**32 else 32
_counter = itertools.count(1)


def materialize_string_constant_if_needed(gpu_module, val):
    """If *val* is a Python str, materialize it as an MLIR unicode struct.

    Returns *val* unchanged if it is already an ``ir.Value``.
    """
    if isinstance(val, str):
        return materialize_string_constant(gpu_module, val)
    return val


def materialize_string_constant(gpu_module, pystr: str) -> ir.Value:
    """Materialize a Python string as an MLIR unicode struct SSA value.

    Emits an llvm.mlir.global with the string's byte data in *gpu_module*,
    then builds a unicode_type struct pointing at it.  The meminfo and
    parent fields are NULL (constant strings are not refcounted).

    Returns an SSA value of type
    ``!llvm.struct<(ptr, i64, i32, i32, i64, ptr, ptr)>``.
    """
    from numba_cuda_mlir.numba_cuda.cpython.unicode import compile_time_get_string_data

    databytes, length, kind, is_ascii, _ = compile_time_get_string_data(pystr)

    global_name = f"__numba_cuda_mlir_str_{next(_counter)}"

    arr_type = ir.Type.parse(f"!llvm.array<{len(databytes)} x i8>")
    gpu_block = gpu_module.bodyRegion.blocks[0]
    with ir.InsertionPoint.at_block_begin(gpu_block):
        linkage = ir.Attribute.parse("#llvm.linkage<internal>")
        str_value = ir.StringAttr.get(bytes(databytes).decode("latin-1"))
        llvm.GlobalOp(
            arr_type,
            global_name,
            linkage,
            addr_space=0,
            constant=True,
            value=str_value,
        )

    data_ptr = llvm.addressof(global_name)

    hash_type = ir.IntegerType.get_signless(_HASH_WIDTH)
    struct_type = llvm.StructType.get_literal(
        [llvm.ptr(), T.i64(), T.i32(), T.i32(), hash_type, llvm.ptr(), llvm.ptr()]
    )
    desc = llvm.mlir_undef(res=struct_type)
    desc = llvm.insertvalue(desc, data_ptr, 0)
    desc = llvm.insertvalue(desc, constant(length, T.i64()), 1)
    desc = llvm.insertvalue(desc, constant(kind, T.i32()), 2)
    desc = llvm.insertvalue(desc, constant(is_ascii, T.i32()), 3)
    desc = llvm.insertvalue(desc, constant(-1, hash_type), 4)
    desc = llvm.insertvalue(desc, llvm.mlir_zero(res=llvm.ptr()), 5)
    desc = llvm.insertvalue(desc, llvm.mlir_zero(res=llvm.ptr()), 6)
    return desc
