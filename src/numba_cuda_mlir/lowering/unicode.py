# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import arith, scf
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.mlir.dialect_exts import llvm

from numba_cuda_mlir.extending import (
    overload,
    overload_method,
    register_jitable,
    typing_registry as extending_typing_registry,
)
from numba_cuda_mlir.lowering_registry import LoweringRegistry
from numba_cuda_mlir.lowering_utilities import GEP_DYNAMIC_INDEX, true, false

from numba_cuda_mlir.numba_cuda.cpython.unicode import (
    _malloc_string,
    _get_code_point,
    _kind_to_byte_width,
    set_uint8,
    set_uint16,
    set_uint32,
    deref_uint8,
    deref_uint16,
    deref_uint32,
)

from numba_cuda_mlir.numba_cuda.core.pythonapi import (
    PY_UNICODE_1BYTE_KIND,
    PY_UNICODE_2BYTE_KIND,
    PY_UNICODE_4BYTE_KIND,
)

from numba_cuda_mlir.numba_cuda.core import errors

import numpy as np

registry = LoweringRegistry()
lower = registry.lower


def _unicode_eq_lower(builder, target, args, kwargs):
    """Lower unicode_type == unicode_type / StringLiteral.

    Constant-folds when both values are known Python strings at compile time.
    Otherwise compares length first, then byte-by-byte (ASCII/kind=1).
    Uses select + for loop (no scf.if with results needed).
    """
    from numba_cuda_mlir.lowering_utilities.string import (
        materialize_string_constant_if_needed,
    )

    lhs_raw = builder.load_var(args[0])
    rhs_raw = builder.load_var(args[1])

    if isinstance(lhs_raw, str) and isinstance(rhs_raw, str):
        result = arith.constant(result=T.bool(), value=(lhs_raw == rhs_raw))
        builder.store_var(target, result)
        return

    lhs_val = materialize_string_constant_if_needed(builder.mlir_gpu_module, lhs_raw)
    rhs_val = materialize_string_constant_if_needed(builder.mlir_gpu_module, rhs_raw)

    # Extract lengths (field 1)
    lhs_len = llvm.extractvalue(T.i64(), lhs_val, [1])
    rhs_len = llvm.extractvalue(T.i64(), rhs_val, [1])

    # Extract data pointers (field 0)
    lhs_data = llvm.extractvalue(llvm.ptr(), lhs_val, [0])
    rhs_data = llvm.extractvalue(llvm.ptr(), rhs_val, [0])

    # Compare lengths
    len_eq = arith.cmpi(arith.CmpIPredicate.eq, lhs_len, rhs_len)

    # Compare bytes in a loop: accumulate mismatch count.
    # We compare raw bytes (works for kind=1 ASCII strings).
    zero_i64 = arith.constant(T.i64(), 0)
    one_i64 = arith.constant(T.i64(), 1)
    zero_mismatches = arith.constant(T.i64(), 0)

    # scf.for with iter_arg: count mismatches
    loop = scf.ForOp(zero_i64, lhs_len, one_i64, [zero_mismatches])
    iv = loop.body.arguments[0]
    mismatch_count = loop.body.arguments[1]
    with ir.InsertionPoint(loop.body):
        lhs_byte_ptr = llvm.getelementptr(
            llvm.ptr(), lhs_data, [iv], [GEP_DYNAMIC_INDEX], T.i8(), None
        )
        rhs_byte_ptr = llvm.getelementptr(
            llvm.ptr(), rhs_data, [iv], [GEP_DYNAMIC_INDEX], T.i8(), None
        )
        lhs_byte = llvm.load(T.i8(), lhs_byte_ptr)
        rhs_byte = llvm.load(T.i8(), rhs_byte_ptr)
        bytes_ne = arith.cmpi(arith.CmpIPredicate.ne, lhs_byte, rhs_byte)
        ne_as_i64 = arith.extui(T.i64(), bytes_ne)
        new_count = arith.addi(mismatch_count, ne_as_i64)
        scf.YieldOp([new_count])

    # Result: lengths equal AND no mismatches
    no_mismatches = arith.cmpi(arith.CmpIPredicate.eq, loop.results[0], zero_i64)
    result = arith.andi(len_eq, no_mismatches)
    builder.store_var(target, result)


# Register for all combinations of UnicodeType and StringLiteral
@lower(operator.eq, types.UnicodeType, types.UnicodeType)
@lower(operator.eq, types.UnicodeType, types.StringLiteral)
@lower(operator.eq, types.StringLiteral, types.UnicodeType)
def _lower_eq(builder, target, args, kwargs):
    _unicode_eq_lower(builder, target, args, kwargs)


@lower(operator.ne, types.UnicodeType, types.UnicodeType)
@lower(operator.ne, types.UnicodeType, types.StringLiteral)
@lower(operator.ne, types.StringLiteral, types.UnicodeType)
def _lower_ne(builder, target, args, kwargs):
    _unicode_eq_lower(builder, target, args, kwargs)
    eq_result = builder.load_var(target)
    ne_result = arith.xori(eq_result, true())
    builder.store_var(target, ne_result)


@register_jitable(typing_registry=extending_typing_registry)
def _empty_string_numba_cuda_mlir(kind, length, is_ascii=0):
    char_width = _kind_to_byte_width(kind)
    s = _malloc_string(kind, char_width, length, is_ascii)
    _set_code_point(s, length, np.uint32(0))
    return s


@register_jitable(typing_registry=extending_typing_registry)
def _ascii_upper_numba_cuda_mlir(data, res):
    for idx in range(len(data)):
        ch = _get_code_point(data, idx)
        if 97 <= ch <= 122:  # ord('a') .. ord('z')
            ch = ch - 32
        _set_code_point(res, idx, ch)


@register_jitable(typing_registry=extending_typing_registry)
def _ascii_lower_numba_cuda_mlir(data, res):
    for idx in range(len(data)):
        ch = _get_code_point(data, idx)
        if 65 <= ch <= 90:  # ord('A') .. ord('Z')
            ch = ch + 32
        _set_code_point(res, idx, ch)


@overload_method(types.UnicodeType, "upper", typing_registry=extending_typing_registry)
def unicode_upper(data):
    def impl(data):
        length = len(data)
        if length == 0:
            return _empty_string_numba_cuda_mlir(data._kind, length, data._is_ascii)
        res = _empty_string_numba_cuda_mlir(data._kind, length, 1)
        _ascii_upper_numba_cuda_mlir(data, res)
        return res

    return impl


@overload_method(types.UnicodeType, "lower", typing_registry=extending_typing_registry)
def unicode_lower(data):
    def impl(data):
        length = len(data)
        if length == 0:
            return _empty_string_numba_cuda_mlir(data._kind, length, data._is_ascii)
        res = _empty_string_numba_cuda_mlir(data._kind, length, 1)
        _ascii_lower_numba_cuda_mlir(data, res)
        return res

    return impl


@overload(len, typing_registry=extending_typing_registry)
def unicode_len(s):
    if isinstance(s, types.UnicodeType):

        def len_impl(s):
            return s._length

        return len_impl


@register_jitable(_nrt=False, typing_registry=extending_typing_registry)
def _set_code_point(a, i, ch):
    # WARNING: This method is very dangerous:
    #   * Assumes that data contents can be changed (only allowed for new
    #     strings)
    #   * Assumes that the kind of unicode string is sufficiently wide to
    #     accept ch.  Will truncate ch to make it fit.
    #   * Assumes that i is within the valid boundaries of the function
    if a._kind == PY_UNICODE_1BYTE_KIND:
        set_uint8(a._data, i, ch)
    elif a._kind == PY_UNICODE_2BYTE_KIND:
        set_uint16(a._data, i, ch)
    elif a._kind == PY_UNICODE_4BYTE_KIND:
        set_uint32(a._data, i, ch)
    else:
        raise AssertionError("Unexpected unicode representation in _set_code_point")


# Disable RefCt for performance.
@register_jitable(_nrt=False, typing_registry=extending_typing_registry)
def _get_code_point(a, i):
    if a._kind == PY_UNICODE_1BYTE_KIND:
        return deref_uint8(a._data, i)
    elif a._kind == PY_UNICODE_2BYTE_KIND:
        return deref_uint16(a._data, i)
    elif a._kind == PY_UNICODE_4BYTE_KIND:
        return deref_uint32(a._data, i)
    else:
        # there's also a wchar kind, but that's one of the above,
        # so skipping for this example
        return 0
