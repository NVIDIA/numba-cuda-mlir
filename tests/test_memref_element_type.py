# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for _is_valid_memref_element_type — ensuring that externally defined
extension types whose MLIR representation is an ``!llvm.struct`` are routed
through ``llvm.alloca`` / ``llvm.load`` / ``llvm.store`` rather than
``memref.alloca``, which rejects non-builtin element types.

When a variable is assigned more than once in a kernel, cusimt stack-allocates
it.  For builtin scalars (``i32``, ``f64``, …) a ``memref<1xT>`` works fine.
For LLVM dialect types (``!llvm.struct<…>``), ``memref`` is not valid and
``llvm.alloca`` must be used instead.  This situation arises with any
extension type whose data model lowers to ``!llvm.struct<…>``.

The integration tests below define a minimal extension type that lowers to
``!llvm.struct<(i32, i32)>`` and force multi-assignment so the stack
allocation path is exercised end-to-end.
"""

import numpy as np

from cusimt import cuda, extending, types
from cusimt.extending import lower_cast, lowering_registry
from cusimt.lowering_utilities import convert
from cusimt.models import PrimitiveModel, register_model
from cusimt._mlir import ir as mlir_ir
from cusimt._mlir.dialects import llvm
from numba.core.typeconv import Conversion


# ---------------------------------------------------------------------------
# Custom extension type: two i32 fields packed in an LLVM struct.
# Represents a (value, valid_bit) pair — a minimal "masked" scalar.
# ---------------------------------------------------------------------------


class MiniMaskedType(types.Type):
    def __init__(self):
        super().__init__(name="MiniMasked")

    @property
    def key(self):
        return self.__class__

    def can_convert_to(self, typingctx, other):
        if isinstance(other, types.Integer):
            return Conversion.safe
        return None


mini_masked = MiniMaskedType()


@register_model(MiniMaskedType)
class MiniMaskedModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        i32 = mlir_ir.IntegerType.get_signless(32)
        be_type = llvm.StructType.get_literal([i32, i32])
        super().__init__(dmm, fe_type, be_type)


# ---------------------------------------------------------------------------
# Constructor: make_masked(value, valid) -> MiniMasked
# ---------------------------------------------------------------------------


def make_masked(value, valid):
    raise NotImplementedError("only callable inside a cusimt kernel")


@extending.type_callable(make_masked)
def _type_make_masked(context):
    def typer(value, valid):
        if isinstance(value, types.Integer) and isinstance(valid, types.Integer):
            return mini_masked

    return typer


@lowering_registry.lower(make_masked, types.Integer, types.Integer)
def _lower_make_masked(builder, target, args, kwargs):
    value = builder.load_var(args[0])
    valid = builder.load_var(args[1])
    i32 = mlir_ir.IntegerType.get_signless(32)
    value = convert(value, i32)
    valid = convert(valid, i32)
    struct_ty = builder.get_mlir_type(mini_masked)
    undef = llvm.UndefOp(struct_ty)
    with_value = llvm.insertvalue(
        container=undef,
        value=value,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    result = llvm.insertvalue(
        container=with_value,
        value=valid,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    builder.store_var(target, result)


# ---------------------------------------------------------------------------
# @lower_cast: MiniMasked -> int32 (extract value field)
# ---------------------------------------------------------------------------


@lower_cast(MiniMaskedType, types.Integer)
def _cast_mini_masked_to_int(context, builder, fromty, toty, val):
    result_ty = builder.get_mlir_type(toty)
    return llvm.extractvalue(
        res=result_ty,
        container=val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_extension_type_multi_assign_uses_alloca():
    """A variable of extension type (!llvm.struct) assigned multiple times
    must use llvm.alloca — not memref.alloca which would be invalid.

    ``m`` is assigned twice (both as MiniMasked) so
    allocate_stack_space_for_vars_with_multiple_assigns fires.  Without the
    _is_valid_memref_element_type guard this would crash at MLIR verification
    because memref<!llvm.struct<(i32, i32)>> is not legal.

    Branch unification (``m`` vs ``int32``) forces a cast that reads ``m``
    back from its alloca slot, verifying the full store-load round trip.
    """

    @cuda.jit
    def kernel(flag, out):
        m = make_masked(1, 0)
        if flag[0]:
            m = make_masked(42, 1)
        if flag[0]:
            x = m
        else:
            x = np.int32(99)
        out[0] = x

    flag = np.array([True], dtype=np.bool_)
    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](flag, out)
    assert out[0] == 42, f"expected 42, got {out[0]}"

    flag[0] = False
    out[0] = 0
    kernel[1, 1](flag, out)
    assert out[0] == 99, f"expected 99, got {out[0]}"


def test_extension_type_loop_reassign():
    """Extension type variable re-assigned inside a loop.

    Each iteration overwrites ``m`` via llvm.store into its alloca slot.
    After the loop, branch unification reads ``m`` back via llvm.load,
    verifying that store/load work correctly across iterations.
    """

    @cuda.jit
    def kernel(n, out):
        m = make_masked(0, 0)
        for i in range(n[0]):
            m = make_masked(i, 1)
        if n[0] > 0:
            x = m
        else:
            x = np.int32(-1)
        out[0] = x

    n = np.array([5], dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](n, out)
    assert out[0] == 4, f"expected 4, got {out[0]}"

    n[0] = 1
    out[0] = 0
    kernel[1, 1](n, out)
    assert out[0] == 0, f"expected 0, got {out[0]}"

    n[0] = 0
    out[0] = 0
    kernel[1, 1](n, out)
    assert out[0] == -1, f"expected -1, got {out[0]}"
