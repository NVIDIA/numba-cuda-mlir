# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for numba_cuda_mlir.extending.lower_cast — the external API that allows
extension libraries to register custom implicit cast lowerings.

The @lower_cast decorator registers a function that implements an implicit
conversion between two Numba types.  When Numba's type inference inserts
a ``cast`` IR node (e.g. from branch unification, setitem type mismatch,
or explicit coercion), numba_cuda_mlir's lowering consults the cast registry before
falling back to the default ``mlir_convert`` path.
"""

import numpy as np

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir.extending import lower_cast, lowering_registry
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.numba_cuda.typeconv import Conversion


# ---------------------------------------------------------------------------
# Custom Numba type: a 1-field struct wrapping int64
# ---------------------------------------------------------------------------


class MyBoxedIntType(types.Type):
    def __init__(self):
        super().__init__(name="MyBoxedInt")

    @property
    def key(self):
        return self.__class__

    def can_convert_to(self, typingctx, other):
        if isinstance(other, types.Integer):
            return Conversion.safe
        return None


my_boxed_int = MyBoxedIntType()


@register_model(MyBoxedIntType)
class MyBoxedIntModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = llvm.StructType.get_literal([mlir_ir.IntegerType.get_signless(64)])
        super().__init__(dmm, fe_type, be_type)


# ---------------------------------------------------------------------------
# Constructor: make_boxed_int(x) -> MyBoxedInt
# ---------------------------------------------------------------------------


def make_boxed_int(x):
    raise NotImplementedError("only callable inside a numba_cuda_mlir kernel")


@extending.type_callable(make_boxed_int)
def _type_make_boxed_int(context):
    def typer(x):
        if isinstance(x, types.Integer):
            return my_boxed_int

    return typer


@lowering_registry.lower(make_boxed_int, types.Integer)
def _lower_make_boxed_int(builder, target, args, kwargs):
    val = builder.load_var(args[0])
    struct_ty = builder.get_mlir_type(my_boxed_int)
    i64_ty = mlir_ir.IntegerType.get_signless(64)
    val = convert(val, i64_ty)
    undef = llvm.UndefOp(struct_ty)
    result = llvm.insertvalue(
        container=undef,
        value=val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    builder.store_var(target, result)


# ---------------------------------------------------------------------------
# @lower_cast: MyBoxedInt -> int64   (the API under test)
# ---------------------------------------------------------------------------


@lower_cast(MyBoxedIntType, types.Integer)
def _cast_boxed_to_int(context, builder, fromty, toty, val):
    result_ty = builder.get_mlir_type(toty)
    return llvm.extractvalue(
        res=result_ty,
        container=val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )


extending.refresh_registries()


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_lower_cast_custom_type_to_int64():
    """Branch unification forces Numba to insert a ``cast`` IR node.

    One branch assigns a MyBoxedInt, the other assigns a plain int64.
    Numba unifies to int64 (via can_convert_to) and inserts
    cast(MyBoxedInt -> int64), which invokes our @lower_cast.
    """

    @cuda.jit
    def kernel(flag, out):
        if flag[0]:
            x = make_boxed_int(42)
        else:
            x = np.int64(99)
        out[0] = x

    flag = np.array([True], dtype=np.bool_)
    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](flag, out)
    assert out[0] == 42

    flag[0] = False
    out[0] = 0
    kernel[1, 1](flag, out)
    assert out[0] == 99
