# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir.extending import lower_cast, lowering_registry
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.numba_cuda.typeconv import Conversion


# ---------------------------------------------------------------------------
# Custom extension type: a pair of (int32, int32) packed in a struct.
# Exercises the pattern where an extension type has a custom MLIR data model.
# ---------------------------------------------------------------------------


class MyPairType(types.Type):
    def __init__(self):
        super().__init__(name="MyPair")

    @property
    def key(self):
        return self.__class__

    def can_convert_to(self, typingctx, other):
        if isinstance(other, types.Integer):
            return Conversion.safe
        return None


my_pair = MyPairType()


@register_model(MyPairType)
class MyPairModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        i32 = mlir_ir.IntegerType.get_signless(32)
        be_type = llvm.StructType.get_literal([i32, i32])
        super().__init__(dmm, fe_type, be_type)


# ---------------------------------------------------------------------------
# Constructor: make_pair(a, b) -> MyPair
# The lowering uses to_mlir_type(my_pair) — the code path under test.
# ---------------------------------------------------------------------------


def make_pair(a, b):
    raise NotImplementedError("only callable inside a numba_cuda_mlir kernel")


@extending.type_callable(make_pair)
def _type_make_pair(context):
    def typer(a, b):
        if isinstance(a, types.Integer) and isinstance(b, types.Integer):
            return my_pair

    return typer


@lowering_registry.lower(make_pair, types.Integer, types.Integer)
def _lower_make_pair(builder, target, args, kwargs):
    a = builder.load_var(args[0])
    b = builder.load_var(args[1])
    i32 = mlir_ir.IntegerType.get_signless(32)
    a = convert(a, i32)
    b = convert(b, i32)
    struct_ty = to_mlir_type(my_pair)
    undef = llvm.UndefOp(struct_ty)
    with_a = llvm.insertvalue(
        container=undef,
        value=a,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    result = llvm.insertvalue(
        container=with_a,
        value=b,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    builder.store_var(target, result)


# ---------------------------------------------------------------------------
# @lower_cast: MyPair -> int32 (extract first field)
# ---------------------------------------------------------------------------


@lower_cast(MyPairType, types.Integer)
def _cast_pair_to_int(context, builder, fromty, toty, val):
    result_ty = builder.get_mlir_type(toty)
    return llvm.extractvalue(
        res=result_ty,
        container=val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )


extending.refresh_registries()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_to_mlir_type_resolves_extension_type():
    """to_mlir_type resolves a custom type via the mlir_data_manager fallback.

    Without the fallback, this would raise:
        TypeError: Unsupported MLIR dtype: MyPair
    """
    with mlir_ir.Context(), mlir_ir.Location.unknown():
        mlir_ty = to_mlir_type(my_pair)
        assert isinstance(mlir_ty, mlir_ir.Type)
        assert str(mlir_ty) == "!llvm.struct<(i32, i32)>"


@pytest.mark.parametrize("flag_value, expected", [(True, 42), (False, 99)])
def test_to_mlir_type_extension_in_kernel(flag_value, expected):
    """End-to-end: a kernel constructs a MyPair (whose lowering calls
    to_mlir_type) and then casts it to int via branch unification.

    This exercises the code path where an extension type's MLIR struct type
    is resolved via mlir_data_manager.lookup inside to_mlir_type.
    """

    @cuda.jit
    def kernel(flag, out):
        if flag[0]:
            p = make_pair(42, 7)
        else:
            p = np.int32(99)
        out[0] = p

    flag = np.array([flag_value], dtype=np.bool_)
    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](flag, out)
    assert out[0] == expected
