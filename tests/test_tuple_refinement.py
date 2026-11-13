# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for refining the items of a tuple from a later use of the item.

``inferred_array(n)`` stands in for a constructor whose type is completed by
type inference: it returns a 1-D array of ``n`` elements whose dtype is
undefined until an indexed write gives it one.

``ranked_array(n)`` returns a ``RankedArray``, a float64 array whose type also
carries a rank of 0 or 1; ``a.promote()`` gives ``a`` rank 1, and unifying
two ranks keeps the higher one. It stands in for an extension type whose precise
type a later use can still strengthen.
"""

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import memref
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.models import ArrayModel, register_model
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.types import undefined
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    bound_function,
    signature,
)


def inferred_array(n):
    raise NotImplementedError("kernel only")


@extending.type_callable(inferred_array)
def _type_inferred_array(context):
    def typer(n):
        if isinstance(n, types.IntegerLiteral):
            return types.Array(undefined, 1, "C")

    return typer


@lowering_registry.lower(inferred_array, types.IntegerLiteral)
def _lower_inferred_array(builder, target, args, kwargs):
    arr = builder.get_numba_type(target.name)
    n = builder.get_numba_type(args[0].name).literal_value
    with builder.alloca_insertion_point():
        storage = builder.get_storage_type(arr.dtype)
        mr = memref.alloca(mlir_ir.MemRefType.get([n], storage), [], [])
    builder.store_var(target, mr)


class RankedArray(types.Array):
    def __init__(self, rank):
        self.rank = rank
        super().__init__(types.float64, 1, "C", name=f"ranked_array(rank={rank})")

    @property
    def key(self):
        return (super().key, self.rank)

    def unify(self, typingctx, other):
        if isinstance(other, RankedArray):
            return self if self.rank >= other.rank else other
        return None


register_model(RankedArray)(ArrayModel)


def ranked_array(n):
    raise NotImplementedError("kernel only")


@extending.type_callable(ranked_array)
def _type_ranked_array(context):
    def typer(n):
        if isinstance(n, types.IntegerLiteral):
            return RankedArray(0)

    return typer


@lowering_registry.lower(ranked_array, types.IntegerLiteral)
def _lower_ranked_array(builder, target, args, kwargs):
    n = builder.get_numba_type(args[0].name).literal_value
    with builder.alloca_insertion_point():
        storage = builder.get_storage_type(types.float64)
        mr = memref.alloca(mlir_ir.MemRefType.get([n], storage), [], [])
    builder.store_var(target, mr)


@typing_registry.register_attr
class _RankedArrayAttributes(AttributeTemplate):
    key = RankedArray

    @bound_function("ranked_array.promote")
    def resolve_promote(self, ary, args, kws):
        return signature(types.none, recvr=RankedArray(1))


@lowering_registry.lower("ranked_array.promote", RankedArray)
def _lower_promote(builder, target, args, kwargs):
    pass


extending.refresh_registries()

F64_ARRAY = types.Array(types.float64, 1, "C")
I32_ARRAY = types.Array(types.int32, 1, "C")


def _typemap(kernel):
    return next(iter(kernel.overloads.values()))._type_annotation.typemap


def _run(kernel, dtype=np.float64):
    out = np.zeros(1, dtype=dtype)
    kernel[1, 1](out)
    return out[0]


def test_projection_refines_the_item():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        t = (a,)
        x = t[0]
        x[0] = 1.5
        out[0] = a[0]

    assert _run(kernel) == 1.5
    typemap = _typemap(kernel)
    assert typemap["a"] == F64_ARRAY
    assert typemap["t"] == types.UniTuple(F64_ARRAY, 1)


def test_projection_of_a_tuple_expression():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        x = (a,)[0]
        x[0] = 1.5
        out[0] = x[0]

    assert _run(kernel) == 1.5
    assert _typemap(kernel)["a"] == F64_ARRAY


def test_projection_through_copies_of_the_tuple():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        t = (a, 1.0)
        u = t
        v = u
        x = v[0]
        x[0] = 1.5
        out[0] = a[0] + t[1]

    assert _run(kernel) == 2.5
    typemap = _typemap(kernel)
    assert typemap["a"] == F64_ARRAY
    assert typemap["t"] == typemap["u"] == types.Tuple((F64_ARRAY, types.float64))


def test_heterogeneous_tuple_refines_only_its_imprecise_items():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        b = inferred_array(2)
        t = (a, np.int32(2), 3, b)
        x = t[0]
        y = t[3]
        x[0] = 1.5
        y[0] = np.int32(4)
        out[0] = a[0] + t[1] + t[2] + b[0]

    assert _run(kernel) == 10.5
    typemap = _typemap(kernel)
    assert typemap["a"] == F64_ARRAY
    assert typemap["b"] == I32_ARRAY
    assert typemap["t"] == types.Tuple((F64_ARRAY, types.int32, types.IntegerLiteral(3), I32_ARRAY))


def test_negative_and_named_constant_indices():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        b = inferred_array(2)
        t = (a, b)
        i = 1
        x = t[-2]
        y = t[i]
        x[0] = 1.5
        y[0] = 2.0
        out[0] = a[0] + b[0]

    assert _run(kernel) == 3.5
    typemap = _typemap(kernel)
    assert typemap["a"] == typemap["b"] == F64_ARRAY


def test_nested_tuple():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        b = inferred_array(2)
        t = ((a, b), 1.0)
        inner = t[0]
        inner[1][0] = 2.5
        inner[0][0] = 1.5
        out[0] = a[0] + b[0] + t[1]

    assert _run(kernel) == 5.0
    typemap = _typemap(kernel)
    assert typemap["a"] == typemap["b"] == F64_ARRAY
    assert typemap["t"] == types.Tuple((types.UniTuple(F64_ARRAY, 2), types.float64))


def test_unpacking_a_tuple():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        b = inferred_array(2)
        t = (a, b)
        x, y = t
        x[0] = 1.5
        y[0] = 2.5
        out[0] = a[0] + b[0]

    assert _run(kernel) == 4.0
    typemap = _typemap(kernel)
    assert typemap["a"] == typemap["b"] == F64_ARRAY


def test_multiple_assignment():
    # Python builds and unpacks a tuple for four or more values.
    @cuda.jit
    def kernel(out):
        a, b, c, s = inferred_array(2), inferred_array(2), inferred_array(2), 0.5
        a[0] = 1.0
        b[0] = 2.0
        c[0] = 3.0
        out[0] = a[0] + b[0] + c[0] + s

    assert _run(kernel) == 6.5
    typemap = _typemap(kernel)
    assert typemap["a"] == typemap["b"] == typemap["c"] == F64_ARRAY


def test_a_precise_item_takes_a_stronger_type_from_its_projection():
    @cuda.jit
    def kernel(out):
        a = ranked_array(2)
        t = (a, 1.0)
        x = t[0]
        x.promote()
        x[0] = 1.5
        out[0] = a[0] + t[1]

    assert _run(kernel) == 2.5
    typemap = _typemap(kernel)
    assert typemap["a"] == RankedArray(1)
    assert typemap["t"] == types.Tuple((RankedArray(1), types.float64))


# Behaviour that the change must not alter.


def test_tuple_of_precise_types():
    @cuda.jit
    def kernel(out):
        a = cuda.local.array(2, np.float64)
        t = (a, np.int32(3))
        x = t[0]
        x[0] = t[1]
        out[0] = a[0]

    assert _run(kernel) == 3.0
    assert _typemap(kernel)["t"] == types.Tuple((F64_ARRAY, types.int32))


def test_unrefined_projection_is_still_refused():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        t = (a,)
        x = t[0]
        out[0] = 1.0

    with pytest.raises(TypingError, match=r"Cannot infer the type of variable 'a'"):
        _run(kernel)


def test_slice_does_not_refine_the_tuple():
    @cuda.jit
    def kernel(out):
        a = inferred_array(2)
        t = (a, 1.0)
        s = t[0:1]
        x = s[0]
        x[0] = 1.5
        out[0] = x[0]

    with pytest.raises(TypingError, match=r"Cannot infer the type of variable 'a'"):
        _run(kernel)


def test_runtime_index_does_not_refine_the_tuple():
    @cuda.jit
    def kernel(out, i):
        a = inferred_array(2)
        b = inferred_array(2)
        t = (a, b)
        x = t[i]
        x[0] = 1.5
        out[0] = x[0]

    with pytest.raises(TypingError, match=r"Cannot infer the type of variable 'a'"):
        kernel[1, 1](np.zeros(1), 0)
