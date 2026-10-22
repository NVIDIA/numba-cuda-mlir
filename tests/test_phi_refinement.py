# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for refining the inputs of a phi whose type is inferred from a later use.

``inferred_array(n)`` stands in for a constructor whose type is completed by
type inference: it returns a 1-D array of ``n`` elements whose dtype is
undefined until an indexed write gives it one.
"""

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import memref
from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.types import undefined


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


extending.refresh_registries()

F64_ARRAY = types.Array(types.float64, 1, "C")
RO_F64_ARRAY = types.Array(types.float64, 1, "C", readonly=True)


def _typemap(kernel):
    return next(iter(kernel.overloads.values()))._type_annotation.typemap


def _versions(typemap, name):
    """Types of ``name`` and of its SSA versions (``name.1``, ``name.2``, ...)."""
    return {k: v for k, v in typemap.items() if k == name or k.startswith(name + ".")}


def _run(kernel, flags, dtype=np.float64):
    results = []
    for flag in flags:
        out = np.zeros(1, dtype=dtype)
        kernel[1, 1](out, flag)
        results.append(out[0])
    return results


def test_two_constructors_joined_then_written():
    @cuda.jit
    def kernel(out, flag):
        if flag:
            buf = inferred_array(2)
        else:
            buf = inferred_array(2)
        buf[0] = 1.5
        buf[1] = 2.0
        out[0] = buf[0] + buf[1]

    assert _run(kernel, (1, 0)) == [3.5, 3.5]
    versions = _versions(_typemap(kernel), "buf")
    assert len(versions) == 3
    assert set(versions.values()) == {F64_ARRAY}


def test_three_constructors_joined_then_written():
    @cuda.jit
    def kernel(out, flag):
        if flag == 1:
            buf = inferred_array(2)
        elif flag == 2:
            buf = inferred_array(2)
        else:
            buf = inferred_array(2)
        buf[0] = 1.5
        buf[1] = 2.0
        out[0] = buf[0] + buf[1]

    assert _run(kernel, (1, 2, 0)) == [3.5, 3.5, 3.5]
    versions = _versions(_typemap(kernel), "buf")
    assert len(versions) == 4
    assert set(versions.values()) == {F64_ARRAY}


@pytest.mark.parametrize("constructor_first", [True, False])
def test_constructor_joined_with_a_precise_array(constructor_first):
    if constructor_first:

        def body(out, flag):
            if flag:
                buf = inferred_array(2)
            else:
                buf = cuda.local.array(2, np.float64)
            buf[0] = 1.5
            buf[1] = 2.0
            out[0] = buf[0] + buf[1]

    else:

        def body(out, flag):
            if flag:
                buf = cuda.local.array(2, np.float64)
            else:
                buf = inferred_array(2)
            buf[0] = 1.5
            buf[1] = 2.0
            out[0] = buf[0] + buf[1]

    kernel = cuda.jit(body)
    assert _run(kernel, (1, 0)) == [3.5, 3.5]
    assert set(_versions(_typemap(kernel), "buf").values()) == {F64_ARRAY}


def test_constructors_joined_through_copies():
    @cuda.jit
    def kernel(out, flag):
        a = inferred_array(2)
        b = inferred_array(2)
        if flag:
            x = a
        else:
            x = b
        x[0] = 1.5
        x[1] = 2.0
        out[0] = x[0] + x[1]

    assert _run(kernel, (1, 0)) == [3.5, 3.5]
    typemap = _typemap(kernel)
    assert typemap["a"] == F64_ARRAY
    assert typemap["b"] == F64_ARRAY


def test_loop_carried_constructor():
    @cuda.jit
    def kernel(out, n):
        buf = inferred_array(2)
        s = 0.0
        for i in range(n):
            buf[0] = 1.5
            buf[1] = 2.0
            s += buf[0] + buf[1]
            buf = inferred_array(2)
        out[0] = s

    assert _run(kernel, (3,)) == [10.5]
    assert set(_versions(_typemap(kernel), "buf").values()) == {F64_ARRAY}


def _run_with_input(kernel, src, flag):
    out = cuda.to_device(np.zeros(1))
    kernel[1, 1](out, cuda.to_device(src), flag)
    return out.copy_to_host()[0]


@pytest.mark.parametrize("write", [True, False])
def test_constructor_joined_with_a_readonly_array_stays_mutable(write):
    # `a[0] = a[1]` writes `a` without giving it a dtype, so the dtype comes
    # from the phi alone; the phi is readonly, `a` is not.
    if write:

        def body(out, ro, flag):
            a = inferred_array(2)
            if flag:
                x = a
            else:
                x = ro
            a[0] = a[1]
            out[0] = x[0]

    else:

        def body(out, ro, flag):
            a = inferred_array(2)
            if flag:
                x = a
            else:
                x = ro
            out[0] = x[0]

    kernel = cuda.jit([(F64_ARRAY, RO_F64_ARRAY, types.int64)])(body)
    assert _run_with_input(kernel, np.full(2, 7.0), 0) == 7.0
    typemap = _typemap(kernel)
    assert typemap["a"] == F64_ARRAY
    assert RO_F64_ARRAY in _versions(typemap, "x").values()


def test_constructor_joined_with_a_strided_array_keeps_its_layout():
    @cuda.jit
    def kernel(out, src, flag):
        a = inferred_array(2)
        if flag:
            x = a
        else:
            x = src[::2]
        a[0] = a[1]
        out[0] = x[0]

    assert _run_with_input(kernel, np.arange(4.0) + 5.0, 0) == 5.0
    typemap = _typemap(kernel)
    assert typemap["a"] == F64_ARRAY
    assert types.Array(types.float64, 1, "A") in _versions(typemap, "x").values()


# Behaviour that the change must not alter.


def test_phi_of_the_same_precise_type():
    @cuda.jit
    def kernel(out, flag):
        if flag:
            buf = cuda.local.array(2, np.int32)
            v = np.int32(3)
        else:
            buf = cuda.local.array(2, np.int32)
            v = np.int32(4)
        buf[0] = v
        out[0] = buf[0]

    assert _run(kernel, (1, 0), dtype=np.int32) == [3, 4]
    typemap = _typemap(kernel)
    assert set(_versions(typemap, "buf").values()) == {types.Array(types.int32, 1, "C")}
    assert set(_versions(typemap, "v").values()) == {types.int32}


def test_phi_of_different_precise_types_still_unifies():
    @cuda.jit
    def kernel(out, flag):
        if flag:
            v = np.int32(3)
        else:
            v = np.float64(0.5)
        out[0] = v

    assert _run(kernel, (1, 0)) == [3.0, 0.5]
    versions = _versions(_typemap(kernel), "v")
    assert sorted(map(str, versions.values())) == ["float64", "float64", "int32"]


def test_phi_of_different_array_dtypes_is_still_refused():
    @cuda.jit
    def kernel(out, flag):
        if flag:
            buf = cuda.local.array(2, np.int32)
        else:
            buf = cuda.local.array(2, np.float64)
        buf[0] = 1
        out[0] = buf[0]

    with pytest.raises(TypingError, match="Cannot unify array"):
        kernel[1, 1](np.zeros(1, dtype=np.float64), 1)


def test_unrefined_join_is_still_refused():
    @cuda.jit
    def kernel(out, flag):
        if flag:
            buf = inferred_array(2)
        else:
            buf = inferred_array(2)
        out[0] = flag

    with pytest.raises(TypingError, match=r"Cannot infer the type of variable 'buf"):
        kernel[1, 1](np.zeros(1, dtype=np.float64), 1)
