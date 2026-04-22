# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cusimt.mlir.dialect_exts.llvm convenience helpers."""

from cusimt._mlir import ir
from cusimt._mlir.extras import types as T

from cusimt.mlir.dialect_exts import llvm


def _with_context(f):
    """Run *f* inside an MLIR context + module."""

    def wrapper():
        with ir.Context(), ir.Location.unknown():
            m = ir.Module.create()
            with ir.InsertionPoint(m.body):
                f(m)

    return wrapper


@_with_context
def test_ptr(_m):
    p = llvm.ptr()
    assert str(p) == "!llvm.ptr"


@_with_context
def test_insertvalue_int_position(_m):
    st = llvm.StructType.get_literal([T.i32(), T.i64()])
    from cusimt._mlir.dialects import func, arith

    f = func.FuncOp("test", ir.FunctionType.get([], []))
    entry = f.add_entry_block()
    with ir.InsertionPoint(entry):
        undef = llvm.mlir_undef(res=st)
        val = llvm.insertvalue(undef, arith.constant(T.i32(), 42), 0)
        func.ReturnOp([])
    asm = str(f)
    assert "llvm.insertvalue" in asm
    assert "[0]" in asm


@_with_context
def test_insertvalue_list_position(_m):
    st = llvm.StructType.get_literal([T.i32(), T.i64()])
    from cusimt._mlir.dialects import func, arith

    f = func.FuncOp("test", ir.FunctionType.get([], []))
    entry = f.add_entry_block()
    with ir.InsertionPoint(entry):
        undef = llvm.mlir_undef(res=st)
        val = llvm.insertvalue(undef, arith.constant(T.i64(), 99), [1])
        func.ReturnOp([])
    asm = str(f)
    assert "[1]" in asm


@_with_context
def test_addressof(m):
    from cusimt._mlir.dialects import func

    linkage = ir.Attribute.parse("#llvm.linkage<external>")
    llvm.GlobalOp(
        T.i32(),
        "my_global",
        linkage,
        addr_space=0,
        value=ir.IntegerAttr.get(T.i32(), 0),
    )
    f = func.FuncOp("test", ir.FunctionType.get([], [llvm.ptr()]))
    entry = f.add_entry_block()
    with ir.InsertionPoint(entry):
        p = llvm.addressof("my_global")
        func.ReturnOp([p])
    asm = str(f)
    assert "@my_global" in asm
    assert "!llvm.ptr" in asm


@_with_context
def test_addressof_default_result_type(m):
    from cusimt._mlir.dialects import func

    linkage = ir.Attribute.parse("#llvm.linkage<external>")
    llvm.GlobalOp(
        T.i32(), "g2", linkage, addr_space=0, value=ir.IntegerAttr.get(T.i32(), 0)
    )
    f = func.FuncOp("test", ir.FunctionType.get([], [llvm.ptr()]))
    entry = f.add_entry_block()
    with ir.InsertionPoint(entry):
        # No explicit res= — should default to !llvm.ptr
        p = llvm.addressof("g2")
        assert str(p.type) == "!llvm.ptr"
        func.ReturnOp([p])
