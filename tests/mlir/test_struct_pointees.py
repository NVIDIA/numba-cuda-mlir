# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for numba_cuda_mlir.struct_pointees.

Pointee types for pointer members of identified structs are *derived* from the
Numba types the compiler already records; nothing declares them. These tests
pin that derivation and the attribute it produces.

The translation side -- what `llvm70.struct_pointees` does to the emitted LLVM 7
types -- is covered by `cext/mlir-llvm70/test/struct_pointees.mlir`.
"""

import pytest

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.numba_cuda import types as nbt
import numba_cuda_mlir.types as nt

from numba_cuda_mlir.lowering_utilities.context import get_context
from numba_cuda_mlir.models import mlir_data_manager
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type
from numba_cuda_mlir.struct_pointees import (
    _is_translatable,
    _pointee_type,
    _pointee_types,
    identified_struct_names,
    struct_pointees_attr,
)
from numba_cuda_mlir.type_defs.aggregate_types import (
    AggregateType,
    _map_be_type_names_to_fe_types,
)


@pytest.fixture
def ctx():
    """The compilation context, with the recorded-struct registry restored after.

    `AggregateType.__init__` files itself in a process-wide registry, so a test
    that builds a struct would otherwise leak it into every later compile.
    """
    saved = dict(_map_be_type_names_to_fe_types)
    try:
        with get_context() as ctx, ir.Location.unknown():
            yield ctx
    finally:
        _map_be_type_names_to_fe_types.clear()
        _map_be_type_names_to_fe_types.update(saved)


def _only(attr, name):
    """The ArrayAttr recorded for `name`, as a string."""
    return str(attr[name])


def test_pointee_recovered_at_the_right_member_index(ctx):
    st = AggregateType(
        "P_Mixed",
        [
            ("size", nbt.int64),
            ("data", nbt.CPointer(nbt.float32)),
            ("flag", nbt.int8),
            ("mask", nbt.CPointer(nbt.int8)),
        ],
    )
    # Non-pointer members must not shift the rest down: an off-by-one here
    # would refine `size` instead of `data`.
    assert _pointee_types(st) == {1: T.f32(), 3: T.i8()}


def test_bool_pointee_is_the_storage_type(ctx):
    """`bool` is `i8` in memory, not `i1`.

    Taking the value type would spell the member `i1*` where a typed producer
    has `i8*` -- actively *creating* the mismatch this mechanism removes, since
    the unrefined default `i8*` would already have been right.
    """
    st = AggregateType("P_Bool", [("flag", nbt.CPointer(nbt.boolean))])
    assert _pointee_types(st) == {0: T.i8()}


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (nbt.float16, "i16"),
        (nt.bf16, "i16"),
        (nt.f8E5M2, "i8"),
        (nt.tf32, "i32"),
        (nbt.float32, "f32"),
        (nbt.int64, "i64"),
    ],
)
def test_storage_backed_pointees(ctx, dtype, expected):
    """Types the datamodel stores as integers must point to those integers.

    `f8E5M2` and `tf32` additionally have no `MLIRToLLVM70::convertType` case,
    so passing the value type down would abort the process, not raise.
    """
    st = AggregateType("P_Storage", [("p", nbt.CPointer(dtype))])
    assert str(_pointee_types(st)[0]) == expected


def test_aggregate_pointee_uses_the_canonical_identified_struct(ctx):
    """A pointer to a struct must point at *the* struct, not a forked copy.

    `new_identified` only fills an opaque struct of the same name; against a
    bodied one it mints `Name.1`, a different LLVM type -- exactly the mismatch
    this mechanism exists to remove. Materialising the struct first is what
    building any body that mentions it does, and is the order that forks.
    """
    inner = AggregateType("P_Canon", [("x", nbt.int32)])
    st = AggregateType("P_HasCanon", [("child", nbt.CPointer(inner))])

    assert str(to_mlir_type(inner)) == '!llvm.struct<"P_Canon", (i32)>'
    assert str(_pointee_types(st)[0]) == '!llvm.struct<"P_Canon", (i32)>'


@pytest.mark.parametrize("materialise_first", [False, True], ids=["model-first", "type-first"])
def test_datamodel_and_to_mlir_type_agree_on_the_struct(ctx, materialise_first):
    """`AggregateTypeModel` must resolve an existing struct, not fork `Name.1`.

    The two paths that build an identified struct used to differ:
    `to_mlir_type` resolves then creates, the datamodel created unconditionally.
    Whichever ran second forked the type, silently producing two LLVM structs
    for one Numba type.
    """
    name = "P_Agree_" + ("type" if materialise_first else "model")
    fe = AggregateType(name, [("x", nbt.int32)])
    if materialise_first:
        to_mlir_type(fe)

    be = mlir_data_manager.lookup(fe).get_value_type()
    assert be.name == name, f"datamodel forked the struct: {be.name}"
    assert be == to_mlir_type(fe)


@pytest.mark.parametrize(
    "dtype",
    [nbt.void, nbt.Array(nbt.float32, 1, "C")],
    ids=["void", "array"],
)
def test_untranslatable_pointee_is_dropped(ctx, dtype):
    """`convertType` aborts the process on these, so they must not be emitted.

    `CPointer(void)` makes this reachable: its pointee converts to `none`.
    Dropping the entry leaves the member at `i8*`, which is what translation
    would have produced anyway.
    """
    st = AggregateType("P_Untranslatable", [("p", nbt.CPointer(dtype))])
    assert _pointee_types(st) == {}


def test_is_translatable_recurses_into_aggregates(ctx):
    """A struct reached only as a pointee still has its body converted."""
    assert _is_translatable(ir.Type.parse("!llvm.struct<(i32, f32)>"))
    assert not _is_translatable(ir.Type.parse("!llvm.struct<(i32, f8E5M2)>"))
    assert _is_translatable(ir.Type.parse("!llvm.array<4 x i32>"))
    assert not _is_translatable(ir.Type.parse("!llvm.array<4 x f8E5M2>"))


def test_pointee_type_returns_none_for_non_types(ctx):
    """Tuple-valued models (UniTuple) return a tuple, not an `ir.Type`."""
    assert _pointee_type(nbt.UniTuple(nbt.int32, 2)) is None


def test_pointer_to_identified_struct(ctx):
    inner = AggregateType("P_Inner", [("x", nbt.int32)])
    st = AggregateType("P_Outer", [("child", nbt.CPointer(inner))])
    assert str(_pointee_types(st)[0]) == '!llvm.struct<"P_Inner", (i32)>'


def test_struct_without_pointers_contributes_nothing(ctx):
    st = AggregateType("P_Plain", [("x", nbt.int32), ("y", nbt.int64)])
    assert _pointee_types(st) == {}


def test_bitfield_struct_is_skipped(ctx):
    """Bitfield structs collapse to one storage member, so indices do not line up."""
    st = AggregateType(
        "P_Bits",
        [("data", nbt.CPointer(nbt.float32), 4), ("rest", nbt.int32, 4)],
    )
    assert st.is_bitfield_struct
    assert _pointee_types(st) == {}


def _module(body):
    """A gpu.module whose kernel mentions each type in `body`."""
    lines = "\n".join(f"      %v{i} = llvm.mlir.undef : {ty}" for i, ty in enumerate(body))
    return ir.Module.parse(
        "module {\n"
        "  gpu.module @kernels {\n"
        "    llvm.func @k() attributes {gpu.kernel} {\n"
        f"{lines}\n"
        "      llvm.return\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def test_walk_finds_structs_through_every_reachable_path(ctx):
    m = ir.Module.parse(
        """
        module {
          gpu.module @kernels {
            llvm.func @k(%a: !llvm.struct<"W_BlockArg", (i32)>) attributes {gpu.kernel} {
              %c1 = llvm.mlir.constant(1 : i64) : i64
              %p = llvm.alloca %c1 x !llvm.struct<"W_ElemTypeAttr", (i32)> : (i64) -> !llvm.ptr
              %r = llvm.mlir.undef : !llvm.struct<"W_Result", (i32)>
              %n = llvm.mlir.undef : !llvm.array<2 x struct<"W_InArray", (i32)>>
              %s = llvm.mlir.undef : !llvm.struct<"W_Outer", (struct<"W_InStruct", (i32)>, i64)>
              llvm.return
            }
            llvm.func @decl(!llvm.struct<"W_FuncSig", (i32)>)
          }
        }
        """
    )
    assert identified_struct_names(m.operation) == {
        "W_BlockArg",
        "W_ElemTypeAttr",
        "W_Result",
        "W_InArray",
        "W_Outer",
        "W_InStruct",
        "W_FuncSig",
    }


def test_walk_finds_byval_argument_attribute_types(ctx):
    """`arg_attrs` nests an array of dictionaries, so the walk must go two deep.

    `llvm.byval` carries a struct type that appears nowhere else in the IR --
    the argument itself is just `!llvm.ptr` -- so missing it silently drops the
    refinement for that struct.
    """
    m = ir.Module.parse(
        """
        module {
          gpu.module @kernels {
            llvm.func @k(%p: !llvm.ptr {llvm.byval = !llvm.struct<"W_ByVal", (ptr, i64)>})
                attributes {gpu.kernel} {
              llvm.return
            }
          }
        }
        """
    )
    assert identified_struct_names(m.operation) == {"W_ByVal"}


def test_walk_ignores_literal_structs(ctx):
    m = _module(["!llvm.struct<(i32, i64)>"])
    assert identified_struct_names(m.operation) == set()


def test_attribute_shape_matches_translation_contract(ctx):
    """Each entry is an ArrayAttr of TypeAttr-or-UnitAttr, keyed by struct name.

    This is what `MLIRToLLVM70::loadStructPointees` parses; the C++ side reads
    anything that is not a TypeAttr as "no hint for this member".
    """
    AggregateType("P_Shape", [("size", nbt.int64), ("data", nbt.CPointer(nbt.float32))])
    m = _module(['!llvm.struct<"P_Shape", (i64, ptr)>'])
    attr = struct_pointees_attr(m.operation)

    assert isinstance(attr, ir.DictAttr)
    members = attr["P_Shape"]
    assert isinstance(members, ir.ArrayAttr)
    assert len(members) == 2
    assert not isinstance(members[0], ir.TypeAttr)
    assert ir.TypeAttr(members[1]).value == T.f32()


def test_only_structs_the_module_uses_are_described(ctx):
    """The recorded-type registry is process-wide; the attribute must not be."""
    AggregateType("P_Used", [("data", nbt.CPointer(nbt.float32))])
    AggregateType("P_Unused", [("data", nbt.CPointer(nbt.int32))])
    m = _module(['!llvm.struct<"P_Used", (ptr)>'])
    attr = struct_pointees_attr(m.operation)
    assert _only(attr, "P_Used") == "[f32]"
    assert "P_Unused" not in attr


def test_struct_used_only_via_a_nested_member_is_described(ctx):
    AggregateType("P_Nested", [("data", nbt.CPointer(nbt.float32))])
    m = _module(['!llvm.struct<"P_Holder", (struct<"P_Nested", (ptr)>, i64)>'])
    attr = struct_pointees_attr(m.operation)
    assert _only(attr, "P_Nested") == "[f32]"


def test_module_using_no_pointer_bearing_struct_yields_no_attribute(ctx):
    AggregateType("P_Nothing", [("x", nbt.int32)])
    m = _module(['!llvm.struct<"P_Nothing", (i32)>'])
    assert struct_pointees_attr(m.operation) is None


def test_struct_in_module_but_not_in_registry_is_ignored(ctx):
    """A struct the compiler never recorded (hand-written IR) must not crash."""
    m = _module(['!llvm.struct<"P_Unknown", (ptr, i64)>'])
    assert struct_pointees_attr(m.operation) is None
