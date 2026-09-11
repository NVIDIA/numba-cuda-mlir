# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover pointee types for the pointer members of identified LLVM structs.

Below sm_100 a module is translated to the LLVM 7 dialect, whose pointers are
typed. MLIR's are opaque, so every pointer becomes ``i8*`` -- fine on its own,
but a struct from a typed producer (an NVRTC-compiled device function, say)
spells the same member ``float*``, and two identified structs of the same shape
with different pointee types are different LLVM types. The call between them
then links through a bitcast and stops being inlinable.

Numba-CUDA has no such problem: its ``CPointerModel`` builds
``dmm.lookup(fe_type.dtype).get_data_type().as_pointer()``. Here ``to_mlir_type``
maps ``CPointer`` to a bare ``!llvm.ptr``, so the pointee is lost from the *IR* --
but not from the *type*, which ``AggregateType.record_named_type`` already filed
under the struct's name. We read it back from there and pass it to the
translation in the ``llvm70.struct_pointees`` attribute, which
``MLIRToLLVM70::loadStructPointees`` consumes. Nothing user-facing is involved.

Both halves are deliberately lossy in the safe direction: a member we cannot
describe keeps ``i8*``, which is what translation would have produced anyway.

What this does *not* fix
------------------------
Only pointer members are refined. Non-pointer members keep the value type the
datamodel gives them, which is not always what a C++ producer spells for the
same field:

===============  ==================  =========================
field            here                clang (``bool`` is a byte)
===============  ==================  =========================
``bool``         ``i1``              ``i8``
``float16``      ``f16``             ``i16`` (via ``__half``)
===============  ==================  =========================

So a struct whose body differs stays a distinct LLVM type no matter what its
pointers say, and a call passing it still links through a bitcast. Worse, that
defeats the refinement transitively: a member declared ``CPointer(S)`` becomes
``%S*``, but if ``%S``'s own body disagrees with the producer's ``%S``, the two
``%S*`` are still different types. The cudf fixtures in
``cext/mlir-llvm70/test/cudf`` show this shape today -- ``Masked(bool)`` is
``(i1, i1)`` here where the C++ equivalent is ``{ i8, i8 }``.

Aligning member types is an ABI and layout change well beyond recovering a
pointee, so it is left alone. Anyone defining the same struct on both sides
should expect an exact body match to be required, not just matching pointers.
"""

from __future__ import annotations

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_storage_type
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.type_defs.aggregate_types import (
    AggregateType,
    _map_be_type_names_to_fe_types,
)


def _is_translatable(ty):
    """Whether ``MLIRToLLVM70::convertType`` can convert ``ty``.

    It calls ``report_fatal_error`` on anything outside its list, which aborts
    the process rather than raising, so an untranslatable pointee has to be
    dropped here instead of caught later. ``CPointer(void)`` alone makes this
    reachable: its pointee is ``none``.
    """
    ty = ty.maybe_downcast()
    if isinstance(ty, llvm.StructType):
        return ty.opaque or all(_is_translatable(m) for m in ty.body)
    if isinstance(ty, llvm.ArrayType):
        return _is_translatable(ty.element_type)
    if isinstance(ty, ir.VectorType):
        return _is_translatable(ty.element_type)
    return isinstance(
        ty,
        (
            ir.IntegerType,
            ir.IndexType,
            ir.F16Type,
            ir.BF16Type,
            ir.F32Type,
            ir.F64Type,
            llvm.PointerType,
        ),
    )


def _pointee_type(field_type):
    """The MLIR type a ``CPointer`` member points to, or None if unusable.

    The pointee is memory, so it is the *storage* type: ``bool`` is ``i8`` there
    and not ``i1``, and ``float16``/``bfloat16``/the float8 family are integers.
    This is the rule ``types.Array`` already uses for its element type, and the
    one numba-cuda's ``CPointerModel`` uses (``get_data_type``). Taking the value
    type instead would spell a member ``i1*`` where the typed producer has
    ``i8*`` -- losing the very match this exists to create.

    An identified struct is its own layout, and its storage type is that same
    struct: ``AggregateTypeModel`` resolves the name before creating, so this
    does not fork a second ``Name.1``.
    """
    ty = to_mlir_storage_type(field_type)
    # Tuple-valued models (UniTuple, NestedArray) return a tuple, not a type.
    if not isinstance(ty, ir.Type) or not _is_translatable(ty):
        return None
    return ty


def _pointee_types(fe_type):
    """Member index -> pointee MLIR type, for the pointer members of a struct.

    Empty for a bitfield struct, whose members collapse to one storage member,
    so an index here would refine the wrong one.
    """
    if fe_type.is_bitfield_struct:
        return {}

    pointees = {}
    for i, (_, field_type, *_) in enumerate(fe_type.fields):
        if not isinstance(field_type, types.CPointer):
            continue
        try:
            pointee = _pointee_type(field_type.dtype)
        except Exception as exc:
            # Nothing else converts the pointee -- the member itself is a bare
            # `!llvm.ptr` -- so an exotic one must not fail a compile that
            # worked before. Omitting it just keeps `i8*`.
            trace("no pointee for %s member %d: %s", fe_type.name, i, exc)
            continue
        if pointee is not None:
            pointees[i] = pointee
    return pointees


def identified_struct_names(op):
    """Names of the identified struct types reachable from ``op``.

    Reached through a block argument, an op result, or a type attribute
    (``llvm.alloca``'s ``elem_type``, a function signature, an ``llvm.byval``
    argument attribute), then transitively through struct members, array
    elements and function signatures. Operands need no visit of their own: each
    is a block argument or a result, so it has been seen already.
    """
    names, seen = set(), set()

    def visit_type(ty):
        ty = ty.maybe_downcast()
        if ty in seen:
            return
        seen.add(ty)
        if isinstance(ty, llvm.StructType):
            if ty.name is not None:
                names.add(ty.name)
            if not ty.opaque:
                for member in ty.body:
                    visit_type(member)
        elif isinstance(ty, llvm.ArrayType):
            visit_type(ty.element_type)
        elif isinstance(ty, llvm.FunctionType):
            visit_type(ty.return_type)
            for param in ty.inputs:
                visit_type(param)

    def visit_attr(attr):
        # `arg_attrs` nests two deep: an array of per-argument dictionaries.
        if isinstance(attr, ir.TypeAttr):
            visit_type(attr.value)
        elif isinstance(attr, ir.ArrayAttr):
            for element in attr:
                visit_attr(element)
        elif isinstance(attr, ir.DictAttr):
            for i in range(len(attr)):
                visit_attr(attr[i].attr)

    def visit_op(o):
        for region in o.regions:
            for block in region.blocks:
                for arg in block.arguments:
                    visit_type(arg.type)
                for inner in block.operations:
                    for result in inner.results:
                        visit_type(result.type)
                    for i in range(len(inner.attributes)):
                        visit_attr(inner.attributes[i].attr)
                    visit_op(inner)

    visit_op(op)
    return names


def struct_pointees_attr(op):
    """The ``llvm70.struct_pointees`` DictionaryAttr for ``op``, or None.

    Restricted to the structs ``op`` uses: the recorded-type registry is
    process-wide and accumulates every struct seen so far, which the translation
    would ignore but a dumped module would not.

    A member with no pointee becomes ``UnitAttr``, read as "no hint", so a
    struct can have some members described and the rest left as ``i8*``.
    """
    entries = {}
    with op.context:
        for name in identified_struct_names(op):
            fe_type = _map_be_type_names_to_fe_types.get(name)
            # Unions are recorded here too, and structs the compiler never built
            # (hand-written IR) are not recorded at all.
            if not isinstance(fe_type, AggregateType):
                continue
            pointees = _pointee_types(fe_type)
            if not pointees:
                continue
            entries[name] = ir.ArrayAttr.get(
                [
                    ir.TypeAttr.get(pointees[i]) if i in pointees else ir.UnitAttr.get()
                    for i in range(max(pointees) + 1)
                ]
            )
        return ir.DictAttr.get(entries) if entries else None
