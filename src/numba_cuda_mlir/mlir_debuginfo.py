# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import linecache
import os
from typing import NamedTuple

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.lowering_utilities import context as numba_cuda_mlir_context
from numba_cuda_mlir.numba_cuda.datamodel.models import ArrayModel
from numba_cuda_mlir.numba_cuda.types.ext_types import Bfloat16, GridGroup


class _DILocalVarInfo(NamedTuple):
    name: str
    line: int
    type_str: str
    arg_index: int | None


_BYTE_SIZE_BITS = 8
_BFLOAT16_BITS = 16
_GRID_GROUP_BITS = 64
_INT_LITERAL_BITS = 64
_POINTER_BITS = 64


def _basic_di_type(name, bits, encoding):
    return (
        f'#llvm.di_basic_type<tag = DW_TAG_base_type, name = "{name}", '
        f"sizeInBits = {bits}, encoding = DW_ATE_{encoding}>"
    )


def _cpointer_di_type(numba_type):
    pointee_di = _numba_type_to_di_type_str(numba_type.dtype)
    if pointee_di is None:
        pointee_di = _basic_di_type("byte", _BYTE_SIZE_BITS, "unsigned_char")
    return _derived_di_type(
        "DW_TAG_pointer_type",
        base_type=pointee_di,
        size_bits=_POINTER_BITS,
    )


def _enum_member_di_type(numba_type):
    dtype = numba_type.dtype
    if not isinstance(dtype, types.Integer):
        return None
    encoding = "signed" if dtype.signed else "unsigned"
    # NOTE: The current MLIR toolchain does not provide #llvm.di_enumerator and
    # lowering rejects DW_TAG_enumerator encoded via #llvm.di_derived_type in
    # the modern LLVM translation path. Until #llvm.di_enumerator is available,
    # emit a stable scalar DI type instead of invalid enum nodes.
    # Keep str(numba_type) by design.
    return _basic_di_type(str(numba_type), dtype.bitwidth, encoding)


def _complex_di_type(name, float_bits):
    """Build a DICompositeType string for a complex number."""
    float_type = _basic_di_type(f"float{float_bits}", float_bits, "float")
    real_member = (
        f"#llvm.di_derived_type<tag = DW_TAG_member, "
        f'name = "real", baseType = {float_type}, '
        f"sizeInBits = {float_bits}, offsetInBits = 0>"
    )
    imag_member = (
        f"#llvm.di_derived_type<tag = DW_TAG_member, "
        f'name = "imag", baseType = {float_type}, '
        f"sizeInBits = {float_bits}, offsetInBits = {float_bits}>"
    )
    return (
        f"#llvm.di_composite_type<tag = DW_TAG_structure_type, "
        f'name = "{name}", sizeInBits = {float_bits * 2}, '
        f"elements = {real_member}, {imag_member}>"
    )


def _derived_di_type(
    tag,
    *,
    name=None,
    base_type=None,
    size_bits=None,
    offset_bits=None,
    flags=None,
    extra_data=None,
):
    params = [f"tag = {tag}"]
    if name is not None:
        params.append(f'name = "{name}"')
    if base_type is not None:
        params.append(f"baseType = {base_type}")
    if size_bits is not None:
        params.append(f"sizeInBits = {size_bits}")
    if offset_bits is not None:
        params.append(f"offsetInBits = {offset_bits}")
    if flags is not None:
        params.append(f"flags = {flags}")
    if extra_data is not None:
        params.append(f"extraData = {extra_data} : i8")
    return f"#llvm.di_derived_type<{', '.join(params)}>"


def _composite_di_type(
    tag,
    *,
    name=None,
    base_type=None,
    size_bits=None,
    elements=None,
    identifier=None,
    discriminator=None,
):
    params = [f"tag = {tag}"]
    if name is not None:
        params.append(f'name = "{name}"')
    if base_type is not None:
        params.append(f"baseType = {base_type}")
    if size_bits is not None:
        params.append(f"sizeInBits = {size_bits}")
    if identifier is not None:
        params.append(f'identifier = "{identifier}"')
    if discriminator is not None:
        params.append(f"discriminator = {discriminator}")
    if elements:
        params.append(f"elements = {', '.join(elements)}")
    return f"#llvm.di_composite_type<{', '.join(params)}>"


def _subrange_di_type(count):
    return f"#llvm.di_subrange<count = {count}>"


def _llvm_type_str(numba_type):
    numba_type = _strip_literal_type(numba_type)
    match numba_type:
        case types.Boolean():
            return "i8"
        case types.Integer(bitwidth=bw):
            return f"i{bw}"
        case types.Float(bitwidth=16):
            return "half"
        case types.Float(bitwidth=32):
            return "float"
        case types.Float(bitwidth=64):
            return "double"
        case types.UniTuple(dtype=dtype, count=count):
            elem_str = _llvm_type_str(dtype)
            return None if elem_str is None else f"[{count} x {elem_str}]"
        case types.BaseTuple():
            elem_strs = [_llvm_type_str(t) for t in numba_type.types]
            if any(elem_str is None for elem_str in elem_strs):
                return None
            return "{" + ", ".join(elem_strs) + "}"
        case _:
            return None


def _align_to_bits(offset_bits, alignment_bits):
    return (
        offset_bits
        if offset_bits % alignment_bits == 0
        else offset_bits + alignment_bits - offset_bits % alignment_bits
    )


def _type_alignment_bits(numba_type):
    numba_type = _strip_literal_type(numba_type)
    match numba_type:
        case types.Boolean():
            return _BYTE_SIZE_BITS
        case types.Integer(bitwidth=bw) | types.Float(bitwidth=bw):
            return bw
        case types.UniTuple(dtype=dtype):
            return _type_alignment_bits(dtype)
        case types.BaseTuple():
            layout = _llvm_struct_layout_bits(numba_type.types)
            return None if layout is None else layout[2]
        case _:
            return None


def _llvm_struct_layout_bits(field_types):
    """Compute LLVM literal-struct layout, including alignment padding."""
    member_offsets = []
    offset_bits = 0
    max_alignment_bits = _BYTE_SIZE_BITS
    for t in field_types:
        field_size_bits = _type_size_bits(t)
        field_alignment_bits = _type_alignment_bits(t)
        if field_size_bits is None or field_alignment_bits is None:
            return None
        offset_bits = _align_to_bits(offset_bits, field_alignment_bits)
        member_offsets.append(offset_bits)
        offset_bits += field_size_bits
        max_alignment_bits = max(max_alignment_bits, field_alignment_bits)
    total_size_bits = _align_to_bits(offset_bits, max_alignment_bits)
    return (member_offsets, total_size_bits, max_alignment_bits)


def _type_size_bits(numba_type):
    numba_type = _strip_literal_type(numba_type)
    match numba_type:
        case types.Boolean():
            return _BYTE_SIZE_BITS
        case types.Integer(bitwidth=bw) | types.Float(bitwidth=bw):
            return bw
        case types.UniTuple(dtype=dtype, count=count):
            elem_bits = _type_size_bits(dtype)
            return None if elem_bits is None else elem_bits * count
        case types.BaseTuple():
            layout = _llvm_struct_layout_bits(numba_type.types)
            return None if layout is None else layout[1]
        case types.Record():
            return numba_type.size * _BYTE_SIZE_BITS
        case _:
            return None


def _strip_literal_type(numba_type):
    return getattr(numba_type, "literal_type", numba_type)


def _uni_tuple_di_type(numba_type):
    elem_di = _numba_type_to_di_type_str(numba_type.dtype)
    elem_bits = _type_size_bits(numba_type.dtype)
    elem_str = _llvm_type_str(numba_type.dtype)
    if elem_di is None or elem_bits is None or elem_str is None:
        return None

    return _composite_di_type(
        "DW_TAG_array_type",
        name=f"UniTuple({numba_type.dtype} x {numba_type.count}) ([{numba_type.count} x {elem_str}])",
        base_type=elem_di,
        size_bits=numba_type.count * elem_bits,
        elements=[_subrange_di_type(numba_type.count)],
    )


def _base_tuple_di_type(numba_type):
    members_di = []
    llvm_member_types = []
    tuple_layout = _llvm_struct_layout_bits(numba_type.types)
    if tuple_layout is None:
        return None
    member_offsets, total_size_bits, _ = tuple_layout
    for i, (field_type, offset_bits) in enumerate(
        zip(numba_type.types, member_offsets, strict=True)
    ):
        field_di = _numba_type_to_di_type_str(field_type)
        field_bits = _type_size_bits(field_type)
        if field_di is None or field_bits is None:
            return None
        members_di.append(
            _derived_di_type(
                "DW_TAG_member",
                name=f"f{i}",
                base_type=field_di,
                offset_bits=offset_bits,
                size_bits=field_bits,
            )
        )
        llvm_str = _llvm_type_str(field_type)
        if llvm_str is None:
            return None
        llvm_member_types.append(llvm_str)
    type_name = ", ".join(str(t) for t in numba_type.types)
    llvm_type = ", ".join(llvm_member_types)
    return _composite_di_type(
        "DW_TAG_structure_type",
        name=f"Tuple({type_name}) ({{{llvm_type}}})",
        size_bits=total_size_bits,
        elements=members_di,
    )


def _record_di_type(numba_type):
    members_di = []
    for field_name in numba_type.fields:
        field_type = numba_type.typeof(field_name)
        field_di = _numba_type_to_di_type_str(field_type)
        field_bits = _type_size_bits(field_type)
        if field_di is None or field_bits is None:
            return None
        field_offset_bits = numba_type.offset(field_name) * _BYTE_SIZE_BITS
        members_di.append(
            _derived_di_type(
                "DW_TAG_member",
                name=field_name,
                base_type=field_di,
                offset_bits=field_offset_bits,
                size_bits=field_bits,
            )
        )
    return _composite_di_type(
        "DW_TAG_structure_type",
        name=str(numba_type),
        size_bits=numba_type.size * _BYTE_SIZE_BITS,
        elements=members_di,
    )


def _array_di_type(numba_type):
    elem = numba_type.dtype
    ndim = numba_type.ndim
    layout = numba_type.layout

    i8_di = _basic_di_type("i8", _BYTE_SIZE_BITS, "unsigned")
    i8_ptr_di = _derived_di_type(
        "DW_TAG_pointer_type",
        base_type=i8_di,
        size_bits=_POINTER_BITS,
    )

    member = lambda n, b, o, s: _derived_di_type(
        "DW_TAG_member",
        name=n,
        base_type=b,
        size_bits=s,
        offset_bits=o,
    )

    def field_debug_info(field_type):
        if isinstance(field_type, (types.MemInfoPointer, types.PyObject)):
            return i8_ptr_di, _POINTER_BITS, "i8*"
        if isinstance(field_type, types.CPointer):
            field_di = _cpointer_di_type(field_type)
            pointee_llvm = _llvm_type_str(field_type.dtype) or "i8"
            return field_di, _POINTER_BITS, f"{pointee_llvm}*"
        field_di = _numba_type_to_di_type_str(field_type)
        field_size = _type_size_bits(field_type)
        field_llvm = _llvm_type_str(field_type)
        return field_di, field_size, field_llvm

    members = []
    total_size = 0
    llvm_members = []
    for field_name, field_type in ArrayModel.get_members(numba_type):
        field_di, field_size, field_llvm = field_debug_info(field_type)
        if field_di is None or field_size is None or field_llvm is None:
            return None
        members.append(member(field_name, field_di, total_size, field_size))
        total_size += field_size
        llvm_members.append(field_llvm)
    llvm_struct = ", ".join(llvm_members)

    return _composite_di_type(
        "DW_TAG_structure_type",
        name=f"array({elem}, {ndim}d, {layout}) ({{{llvm_struct}}})",
        size_bits=total_size,
        elements=members,
    )


def _union_di_type(numba_type):
    variants = numba_type.types
    variant_sizes = [_type_size_bits(t) for t in variants]
    if not variant_sizes or any(bits is None for bits in variant_sizes):
        return None
    size_bits = max(variant_sizes)

    variant_members = []
    for tag, (variant_type, variant_bits) in enumerate(zip(variants, variant_sizes, strict=True)):
        variant_di = _numba_type_to_di_type_str(variant_type)
        if variant_di is None or variant_bits is None:
            return None
        variant_members.append(
            _derived_di_type(
                "DW_TAG_member",
                name=f"_{variant_type}",
                base_type=variant_di,
                size_bits=variant_bits,
                offset_bits=variant_bits,
                extra_data=tag,
            )
        )

    variant_key = ".".join(str(t) for t in variants)
    discriminator = _derived_di_type(
        "DW_TAG_member",
        name=f"discriminator-{variant_key}",
        base_type=_basic_di_type("uint8", _BYTE_SIZE_BITS, "unsigned"),
        size_bits=_BYTE_SIZE_BITS,
        flags="Artificial",
    )
    variant_part = _composite_di_type(
        "DW_TAG_variant_part",
        name="variant_part",
        size_bits=size_bits,
        elements=variant_members,
        identifier=f"variant-{variant_key}",
        discriminator=discriminator,
    )
    return _composite_di_type(
        "DW_TAG_structure_type",
        name="variant_wrapper_struct",
        size_bits=2 * size_bits,
        elements=[discriminator, variant_part],
        identifier=f"variant-wrapper-{variant_key}",
    )


def _numba_type_to_di_type_str(numba_type):
    """Map a Numba type to an MLIR #llvm.di_basic_type string."""
    numba_type = _strip_literal_type(numba_type)
    match numba_type:
        case types.Boolean():
            return _basic_di_type("bool", _BYTE_SIZE_BITS, "boolean")
        case types.Float(bitwidth=bw):
            return _basic_di_type(f"float{bw}", bw, "float")
        case types.Integer(signed=True, bitwidth=bw):
            return _basic_di_type(f"int{bw}", bw, "signed")
        case types.Integer(signed=False, bitwidth=bw):
            return _basic_di_type(f"uint{bw}", bw, "unsigned")
        case types.CPointer():
            return _cpointer_di_type(numba_type)
        case types.EnumMember():
            return _enum_member_di_type(numba_type)
        case types.NPDatetime():
            # NumPy stores datetime64 as signed int64.
            return _basic_di_type(f"datetime64[{numba_type.unit}]", _INT_LITERAL_BITS, "signed")
        case types.NPTimedelta():
            # NumPy stores timedelta64 as signed int64.
            return _basic_di_type(f"timedelta64[{numba_type.unit}]", _INT_LITERAL_BITS, "signed")
        case types.Complex(underlying_float=types.Float(bitwidth=bw)):
            return _complex_di_type(f"complex{bw * 2}", bw)
        case types.UniTuple():
            return _uni_tuple_di_type(numba_type)
        case types.BaseTuple():
            return _base_tuple_di_type(numba_type)
        case types.Record():
            return _record_di_type(numba_type)
        case types.Array():
            return _array_di_type(numba_type)
        case types.UnionType():
            return _union_di_type(numba_type)
        case Bfloat16():
            return _basic_di_type("__nv_bfloat16", _BFLOAT16_BITS, "float")
        case GridGroup():
            # GridGroup is an opaque cooperative-groups handle.
            return _basic_di_type("GridGroup", _GRID_GROUP_BITS, "unsigned")
        case _:
            return None


class DIBuilder:
    """Builds LLVM debug info MLIR attributes.

    All DI attributes are collected as text fragments and parsed in a single
    Module.parse call so that ``distinct[N]<>`` IDs resolve to the same
    DistinctAttr objects.
    """

    def __init__(self, loc, func_name, *, line_only=False, opt=True, context=None):
        self.context = context or numba_cuda_mlir_context.get_context()
        self._local_vars: list[_DILocalVarInfo] = []
        self.di_subprogram = None
        self.di_local_vars: dict[str, ir.Attribute] = {}
        self.di_expression = None
        self.arg_names: set[str] = set()

        raw_path = getattr(loc, "filename", None)
        if not raw_path or not raw_path.strip():
            self.valid = False
            return
        if raw_path.startswith("<") and raw_path.endswith(">"):
            # Synthetic paths like "<ipython-input-N>" (Jupyter cells) have no
            # file on disk, but IPython registers their source in Python's
            # linecache so DWARF can still reference meaningful line numbers.
            if not linecache.getlines(raw_path):
                self.valid = False
                return

        self.valid = True
        filename = os.path.basename(raw_path)
        directory = os.path.dirname(raw_path) or "."

        # Caller guarantees at least one of debug or lineinfo is True.
        emission = "DebugDirectivesOnly" if line_only else "Full"

        self._di_file_str = f'#llvm.di_file<"{filename}" in "{directory}">'
        self._di_cu_str = (
            f"#llvm.di_compile_unit<"
            f"id = distinct[0]<>, "
            f"sourceLanguage = DW_LANG_C_plus_plus, "
            f"file = {self._di_file_str}, "
            f'producer = "clang (numba-cuda-mlir)", '
            f"isOptimized = {str(opt).lower()}, "
            f"emissionKind = {emission}"
            f">"
        )
        subprogram_flags = "Definition|Optimized" if opt else "Definition"
        self._di_subroutine_type_str = "#llvm.di_subroutine_type<types = #llvm.di_null_type>"
        self._di_sp_str = (
            f"#llvm.di_subprogram<"
            f"id = distinct[1]<>, "
            f"compileUnit = {self._di_cu_str}, "
            f"scope = {self._di_file_str}, "
            f'name = "{func_name}", '
            f"file = {self._di_file_str}, "
            f"type = {self._di_subroutine_type_str}, "
            f"line = {loc.line}, "
            f"scopeLine = {loc.line}, "
            f'subprogramFlags = "{subprogram_flags}"'
            f">"
        )

    def add_local_variable(self, name, line, numba_type, *, arg_index=None):
        """Register a local variable for debug info emission."""
        if not self.valid:
            return
        type_str = _numba_type_to_di_type_str(numba_type)
        if type_str is None:
            return
        if arg_index is not None:
            self.arg_names.add(name)
        self._local_vars.append(_DILocalVarInfo(name, line, type_str, arg_index))

    def build(self):
        """Parse all DI metadata in a single call for consistent distinct IDs.

        Returns the di_subprogram attribute, or None if valid is False.
        """
        if not self.valid:
            return None

        attrs = {"numba_cuda_mlir.di_sp": self._di_sp_str}
        var_keys = []
        for i, var in enumerate(self._local_vars):
            arg_field = f", arg = {var.arg_index}" if var.arg_index is not None else ""
            var_str = (
                f"#llvm.di_local_variable<"
                f"scope = {self._di_sp_str}, "
                f'name = "{var.name}"{arg_field}, '
                f"file = {self._di_file_str}, "
                f"line = {var.line}, "
                f"type = {var.type_str}"
                f">"
            )
            key = f"numba_cuda_mlir.di_var_{var.name}_{i}"
            attrs[key] = var_str
            var_keys.append((var.name, key))
        attrs["numba_cuda_mlir.di_expr"] = "#llvm.di_expression<>"

        attr_strs = ", ".join(f"{k} = {v}" for k, v in attrs.items())
        module_str = f"module attributes {{{attr_strs}}} {{}}"

        with self.context:
            helper = ir.Module.parse(module_str)
            self.di_subprogram = helper.operation.attributes["numba_cuda_mlir.di_sp"]
            self.di_expression = helper.operation.attributes["numba_cuda_mlir.di_expr"]
            for name, key in var_keys:
                self.di_local_vars[name] = helper.operation.attributes[key]

        return self.di_subprogram
