# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import linecache
import os

from typing import NamedTuple

from numba import types
from cusimt._mlir import ir
from cusimt.lowering_utilities import context as cusimt_context


class _DILocalVarInfo(NamedTuple):
    name: str
    line: int
    type_str: str
    arg_index: int | None


_BYTE_SIZE_BITS = 8
_INT_LITERAL_BITS = 64


def _di_basic_type(name, bits, encoding):
    return (
        f'#llvm.di_basic_type<tag = DW_TAG_base_type, name = "{name}", '
        f"sizeInBits = {bits}, encoding = DW_ATE_{encoding}>"
    )


def _numba_type_to_di_type_str(numba_type):
    """Map a Numba type to an MLIR #llvm.di_basic_type string."""
    match numba_type:
        case types.Boolean() | types.BooleanLiteral():
            return _di_basic_type("bool", _BYTE_SIZE_BITS, "boolean")
        case types.Float(bitwidth=bw):
            return _di_basic_type(f"float{bw}", bw, "float")
        case types.Integer(signed=True, bitwidth=bw):
            return _di_basic_type(f"int{bw}", bw, "signed")
        case types.Integer(signed=False, bitwidth=bw):
            return _di_basic_type(f"uint{bw}", bw, "unsigned")
        case types.IntegerLiteral():
            return _di_basic_type("int64", _INT_LITERAL_BITS, "signed")
        case _:
            return None


class DIBuilder:
    """Builds LLVM debug info MLIR attributes.

    All DI attributes are collected as text fragments and parsed in a single
    Module.parse call so that ``distinct[N]<>`` IDs resolve to the same
    DistinctAttr objects.
    """

    def __init__(self, loc, func_name, *, line_only=False, opt=True, context=None):
        self.context = context or cusimt_context.get_context()
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
            f'producer = "clang (cuSIMT)", '
            f"isOptimized = {str(opt).lower()}, "
            f"emissionKind = {emission}"
            f">"
        )
        subprogram_flags = "Definition|Optimized" if opt else "Definition"
        self._di_subroutine_type_str = (
            "#llvm.di_subroutine_type<types = #llvm.di_null_type>"
        )
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

        attrs = {"cusimt.di_sp": self._di_sp_str}
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
            key = f"cusimt.di_var_{var.name}_{i}"
            attrs[key] = var_str
            var_keys.append((var.name, key))
        attrs["cusimt.di_expr"] = "#llvm.di_expression<>"

        attr_strs = ", ".join(f"{k} = {v}" for k, v in attrs.items())
        module_str = f"module attributes {{{attr_strs}}} {{}}"

        with self.context:
            helper = ir.Module.parse(module_str)
            self.di_subprogram = helper.operation.attributes["cusimt.di_sp"]
            self.di_expression = helper.operation.attributes["cusimt.di_expr"]
            for name, key in var_keys:
                self.di_local_vars[name] = helper.operation.attributes[key]

        return self.di_subprogram
