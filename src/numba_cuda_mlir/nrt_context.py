# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR NRT context: provides incref/decref emission for the MLIR lowering pass.

Mirrors numba-cuda's NRTContext but emits MLIR LLVM dialect ops instead of
llvmlite IR.  The key entry points are ``incref()`` and ``decref()`` which
walk a type's data model to find all contained NRT MemInfos and emit the
appropriate NRT_incref / NRT_decref calls.

NRT refcounting contract for expression lowerings
--------------------------------------------------
Every Numba IR variable is independently del'd by liveness analysis.  ``del``
calls ``decref``, which walks into contained NRT members via ``traverse_mlir``.
To keep refcounts balanced, every expression lowering (registered builder,
overload impl, getattr impl, etc.) that **stores its result** must follow
this rule:

  * If the expression **creates a fresh NRT object** (e.g. allocates a new
    managed_udf_string via a shim call), the birth gives rc=1.  That single
    reference belongs to the target variable.  **No incref needed.**

  * If the expression **copies or extracts an existing NRT reference** from
    one Numba IR variable into another (struct wrapping, struct field
    extraction, pass-through alias), it must call ``builder.incref(typ, val)``
    on the result before ``store_var``.  Without this, two variables alias the
    same inner NRT pointer, both get del'd, and the second decref is a
    use-after-free.

This matches the Numba CPU convention: ``lower_expr`` in
``numba.core.lowering`` increfs results of ``build_tuple``, ``cast``,
``pair_first``/``pair_second``, and variable-to-variable assignments.
Function call results are NOT incref'd by the framework — the impl/builder
is responsible.

Examples of expressions that NEED incref (copiers/extractors):
  - Struct constructor wrapping an existing NRT member (e.g. NRTWrapperType(inner))
  - Pass-through alias (e.g. pack_return just forwarding a value)
  - Field extraction from a struct (e.g. getattr .inner on NRTWrapperType)
  - Tuple construction from existing NRT elements (handled by numba_cuda_mlir core)
  - Cast of an NRT-managed value (handled by numba_cuda_mlir core)

Examples that do NOT need incref (producers):
  - Shim calls that allocate new NRT objects (concat, upper, lower, etc.)
  - Any operation whose result contains only freshly-born NRT pointers
"""

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import func, llvm
from numba_cuda_mlir.lowering_utilities import get_or_insert_function
from numba_cuda_mlir.types import BaseTuple


def _get_or_declare_nrt_func(module, name):
    """Declare ``void @name(ptr)`` in *module* if not already present."""

    func_type = ir.FunctionType.get([llvm.PointerType.get()], [])
    return get_or_insert_function(name, func_type, module)


def _emit_nrt_call(module, funcname, meminfo_val):
    """Emit a call to NRT_incref or NRT_decref on a single meminfo pointer."""
    callee = _get_or_declare_nrt_func(module, funcname)
    func.call(result=[], callee=callee.name.value, operands_=[meminfo_val])


class MLIRNRTContext:
    """NRT helpers for the MLIR lowering pass.

    Parameters
    ----------
    data_model_manager : DataModelManager
        The numba_cuda_mlir data model manager used to look up models by Numba type.
    """

    def __init__(self, data_model_manager):
        self._dmm = data_model_manager

    # ------------------------------------------------------------------
    # Public API used by MLIRLower
    # ------------------------------------------------------------------

    def incref(self, module, typ, value):
        """Emit NRT_incref for every MemInfo reachable from *value*."""
        for mi in self._get_meminfos(typ, value):
            _emit_nrt_call(module, "NRT_incref", mi)

    def decref(self, module, typ, value):
        """Emit NRT_decref for every MemInfo reachable from *value*."""
        for mi in self._get_meminfos(typ, value):
            _emit_nrt_call(module, "NRT_decref", mi)

    def type_has_nrt_meminfo(self, typ):
        """Return True if *typ* (or any contained type) needs NRT tracking."""
        try:
            model = self._dmm.lookup(typ)
        except KeyError:
            return False
        if model.has_nrt_meminfo():
            return True
        if isinstance(typ, BaseTuple):
            return any(self.type_has_nrt_meminfo(t) for t in typ)
        if hasattr(model, "traverse_mlir"):
            for member_typ, _getter in model.traverse_mlir():
                if self.type_has_nrt_meminfo(member_typ):
                    return True
        if model.contains_nrt_meminfo():
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_meminfos(self, typ, value):
        """Yield all MemInfo pointer values reachable from *value*.

        Walks the data model tree exactly like numba-cuda's
        ``NRTContext.get_meminfos``.
        """
        try:
            model = self._dmm.lookup(typ)
        except KeyError:
            return

        if model.has_nrt_meminfo():
            mi = model.get_nrt_meminfo(value)
            if mi is not None:
                yield mi

        if hasattr(model, "traverse_mlir"):
            for member_typ, getter in model.traverse_mlir():
                member_val = getter(value)
                yield from self._get_meminfos(member_typ, member_val)
