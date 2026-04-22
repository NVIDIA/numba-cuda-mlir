# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt._mlir.dialects.llvm import *  # noqa: F401,F403
from cusimt._mlir.dialects import llvm as _llvm
from cusimt._mlir import ir


def ptr():
    """Shorthand for ``llvm.PointerType.get()``."""
    return _llvm.PointerType.get()


def insertvalue(container, value, position, **kwargs):
    """Like ``llvm.insertvalue`` but *position* can be a plain ``int`` or ``list[int]``."""
    if isinstance(position, int):
        position = ir.DenseI64ArrayAttr.get([position])
    elif isinstance(position, (list, tuple)):
        position = ir.DenseI64ArrayAttr.get(list(position))
    return _llvm.insertvalue(
        container=container, value=value, position=position, **kwargs
    )


def addressof(global_name, *, res=None, loc=None, ip=None):
    """Return a pointer to the named global.

    *global_name* can be a plain ``str`` or a ``FlatSymbolRefAttr``.
    *res* defaults to ``!llvm.ptr`` if not given.
    """
    if res is None:
        res = ptr()
    return _llvm.mlir_addressof(res, global_name, loc=loc, ip=ip)
