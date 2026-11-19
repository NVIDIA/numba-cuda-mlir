# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-driven function layouts with MLIR's compile-time-only leaves erased."""

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.numba_cuda.datamodel.packer import (
    ArgPacker as _ArgPacker,
    _flatten,
    _Unflattener,
)


def _map_layout(layout, value, leaf):
    if isinstance(layout, (tuple, list)):
        return tuple(_map_layout(ty, val, leaf) for ty, val in zip(layout, value))
    return leaf(layout, value)


class _Model:
    def __init__(self, model, representation):
        self.model = model
        self.representation = representation
        self.layout = getattr(model, f"get_{representation}_type")()

    def get_argument_type(self):
        def erase(ty, value):
            return () if isinstance(ty, ir.NoneType) else ty

        return _map_layout(self.layout, self.layout, erase)

    def as_argument(self, builder, value):
        from numba_cuda_mlir.lowering_utilities import convert

        value = getattr(self.model, f"as_{self.representation}")(builder, value)

        def materialize(ty, val):
            return () if isinstance(ty, ir.NoneType) else convert(val, ty)

        return _map_layout(self.layout, value, materialize)

    def from_argument(self, builder, value):
        def restore(ty, val):
            return ir.NoneType.get() if isinstance(ty, ir.NoneType) else val

        value = _map_layout(self.layout, value, restore)
        return getattr(self.model, f"from_{self.representation}")(builder, value)


class ArgPacker(_ArgPacker):
    """Share declared layouts, outgoing conversion and incoming reconstruction.

    Unlike LLVM's argument packer, MLIR has NoneType leaves in otherwise native
    tuple/memref representations. Erase these leaves from the *layout* as well
    as the operands, then restore their placeholders before model conversion.
    Return layouts use the same machinery with the models' return methods, so
    storage conversions (notably boolean i1/i8) stay distinct from arguments.
    """

    def __init__(self, dmm, fe_args, *, representation="argument"):
        self._nargs = len(fe_args)
        self._dm_args = [_Model(dmm.lookup(ty), representation) for ty in fe_args]
        layouts = [model.get_argument_type() for model in self._dm_args]
        self._unflattener = _Unflattener(layouts)
        self._be_args = list(_flatten(layouts))
