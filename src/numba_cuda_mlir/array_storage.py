# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logical array elements and the units used by their storage descriptors."""

from dataclasses import dataclass

import numpy as np

from numba_cuda_mlir.numba_cuda import types


def array_is_aligned(dtype, pointer, strides):
    """Whether typed storage can express the array and its alignment guarantees.

    Conservatively classify non-integral element strides as unaligned too: a
    naturally aligned complex array can still require byte-based addressing.
    """
    dtype = np.dtype(dtype)
    return (
        (not dtype.fields or dtype.isalignedstruct)
        and dtype.itemsize > 0
        and pointer % max(1, dtype.alignment) == 0
        and (strides is None or all(s % dtype.itemsize == 0 for s in strides))
    )


@dataclass(frozen=True)
class ArrayStorageLayout:
    """Storage policy; shapes always count logical elements, never bytes.

    Unaligned arrays use i8 memrefs whose offsets and strides count bytes.
    Aligned arrays retain typed memrefs and element-unit descriptors.
    """

    array_type: types.Array

    @property
    def byte_addressed(self):
        return not self.array_type.aligned or isinstance(
            self.array_type.dtype, (types.CharSeq, types.UnicodeCharSeq)
        )

    @property
    def itemsize(self):
        from numba_cuda_mlir.numba_cuda.np.numpy_support import as_dtype
        from numba_cuda_mlir.numba_cuda.core.errors import NumbaNotImplementedError

        dtype = self.array_type.dtype
        from numba_cuda_mlir.type_defs.vector_types import VectorType

        if isinstance(dtype, VectorType):
            scalar = ArrayStorageLayout(types.Array(dtype.dtype, 1, "C"))
            return scalar.itemsize * dtype.length
        try:
            return as_dtype(dtype).itemsize
        except (NotImplementedError, NumbaNotImplementedError, TypeError):
            from numba_cuda_mlir.type_defs.float_types import numeric_storage_bitwidth

            width = numeric_storage_bitwidth(dtype)
            if width is None:
                raise
            return width // 8

    @property
    def unit_bytes(self):
        return 1 if self.byte_addressed else self.itemsize

    @property
    def element_type(self):
        from numba_cuda_mlir._mlir.extras import types as T
        from numba_cuda_mlir.lowering_utilities import get_storage_type

        if self.byte_addressed:
            return T.i8()
        if isinstance(self.array_type.dtype, types.Record):
            return T.vector(self.itemsize, element_type=T.i8())
        return get_storage_type(self.array_type.dtype)

    def memref_type(self, *, memory_space=None):
        from numba_cuda_mlir._mlir import ir

        dyn = ir.ShapedType.get_dynamic_size()
        layout = ir.StridedLayoutAttr.get(dyn, [dyn] * self.array_type.ndim)
        return ir.MemRefType.get(
            [dyn] * self.array_type.ndim,
            self.element_type,
            layout=layout,
            memory_space=memory_space,
        )

    @property
    def alignment(self):
        from numba_cuda_mlir.numba_cuda.np.numpy_support import as_dtype

        if not self.array_type.aligned:
            return 1
        dtype = self.array_type.dtype
        from numba_cuda_mlir.type_defs.vector_types import VectorType

        if isinstance(dtype, VectorType):
            return ArrayStorageLayout(types.Array(dtype.dtype, 1, "C")).alignment
        from numba_cuda_mlir.numba_cuda.core.errors import NumbaNotImplementedError

        try:
            return max(1, as_dtype(dtype).alignment)
        except (NotImplementedError, NumbaNotImplementedError, TypeError):
            return 1


def record_field_array_type(array_type, field):
    """Project a record field without losing mutability or storage guarantees."""
    from numba_cuda_mlir.numba_cuda.np.numpy_support import as_dtype

    dtype = field.type
    ndim = array_type.ndim
    if isinstance(dtype, types.NestedArray):
        ndim += dtype.ndim
        dtype = dtype.dtype
    itemsize = as_dtype(dtype).itemsize
    aligned = (
        array_type.aligned
        and itemsize > 0
        and array_type.dtype.size % itemsize == 0
        and field.offset % itemsize == 0
    )
    return types.Array(dtype, ndim, "A", readonly=not array_type.mutable, aligned=aligned)
