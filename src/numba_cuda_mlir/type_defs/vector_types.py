# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.types import Type


class VectorType(Type):
    def __init__(self, dtype, shape, alignment=None):
        self._dtype = dtype
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = tuple(shape)

        if alignment is None:
            length = 1
            for d in self._shape:
                length *= d
            align_length = 1 if length == 3 else length
            self._alignment = min(16, (align_length * self._dtype.bitwidth) // 8)
            self._explicit_alignment = False
        else:
            self._alignment = alignment
            self._explicit_alignment = True

        shape_str = "x".join(str(d) for d in self._shape)
        name = f"{dtype}x{shape_str}"
        if self._explicit_alignment:
            name += f"_{alignment}a"
        super().__init__(name)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def alignment(self):
        return self._alignment

    @property
    def key(self):
        return (self._dtype, self._shape, self._alignment, self._explicit_alignment)

    @property
    def length(self):
        result = 1
        for d in self._shape:
            result *= d
        return result
