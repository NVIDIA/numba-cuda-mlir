# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.types import Type


class VectorType(Type):
    def __init__(self, dtype, shape):
        self._dtype = dtype
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = tuple(shape)
        shape_str = "x".join(str(d) for d in self._shape)
        name = f"{dtype}x{shape_str}"
        super().__init__(name)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def key(self):
        return (self._dtype, self._shape)

    @property
    def length(self):
        result = 1
        for d in self._shape:
            result *= d
        return result
