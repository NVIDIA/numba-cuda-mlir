# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda.types import Type


class VectorType(Type):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        shape_str = "x".join(str(d) for d in self.shape)
        super().__init__(f"vector[{dtype}, {shape_str}]")

    @property
    def key(self):
        return (self.dtype, self.shape)

    @property
    def length(self):
        result = 1
        for d in self.shape:
            result *= d
        return result
