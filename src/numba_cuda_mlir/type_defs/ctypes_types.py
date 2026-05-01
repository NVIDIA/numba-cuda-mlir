# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes
import operator
from numba_cuda_mlir.lowering_utilities.type_conversions import to_numba_type
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    AbstractTemplate,
    Registry,
    bound_function,
    signature,
)
from numba_cuda_mlir import types


class CTypesType(types.Type):
    def __init__(self, ctype):
        self.ctype = ctype
        super().__init__(f"CTypesType({ctype.__name__})")

    def __repr__(self):
        return f"CTypesType({self.ctype})"
