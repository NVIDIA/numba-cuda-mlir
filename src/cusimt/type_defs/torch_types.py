# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Assume we already have torch importable--we should not be imported unless
this has already been checked.
"""

from ast import Not
from numba.core.typing import typeof
from numba import types
from cusimt.lowering_utilities.type_conversions import to_numba_type
import torch
from numba.core.typing.templates import AttributeTemplate, Registry

registry = Registry()


@typeof.typeof_impl.register(torch.Tensor)
def typeof_torch_tensor(val, c):
    dtype = to_numba_type(val.dtype)
    ty = types.Array(dtype=dtype, ndim=len(val.shape), layout="C")
    return ty


@registry.register_attr
class Torch_stub_resolver(AttributeTemplate):
    key = types.Module(torch)

    def resolve(self, mod, attrname):
        if not hasattr(torch, attrname):
            raise AttributeError(f"torch has no attribute {attrname}")

        attr = getattr(torch, attrname)
        if isinstance(attr, torch.dtype):
            numba_dtype = to_numba_type(attr)
            return types.DType(numba_dtype)

        raise NotImplementedError(f"Unhandled torch attribute {attrname}")
