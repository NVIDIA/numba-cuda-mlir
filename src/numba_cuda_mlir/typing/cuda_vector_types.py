# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Typing support for CUDA vector types (float32x4, int32x2, etc.)
"""

from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    Registry,
    signature,
)
from numba_cuda_mlir import types
from numba_cuda_mlir.type_defs.vector_types import VectorType
from numba_cuda_mlir.cuda.vector_types import (
    _vector_types,
)

from numba_cuda_mlir.numba_cuda.types import Callable, DTypeSpec
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir.numba_cuda.datamodel.models import OpaqueModel
from numba_cuda_mlir.numba_cuda.typing import typeof


class VectorTypeClass(Callable, DTypeSpec):
    """
    A class representing a vector type that can be called to construct instances
    and used as a dtype specifier.
    """

    def __init__(self, instance_type, constructor_template):
        self.instance_type = instance_type
        self._template = constructor_template
        self.typing_key = instance_type
        super().__init__(f"class({instance_type})")

    @property
    def dtype(self):
        return self.instance_type

    def get_call_type(self, context, args, kws):
        return self._template(context).apply(args, kws)

    def get_call_signatures(self):
        sigs = getattr(self._template, "cases", [])
        is_param = hasattr(self._template, "generic")
        return sigs, is_param

    def get_impl_key(self, sig):
        return self.typing_key


registry = Registry()

# Attribute index mapping
ATTR_INDEX = {"x": 0, "y": 1, "z": 2, "w": 3}


register_model(VectorTypeClass)(OpaqueModel)


_constructor_template_cache = {}


def make_constructor_template(vec_type):
    """Create a typing template for a vector type constructor (cached)."""
    if vec_type in _constructor_template_cache:
        return _constructor_template_cache[vec_type]

    num_elements = vec_type.length

    class ConstructorTemplate(AbstractTemplate):
        key = vec_type

        def generic(self, args, kws):
            if kws:
                return None

            # Single argument cases (must check before multi-arg scalar case)
            if len(args) == 1:
                arg = args[0]
                # Copy from compatible vector (same number of elements)
                if isinstance(arg, VectorType):
                    if arg.length == num_elements:
                        return signature(vec_type, arg)
                # Scalar broadcast
                if isinstance(arg, (types.Integer, types.Float)):
                    return signature(vec_type, arg)

            # All scalar arguments matching element count
            if len(args) == num_elements:
                all_scalars = all(
                    isinstance(arg, (types.Integer, types.Float, types.Boolean)) for arg in args
                )
                if all_scalars:
                    return signature(vec_type, *args)

            # Mixed vector/scalar arguments
            total_elements = 0
            for arg in args:
                if isinstance(arg, VectorType):
                    total_elements += arg.length
                elif isinstance(arg, (types.Integer, types.Float, types.Boolean)):
                    total_elements += 1
                else:
                    return None

            if total_elements == num_elements:
                return signature(vec_type, *args)

            return None

    ConstructorTemplate.__name__ = f"{vec_type.name}ConstructorTemplate"
    _constructor_template_cache[vec_type] = ConstructorTemplate
    return ConstructorTemplate


# Register all vector type constructors
for vec_type in _vector_types:
    template = make_constructor_template(vec_type)
    registry.register(template)
    registry.register_global(vec_type, VectorTypeClass(vec_type, template))


@registry.register_attr
class VectorTypeAttributeTemplate(AttributeTemplate):
    """Typing for vector type attribute access (.x, .y, .z, .w)."""

    key = VectorType

    def resolve_x(self, ty):
        if ty.length >= 1:
            return ty.dtype

    def resolve_y(self, ty):
        if ty.length >= 2:
            return ty.dtype

    def resolve_z(self, ty):
        if ty.length >= 3:
            return ty.dtype

    def resolve_w(self, ty):
        if ty.length >= 4:
            return ty.dtype


# Register typeof_impl for all vector types so they can be used as closure variables


@typeof.typeof_impl.register(VectorType)
def typeof_vector_type(val, c):
    """Create a typeof implementation for a vector type."""
    template = make_constructor_template(val)
    return VectorTypeClass(val, template)
