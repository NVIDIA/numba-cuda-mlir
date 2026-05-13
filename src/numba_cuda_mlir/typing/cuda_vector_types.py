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
    _vector_type_stubs,
)

registry = Registry()

# Map base type names to numba types
BASE_TYPE_MAP = {
    "int8": types.int8,
    "int16": types.int16,
    "int32": types.int32,
    "int64": types.int64,
    "uint8": types.uint8,
    "uint16": types.uint16,
    "uint32": types.uint32,
    "uint64": types.uint64,
    "float32": types.float32,
    "float64": types.float64,
}

# Attribute index mapping
ATTR_INDEX = {"x": 0, "y": 1, "z": 2, "w": 3}


def get_vector_type_for_stub(stub_class):
    """Get the VectorType corresponding to a stub class."""
    base_type = BASE_TYPE_MAP[stub_class._base_type_name]
    return VectorType(base_type, (stub_class._num_elements,))


_constructor_template_cache = {}


def make_constructor_template(stub_class):
    """Create a typing template for a vector type constructor (cached)."""
    if stub_class in _constructor_template_cache:
        return _constructor_template_cache[stub_class]

    base_type = BASE_TYPE_MAP[stub_class._base_type_name]
    num_elements = stub_class._num_elements
    result_type = VectorType(base_type, (num_elements,))

    class ConstructorTemplate(AbstractTemplate):
        key = stub_class

        def generic(self, args, kws):
            if kws:
                return None

            # Single argument cases (must check before multi-arg scalar case)
            if len(args) == 1:
                arg = args[0]
                # Copy from compatible vector (same number of elements)
                if isinstance(arg, VectorType):
                    if arg.length == num_elements:
                        return signature(result_type, arg)
                # Scalar broadcast
                if isinstance(arg, (types.Integer, types.Float)):
                    return signature(result_type, arg)

            # All scalar arguments matching element count
            if len(args) == num_elements:
                all_scalars = all(
                    isinstance(arg, (types.Integer, types.Float, types.Boolean)) for arg in args
                )
                if all_scalars:
                    return signature(result_type, *args)

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
                return signature(result_type, *args)

            return None

    ConstructorTemplate.__name__ = f"{stub_class.__name__}ConstructorTemplate"
    _constructor_template_cache[stub_class] = ConstructorTemplate
    return ConstructorTemplate


# Register all vector type constructors
for stub in _vector_type_stubs:
    template = make_constructor_template(stub)
    registry.register(template)
    registry.register_global(stub, types.Function(template))


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


# Register typeof_impl for all vector type stubs so they can be used as closure variables
from numba_cuda_mlir.numba_cuda.typing import typeof


def _make_typeof_impl(stub_class):
    """Create a typeof implementation for a vector type stub."""
    template = make_constructor_template(stub_class)

    def typeof_stub(val, c):
        return types.Function(template)

    return typeof_stub


for stub in _vector_type_stubs:
    typeof.typeof_impl.register(stub)(_make_typeof_impl(stub))
