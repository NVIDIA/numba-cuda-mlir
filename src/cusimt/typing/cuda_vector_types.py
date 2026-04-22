# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Typing support for CUDA vector types (float32x4, int32x2, etc.)
"""

from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    Registry,
    signature,
)
from cusimt import types
from cusimt.type_defs.vector_types import VectorType
from cusimt.cuda.vector_types import (
    _vector_type_stubs,
    vector_type_stubs_by_name,
    vector_type_stubs_by_alias,
    VectorTypeStub,
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
                    isinstance(arg, (types.Integer, types.Float, types.Boolean))
                    for arg in args
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
from numba.core.typing import typeof


def _make_typeof_impl(stub_class):
    """Create a typeof implementation for a vector type stub."""
    template = make_constructor_template(stub_class)

    def typeof_stub(val, c):
        return types.Function(template)

    return typeof_stub


for stub in _vector_type_stubs:
    typeof.typeof_impl.register(stub)(_make_typeof_impl(stub))


def _make_nc_constructor_template(nc_stub, our_stub):
    """Create a typing template for a numba-cuda vector type stub."""
    base_type = BASE_TYPE_MAP[our_stub._base_type_name]
    num_elements = our_stub._num_elements
    result_type = VectorType(base_type, (num_elements,))

    class NCConstructorTemplate(AbstractTemplate):
        key = nc_stub

        def generic(self, args, kws):
            if kws:
                return None
            # Single argument cases (must check before multi-arg scalar case)
            if len(args) == 1:
                arg = args[0]
                if isinstance(arg, VectorType) and arg.length == num_elements:
                    return signature(result_type, arg)
                if isinstance(arg, (types.Integer, types.Float)):
                    return signature(result_type, arg)
            # All scalar arguments
            if len(args) == num_elements:
                all_scalars = all(
                    isinstance(arg, (types.Integer, types.Float, types.Boolean))
                    for arg in args
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

    NCConstructorTemplate.__name__ = f"NC{nc_stub.__name__}ConstructorTemplate"
    return NCConstructorTemplate


# Also register templates and typeof_impl for numba-cuda's stubs
def _register_numba_cuda_stubs():
    """Register typing templates and typeof_impl for numba-cuda's vector type stubs."""
    try:
        from cusimt.numba_cuda.stubs import _vector_type_stubs as numba_cuda_stubs
    except ImportError:
        return

    for nc_stub in numba_cuda_stubs:
        name = nc_stub.__name__
        our_stub = vector_type_stubs_by_name.get(name)
        if our_stub is None:
            continue

        # Check if already cached
        if nc_stub in _constructor_template_cache:
            template = _constructor_template_cache[nc_stub]
        else:
            template = _make_nc_constructor_template(nc_stub, our_stub)
            _constructor_template_cache[nc_stub] = template

        # Register the template
        registry.register(template)
        registry.register_global(nc_stub, types.Function(template))

        # Also register typeof_impl for closure capture support
        def make_typeof(tmpl):
            def typeof_numba_cuda_stub(val, c):
                return types.Function(tmpl)

            return typeof_numba_cuda_stub

        typeof.typeof_impl.register(nc_stub)(make_typeof(template))


_register_numba_cuda_stubs()
