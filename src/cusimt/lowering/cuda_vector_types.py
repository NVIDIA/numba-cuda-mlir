# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lowering support for CUDA vector types (float32x4, int32x2, etc.)
"""

from cusimt.mlir_lowering_registry import MLIRLoweringRegistry

registry = MLIRLoweringRegistry()
lower = registry.lower
lower_getattr = registry.lower_getattr
from cusimt.mlir_lowering import MLIRLower
from cusimt.lowering_utilities import convert
from cusimt.lowering_utilities.type_conversions import to_mlir_type
from cusimt.cuda.vector_types import _vector_type_stubs, VectorTypeStub
from cusimt.type_defs.vector_types import VectorType
from cusimt import types
from cusimt._mlir.dialects import vector, arith
from cusimt._mlir import ir
from typing import Any

ATTR_INDEX = {"x": 0, "y": 1, "z": 2, "w": 3}


def _build_vector_from_scalars(scalars: list, vec_type: ir.VectorType) -> ir.Value:
    """Build an MLIR vector from a list of scalar values."""
    elem_type = vec_type.element_type

    # Convert all scalars to the correct element type
    converted = [convert(s, elem_type) for s in scalars]

    # Use vector.from_elements to build the vector
    return vector.from_elements(vec_type, converted)


def _extract_vector_elements(vec: ir.Value) -> list:
    """Extract all elements from an MLIR vector."""
    vec_type = vec.type
    num_elements = vec_type.shape[0]
    elements = []
    for i in range(num_elements):
        # Use static position for extraction (empty dynamic_position list)
        elem = vector.extract(vec, [], [i])
        elements.append(elem)
    return elements


def make_constructor_lowering(stub_class):
    """Create a lowering function for a vector type constructor."""

    def constructor_lowering(lower_ctx: MLIRLower, target, args: list[Any], kwargs):
        target_type = lower_ctx.get_numba_type(target.name)
        vec_type = to_mlir_type(target_type)
        elem_type = vec_type.element_type
        num_elements = vec_type.shape[0]

        # Collect all scalar elements from args (may be scalars or vectors)
        scalars = []
        for arg in args:
            val = lower_ctx.load_var(arg)
            arg_type = lower_ctx.get_numba_type(arg.name)

            if isinstance(arg_type, VectorType):
                # Extract elements from input vector
                scalars.extend(_extract_vector_elements(val))
            else:
                # Scalar value
                scalars.append(val)

        # Handle broadcast case (single scalar -> all elements)
        if len(scalars) == 1 and num_elements > 1:
            scalars = scalars * num_elements

        result = _build_vector_from_scalars(scalars, vec_type)
        lower_ctx.store_var(target, result)

    return constructor_lowering


def make_scalar_constructor_lowering(stub_class, num_scalars):
    """Create a lowering for constructor with specific number of scalar args."""
    lowering_fn = make_constructor_lowering(stub_class)

    # Generate the type signature for this overload
    base_type_name = stub_class._base_type_name
    if "float" in base_type_name:
        scalar_type = types.Float
    elif "uint" in base_type_name:
        scalar_type = types.Integer
    else:
        scalar_type = types.Integer

    return lowering_fn


# Register lowerings for all vector type constructors
for stub in _vector_type_stubs:
    num_elements = stub._num_elements
    lowering_fn = make_constructor_lowering(stub)

    # Register for various argument patterns
    # All scalars - include both concrete types and base types for literal matching
    int_types = [
        types.int8,
        types.int16,
        types.int32,
        types.int64,
        types.Integer,
        types.IntegerLiteral,
    ]
    for int_type in int_types:
        lower(stub, *([int_type] * num_elements))(lowering_fn)
        # Single scalar broadcast
        lower(stub, int_type)(lowering_fn)

    uint_types = [types.uint8, types.uint16, types.uint32, types.uint64]
    for uint_type in uint_types:
        lower(stub, *([uint_type] * num_elements))(lowering_fn)
        lower(stub, uint_type)(lowering_fn)

    float_types = [types.float32, types.float64, types.Float]
    for float_type in float_types:
        lower(stub, *([float_type] * num_elements))(lowering_fn)
        lower(stub, float_type)(lowering_fn)

    # Boolean (for integer vectors)
    lower(stub, *([types.boolean] * num_elements))(lowering_fn)

    # Vector copy/conversion (single vector arg with same element count)
    for other_stub in _vector_type_stubs:
        if other_stub._num_elements == num_elements:
            other_vec_type = VectorType(
                getattr(types, other_stub._base_type_name),
                (other_stub._num_elements,),
            )
            lower(stub, other_vec_type)(lowering_fn)

    # Mixed vector+scalar patterns
    # Generate all valid combinations of vectors and scalars that sum to num_elements
    base_type_name = stub._base_type_name
    base_type = getattr(types, base_type_name)
    scalar_types = (
        [base_type, types.Float]
        if "float" in base_type_name
        else [base_type, types.Integer]
    )

    # Helper: get all vector types with n elements for this base type
    def get_vec_type(n):
        return VectorType(base_type, (n,))

    # Register patterns with vectors of smaller sizes
    for scalar_type in scalar_types:
        if num_elements >= 2:
            # vec1 + scalar, scalar + vec1, vec1 + vec1
            vec1 = get_vec_type(1)
            lower(stub, vec1, scalar_type)(lowering_fn)
            lower(stub, scalar_type, vec1)(lowering_fn)
            lower(stub, vec1, vec1)(lowering_fn)

        if num_elements >= 3:
            vec1, vec2 = get_vec_type(1), get_vec_type(2)
            # vec2 + scalar, scalar + vec2
            lower(stub, vec2, scalar_type)(lowering_fn)
            lower(stub, scalar_type, vec2)(lowering_fn)
            # vec1 + vec2, vec2 + vec1
            lower(stub, vec1, vec2)(lowering_fn)
            lower(stub, vec2, vec1)(lowering_fn)
            # vec1 + scalar + scalar, scalar + vec1 + scalar, scalar + scalar + vec1
            lower(stub, vec1, scalar_type, scalar_type)(lowering_fn)
            lower(stub, scalar_type, vec1, scalar_type)(lowering_fn)
            lower(stub, scalar_type, scalar_type, vec1)(lowering_fn)
            # vec1 + vec1 + scalar, etc.
            lower(stub, vec1, vec1, scalar_type)(lowering_fn)
            lower(stub, vec1, scalar_type, vec1)(lowering_fn)
            lower(stub, scalar_type, vec1, vec1)(lowering_fn)
            # vec1 + vec1 + vec1
            lower(stub, vec1, vec1, vec1)(lowering_fn)

        if num_elements >= 4:
            vec1, vec2, vec3 = get_vec_type(1), get_vec_type(2), get_vec_type(3)
            # vec3 + scalar, scalar + vec3
            lower(stub, vec3, scalar_type)(lowering_fn)
            lower(stub, scalar_type, vec3)(lowering_fn)
            # vec2 + vec2
            lower(stub, vec2, vec2)(lowering_fn)
            # vec2 + scalar + scalar, scalar + vec2 + scalar, scalar + scalar + vec2
            lower(stub, vec2, scalar_type, scalar_type)(lowering_fn)
            lower(stub, scalar_type, vec2, scalar_type)(lowering_fn)
            lower(stub, scalar_type, scalar_type, vec2)(lowering_fn)
            # vec1 + vec3, vec3 + vec1
            lower(stub, vec1, vec3)(lowering_fn)
            lower(stub, vec3, vec1)(lowering_fn)
            # vec2 + vec1 + scalar, etc.
            lower(stub, vec2, vec1, scalar_type)(lowering_fn)
            lower(stub, vec2, scalar_type, vec1)(lowering_fn)
            lower(stub, vec1, vec2, scalar_type)(lowering_fn)
            lower(stub, vec1, scalar_type, vec2)(lowering_fn)
            lower(stub, scalar_type, vec2, vec1)(lowering_fn)
            lower(stub, scalar_type, vec1, vec2)(lowering_fn)
            # vec1 + vec1 + vec2, etc.
            lower(stub, vec1, vec1, vec2)(lowering_fn)
            lower(stub, vec1, vec2, vec1)(lowering_fn)
            lower(stub, vec2, vec1, vec1)(lowering_fn)
            # vec1 + vec1 + scalar + scalar, etc. (various 4-arg patterns)
            lower(stub, vec1, vec1, scalar_type, scalar_type)(lowering_fn)
            lower(stub, vec1, scalar_type, vec1, scalar_type)(lowering_fn)
            lower(stub, vec1, scalar_type, scalar_type, vec1)(lowering_fn)
            lower(stub, scalar_type, vec1, vec1, scalar_type)(lowering_fn)
            lower(stub, scalar_type, vec1, scalar_type, vec1)(lowering_fn)
            lower(stub, scalar_type, scalar_type, vec1, vec1)(lowering_fn)
            # vec1 + vec1 + vec1 + scalar, etc.
            lower(stub, vec1, vec1, vec1, scalar_type)(lowering_fn)
            lower(stub, vec1, vec1, scalar_type, vec1)(lowering_fn)
            lower(stub, vec1, scalar_type, vec1, vec1)(lowering_fn)
            lower(stub, scalar_type, vec1, vec1, vec1)(lowering_fn)
            # vec1 + vec1 + vec1 + vec1
            lower(stub, vec1, vec1, vec1, vec1)(lowering_fn)


# Register attribute access lowerings
def _make_attr_lowering(attr_name):
    """Create a lowering function for vector attribute access."""
    idx = ATTR_INDEX[attr_name]

    def attr_lowering(context, builder: MLIRLower, target, value):
        vec = builder.load_var(value)
        # Use static position for extraction (empty dynamic_position list)
        elem = vector.extract(vec, [], [idx])
        builder.store_var(target, elem)

    return attr_lowering


lower_getattr(VectorType, "x")(_make_attr_lowering("x"))
lower_getattr(VectorType, "y")(_make_attr_lowering("y"))
lower_getattr(VectorType, "z")(_make_attr_lowering("z"))
lower_getattr(VectorType, "w")(_make_attr_lowering("w"))


# Also register lowerings for numba-cuda's stub classes
def _register_numba_cuda_stubs():
    """Register lowerings for numba-cuda's vector type stubs."""
    try:
        from cusimt.numba_cuda.stubs import _vector_type_stubs as numba_cuda_stubs
    except ImportError:
        return

    from cusimt.cuda.vector_types import vector_type_stubs_by_name

    for nc_stub in numba_cuda_stubs:
        name = nc_stub.__name__
        our_stub = vector_type_stubs_by_name.get(name)
        if our_stub is None:
            continue

        num_elements = our_stub._num_elements
        lowering_fn = make_constructor_lowering(our_stub)

        # Register for various argument patterns (same as above)
        int_types = [
            types.int8,
            types.int16,
            types.int32,
            types.int64,
            types.Integer,
            types.IntegerLiteral,
        ]
        for int_type in int_types:
            lower(nc_stub, *([int_type] * num_elements))(lowering_fn)
            lower(nc_stub, int_type)(lowering_fn)

        uint_types = [types.uint8, types.uint16, types.uint32, types.uint64]
        for uint_type in uint_types:
            lower(nc_stub, *([uint_type] * num_elements))(lowering_fn)
            lower(nc_stub, uint_type)(lowering_fn)

        float_types_list = [types.float32, types.float64, types.Float]
        for float_type in float_types_list:
            lower(nc_stub, *([float_type] * num_elements))(lowering_fn)
            lower(nc_stub, float_type)(lowering_fn)

        lower(nc_stub, *([types.boolean] * num_elements))(lowering_fn)

        # Vector copy/conversion
        for other_stub in _vector_type_stubs:
            if other_stub._num_elements == num_elements:
                other_vec_type = VectorType(
                    getattr(types, other_stub._base_type_name),
                    (other_stub._num_elements,),
                )
                lower(nc_stub, other_vec_type)(lowering_fn)

        # Mixed vector+scalar patterns (same as for our stubs)
        base_type_name = our_stub._base_type_name
        base_type = getattr(types, base_type_name)
        scalar_types = (
            [base_type, types.Float]
            if "float" in base_type_name
            else [base_type, types.Integer]
        )

        def get_vec_type(n):
            return VectorType(base_type, (n,))

        for scalar_type in scalar_types:
            if num_elements >= 2:
                vec1 = get_vec_type(1)
                lower(nc_stub, vec1, scalar_type)(lowering_fn)
                lower(nc_stub, scalar_type, vec1)(lowering_fn)
                lower(nc_stub, vec1, vec1)(lowering_fn)

            if num_elements >= 3:
                vec1, vec2 = get_vec_type(1), get_vec_type(2)
                lower(nc_stub, vec2, scalar_type)(lowering_fn)
                lower(nc_stub, scalar_type, vec2)(lowering_fn)
                lower(nc_stub, vec1, vec2)(lowering_fn)
                lower(nc_stub, vec2, vec1)(lowering_fn)
                lower(nc_stub, vec1, scalar_type, scalar_type)(lowering_fn)
                lower(nc_stub, scalar_type, vec1, scalar_type)(lowering_fn)
                lower(nc_stub, scalar_type, scalar_type, vec1)(lowering_fn)
                lower(nc_stub, vec1, vec1, scalar_type)(lowering_fn)
                lower(nc_stub, vec1, scalar_type, vec1)(lowering_fn)
                lower(nc_stub, scalar_type, vec1, vec1)(lowering_fn)
                lower(nc_stub, vec1, vec1, vec1)(lowering_fn)

            if num_elements >= 4:
                vec1, vec2, vec3 = get_vec_type(1), get_vec_type(2), get_vec_type(3)
                lower(nc_stub, vec3, scalar_type)(lowering_fn)
                lower(nc_stub, scalar_type, vec3)(lowering_fn)
                lower(nc_stub, vec2, vec2)(lowering_fn)
                lower(nc_stub, vec2, scalar_type, scalar_type)(lowering_fn)
                lower(nc_stub, scalar_type, vec2, scalar_type)(lowering_fn)
                lower(nc_stub, scalar_type, scalar_type, vec2)(lowering_fn)
                lower(nc_stub, vec1, vec3)(lowering_fn)
                lower(nc_stub, vec3, vec1)(lowering_fn)
                lower(nc_stub, vec2, vec1, scalar_type)(lowering_fn)
                lower(nc_stub, vec2, scalar_type, vec1)(lowering_fn)
                lower(nc_stub, vec1, vec2, scalar_type)(lowering_fn)
                lower(nc_stub, vec1, scalar_type, vec2)(lowering_fn)
                lower(nc_stub, scalar_type, vec2, vec1)(lowering_fn)
                lower(nc_stub, scalar_type, vec1, vec2)(lowering_fn)
                lower(nc_stub, vec1, vec1, vec2)(lowering_fn)
                lower(nc_stub, vec1, vec2, vec1)(lowering_fn)
                lower(nc_stub, vec2, vec1, vec1)(lowering_fn)
                lower(nc_stub, vec1, vec1, scalar_type, scalar_type)(lowering_fn)
                lower(nc_stub, vec1, scalar_type, vec1, scalar_type)(lowering_fn)
                lower(nc_stub, vec1, scalar_type, scalar_type, vec1)(lowering_fn)
                lower(nc_stub, scalar_type, vec1, vec1, scalar_type)(lowering_fn)
                lower(nc_stub, scalar_type, vec1, scalar_type, vec1)(lowering_fn)
                lower(nc_stub, scalar_type, scalar_type, vec1, vec1)(lowering_fn)
                lower(nc_stub, vec1, vec1, vec1, scalar_type)(lowering_fn)
                lower(nc_stub, vec1, vec1, scalar_type, vec1)(lowering_fn)
                lower(nc_stub, vec1, scalar_type, vec1, vec1)(lowering_fn)
                lower(nc_stub, scalar_type, vec1, vec1, vec1)(lowering_fn)
                lower(nc_stub, vec1, vec1, vec1, vec1)(lowering_fn)


_register_numba_cuda_stubs()
