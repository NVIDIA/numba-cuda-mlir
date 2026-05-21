# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lowering support for CUDA vector types (float32x4, int32x2, etc.)
"""

import itertools
from typing import Any

from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
_raw_lower = registry.lower
lower_getattr = registry.lower_getattr
from numba_cuda_mlir.mlir_lowering import MLIRLower
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type
from numba_cuda_mlir.cuda.vector_types import _vector_type_stubs
from numba_cuda_mlir.type_defs.vector_types import VectorType
from numba_cuda_mlir import types
from numba_cuda_mlir._mlir.dialects import vector, arith
from numba_cuda_mlir._mlir import ir

ATTR_INDEX = {"x": 0, "y": 1, "z": 2, "w": 3}


def _num_vector_elements(vec_type: ir.VectorType) -> int:
    num_elements = 1
    for dim in vec_type.shape:
        num_elements *= dim
    return num_elements


def _build_vector_from_scalars(scalars: list, vec_type: ir.VectorType) -> ir.Value:
    """Build an MLIR vector from a list of scalar values."""
    elem_type = vec_type.element_type
    num_elements = _num_vector_elements(vec_type)

    if len(scalars) != num_elements:
        raise ValueError(
            f"Expected {num_elements} scalar elements for {vec_type}, got {len(scalars)}"
        )

    # Convert all scalars to the target element type before building the vector.
    converted = [convert(s, elem_type) for s in scalars]

    # Use vector.from_elements to build the vector.
    return vector.from_elements(vec_type, converted)


def _extract_vector_elements(vec: ir.Value) -> list:
    """Extract all elements from an MLIR vector."""
    vec_type = vec.type
    elements = []
    for indices in itertools.product(*(range(dim) for dim in vec_type.shape)):
        # Use static positions for extraction (empty dynamic_position list).
        elem = vector.extract(vec, [], list(indices))
        elements.append(elem)
    return elements


def _constructor_lowering(lower_ctx: MLIRLower, target, args: list[Any], kwargs):
    """Generic lowering for all vector type constructors.

    Handles any combination of scalar and vector arguments: broadcasts a
    single scalar, concatenates mixed scalar/vector args, or copies/converts
    a vector of the same width.
    """
    target_type = lower_ctx.get_numba_type(target.name)
    vec_type = to_mlir_type(target_type)
    num_elements = _num_vector_elements(vec_type)

    scalars = []
    for arg in args:
        val = lower_ctx.load_var(arg)
        arg_type = lower_ctx.get_numba_type(arg.name)

        if isinstance(arg_type, VectorType):
            scalars.extend(_extract_vector_elements(val))
        else:
            scalars.append(val)

    if len(scalars) == 1 and num_elements > 1:
        scalars = scalars * num_elements

    result = _build_vector_from_scalars(scalars, vec_type)
    lower_ctx.store_var(target, result)


# Register fixed constructor arities instead of a single VarArg lowering.  This
# avoids platform-dependent dispatch of mixed scalar/vector constructors while
# keeping typing responsible for rejecting invalid element counts.
for stub in _vector_type_stubs:
    for arity in range(1, 5):
        _raw_lower(stub, *([types.Any] * arity))(_constructor_lowering)


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
