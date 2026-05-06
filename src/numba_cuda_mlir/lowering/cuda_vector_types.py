# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lowering support for CUDA vector types (float32x4, int32x2, etc.)
"""

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
from typing import Any

ATTR_INDEX = {"x": 0, "y": 1, "z": 2, "w": 3}


def _build_vector_from_scalars(scalars: list, vec_type: ir.VectorType) -> ir.Value:
    """Build an MLIR vector from a list of scalar values."""
    elem_type = vec_type.element_type
    converted = [convert(s, elem_type) for s in scalars]
    return vector.from_elements(vec_type, converted)


def _extract_vector_elements(vec: ir.Value) -> list:
    """Extract all elements from an MLIR vector."""
    vec_type = vec.type
    num_elements = vec_type.shape[0]
    elements = []
    for i in range(num_elements):
        elem = vector.extract(vec, [], [i])
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
    num_elements = vec_type.shape[0]

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


# One generic registration per vector-type stub instead of enumerating
# every permutation of scalar/vector argument types.
for stub in _vector_type_stubs:
    _raw_lower(stub, types.VarArg(types.Any))(_constructor_lowering)


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
