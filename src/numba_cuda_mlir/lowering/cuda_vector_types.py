# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lowering support for CUDA vector types (float32x4, int32x2, etc.)
"""

import itertools
import operator
from typing import Any
from numba_cuda_mlir.numba_cuda.types import StarArgTuple, StarArgUniTuple

from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
_raw_lower = registry.lower
lower_getattr = registry.lower_getattr
from numba_cuda_mlir.mlir_lowering import MLIRLower
from numba_cuda_mlir.lowering_utilities import (
    convert,
    get_conversion_signedness,
    _get_mlir_bin_op_for_operator,
)
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type
from numba_cuda_mlir.cuda.vector_types import _vector_types
from numba_cuda_mlir.type_defs.vector_types import VectorType
from numba_cuda_mlir import types
from numba_cuda_mlir._mlir.dialects import vector, arith
from numba_cuda_mlir._mlir.dialects import complex as complex_dialect
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
    converted = [convert(s, elem_type, signed=signed) for s, signed in scalars]

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


def _constructor_lowering(lower_ctx: MLIRLower, signature, args: list[Any]):
    """Generic lowering for all vector type constructors.

    Handles any combination of scalar and vector arguments: broadcasts a
    single scalar, concatenates mixed scalar/vector args, or copies/converts
    a vector of the same width.
    """
    target_type = signature.return_type
    vec_type = to_mlir_type(target_type)
    num_elements = _num_vector_elements(vec_type)

    scalars = []
    operands = []
    for val, arg_type in zip(args, signature.args):
        if isinstance(arg_type, (StarArgTuple, StarArgUniTuple)):
            operands.extend(zip(val, arg_type.types))
        else:
            operands.append((val, arg_type))
    for val, arg_type in operands:
        signed = get_conversion_signedness(arg_type, target_type)

        if isinstance(arg_type, VectorType):
            scalars.extend((scalar, signed) for scalar in _extract_vector_elements(val))
        elif isinstance(arg_type, types.Complex):
            scalars.append((complex_dialect.re(val), signed))
            scalars.append((complex_dialect.im(val), signed))
        else:
            scalars.append((val, signed))

    if len(scalars) == 1 and num_elements > 1:
        scalars = scalars * num_elements

    result = _build_vector_from_scalars(scalars, vec_type)
    return result


# One generic registration per vector-type instead of enumerating
# every permutation of scalar/vector argument types.
for vec_type in _vector_types:
    _raw_lower(vec_type, types.VarArg(types.Any), api="values")(_constructor_lowering)


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


def _make_vector_binop_lowering(op):
    iop, fop = _get_mlir_bin_op_for_operator(op)

    def binop_lowering(lower_ctx: MLIRLower, signature, args: list[Any]):
        lhs = args[0]
        rhs = args[1]

        lhs_type = signature.args[0]
        rhs_type = signature.args[1]

        target_type = signature.return_type
        mlir_target_type = to_mlir_type(target_type)
        elem_type = mlir_target_type.element_type
        lhs_signed = get_conversion_signedness(lhs_type, target_type)
        rhs_signed = get_conversion_signedness(rhs_type, target_type)

        if isinstance(lhs_type, VectorType):
            if not isinstance(rhs_type, VectorType):
                # rhs is scalar, broadcast it
                rhs_val = convert(rhs, elem_type, signed=rhs_signed)
                rhs = vector.broadcast(mlir_target_type, rhs_val)
            else:
                rhs = convert(rhs, mlir_target_type, signed=rhs_signed)
            lhs = convert(lhs, mlir_target_type, signed=lhs_signed)
        else:
            # lhs is scalar, broadcast it
            lhs_val = convert(lhs, elem_type, signed=lhs_signed)
            lhs = vector.broadcast(mlir_target_type, lhs_val)
            rhs = convert(rhs, mlir_target_type, signed=rhs_signed)

        if isinstance(elem_type, ir.IntegerType):
            result = iop(lhs, rhs)
        else:
            result = fop(lhs, rhs)

        return result

    return binop_lowering


for op in [
    operator.add,
    operator.iadd,
    operator.sub,
    operator.isub,
    operator.mul,
    operator.imul,
    operator.truediv,
    operator.itruediv,
    operator.floordiv,
    operator.ifloordiv,
    operator.mod,
    operator.imod,
]:
    _raw_lower(op, VectorType, VectorType, api="values")(_make_vector_binop_lowering(op))
    _raw_lower(op, VectorType, types.Number, api="values")(_make_vector_binop_lowering(op))
    _raw_lower(op, types.Number, VectorType, api="values")(_make_vector_binop_lowering(op))


def _make_vector_unary_lowering(op):
    def unary_lowering(lower_ctx: MLIRLower, signature, args: list[Any]):
        val = args[0]

        target_type = signature.return_type
        mlir_target_type = to_mlir_type(target_type)
        elem_type = mlir_target_type.element_type

        val = convert(val, mlir_target_type)

        if op == operator.neg:
            if isinstance(elem_type, ir.IntegerType):
                zero = arith.constant(elem_type, 0)
                zero_vec = vector.broadcast(mlir_target_type, zero)
                result = arith.subi(zero_vec, val)
            else:
                result = arith.negf(val)
        elif op == abs:
            if isinstance(elem_type, ir.IntegerType):
                from numba_cuda_mlir._mlir.dialects import math

                result = math.absi(val)
            else:
                from numba_cuda_mlir._mlir.dialects import math

                result = math.absf(val)
        else:
            raise NotImplementedError(f"Unary operator {op} not implemented for vector types")

        return result

    return unary_lowering


_raw_lower(operator.neg, VectorType, api="values")(_make_vector_unary_lowering(operator.neg))
_raw_lower(abs, VectorType, api="values")(_make_vector_unary_lowering(abs))


def _vector_to_complex_cast(lower_ctx: MLIRLower, signature, args: list[Any]):
    target_type = signature.return_type
    mlir_target_type = lower_ctx.get_mlir_type(target_type)
    val = args[0]
    source_type = signature.args[0]
    signed = get_conversion_signedness(source_type, target_type)

    real = vector.extract(val, [], [0])
    imag = vector.extract(val, [], [1])

    real = convert(real, mlir_target_type.element_type, signed=signed)
    imag = convert(imag, mlir_target_type.element_type, signed=signed)

    result = complex_dialect.create_(
        complex=mlir_target_type,
        real=real,
        imaginary=imag,
    )
    return result


@_raw_lower(complex, VectorType, api="values")
def _complex_from_vector_lowering(lower_ctx: MLIRLower, signature, args: list[Any]):
    """Lowering for complex(vector_type)."""
    return _vector_to_complex_cast(lower_ctx, signature, args)


@_raw_lower(types.NumberClass, VectorType, api="values")
def _number_class_from_vector_lowering(lower_ctx: MLIRLower, signature, args: list[Any]):
    """Lowering for np.complex64(vec2) / np.complex128(vec2)."""
    target_type = signature.return_type
    if not isinstance(target_type, types.Complex):
        raise NotImplementedError(
            f"NumberClass({target_type})(VectorType) lowering only supports Complex targets"
        )

    return _vector_to_complex_cast(lower_ctx, signature, args)
