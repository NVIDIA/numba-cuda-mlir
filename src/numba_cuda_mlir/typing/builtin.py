# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.errors import ForceLiteralArg
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    AbstractTemplate,
    Registry,
    signature,
)
from numba_cuda_mlir import types
from numba_cuda_mlir.type_defs.vector_types import VectorType
import operator

registry = Registry()


@registry.register_global(operator.mul)
class TupleMultiplyTemplate(AbstractTemplate):
    def generic(self, args, kws):
        tup, how_many = args
        match tup, how_many:
            case types.Tuple(), types.Integer():
                pass
            case types.Integer(), types.Tuple():
                tup, how_many = how_many, tup
            case _:
                return None

        if not isinstance(how_many, types.Literal):
            raise ForceLiteralArg(set([args.index(how_many)]))

        return signature(types.Tuple((tup.dtype,) * how_many.literal_value), tup, how_many)


@registry.register_global(operator.setitem)
class ArraySetitemTemplate(AbstractTemplate):
    def generic(self, args, kws):
        if len(args) != 3:
            return None
        array, idx, value = args
        if not isinstance(array, types.Array):
            return None
        if isinstance(value, types.Complex):
            return signature(types.none, array, idx, value)
        if (
            isinstance(value, VectorType)
            and value.length == 2
            and isinstance(array.dtype, types.Complex)
        ):
            return signature(types.none, array, idx, value)
        return None


@registry.register_global(operator.contains)
class ContainsTemplate(AbstractTemplate):
    """Typing template for operator.contains (in operator)"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        container, item = args

        # dont return a sig for extension types
        if not isinstance(item, (types.Number, types.Boolean)):
            return None

        # Tuple contains
        if isinstance(container, types.Tuple):
            # For literal tuples, we can evaluate at compile time
            if all(isinstance(x, types.Literal) for x in container.types):
                return signature(types.boolean, container, item)
            return None

        # UniTuple contains
        elif isinstance(container, types.UniTuple):
            return signature(types.boolean, container, item)

        return None


@registry.register_global(operator.neg)
class NegTemplate(AbstractTemplate):
    """Typing template for operator.neg (unary -)"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arg = args[0]

        # For arrays, return same array type
        if isinstance(arg, types.Array):
            return signature(arg, arg)

        # For numbers, return same number type (or promoted)
        if isinstance(arg, types.Number):
            # Integer negation can increase bit width
            if isinstance(arg, types.Integer):
                # Promote to at least int64 for signed operations
                if arg.bitwidth < 64:
                    return signature(types.int64, arg)
                return signature(arg, arg)
            # Floats stay the same
            return signature(arg, arg)

        return None


@registry.register_global(operator.pos)
class PosTemplate(AbstractTemplate):
    """Typing template for operator.pos (unary +)"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arg = args[0]

        # Returns the same type as input
        if isinstance(arg, (types.Number, types.Array)):
            return signature(arg, arg)

        return None


@registry.register_global(operator.floordiv)
@registry.register_global(operator.ifloordiv)
class FloorDivTemplate(AbstractTemplate):
    """Typing template for operator.floordiv (//) and operator.ifloordiv (//=)"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        # For arrays, use integer type promotion
        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            from numba_cuda_mlir.lower import (
                type_conversions,
                numpy_implicit_type_promotion,
            )

            lhs_dtype = lhs.dtype if isinstance(lhs, types.Array) else lhs
            rhs_dtype = rhs.dtype if isinstance(rhs, types.Array) else rhs

            # Floor division always returns integer type
            result_dtype = numpy_implicit_type_promotion(lhs_dtype, rhs_dtype)

            # Ensure result is integer type for floor division
            if isinstance(result_dtype, types.Float):
                # Convert to integer of same width
                result_dtype = type_conversions.integer_of_width(result_dtype.bitwidth)
            elif not isinstance(result_dtype, types.Integer):
                result_dtype = types.intp

            # Determine output dimensionality
            if isinstance(lhs, types.Array) and isinstance(rhs, types.Array):
                ndim = max(lhs.ndim, rhs.ndim)
                layout = lhs.layout
            elif isinstance(lhs, types.Array):
                ndim = lhs.ndim
                layout = lhs.layout
            else:
                ndim = rhs.ndim
                layout = rhs.layout

            return signature(types.Array(result_dtype, ndim, layout), lhs, rhs)

        # For scalars, handled by default
        return None


@registry.register_attr
class NumberAttributeTemplate(AttributeTemplate):
    """Typing template for number methods"""

    key = types.Number

    def resolve_bit_count(self, num):
        """Type num.bit_count() -> int"""
        return types.BoundFunction(NumberBitCountMethodTemplate, num)


class NumberBitCountMethodTemplate(AbstractTemplate):
    """Typing for num.bit_count() method"""

    key = "Number.bit_count"

    def generic(self, args, kws):
        if len(args) == 0:
            # bit_count returns an integer
            return signature(types.intp, recvr=self.this)
        return None
