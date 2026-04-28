# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    Registry,
    signature,
)
from numba import types

registry = Registry()

# Mapping of function names to their templates
_math_functions = {}


def _make_unary_math_template(key, return_type_fn=None):
    """
    Create a typing template for unary math functions.

    Args:
        key: The function to register (e.g., math.sin)
        return_type_fn: Optional function to compute return type from arg type.
                       If None, returns the same type as input.
    """

    class UnaryMathTemplate(AbstractTemplate):
        def generic(self, args, kws):
            if len(args) == 1 and isinstance(args[0], (types.Integer, types.Float)):
                return_type = return_type_fn(args[0]) if return_type_fn else args[0]
                return signature(return_type, args[0])

    UnaryMathTemplate.key = key
    func_name = key.__name__
    template = registry.register(UnaryMathTemplate)
    _math_functions[func_name] = template
    # Also register as global so the function object can be looked up directly
    registry.register_global(key, types.Function(template))
    return template


def _float_to_integer_return_type(arg_type):
    if isinstance(arg_type, types.Integer):
        return arg_type
    elif arg_type in (types.float16, types.float32):
        return types.int32
    else:
        return types.int64


# Functions that return integer (matching Python 3 behavior)
for func in [math.ceil, math.floor, math.trunc]:
    _make_unary_math_template(func, _float_to_integer_return_type)


# Functions that return the same type as input
for func in [
    math.sin,
    math.cos,
    math.tan,
    math.sqrt,
    math.exp,
    math.exp2,
    math.expm1,
    math.log,
    math.log2,
    math.log10,
    math.log1p,
    math.fabs,
    math.tanh,
    math.sinh,
    math.cosh,
    math.asin,
    math.acos,
    math.atan,
    math.asinh,
    math.acosh,
    math.atanh,
    math.degrees,
    math.radians,
    math.gamma,
    math.lgamma,
    math.erf,
    math.erfc,
]:
    _make_unary_math_template(func)


# Functions that return boolean
for func in [math.isfinite, math.isnan, math.isinf]:
    _make_unary_math_template(func, lambda _: types.boolean)


def _make_binary_math_template(key, return_type_fn=None):
    """
    Create a typing template for binary math functions.

    Args:
        key: The function to register (e.g., math.atan2)
        return_type_fn: Optional function to compute return type from arg types.
                       If None, returns the type of the first argument.
    """

    class BinaryMathTemplate(AbstractTemplate):
        def generic(self, args, kws):
            if (
                len(args) == 2
                and isinstance(args[0], types.Number)
                and isinstance(args[1], types.Number)
            ):
                return_type = return_type_fn(args[0], args[1]) if return_type_fn else args[0]
                return signature(return_type, args[0], args[1])

    BinaryMathTemplate.key = key
    func_name = key.__name__
    template = registry.register(BinaryMathTemplate)
    _math_functions[func_name] = template
    registry.register_global(key, types.Function(template))
    return template


# Binary math functions that return same type as input
for func in [
    math.atan2,
    math.copysign,
    math.hypot,
    math.fmod,
    math.remainder,
    math.pow,
    math.nextafter,
]:
    _make_binary_math_template(func)


# Special case: frexp returns a tuple (mantissa: float, exponent: int32)
@registry.register
class FrexpTemplate(AbstractTemplate):
    key = math.frexp

    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], types.Float):
            # frexp returns (mantissa, exponent) where mantissa is same type as input
            return_type = types.Tuple([args[0], types.int32])
            return signature(return_type, args[0])


_math_functions["frexp"] = FrexpTemplate
registry.register_global(math.frexp, types.Function(FrexpTemplate))


# Special case: ldexp takes (float, int) and returns float
@registry.register
class LdexpTemplate(AbstractTemplate):
    key = math.ldexp

    def generic(self, args, kws):
        if len(args) == 2 and isinstance(args[0], types.Float):
            # ldexp takes (mantissa, exponent) and returns same type as mantissa
            if isinstance(args[1], types.Integer):
                return signature(args[0], args[0], args[1])


_math_functions["ldexp"] = LdexpTemplate
registry.register_global(math.ldexp, types.Function(LdexpTemplate))


# Special case: modf returns a tuple (fractional_part, integer_part)
@registry.register
class ModfTemplate(AbstractTemplate):
    key = math.modf

    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], types.Float):
            # modf returns (fractional, integer) both same type as input
            return_type = types.Tuple([args[0], args[0]])
            return signature(return_type, args[0])


_math_functions["modf"] = ModfTemplate
registry.register_global(math.modf, types.Function(ModfTemplate))


@registry.register_attr
class MathModuleAttributeTemplate(AttributeTemplate):
    """
    Resolve attributes of the math module (e.g., math.sin, math.cos).
    """

    key = types.Module(math)

    def resolve(self, mod, attrname):
        """Resolve math module attributes to their function templates."""
        if attrname in _math_functions:
            # Must use the typing context's merged global (augmented with extending
            # registries, e.g. cuDF Masked unary ops). Returning types.Function from
            # _math_functions alone drops extra templates and breaks sin(Masked(float64))
            # when the attribute is resolved via this template before globals merge.
            fn = getattr(math, attrname, None)
            if fn is not None:
                gty = self.context._lookup_global(fn)
                if gty is not None:
                    return gty
            return types.Function(_math_functions[attrname])
        return None


@registry.register_attr
class ComplexAttributeTemplate(AttributeTemplate):
    """
    Resolve attributes of complex numbers (e.g., complex64.real, complex128.imag).
    """

    key = types.Complex

    def resolve_real(self, val):
        """Resolve .real attribute to return the underlying float type."""
        if isinstance(val, types.Complex):
            # complex64 -> float32, complex128 -> float64
            return val.underlying_float
        return None

    def resolve_imag(self, val):
        """Resolve .imag attribute to return the underlying float type."""
        if isinstance(val, types.Complex):
            # complex64 -> float32, complex128 -> float64
            return val.underlying_float
        return None
