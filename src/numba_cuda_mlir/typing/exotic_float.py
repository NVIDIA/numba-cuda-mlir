# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing for exotic float types (fp8, fp4, fp6, tf32).

Registers operator overloads for all SpecialFloatType instances except bf16
(which has its own typing via numba-cuda's bf16 registry).

Also registers FP8 constructors (fp8_e5m2, fp8_e4m3, fp8_e8m0),
cvt_e8m0_to_bf16raw, and bfloat16_raw attribute access.
"""

import operator
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    Registry,
    signature,
)
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.types.ext_types import bfloat16 as bf16

registry = Registry()

_EXOTIC_FLOAT_TYPES = [
    types.f4E2M1FN,
    types.f6E2M3FN,
    types.f6E3M2FN,
    types.f8E3M4,
    types.f8E4M3B11FNUZ,
    types.f8E4M3FN,
    types.f8E4M3FNUZ,
    types.f8E4M3,
    types.f8E5M2FNUZ,
    types.f8E5M2,
    types.f8E8M0FNU,
    types.tf32,
]

_BINARY_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.itruediv,
]

_COMPARISON_OPS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]

_UNARY_OPS = [
    operator.neg,
    operator.pos,
]

_FP8_CTOR_INPUT_TYPES = [
    types.float16,
    bf16,
    types.float32,
    types.float64,
    types.int8,
    types.int16,
    types.int32,
    types.int64,
    types.uint8,
    types.uint16,
    types.uint32,
    types.uint64,
]

# Exotic float arithmetic/comparison/unary ops
for _ty in _EXOTIC_FLOAT_TYPES:
    for _op in _BINARY_OPS:

        class ExoticFloatBinaryTemplate(ConcreteTemplate):
            key = _op
            cases = [signature(_ty, _ty, _ty)]

        ExoticFloatBinaryTemplate.__name__ = f"ExoticFloat_{_ty.name}_{_op.__name__}"
        registry.register(ExoticFloatBinaryTemplate)
        registry.register_global(_op, types.Function(ExoticFloatBinaryTemplate))

    for _op in _COMPARISON_OPS:

        class ExoticFloatComparisonTemplate(ConcreteTemplate):
            key = _op
            cases = [signature(types.boolean, _ty, _ty)]

        ExoticFloatComparisonTemplate.__name__ = (
            f"ExoticFloat_{_ty.name}_{_op.__name__}"
        )
        registry.register(ExoticFloatComparisonTemplate)
        registry.register_global(_op, types.Function(ExoticFloatComparisonTemplate))

    for _op in _UNARY_OPS:

        class ExoticFloatUnaryTemplate(ConcreteTemplate):
            key = _op
            cases = [signature(_ty, _ty)]

        ExoticFloatUnaryTemplate.__name__ = f"ExoticFloat_{_ty.name}_{_op.__name__}"
        registry.register(ExoticFloatUnaryTemplate)
        registry.register_global(_op, types.Function(ExoticFloatUnaryTemplate))


def register_fp8_globals():
    """Register FP8 constructors, cvt_e8m0_to_bf16raw, and bfloat16_raw.x."""
    try:
        fp8_e5m2 = types.fp8_e5m2
        fp8_e4m3 = types.fp8_e4m3
        fp8_e8m0 = types.fp8_e8m0
        _type_fp8_e5m2 = types._type_fp8_e5m2
        _type_fp8_e4m3 = types._type_fp8_e4m3
        _type_fp8_e8m0 = types._type_fp8_e8m0
        cvt_e8m0_to_bf16raw = types.cvt_e8m0_to_bf16raw
        bfloat16_raw_type = types.bfloat16_raw_type
    except (AttributeError, ImportError):
        return

    for _py_type, _nb_type in [
        (fp8_e5m2, _type_fp8_e5m2),
        (fp8_e4m3, _type_fp8_e4m3),
        (fp8_e8m0, _type_fp8_e8m0),
    ]:

        class FP8CtorTemplate(ConcreteTemplate):
            key = _py_type
            cases = [signature(_nb_type)] + [
                signature(_nb_type, inp) for inp in _FP8_CTOR_INPUT_TYPES
            ]

        FP8CtorTemplate.__name__ = f"FP8Ctor_{_nb_type.name}"
        registry.register(FP8CtorTemplate)
        registry.register_global(_py_type, types.Function(FP8CtorTemplate))

    class CvtE8m0ToBf16rawTemplate(ConcreteTemplate):
        key = cvt_e8m0_to_bf16raw
        cases = [signature(bfloat16_raw_type, types.uint8)]

    registry.register(CvtE8m0ToBf16rawTemplate)
    registry.register_global(
        cvt_e8m0_to_bf16raw, types.Function(CvtE8m0ToBf16rawTemplate)
    )

    class Bfloat16RawAttrTemplate(AttributeTemplate):
        key = bfloat16_raw_type

        def resolve_x(self, obj):
            return types.uint16

    registry.register_attr(Bfloat16RawAttrTemplate)
