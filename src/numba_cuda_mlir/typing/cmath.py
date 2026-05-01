# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import cmath
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.typing.templates import (
    ConcreteTemplate,
    signature,
    Registry,
)
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    bound_function,
)

registry = Registry()
infer_global = registry.register_global
infer_getattr = registry.register_attr


@infer_global(cmath.acos)
@infer_global(cmath.acosh)
@infer_global(cmath.asin)
@infer_global(cmath.asinh)
@infer_global(cmath.atan)
@infer_global(cmath.atanh)
@infer_global(cmath.cos)
@infer_global(cmath.sin)
@infer_global(cmath.cosh)
@infer_global(cmath.sinh)
@infer_global(cmath.tanh)
@infer_global(cmath.tan)
@infer_global(cmath.exp)
@infer_global(cmath.sqrt)
@infer_global(cmath.log10)
class Math_unary(ConcreteTemplate):
    metadata = {"target": "cuda"}
    cases = [
        signature(types.complex64, types.complex64),
        signature(types.complex128, types.complex128),
    ]


@infer_global(cmath.phase)
class Math_phase(ConcreteTemplate):
    metadata = {"target": "cuda"}
    cases = [
        signature(types.float32, types.complex64),
        signature(types.float64, types.complex128),
    ]


@infer_global(cmath.rect)
class Math_binary(ConcreteTemplate):
    metadata = {"target": "cuda"}
    cases = [
        signature(types.complex128, types.float64, types.float64),
        signature(types.complex64, types.float32, types.float32),
    ]


@infer_global(cmath.log)
class Math_log(ConcreteTemplate):
    metadata = {"target": "cuda"}
    cases = [
        # Single argument (natural log)
        signature(types.complex64, types.complex64),
        signature(types.complex128, types.complex128),
        # Two arguments (log with base)
        signature(types.complex64, types.complex64, types.complex64),
        signature(types.complex128, types.complex128, types.complex128),
    ]


@infer_global(cmath.isnan)
@infer_global(cmath.isinf)
@infer_global(cmath.isfinite)
class Math_predicate(ConcreteTemplate):
    metadata = {"target": "cuda"}
    cases = [
        signature(types.boolean, types.complex64),
        signature(types.boolean, types.complex128),
    ]


@infer_global(cmath.polar)
class Math_polar(ConcreteTemplate):
    metadata = {"target": "cuda"}
    cases = [
        signature(types.Tuple((types.float64, types.float64)), types.complex64),
        signature(types.Tuple((types.float64, types.float64)), types.complex128),
    ]


@infer_getattr
class NumberAttribute(AttributeTemplate):
    key = types.Number

    def resolve___class__(self, ty):
        return types.NumberClass(ty)

    def resolve_real(self, ty):
        return getattr(ty, "underlying_float", ty)

    def resolve_imag(self, ty):
        return getattr(ty, "underlying_float", ty)

    @bound_function("complex.conjugate")
    def resolve_conjugate(self, ty, args, kws):
        assert not args
        assert not kws
        return signature(ty)

    @bound_function("number.item")
    def resolve_item(self, ty, args, kws):
        assert not kws
        if not args:
            return signature(ty)
