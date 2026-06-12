# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Implement the cmath module functions.

The ``@lower`` codegen that used to live here (cmath.isnan/isinf/isfinite,
exp, log, sqrt, cos, sin, tan, acos, asin, atan, asinh) built llvmlite IR and
was installed into the target context, but it is filtered out on the MLIR path
(numba_cuda_mlir.lowering.cmath lowers these). Only the ``@overload``
implementations remain: those provide pure-Python implementations that the MLIR
path type-infers and lowers normally.
"""

import cmath
import math

from numba_cuda_mlir.numba_cuda.core.imputils import Registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.cpython import mathimpl
from numba_cuda_mlir.numba_cuda.extending import overload


registry = Registry("cmathimpl")
lower = registry.lower


@overload(cmath.rect)
def impl_cmath_rect(r, phi):
    if all([isinstance(typ, types.Float) for typ in [r, phi]]):

        def impl(r, phi):
            if not math.isfinite(phi):
                if not r:
                    # cmath.rect(0, phi={inf, nan}) = 0
                    return abs(r)
                if math.isinf(r):
                    # cmath.rect(inf, phi={inf, nan}) = inf + j phi
                    return complex(r, phi)
            real = math.cos(phi)
            imag = math.sin(phi)
            if real == 0.0 and math.isinf(r):
                # 0 * inf would return NaN, we want to keep 0 but xor the sign
                real /= r
            else:
                real *= r
            if imag == 0.0 and math.isinf(r):
                # ditto
                imag /= r
            else:
                imag *= r
            return complex(real, imag)

        return impl


@overload(cmath.log10)
def impl_cmath_log10(z):
    if not isinstance(z, types.Complex):
        return

    LN_10 = 2.302585092994045684

    def log10_impl(z):
        """cmath.log10(z)"""
        z = cmath.log(z)
        # This formula gives better results on +/-inf than cmath.log(z, 10)
        # See http://bugs.python.org/issue22544
        return complex(z.real / LN_10, z.imag / LN_10)

    return log10_impl


@overload(cmath.phase)
def phase_impl(x):
    """cmath.phase(x + y j)"""

    if not isinstance(x, types.Complex):
        return

    def impl(x):
        return math.atan2(x.imag, x.real)

    return impl


@overload(cmath.polar)
def polar_impl(x):
    if not isinstance(x, types.Complex):
        return

    def impl(x):
        r, i = x.real, x.imag
        return math.hypot(r, i), math.atan2(i, r)

    return impl


@overload(cmath.cosh)
def impl_cmath_cosh(z):
    if not isinstance(z, types.Complex):
        return

    def cosh_impl(z):
        """cmath.cosh(z)"""
        x = z.real
        y = z.imag
        if math.isinf(x):
            if math.isnan(y):
                # x = +inf, y = NaN => cmath.cosh(x + y j) = inf + Nan * j
                real = abs(x)
                imag = y
            elif y == 0.0:
                # x = +inf, y = 0 => cmath.cosh(x + y j) = inf + 0j
                real = abs(x)
                imag = y
            else:
                real = math.copysign(x, math.cos(y))
                imag = math.copysign(x, math.sin(y))
            if x < 0.0:
                # x = -inf => negate imaginary part of result
                imag = -imag
            return complex(real, imag)
        return complex(math.cos(y) * math.cosh(x), math.sin(y) * math.sinh(x))

    return cosh_impl


@overload(cmath.sinh)
def impl_cmath_sinh(z):
    if not isinstance(z, types.Complex):
        return

    def sinh_impl(z):
        """cmath.sinh(z)"""
        x = z.real
        y = z.imag
        if math.isinf(x):
            if math.isnan(y):
                # x = +/-inf, y = NaN => cmath.sinh(x + y j) = x + NaN * j
                real = x
                imag = y
            else:
                real = math.cos(y)
                imag = math.sin(y)
                if real != 0.0:
                    real *= x
                if imag != 0.0:
                    imag *= abs(x)
            return complex(real, imag)
        return complex(math.cos(y) * math.sinh(x), math.sin(y) * math.cosh(x))

    return sinh_impl


@overload(cmath.tanh)
def impl_cmath_tanh(z):
    if not isinstance(z, types.Complex):
        return

    def tanh_impl(z):
        """cmath.tanh(z)"""
        x = z.real
        y = z.imag
        if math.isinf(x):
            real = math.copysign(1.0, x)
            if math.isinf(y):
                imag = 0.0
            else:
                imag = math.copysign(0.0, math.sin(2.0 * y))
            return complex(real, imag)
        # This is CPython's algorithm (see c_tanh() in cmathmodule.c).
        # XXX how to force float constants into single precision?
        tx = math.tanh(x)
        ty = math.tan(y)
        cx = 1.0 / math.cosh(x)
        txty = tx * ty
        denom = 1.0 + txty * txty
        return complex(tx * (1.0 + ty * ty) / denom, ((ty / denom) * cx) * cx)

    return tanh_impl


@overload(cmath.acosh)
def impl_cmath_acosh(z):
    if not isinstance(z, types.Complex):
        return

    LN_4 = math.log(4)
    THRES = mathimpl.FLT_MAX / 4

    def acosh_impl(z):
        """cmath.acosh(z)"""
        # CPython's algorithm (see c_acosh() in cmathmodule.c)
        if abs(z.real) > THRES or abs(z.imag) > THRES:
            # Avoid unnecessary overflow for large arguments
            # (also handles infinities gracefully)
            real = math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4
            imag = math.atan2(z.imag, z.real)
            return complex(real, imag)
        else:
            s1 = cmath.sqrt(complex(z.real - 1.0, z.imag))
            s2 = cmath.sqrt(complex(z.real + 1.0, z.imag))
            real = math.asinh(s1.real * s2.real + s1.imag * s2.imag)
            imag = 2.0 * math.atan2(s1.imag, s2.real)
            return complex(real, imag)
        # Condensed formula (NumPy)
        # return cmath.log(z + cmath.sqrt(z + 1.) * cmath.sqrt(z - 1.))

    return acosh_impl
