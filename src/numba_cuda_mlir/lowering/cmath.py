# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Complex number intrinsics and cmath module support for numba-cuda-mlir.

This module provides:
1. Complex attribute operations (.real, .imag, .conjugate())
2. cmath module functions (phase, polar, rect, exp, log, sqrt, trig functions, etc.)
"""

import cmath
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir._mlir.dialects import (
    complex as complex_dialect,
    arith,
    math as math_dialect,
    scf,
    func,
)
from numba_cuda_mlir._mlir.extras import types as T
import numba_cuda_mlir._mlir.ir as ir
from numba_cuda_mlir.mlir_lowering_registry import MLIRLoweringRegistry

registry = MLIRLoweringRegistry()
lower = registry.lower
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir.lowering_utilities import (
    convert,
    DeferredMethodCall,
    get_or_insert_function,
)
from numba_cuda_mlir.descriptor import MLIRTargetContext

# Constants used for overflow protection
FLT_MAX = 3.4028235e38
DBL_MAX = 1.7976931348623157e308
SQRT2 = 1.414213562373095048801688724209698079

# ============================================================================
# Built-in complex() constructor
# ============================================================================


@lower(complex, types.Number, types.Number)
def complex_constructor_2args_cg(mlir_lower, target, args, kwargs):
    """Code generator for complex(real, imag) constructor."""
    assert not kwargs, "complex constructor does not accept keyword arguments"
    assert len(args) == 2, "complex constructor expects exactly 2 arguments"
    real = mlir_lower.load_var(args[0])
    imag = mlir_lower.load_var(args[1])

    # Create complex number from real and imag
    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)
    element_type = target_mlir_type.element_type
    # Convert to target element type
    real = convert(real, element_type)
    imag = convert(imag, element_type)
    result = complex_dialect.create_(target_mlir_type, real, imag)
    mlir_lower.store_var(target, result)


@lower(complex, types.Number)
def complex_constructor_1arg_cg(mlir_lower, target, args, kwargs):
    """Code generator for complex(real) constructor - imag defaults to 0."""
    assert not kwargs, "complex constructor does not accept keyword arguments"
    assert len(args) == 1, "complex constructor expects exactly 1 argument"
    real = mlir_lower.load_var(args[0])

    # Create complex number from real with imag=0
    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)
    element_type = target_mlir_type.element_type
    # Convert to target element type
    real = convert(real, element_type)
    zero = arith.constant(result=element_type, value=0.0)
    result = complex_dialect.create_(target_mlir_type, real, zero)
    mlir_lower.store_var(target, result)


@lower(complex)
def complex_constructor_0arg_cg(mlir_lower, target, args, kwargs):
    """Code generator for complex() constructor with no args - returns 0+0j."""
    assert not kwargs, "complex constructor does not accept keyword arguments"
    assert len(args) == 0, "complex constructor expects no arguments"

    # Create complex number 0+0j
    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)
    element_type = target_mlir_type.element_type
    zero = arith.constant(result=element_type, value=0.0)
    result = complex_dialect.create_(target_mlir_type, zero, zero)
    mlir_lower.store_var(target, result)


# ============================================================================
# Complex Attribute Operations
# ============================================================================


@registry.lower_getattr(types.Complex, "real")
def complex_real_getattr(context: MLIRTargetContext, builder, target, value):
    """Extract the real part of a complex number using complex.re operation."""
    trace("complex.real getattr")
    value_val = builder.load_var(value)
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)

    # Use MLIR's complex.re operation
    real_part = complex_dialect.re(value_val)
    real_part = convert(real_part, target_mlir_type)
    builder.store_var(target, real_part)


@registry.lower_getattr(types.Complex, "imag")
def complex_imag_getattr(context: MLIRTargetContext, builder, target, value):
    """Extract the imaginary part of a complex number using complex.im operation."""
    trace("complex.imag getattr")
    value_val = builder.load_var(value)
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)

    # Use MLIR's complex.im operation
    imag_part = complex_dialect.im(value_val)
    imag_part = convert(imag_part, target_mlir_type)
    builder.store_var(target, imag_part)


# For conjugate, we use DeferredMethodCall like Array methods
def complex_conjugate_cg(mlir_lower, target, args, kwargs):
    """Code generator for complex conjugate operation."""
    assert not kwargs, "conjugate does not accept any keyword arguments"
    assert len(args) == 1, "conjugate expects exactly 1 argument (self)"
    x = mlir_lower.load_var(args[0])
    result = complex_dialect.conj(x)
    mlir_lower.store_var(target, result)


@registry.lower_getattr(types.Complex, "conjugate")
def complex_conjugate_getattr(context: MLIRTargetContext, builder, target, value):
    """
    Handle x.conjugate attribute access.
    Returns a DeferredMethodCall that will execute the conjugate when called.
    """
    trace("complex.conjugate getattr")
    builder.store_var(target, DeferredMethodCall(value, complex_conjugate_cg))


# ============================================================================
# cmath Conversion Functions
# ============================================================================


@lower(cmath.phase, types.Complex)
def cmath_phase_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.phase."""
    assert not kwargs, "phase does not accept any keyword arguments"
    assert len(args) == 1, "phase expects exactly 1 argument"
    x = mlir_lower.load_var(args[0])
    # phase = atan2(imag, real)
    imag = complex_dialect.im(x)
    real = complex_dialect.re(x)
    result = math_dialect.atan2(imag, real)
    mlir_lower.store_var(target, result)


@lower(cmath.polar, types.Complex)
def cmath_polar_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.polar."""
    assert not kwargs, "polar does not accept any keyword arguments"
    assert len(args) == 1, "polar expects exactly 1 argument"
    x = mlir_lower.load_var(args[0])
    real = complex_dialect.re(x)
    imag = complex_dialect.im(x)
    # r = hypot(real, imag)
    r_sq = arith.addf(arith.mulf(real, real), arith.mulf(imag, imag))
    r = math_dialect.sqrt(r_sq)
    # phi = atan2(imag, real)
    phi = math_dialect.atan2(imag, real)
    mlir_lower.store_var(target, (r, phi))


@lower(cmath.rect, types.Number, types.Number)
def cmath_rect_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.rect with special case handling."""
    from numba_cuda_mlir.runtime import cmath as cmath_lib
    from numba_cuda_mlir.lowering_utilities import link

    assert not kwargs, "rect does not accept any keyword arguments"
    assert len(args) == 2, "rect expects exactly 2 arguments"
    r = mlir_lower.load_var(args[0])
    phi = mlir_lower.load_var(args[1])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)
    element_type = target_mlir_type.element_type
    type_suffix = "f32" if element_type == T.f32() else "f64"
    intrinsic_name = f"cmath_rect_type_{type_suffix}"

    # Link the cmath library
    if cmath_lib not in mlir_lower._seen_mlir_libraries:
        mlir_lower._seen_mlir_libraries.add(cmath_lib)
        link.link_inplace(mlir_lower.mlir_module, cmath_lib.source)

    # Ensure r and phi are the right type
    r = convert(r, element_type)
    phi = convert(phi, element_type)

    # Declare or lookup the function: (FTy, FTy) -> (FTy, FTy)
    fn_type = ir.FunctionType.get([element_type, element_type], [element_type, element_type])
    callee = get_or_insert_function(intrinsic_name, fn_type, mlir_lower.mlir_gpu_module)

    # Call the function
    call_results = func.call(
        result=[element_type, element_type],
        callee=callee.name.value,
        operands_=[r, phi],
    )

    # Create complex from results
    result = complex_dialect.create_(target_mlir_type, call_results[0], call_results[1])
    mlir_lower.store_var(target, result)


# ============================================================================
# cmath Classification Functions
# ============================================================================


@lower(cmath.isnan, types.Complex)
def cmath_isnan_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.isnan."""
    assert not kwargs, "isnan does not accept any keyword arguments"
    assert len(args) == 1, "isnan expects exactly 1 argument"
    x = mlir_lower.load_var(args[0])
    real = complex_dialect.re(x)
    imag = complex_dialect.im(x)
    real_isnan = math_dialect.isnan(real)
    imag_isnan = math_dialect.isnan(imag)
    result = arith.ori(real_isnan, imag_isnan)
    mlir_lower.store_var(target, result)


@lower(cmath.isinf, types.Complex)
def cmath_isinf_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.isinf."""
    assert not kwargs, "isinf does not accept any keyword arguments"
    assert len(args) == 1, "isinf expects exactly 1 argument"
    x = mlir_lower.load_var(args[0])
    real = complex_dialect.re(x)
    imag = complex_dialect.im(x)
    real_isinf = math_dialect.isinf(real)
    imag_isinf = math_dialect.isinf(imag)
    result = arith.ori(real_isinf, imag_isinf)
    mlir_lower.store_var(target, result)


@lower(cmath.isfinite, types.Complex)
def cmath_isfinite_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.isfinite."""
    assert not kwargs, "isfinite does not accept any keyword arguments"
    assert len(args) == 1, "isfinite expects exactly 1 argument"
    x = mlir_lower.load_var(args[0])
    real = complex_dialect.re(x)
    imag = complex_dialect.im(x)
    real_isfinite = math_dialect.isfinite(real)
    imag_isfinite = math_dialect.isfinite(imag)
    result = arith.andi(real_isfinite, imag_isfinite)
    mlir_lower.store_var(target, result)


# ============================================================================
# cmath Power and Logarithm Functions
# ============================================================================


def _call_cmath_intrinsic(mlir_lower, func_name, z, target_mlir_type):
    """Helper to call a cmath runtime intrinsic that takes (real, imag) and returns (real, imag)."""
    from numba_cuda_mlir.runtime import cmath as cmath_lib
    from numba_cuda_mlir.lowering_utilities import link

    element_type = target_mlir_type.element_type
    type_suffix = "f32" if element_type == T.f32() else "f64"
    intrinsic_name = f"{func_name}_type_{type_suffix}"

    # Link the cmath library
    if cmath_lib not in mlir_lower._seen_mlir_libraries:
        mlir_lower._seen_mlir_libraries.add(cmath_lib)
        link.link_inplace(mlir_lower.mlir_module, cmath_lib.source)

    x = convert(complex_dialect.re(z), element_type)
    y = convert(complex_dialect.im(z), element_type)

    # Lookup the function
    fn_type = ir.FunctionType.get([element_type, element_type], [element_type, element_type])
    callee = get_or_insert_function(intrinsic_name, fn_type, mlir_lower.mlir_gpu_module)

    # Call the function
    call_results = func.call(
        result=[element_type, element_type], callee=callee.name.value, operands_=[x, y]
    )

    # Create complex from results
    return complex_dialect.create_(target_mlir_type, call_results[0], call_results[1])


@lower(cmath.exp, types.Complex)
def cmath_exp_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.exp with proper special case handling."""
    assert not kwargs, "exp does not accept any keyword arguments"
    assert len(args) == 1, "exp expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_exp", z, target_mlir_type)
    mlir_lower.store_var(target, result)


@lower(cmath.log, types.Complex)
def cmath_log_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.log using log(z) = log|z| + i*arg(z)."""
    assert not kwargs, "log does not accept any keyword arguments"
    if len(args) == 1:
        z = mlir_lower.load_var(args[0])

        target_type = mlir_lower.get_numba_type(target.name)
        target_mlir_type = mlir_lower.get_mlir_type(target_type)

        x = complex_dialect.re(z)
        y = complex_dialect.im(z)

        # log|z| = log(hypot(x, y))
        x_sq = arith.mulf(x, x)
        y_sq = arith.mulf(y, y)
        hypot_xy = math_dialect.sqrt(arith.addf(x_sq, y_sq))
        log_abs = math_dialect.log(hypot_xy)

        # arg(z) = atan2(y, x)
        phase = math_dialect.atan2(y, x)

        result = complex_dialect.create_(target_mlir_type, log_abs, phase)
        mlir_lower.store_var(target, result)
    else:
        raise NotImplementedError("cmath.log with base not implemented")


@lower(cmath.log, types.Complex, types.Complex)
def cmath_log_base_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.log with base."""
    assert not kwargs, "log does not accept any keyword arguments"
    assert len(args) == 2, "log with base expects exactly 2 arguments"
    x = mlir_lower.load_var(args[0])
    base = mlir_lower.load_var(args[1])
    log_x = complex_dialect.log(x)
    log_base = complex_dialect.log(base)
    result = complex_dialect.div(log_x, log_base)
    mlir_lower.store_var(target, result)


@lower(cmath.log10, types.Complex)
def cmath_log10_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.log10 using log10(z) = log(z) / ln(10)."""
    assert not kwargs, "log10 does not accept any keyword arguments"
    assert len(args) == 1, "log10 expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)
    element_type = target_mlir_type.element_type

    x = convert(complex_dialect.re(z), element_type)
    y = convert(complex_dialect.im(z), element_type)

    # log10(z) = log(z) / ln(10), where log(z) = log|z| + i*arg(z)
    x_sq = arith.mulf(x, x)
    y_sq = arith.mulf(y, y)
    hypot_xy = math_dialect.sqrt(arith.addf(x_sq, y_sq))
    log_abs = math_dialect.log(hypot_xy)
    phase = math_dialect.atan2(y, x)

    ln10 = arith.constant(result=element_type, value=2.302585092994046)
    real = arith.divf(log_abs, ln10)
    imag = arith.divf(phase, ln10)

    result = complex_dialect.create_(target_mlir_type, real, imag)
    mlir_lower.store_var(target, result)


@lower(cmath.sqrt, types.Complex)
def cmath_sqrt_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.sqrt with proper special case handling."""
    assert not kwargs, "sqrt does not accept any keyword arguments"
    assert len(args) == 1, "sqrt expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_sqrt", z, target_mlir_type)
    mlir_lower.store_var(target, result)


# ============================================================================
# cmath Trigonometric Functions
# ============================================================================


@lower(cmath.cos, types.Complex)
def cmath_cos_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.cos with proper special case handling."""
    assert not kwargs, "cos does not accept any keyword arguments"
    assert len(args) == 1, "cos expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_cos", z, target_mlir_type)
    mlir_lower.store_var(target, result)


@lower(cmath.sin, types.Complex)
def cmath_sin_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.sin with proper special case handling."""
    assert not kwargs, "sin does not accept any keyword arguments"
    assert len(args) == 1, "sin expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_sin", z, target_mlir_type)
    mlir_lower.store_var(target, result)


@lower(cmath.tan, types.Complex)
def cmath_tan_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.tan."""
    assert not kwargs, "tan does not accept any keyword arguments"
    assert len(args) == 1, "tan expects exactly 1 argument"
    x = mlir_lower.load_var(args[0])
    result = complex_dialect.tan(x)
    mlir_lower.store_var(target, result)


@lower(cmath.acos, types.Complex)
def cmath_acos_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.acos using acos(z) = pi/2 - asin(z)."""
    assert not kwargs, "acos does not accept any keyword arguments"
    assert len(args) == 1, "acos expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)
    element_type = target_mlir_type.element_type

    asin_result = _call_cmath_intrinsic(mlir_lower, "cmath_asin", z, target_mlir_type)
    asin_real = complex_dialect.re(asin_result)
    asin_imag = complex_dialect.im(asin_result)

    pi_over_2 = arith.constant(result=element_type, value=1.5707963267948966)
    result_real = arith.subf(pi_over_2, asin_real)
    result_imag = arith.negf(asin_imag)
    result = complex_dialect.create_(target_mlir_type, result_real, result_imag)
    mlir_lower.store_var(target, result)


@lower(cmath.asin, types.Complex)
def cmath_asin_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.asin with proper special case handling."""
    assert not kwargs, "asin does not accept any keyword arguments"
    assert len(args) == 1, "asin expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_asin", z, target_mlir_type)
    mlir_lower.store_var(target, result)


@lower(cmath.atan, types.Complex)
def cmath_atan_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.atan with proper special case handling."""
    assert not kwargs, "atan does not accept any keyword arguments"
    assert len(args) == 1, "atan expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_atan", z, target_mlir_type)
    mlir_lower.store_var(target, result)


# ============================================================================
# cmath Hyperbolic Functions
# ============================================================================


@lower(cmath.cosh, types.Complex)
def cmath_cosh_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.cosh with proper special case handling."""
    assert not kwargs, "cosh does not accept any keyword arguments"
    assert len(args) == 1, "cosh expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_cosh", z, target_mlir_type)
    mlir_lower.store_var(target, result)


@lower(cmath.sinh, types.Complex)
def cmath_sinh_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.sinh with proper special case handling."""
    assert not kwargs, "sinh does not accept any keyword arguments"
    assert len(args) == 1, "sinh expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_sinh", z, target_mlir_type)
    mlir_lower.store_var(target, result)


@lower(cmath.tanh, types.Complex)
def cmath_tanh_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.tanh."""
    assert not kwargs, "tanh does not accept any keyword arguments"
    assert len(args) == 1, "tanh expects exactly 1 argument"
    x = mlir_lower.load_var(args[0])
    result = complex_dialect.tanh(x)
    mlir_lower.store_var(target, result)


@lower(cmath.acosh, types.Complex)
def cmath_acosh_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.acosh using acosh(z) = sign(z.imag)*i*acos(z).

    CPython: acosh(z) = i*acos(z) if z.imag >= 0, else -i*acos(z)
    Equivalently: acosh.real = |acos.imag|, acosh.imag = copysign(acos.real, z.imag)
    """
    assert not kwargs, "acosh does not accept any keyword arguments"
    assert len(args) == 1, "acosh expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)
    element_type = target_mlir_type.element_type

    # First compute asin(z), then acos = pi/2 - asin
    asin_result = _call_cmath_intrinsic(mlir_lower, "cmath_asin", z, target_mlir_type)
    asin_real = complex_dialect.re(asin_result)
    asin_imag = complex_dialect.im(asin_result)

    pi_over_2 = arith.constant(result=element_type, value=1.5707963267948966)
    acos_real = arith.subf(pi_over_2, asin_real)
    acos_imag = arith.negf(asin_imag)

    # acosh(z) = sign(z.imag) * i * acos(z)
    # i * (acos_real + i*acos_imag) = -acos_imag + i*acos_real
    # So: acosh.real = -acos_imag = |acos_imag| (should be non-negative)
    #     acosh.imag = copysign(acos_real, z.imag)
    z_imag = convert(complex_dialect.im(z), element_type)
    result_real = math_dialect.absf(acos_imag)
    result_imag = math_dialect.copysign(acos_real, z_imag)
    result = complex_dialect.create_(target_mlir_type, result_real, result_imag)
    mlir_lower.store_var(target, result)


@lower(cmath.asinh, types.Complex)
def cmath_asinh_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.asinh with proper special case handling."""
    assert not kwargs, "asinh does not accept any keyword arguments"
    assert len(args) == 1, "asinh expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)

    result = _call_cmath_intrinsic(mlir_lower, "cmath_asinh", z, target_mlir_type)
    mlir_lower.store_var(target, result)


@lower(cmath.atanh, types.Complex)
def cmath_atanh_cg(mlir_lower, target, args, kwargs):
    """Code generator for cmath.atanh using atanh(z) = 0.5 * log((1+z)/(1-z)).

    Special cases:
    - atanh(x ± inf*j) = copysign(0, x) ± pi/2*j for any x
    - For real z with |z| > 1: atanh(z) = atanh(1/z) + copysign(pi/2, imag(z))*j
    """
    assert not kwargs, "atanh does not accept any keyword arguments"
    assert len(args) == 1, "atanh expects exactly 1 argument"
    z = mlir_lower.load_var(args[0])

    target_type = mlir_lower.get_numba_type(target.name)
    target_mlir_type = mlir_lower.get_mlir_type(target_type)
    element_type = target_mlir_type.element_type

    one = arith.constant(result=element_type, value=1.0)
    half = arith.constant(result=element_type, value=0.5)
    zero = arith.constant(result=element_type, value=0.0)
    pi_over_2 = arith.constant(result=element_type, value=1.5707963267948966)
    one_complex = complex_dialect.create_(target_mlir_type, one, zero)
    half_complex = complex_dialect.create_(target_mlir_type, half, zero)

    x = complex_dialect.re(z)
    y = complex_dialect.im(z)

    # Special cases:
    # - imag is infinite: atanh(x ± inf*j) = copysign(0, x) ± pi/2*j
    # - real is infinite: atanh(±inf + y*j) = copysign(0, x) + copysign(pi/2, y)*j
    # - y is nan with x infinite: result = copysign(0, x) + nan*j
    y_is_inf = math_dialect.isinf(y)
    x_is_inf = math_dialect.isinf(x)
    y_is_nan = math_dialect.isnan(y)

    # Check if z is effectively real (y == 0) and |x| > 1 (branch cut case)
    y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, y, zero)
    abs_x = math_dialect.absf(x)
    abs_x_gt_1 = arith.cmpf(arith.CmpFPredicate.OGT, abs_x, one)
    is_branch_cut = arith.andi(y_is_zero, abs_x_gt_1)

    # 1 + z
    one_plus_z = complex_dialect.add(one_complex, z)

    # 1 - z
    one_minus_z = complex_dialect.sub(one_complex, z)

    # (1+z)/(1-z)
    ratio = complex_dialect.div(one_plus_z, one_minus_z)

    # log((1+z)/(1-z))
    log_term = complex_dialect.log(ratio)

    # 0.5 * log(...)
    formula_result = complex_dialect.mul(half_complex, log_term)

    formula_real = complex_dialect.re(formula_result)
    formula_imag = complex_dialect.im(formula_result)

    # For branch cut case: correct the imaginary part sign based on sign of y (even if ±0)
    # The imaginary part should be copysign(pi/2, y) when |x| > 1 and y == ±0
    branch_cut_imag = math_dialect.copysign(pi_over_2, y)
    corrected_formula_imag = arith.select(is_branch_cut, branch_cut_imag, formula_imag)

    # Special case results
    # For y_is_inf: atanh(x ± inf*j) = copysign(0, x) ± pi/2*j
    # For x_is_inf: atanh(±inf + y*j) = copysign(0, x) + copysign(pi/2, y)*j
    # For y_is_nan with x_is_inf: atanh(±inf + nan*j) = copysign(0, x) + nan*j
    signed_zero = math_dialect.copysign(zero, x)
    imag_sign = math_dialect.copysign(pi_over_2, y)
    nan_val = arith.constant(result=element_type, value=float("nan"))

    # When y is nan with infinite x, imag should be nan
    y_nan_with_inf_x = arith.andi(y_is_nan, x_is_inf)

    # Priority: y_is_inf > (y_nan_with_inf_x for imag) > x_is_inf > branch_cut > formula
    final_real = arith.select(
        y_is_inf, signed_zero, arith.select(x_is_inf, signed_zero, formula_real)
    )
    final_imag = arith.select(
        y_is_inf,
        imag_sign,
        arith.select(
            y_nan_with_inf_x,
            nan_val,
            arith.select(x_is_inf, imag_sign, corrected_formula_imag),
        ),
    )
    result = complex_dialect.create_(target_mlir_type, final_real, final_imag)

    mlir_lower.store_var(target, result)
