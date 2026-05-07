# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Complex math intrinsics with IEEE 754 special case handling.
"""

from typing import TypeVar
from numba_cuda_mlir._mlir.dialects import math as math_dialect
from numba_cuda_mlir.mlir.dialect_exts import func, scf, arith
from numba_cuda_mlir.mlir.context import mlir_mod_ctx
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.mlir.dialect_exts.scf import (
    if_ctx_manager as if_,
    else_ctx_manager as else_,
)
from numba_cuda_mlir._mlir.ir import UnitAttr

FTy = TypeVar("FTy")


def _nan(fty):
    return arith.constant(float("nan"), fty)


def _inf(fty):
    return arith.constant(float("inf"), fty)


def _neg_inf(fty):
    return arith.constant(float("-inf"), fty)


def _zero(fty):
    return arith.constant(0.0, fty)


def _neg_zero(fty):
    return arith.constant(-0.0, fty)


@func.func(sym_visibility="private", generics=[FTy])
def cmath_exp(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    nan = _nan(FTy)
    neg_inf = _neg_inf(FTy)

    x_is_finite = math_dialect.isfinite(z_real)
    y_is_finite = math_dialect.isfinite(z_imag)
    x_isnan = math_dialect.isnan(z_real)
    x_is_neg_inf = arith.cmpf(arith.CmpFPredicate.OEQ, z_real, neg_inf)
    y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, z_imag, c0)

    with if_(x_is_finite, results=[FTy, FTy]) as result:
        with if_(y_is_finite, results=[FTy, FTy]) as result_inner:
            exp_x = math_dialect.exp(z_real)
            cos_y = math_dialect.cos(z_imag)
            sin_y = math_dialect.sin(z_imag)
            real_part = arith.mulf(exp_x, cos_y)
            imag_part = arith.mulf(exp_x, sin_y)
            scf.yield_(real_part, imag_part)
        with else_(result_inner):
            scf.yield_(nan, nan)
        scf.yield_(result_inner.results)
    with else_(result):
        with if_(x_isnan, results=[FTy, FTy]) as result_outer:
            with if_(y_is_zero, results=[FTy, FTy]) as result_nan:
                scf.yield_(z_real, z_imag)
            with else_(result_nan):
                scf.yield_(nan, nan)
            scf.yield_(result_nan.results)
        with else_(result_outer):
            with if_(x_is_neg_inf, results=[FTy, FTy]) as result_inf:
                with if_(y_is_finite, results=[FTy, FTy]) as result_neg_inf_yfin:
                    cos_y_neg = math_dialect.cos(z_imag)
                    sin_y_neg = math_dialect.sin(z_imag)
                    real_part_neg = math_dialect.copysign(c0, cos_y_neg)
                    imag_part_neg = math_dialect.copysign(c0, sin_y_neg)
                    scf.yield_(real_part_neg, imag_part_neg)
                with else_(result_neg_inf_yfin):
                    scf.yield_(c0, c0)
                scf.yield_(result_neg_inf_yfin.results)
            with else_(result_inf):
                with if_(y_is_finite, results=[FTy, FTy]) as result_pos_inf:
                    cos_y = math_dialect.cos(z_imag)
                    sin_y = math_dialect.sin(z_imag)
                    cos_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, cos_y, c0)
                    sin_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, sin_y, c0)
                    real_mul = arith.mulf(z_real, cos_y)
                    imag_mul = arith.mulf(z_real, sin_y)
                    real_part = arith.select(cos_is_zero, cos_y, real_mul)
                    imag_part = arith.select(sin_is_zero, sin_y, imag_mul)
                    scf.yield_(real_part, imag_part)
                with else_(result_pos_inf):
                    scf.yield_(z_real, nan)
                scf.yield_(result_pos_inf.results)
            scf.yield_(result_inf.results)
        scf.yield_(result_outer.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_sinh(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    nan = _nan(FTy)

    x_isinf = math_dialect.isinf(z_real)
    y_isnan = math_dialect.isnan(z_imag)
    y_is_finite = math_dialect.isfinite(z_imag)
    x_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, z_real, c0)

    with if_(x_isinf, results=[FTy, FTy]) as result:
        with if_(y_isnan, results=[FTy, FTy]) as result_inf:
            scf.yield_(z_real, z_imag)
        with else_(result_inf):
            cos_y = math_dialect.cos(z_imag)
            sin_y = math_dialect.sin(z_imag)
            cos_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, cos_y, c0)
            sin_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, sin_y, c0)
            abs_x = math_dialect.absf(z_real)
            real_mul = arith.mulf(z_real, cos_y)
            imag_mul = arith.mulf(abs_x, sin_y)
            real_part = arith.select(cos_is_zero, cos_y, real_mul)
            imag_part = arith.select(sin_is_zero, sin_y, imag_mul)
            scf.yield_(real_part, imag_part)
        scf.yield_(result_inf.results)
    with else_(result):
        with if_(y_is_finite, results=[FTy, FTy]) as result_finite:
            sinh_x = math_dialect.sinh(z_real)
            cosh_x = math_dialect.cosh(z_real)
            cos_y = math_dialect.cos(z_imag)
            sin_y = math_dialect.sin(z_imag)
            real_part = arith.mulf(sinh_x, cos_y)
            imag_part = arith.mulf(cosh_x, sin_y)
            scf.yield_(real_part, imag_part)
        with else_(result_finite):
            with if_(x_is_zero, results=[FTy, FTy]) as result_yinf:
                scf.yield_(z_real, nan)
            with else_(result_yinf):
                scf.yield_(nan, nan)
            scf.yield_(result_yinf.results)
        scf.yield_(result_finite.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_cosh(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    nan = _nan(FTy)

    x_isinf = math_dialect.isinf(z_real)
    x_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, z_real, c0)
    y_isnan = math_dialect.isnan(z_imag)
    y_is_finite = math_dialect.isfinite(z_imag)
    y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, z_imag, c0)

    with if_(x_isinf, results=[FTy, FTy]) as result:
        abs_x = math_dialect.absf(z_real)
        with if_(y_isnan, results=[FTy, FTy]) as result_inf:
            scf.yield_(abs_x, z_imag)
        with else_(result_inf):
            with if_(y_is_zero, results=[FTy, FTy]) as result_not_nan:
                one = arith.constant(1.0, FTy)
                x_sign = math_dialect.copysign(one, z_real)
                y_sign = math_dialect.copysign(one, z_imag)
                sign_product = arith.mulf(x_sign, y_sign)
                imag_with_sign = math_dialect.copysign(c0, sign_product)
                scf.yield_(abs_x, imag_with_sign)
            with else_(result_not_nan):
                cos_y = math_dialect.cos(z_imag)
                sin_y = math_dialect.sin(z_imag)
                real_sign = math_dialect.copysign(z_real, cos_y)
                imag_sign = math_dialect.copysign(z_real, sin_y)
                x_is_neg = arith.cmpf(arith.CmpFPredicate.OLT, z_real, c0)
                neg_imag_sign = arith.negf(imag_sign)
                imag_part = arith.select(x_is_neg, neg_imag_sign, imag_sign)
                scf.yield_(real_sign, imag_part)
            scf.yield_(result_not_nan.results)
        scf.yield_(result_inf.results)
    with else_(result):
        with if_(y_is_finite, results=[FTy, FTy]) as result_finite:
            cosh_x = math_dialect.cosh(z_real)
            sinh_x = math_dialect.sinh(z_real)
            cos_y = math_dialect.cos(z_imag)
            sin_y = math_dialect.sin(z_imag)
            real_part = arith.mulf(cosh_x, cos_y)
            imag_part = arith.mulf(sinh_x, sin_y)
            scf.yield_(real_part, imag_part)
        with else_(result_finite):
            with if_(x_is_zero, results=[FTy, FTy]) as result_yinf:
                scf.yield_(nan, c0)
            with else_(result_yinf):
                scf.yield_(nan, nan)
            scf.yield_(result_yinf.results)
        scf.yield_(result_finite.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_sin(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    nan = _nan(FTy)
    inf = _inf(FTy)

    x_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, z_real, c0)
    y_isinf = math_dialect.isinf(z_imag)
    y_isnan = math_dialect.isnan(z_imag)
    x_is_finite = math_dialect.isfinite(z_real)
    y_is_finite = math_dialect.isfinite(z_imag)

    pure_imag_inf = arith.andi(x_is_zero, y_isinf)
    with if_(pure_imag_inf, results=[FTy, FTy]) as result:
        scf.yield_(z_real, z_imag)
    with else_(result):
        pure_imag_nan = arith.andi(x_is_zero, y_isnan)
        with if_(pure_imag_nan, results=[FTy, FTy]) as result_pure:
            scf.yield_(z_real, z_imag)
        with else_(result_pure):
            with if_(x_is_finite, results=[FTy, FTy]) as result_normal:
                with if_(y_is_finite, results=[FTy, FTy]) as result_xfin:
                    sin_x = math_dialect.sin(z_real)
                    cos_x = math_dialect.cos(z_real)
                    sinh_y = math_dialect.sinh(z_imag)
                    cosh_y = math_dialect.cosh(z_imag)
                    real_part = arith.mulf(sin_x, cosh_y)
                    imag_part = arith.mulf(cos_x, sinh_y)
                    scf.yield_(real_part, imag_part)
                with else_(result_xfin):
                    with if_(y_isinf, results=[FTy, FTy]) as result_ynotfin:
                        sin_x = math_dialect.sin(z_real)
                        cos_x = math_dialect.cos(z_real)
                        real_part = math_dialect.copysign(inf, sin_x)
                        imag_sign = arith.mulf(cos_x, z_imag)
                        imag_part = math_dialect.copysign(inf, imag_sign)
                        scf.yield_(real_part, imag_part)
                    with else_(result_ynotfin):
                        scf.yield_(nan, nan)
                    scf.yield_(result_ynotfin.results)
                scf.yield_(result_xfin.results)
            with else_(result_normal):
                with if_(y_is_finite, results=[FTy, FTy]) as result_xinf:
                    y_is_zero_inner = arith.cmpf(arith.CmpFPredicate.OEQ, z_imag, c0)
                    with if_(y_is_zero_inner, results=[FTy, FTy]) as result_yzero:
                        scf.yield_(nan, z_imag)
                    with else_(result_yzero):
                        scf.yield_(nan, nan)
                    scf.yield_(result_yzero.results)
                with else_(result_xinf):
                    scf.yield_(nan, nan)
                scf.yield_(result_xinf.results)
            scf.yield_(result_normal.results)
        scf.yield_(result_pure.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_cos(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    nan = _nan(FTy)

    x_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, z_real, c0)
    x_is_finite = math_dialect.isfinite(z_real)
    y_is_finite = math_dialect.isfinite(z_imag)

    with if_(x_is_finite, results=[FTy, FTy]) as result:
        with if_(y_is_finite, results=[FTy, FTy]) as result_xfin:
            cos_x = math_dialect.cos(z_real)
            sin_x = math_dialect.sin(z_real)
            cosh_y = math_dialect.cosh(z_imag)
            sinh_y = math_dialect.sinh(z_imag)
            real_part = arith.mulf(cos_x, cosh_y)
            sin_sinh = arith.mulf(sin_x, sinh_y)
            imag_part = arith.negf(sin_sinh)
            scf.yield_(real_part, imag_part)
        with else_(result_xfin):
            y_isnan = math_dialect.isnan(z_imag)
            with if_(y_isnan, results=[FTy, FTy]) as result_ynan:
                scf.yield_(nan, nan)
            with else_(result_ynan):
                with if_(x_is_zero, results=[FTy, FTy]) as result_yinf:
                    abs_y = math_dialect.absf(z_imag)
                    one = arith.constant(1.0, FTy)
                    x_sign = math_dialect.copysign(one, z_real)
                    y_sign = math_dialect.copysign(one, z_imag)
                    neg_x_sign = arith.negf(x_sign)
                    imag_sign = arith.mulf(neg_x_sign, y_sign)
                    imag_zero = math_dialect.copysign(c0, imag_sign)
                    scf.yield_(abs_y, imag_zero)
                with else_(result_yinf):
                    inf_val = _inf(FTy)
                    cos_x = math_dialect.cos(z_real)
                    sin_x = math_dialect.sin(z_real)
                    neg_sin_x = arith.negf(sin_x)
                    imag_sign = arith.mulf(neg_sin_x, z_imag)
                    real_part_yinf = math_dialect.copysign(inf_val, cos_x)
                    imag_part_yinf = math_dialect.copysign(inf_val, imag_sign)
                    scf.yield_(real_part_yinf, imag_part_yinf)
                scf.yield_(result_yinf.results)
            scf.yield_(result_ynan.results)
        scf.yield_(result_xfin.results)
    with else_(result):
        with if_(y_is_finite, results=[FTy, FTy]) as result_xinf:
            y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, z_imag, c0)
            with if_(y_is_zero, results=[FTy, FTy]) as result_yzero:
                scf.yield_(nan, z_imag)
            with else_(result_yzero):
                scf.yield_(nan, nan)
            scf.yield_(result_yzero.results)
        with else_(result_xinf):
            scf.yield_(nan, nan)
        scf.yield_(result_xinf.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_rect(r: FTy, phi: FTy):
    c0 = _zero(FTy)
    nan = _nan(FTy)

    r_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, r, c0)
    r_isinf = math_dialect.isinf(r)
    phi_is_finite = math_dialect.isfinite(phi)

    with if_(phi_is_finite, results=[FTy, FTy]) as result:
        cos_phi = math_dialect.cos(phi)
        sin_phi = math_dialect.sin(phi)
        with if_(r_isinf, results=[FTy, FTy]) as result_finite:
            cos_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, cos_phi, c0)
            sin_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, sin_phi, c0)
            real_mul = arith.mulf(r, cos_phi)
            imag_mul = arith.mulf(r, sin_phi)
            one = arith.constant(1.0, FTy)
            r_sign = math_dialect.copysign(one, r)
            cos_sign = math_dialect.copysign(one, cos_phi)
            sin_sign = math_dialect.copysign(one, sin_phi)
            real_sign = arith.mulf(r_sign, cos_sign)
            imag_sign = arith.mulf(r_sign, sin_sign)
            real_zero = math_dialect.copysign(c0, real_sign)
            imag_zero = math_dialect.copysign(c0, imag_sign)
            real_part = arith.select(cos_is_zero, real_zero, real_mul)
            imag_part = arith.select(sin_is_zero, imag_zero, imag_mul)
            scf.yield_(real_part, imag_part)
        with else_(result_finite):
            real_part = arith.mulf(r, cos_phi)
            imag_part = arith.mulf(r, sin_phi)
            scf.yield_(real_part, imag_part)
        scf.yield_(result_finite.results)
    with else_(result):
        with if_(r_is_zero, results=[FTy, FTy]) as result_phi_inf:
            abs_r = math_dialect.absf(r)
            scf.yield_(abs_r, c0)
        with else_(result_phi_inf):
            with if_(r_isinf, results=[FTy, FTy]) as result_r_nonzero:
                scf.yield_(r, nan)
            with else_(result_r_nonzero):
                scf.yield_(nan, nan)
            scf.yield_(result_r_nonzero.results)
        scf.yield_(result_phi_inf.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_sqrt(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    nan = _nan(FTy)
    inf = _inf(FTy)
    neg_inf = _neg_inf(FTy)
    half = arith.constant(0.5, FTy)
    two = arith.constant(2.0, FTy)

    x = z_real
    y = z_imag

    x_isinf = math_dialect.isinf(x)
    y_isinf = math_dialect.isinf(y)
    x_isnan = math_dialect.isnan(x)
    y_isnan = math_dialect.isnan(y)
    x_is_pos_inf = arith.cmpf(arith.CmpFPredicate.OEQ, x, inf)
    x_is_neg_inf = arith.cmpf(arith.CmpFPredicate.OEQ, x, neg_inf)

    with if_(y_isinf, results=[FTy, FTy]) as result:
        scf.yield_(inf, y)
    with else_(result):
        with if_(x_is_neg_inf, results=[FTy, FTy]) as result_xinf:
            with if_(y_isnan, results=[FTy, FTy]) as result_ynan:
                scf.yield_(y, y)
            with else_(result_ynan):
                imag_res = math_dialect.copysign(inf, y)
                scf.yield_(c0, imag_res)
            scf.yield_(result_ynan.results)
        with else_(result_xinf):
            with if_(x_is_pos_inf, results=[FTy, FTy]) as result_xposinf:
                with if_(y_isnan, results=[FTy, FTy]) as result_ynan2:
                    scf.yield_(inf, y)
                with else_(result_ynan2):
                    imag_res = math_dialect.copysign(c0, y)
                    scf.yield_(inf, imag_res)
                scf.yield_(result_ynan2.results)
            with else_(result_xposinf):
                with if_(x_isnan, results=[FTy, FTy]) as result_xnan:
                    scf.yield_(nan, nan)
                with else_(result_xnan):
                    with if_(y_isnan, results=[FTy, FTy]) as result_ynan3:
                        scf.yield_(nan, nan)
                    with else_(result_ynan3):
                        abs_x = math_dialect.absf(x)
                        abs_y = math_dialect.absf(y)
                        x2 = arith.mulf(x, x)
                        y2 = arith.mulf(y, y)
                        modulus = math_dialect.sqrt(arith.addf(x2, y2))
                        t = math_dialect.sqrt(arith.mulf(half, arith.addf(abs_x, modulus)))
                        x_is_neg = arith.cmpf(arith.CmpFPredicate.OLT, x, c0)
                        two_t = arith.mulf(two, t)
                        y_div_2t = arith.divf(y, two_t)
                        abs_y_div_2t = arith.divf(abs_y, two_t)
                        t_signed = math_dialect.copysign(t, y)
                        real_res = arith.select(x_is_neg, abs_y_div_2t, t)
                        imag_res = arith.select(x_is_neg, t_signed, y_div_2t)
                        x_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, x, c0)
                        y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, y, c0)
                        both_zero = arith.andi(x_is_zero, y_is_zero)
                        zero_imag = math_dialect.copysign(c0, y)
                        final_real = arith.select(both_zero, c0, real_res)
                        final_imag = arith.select(both_zero, zero_imag, imag_res)
                        scf.yield_(final_real, final_imag)
                    scf.yield_(result_ynan3.results)
                scf.yield_(result_xnan.results)
            scf.yield_(result_xposinf.results)
        scf.yield_(result_xinf.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_acos(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    c1 = arith.constant(1.0, FTy)
    nan = _nan(FTy)
    inf = _inf(FTy)
    pi = arith.constant(3.141592653589793, FTy)
    pi_over_2 = arith.constant(1.5707963267948966, FTy)

    x = z_real
    y = z_imag

    x_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, x, c0)
    y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, y, c0)
    x_isinf = math_dialect.isinf(x)
    y_isinf = math_dialect.isinf(y)
    x_isnan = math_dialect.isnan(x)
    y_isnan = math_dialect.isnan(y)
    x_is_neg = arith.cmpf(arith.CmpFPredicate.OLT, x, c0)

    both_zero = arith.andi(x_is_zero, y_is_zero)
    with if_(both_zero, results=[FTy, FTy]) as result:
        neg_y = arith.negf(y)
        scf.yield_(pi_over_2, neg_y)
    with else_(result):
        with if_(y_isinf, results=[FTy, FTy]) as result_yinf:
            imag_res = arith.negf(y)
            scf.yield_(pi_over_2, imag_res)
        with else_(result_yinf):
            with if_(x_isinf, results=[FTy, FTy]) as result_xinf:
                with if_(y_isnan, results=[FTy, FTy]) as result_ynan:
                    scf.yield_(nan, nan)
                with else_(result_ynan):
                    neg_y_inf = arith.negf(y)
                    imag_pos_inf = math_dialect.copysign(c0, neg_y_inf)
                    imag_neg_inf = math_dialect.copysign(inf, neg_y_inf)
                    real_res = arith.select(x_is_neg, pi, c0)
                    imag_res = arith.select(x_is_neg, imag_neg_inf, imag_pos_inf)
                    scf.yield_(real_res, imag_res)
                scf.yield_(result_ynan.results)
            with else_(result_xinf):
                with if_(x_isnan, results=[FTy, FTy]) as result_xnan:
                    scf.yield_(nan, nan)
                with else_(result_xnan):
                    with if_(y_isnan, results=[FTy, FTy]) as result_ynan2:
                        scf.yield_(nan, nan)
                    with else_(result_ynan2):
                        x2 = arith.mulf(x, x)
                        y2 = arith.mulf(y, y)
                        one_minus_x2_plus_y2 = arith.subf(c1, arith.subf(x2, y2))
                        two_xy = arith.mulf(arith.constant(2.0, FTy), arith.mulf(x, y))
                        neg_two_xy = arith.negf(two_xy)
                        mod_sq = arith.addf(
                            arith.mulf(one_minus_x2_plus_y2, one_minus_x2_plus_y2),
                            arith.mulf(neg_two_xy, neg_two_xy),
                        )
                        mod = math_dialect.sqrt(mod_sq)
                        phase = math_dialect.atan2(neg_two_xy, one_minus_x2_plus_y2)
                        sqrt_mod = math_dialect.sqrt(mod)
                        half_phase = arith.mulf(arith.constant(0.5, FTy), phase)
                        sqrt_real = arith.mulf(sqrt_mod, math_dialect.cos(half_phase))
                        sqrt_imag = arith.mulf(sqrt_mod, math_dialect.sin(half_phase))
                        sum_real = arith.subf(x, sqrt_imag)
                        sum_imag = arith.addf(y, sqrt_real)
                        log_mod = math_dialect.log(
                            math_dialect.sqrt(
                                arith.addf(
                                    arith.mulf(sum_real, sum_real),
                                    arith.mulf(sum_imag, sum_imag),
                                )
                            )
                        )
                        log_phase = math_dialect.atan2(sum_imag, sum_real)
                        final_real = log_phase
                        final_imag = arith.negf(log_mod)
                        scf.yield_(final_real, final_imag)
                    scf.yield_(result_ynan2.results)
                scf.yield_(result_xnan.results)
            scf.yield_(result_xinf.results)
        scf.yield_(result_yinf.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_acosh(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    c1 = arith.constant(1.0, FTy)
    nan = _nan(FTy)
    inf = _inf(FTy)
    pi = arith.constant(3.141592653589793, FTy)
    pi_over_2 = arith.constant(1.5707963267948966, FTy)

    x = z_real
    y = z_imag

    x_isinf = math_dialect.isinf(x)
    y_isinf = math_dialect.isinf(y)
    x_isnan = math_dialect.isnan(x)
    y_isnan = math_dialect.isnan(y)
    x_is_neg = arith.cmpf(arith.CmpFPredicate.OLT, x, c0)
    y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, y, c0)
    x_is_one = arith.cmpf(arith.CmpFPredicate.OEQ, x, c1)

    one_with_zero = arith.andi(x_is_one, y_is_zero)
    with if_(one_with_zero, results=[FTy, FTy]) as result:
        scf.yield_(c0, y)
    with else_(result):
        with if_(y_isinf, results=[FTy, FTy]) as result_yinf:
            imag_res = math_dialect.copysign(pi_over_2, y)
            scf.yield_(inf, imag_res)
        with else_(result_yinf):
            with if_(x_isinf, results=[FTy, FTy]) as result_xinf:
                with if_(y_isnan, results=[FTy, FTy]) as result_ynan:
                    scf.yield_(inf, nan)
                with else_(result_ynan):
                    imag_pos = math_dialect.copysign(c0, y)
                    imag_neg = math_dialect.copysign(pi, y)
                    imag_res = arith.select(x_is_neg, imag_neg, imag_pos)
                    scf.yield_(inf, imag_res)
                scf.yield_(result_ynan.results)
            with else_(result_xinf):
                with if_(x_isnan, results=[FTy, FTy]) as result_xnan:
                    scf.yield_(nan, nan)
                with else_(result_xnan):
                    with if_(y_isnan, results=[FTy, FTy]) as result_ynan2:
                        scf.yield_(nan, nan)
                    with else_(result_ynan2):
                        x2 = arith.mulf(x, x)
                        y2 = arith.mulf(y, y)
                        two_xy = arith.mulf(arith.constant(2.0, FTy), arith.mulf(x, y))
                        z2_minus_1_real = arith.subf(arith.subf(x2, y2), c1)
                        z2_minus_1_imag = two_xy
                        mod_sq = arith.addf(
                            arith.mulf(z2_minus_1_real, z2_minus_1_real),
                            arith.mulf(z2_minus_1_imag, z2_minus_1_imag),
                        )
                        mod = math_dialect.sqrt(mod_sq)
                        phase = math_dialect.atan2(z2_minus_1_imag, z2_minus_1_real)
                        sqrt_mod = math_dialect.sqrt(mod)
                        half_phase = arith.mulf(arith.constant(0.5, FTy), phase)
                        sqrt_real = arith.mulf(sqrt_mod, math_dialect.cos(half_phase))
                        sqrt_imag = arith.mulf(sqrt_mod, math_dialect.sin(half_phase))
                        sum_real = arith.addf(x, sqrt_real)
                        sum_imag = arith.addf(y, sqrt_imag)
                        log_mod = math_dialect.log(
                            math_dialect.sqrt(
                                arith.addf(
                                    arith.mulf(sum_real, sum_real),
                                    arith.mulf(sum_imag, sum_imag),
                                )
                            )
                        )
                        log_phase = math_dialect.atan2(sum_imag, sum_real)
                        abs_log_mod = math_dialect.absf(log_mod)
                        abs_log_phase = math_dialect.absf(log_phase)
                        final_imag = math_dialect.copysign(abs_log_phase, y)
                        scf.yield_(abs_log_mod, final_imag)
                    scf.yield_(result_ynan2.results)
                scf.yield_(result_xnan.results)
            scf.yield_(result_xinf.results)
        scf.yield_(result_yinf.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_atan(z_real: FTy, z_imag: FTy):
    c0 = _zero(FTy)
    c1 = arith.constant(1.0, FTy)
    nan = _nan(FTy)
    inf = _inf(FTy)
    pi_over_2 = arith.constant(1.5707963267948966, FTy)

    x = z_real
    y = z_imag

    x_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, x, c0)
    y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, y, c0)
    x_isinf = math_dialect.isinf(x)
    y_isinf = math_dialect.isinf(y)
    x_isnan = math_dialect.isnan(x)
    y_isnan = math_dialect.isnan(y)

    both_zero = arith.andi(x_is_zero, y_is_zero)
    with if_(both_zero, results=[FTy, FTy]) as result:
        scf.yield_(x, y)
    with else_(result):
        with if_(y_isinf, results=[FTy, FTy]) as result_yinf:
            real_res = math_dialect.copysign(pi_over_2, x)
            imag_res = math_dialect.copysign(c0, y)
            scf.yield_(real_res, imag_res)
        with else_(result_yinf):
            with if_(x_isinf, results=[FTy, FTy]) as result_xinf:
                with if_(y_isnan, results=[FTy, FTy]) as result_ynan:
                    real_res = math_dialect.copysign(pi_over_2, x)
                    scf.yield_(real_res, nan)
                with else_(result_ynan):
                    real_res = math_dialect.copysign(pi_over_2, x)
                    imag_res = math_dialect.copysign(c0, y)
                    scf.yield_(real_res, imag_res)
                scf.yield_(result_ynan.results)
            with else_(result_xinf):
                with if_(x_isnan, results=[FTy, FTy]) as result_xnan:
                    scf.yield_(nan, nan)
                with else_(result_xnan):
                    with if_(y_isnan, results=[FTy, FTy]) as result_ynan2:
                        scf.yield_(nan, nan)
                    with else_(result_ynan2):
                        with if_(x_is_zero, results=[FTy, FTy]) as result_xzero:
                            abs_y = math_dialect.absf(y)
                            abs_y_gt_1 = arith.cmpf(arith.CmpFPredicate.OGT, abs_y, c1)
                            one_plus_y = arith.addf(c1, y)
                            one_minus_y = arith.subf(c1, y)
                            half = arith.constant(0.5, FTy)
                            atanh_y = arith.mulf(
                                half,
                                math_dialect.log(arith.divf(one_plus_y, one_minus_y)),
                            )
                            y_plus_1 = arith.addf(y, c1)
                            y_minus_1 = arith.subf(y, c1)
                            atanh_inv_y = arith.mulf(
                                half,
                                math_dialect.log(arith.divf(y_plus_1, y_minus_1)),
                            )
                            real_big = math_dialect.copysign(pi_over_2, x)
                            real_small = x
                            real_res = arith.select(abs_y_gt_1, real_big, real_small)
                            imag_res = arith.select(abs_y_gt_1, atanh_inv_y, atanh_y)
                            scf.yield_(real_res, imag_res)
                        with else_(result_xzero):
                            one_plus_iz_real = arith.subf(c1, y)
                            one_plus_iz_imag = x
                            one_minus_iz_real = arith.addf(c1, y)
                            one_minus_iz_imag = arith.negf(x)
                            denom = arith.addf(
                                arith.mulf(one_minus_iz_real, one_minus_iz_real),
                                arith.mulf(one_minus_iz_imag, one_minus_iz_imag),
                            )
                            ratio_real = arith.divf(
                                arith.addf(
                                    arith.mulf(one_plus_iz_real, one_minus_iz_real),
                                    arith.mulf(one_plus_iz_imag, one_minus_iz_imag),
                                ),
                                denom,
                            )
                            ratio_imag = arith.divf(
                                arith.subf(
                                    arith.mulf(one_plus_iz_imag, one_minus_iz_real),
                                    arith.mulf(one_plus_iz_real, one_minus_iz_imag),
                                ),
                                denom,
                            )
                            log_mod = math_dialect.log(
                                math_dialect.sqrt(
                                    arith.addf(
                                        arith.mulf(ratio_real, ratio_real),
                                        arith.mulf(ratio_imag, ratio_imag),
                                    )
                                )
                            )
                            log_phase = math_dialect.atan2(ratio_imag, ratio_real)
                            half = arith.constant(0.5, FTy)
                            final_real = arith.mulf(half, log_phase)
                            raw_imag = arith.mulf(half, log_mod)
                            neg_imag = arith.negf(raw_imag)
                            # Preserve sign of y when result is zero (signed zero)
                            imag_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, neg_imag, c0)
                            signed_zero = math_dialect.copysign(c0, y)
                            final_imag = arith.select(imag_is_zero, signed_zero, neg_imag)
                            scf.yield_(final_real, final_imag)
                        scf.yield_(result_xzero.results)
                    scf.yield_(result_ynan2.results)
                scf.yield_(result_xnan.results)
            scf.yield_(result_xinf.results)
        scf.yield_(result_yinf.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_asin(z_real: FTy, z_imag: FTy):
    """asin(z) using Kahan's algorithm with real arithmetic (matches CPython).

    Uses: s = hypot(1+|x|, |y|), t = hypot(1-|x|, |y|), a = (s+t)/2
    real = copysign(asin(|x|/a), x)
    imag = copysign(log(a + sqrt(a^2-1)), y)
    """
    c0 = _zero(FTy)
    c1 = arith.constant(1.0, FTy)
    nan = _nan(FTy)
    inf = _inf(FTy)
    pi_over_2 = arith.constant(1.5707963267948966, FTy)

    x = z_real
    y = z_imag

    x_isnan = math_dialect.isnan(x)
    y_isnan = math_dialect.isnan(y)
    x_isinf = math_dialect.isinf(x)
    y_isinf = math_dialect.isinf(y)
    y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, y, c0)

    with if_(y_isinf, results=[FTy, FTy]) as result:
        with if_(x_isnan, results=[FTy, FTy]) as result_xnan:
            scf.yield_(x, y)
        with else_(result_xnan):
            real_res = math_dialect.copysign(pi_over_2, x)
            scf.yield_(real_res, y)
        scf.yield_(result_xnan.results)
    with else_(result):
        with if_(x_isnan, results=[FTy, FTy]) as result_xnan2:
            with if_(y_is_zero, results=[FTy, FTy]) as result_yz:
                scf.yield_(nan, y)
            with else_(result_yz):
                scf.yield_(nan, nan)
            scf.yield_(result_yz.results)
        with else_(result_xnan2):
            with if_(y_isnan, results=[FTy, FTy]) as result_ynan:
                with if_(x_isinf, results=[FTy, FTy]) as result_xi:
                    scf.yield_(nan, x)
                with else_(result_xi):
                    scf.yield_(nan, nan)
                scf.yield_(result_xi.results)
            with else_(result_ynan):
                with if_(x_isinf, results=[FTy, FTy]) as result_xinf:
                    real_res = math_dialect.copysign(pi_over_2, x)
                    imag_res = math_dialect.copysign(inf, y)
                    scf.yield_(real_res, imag_res)
                with else_(result_xinf):
                    # Kahan's algorithm using only real arithmetic
                    ax = math_dialect.absf(x)
                    ay = math_dialect.absf(y)

                    # s = hypot(1+ax, ay), t = hypot(1-ax, ay)
                    opax = arith.addf(c1, ax)
                    omax = arith.subf(c1, ax)
                    ay2 = arith.mulf(ay, ay)
                    s = math_dialect.sqrt(arith.addf(arith.mulf(opax, opax), ay2))
                    t = math_dialect.sqrt(arith.addf(arith.mulf(omax, omax), ay2))

                    # a = (s+t)/2, b = ax/a
                    half = arith.constant(0.5, FTy)
                    a = arith.mulf(half, arith.addf(s, t))
                    b = arith.divf(ax, a)

                    # real part = asin(b), clamped to [0, pi/2] via min(b, 1)
                    b_clamped = arith.minimumf(b, c1)
                    real_abs = math_dialect.asin(b_clamped)
                    final_real = math_dialect.copysign(real_abs, x)

                    # imag part = log(a + sqrt(a^2-1))
                    a2m1 = arith.subf(arith.mulf(a, a), c1)
                    # Ensure non-negative (numerical noise near a=1)
                    a2m1_safe = arith.maximumf(a2m1, c0)
                    imag_abs = math_dialect.log(arith.addf(a, math_dialect.sqrt(a2m1_safe)))
                    final_imag = math_dialect.copysign(imag_abs, y)

                    scf.yield_(final_real, final_imag)
                scf.yield_(result_xinf.results)
            scf.yield_(result_ynan.results)
        scf.yield_(result_xnan2.results)
    return result.results[0], result.results[1]


@func.func(sym_visibility="private", generics=[FTy])
def cmath_asinh(z_real: FTy, z_imag: FTy):
    """asinh(z) using Kahan's algorithm: asinh(x+iy) via asin(-y+ix) rotated.

    Uses same real-arithmetic Kahan formula as asin, applied to iz=(-y,x),
    then rotates result: asinh(z) = (asin_imag, -asin_real).
    """
    c0 = _zero(FTy)
    c1 = arith.constant(1.0, FTy)
    nan = _nan(FTy)
    inf = _inf(FTy)
    pi_over_2 = arith.constant(1.5707963267948966, FTy)
    pi_over_4 = arith.constant(0.7853981633974483, FTy)

    x = z_real
    y = z_imag

    x_isnan = math_dialect.isnan(x)
    y_isnan = math_dialect.isnan(y)
    x_isinf = math_dialect.isinf(x)
    y_isinf = math_dialect.isinf(y)

    with if_(x_isinf, results=[FTy, FTy]) as result:
        with if_(y_isnan, results=[FTy, FTy]) as result_ynan:
            scf.yield_(x, nan)
        with else_(result_ynan):
            with if_(y_isinf, results=[FTy, FTy]) as result_yinf:
                real_res = math_dialect.copysign(inf, x)
                imag_res = math_dialect.copysign(pi_over_4, y)
                scf.yield_(real_res, imag_res)
            with else_(result_yinf):
                real_res = math_dialect.copysign(inf, x)
                imag_res = math_dialect.copysign(c0, y)
                scf.yield_(real_res, imag_res)
            scf.yield_(result_yinf.results)
        scf.yield_(result_ynan.results)
    with else_(result):
        with if_(y_isinf, results=[FTy, FTy]) as result_yinf2:
            with if_(x_isnan, results=[FTy, FTy]) as result_xnan:
                scf.yield_(nan, y)
            with else_(result_xnan):
                real_res = math_dialect.copysign(inf, x)
                imag_res = math_dialect.copysign(pi_over_2, y)
                scf.yield_(real_res, imag_res)
            scf.yield_(result_xnan.results)
        with else_(result_yinf2):
            with if_(x_isnan, results=[FTy, FTy]) as result_xnan2:
                y_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, y, c0)
                with if_(y_is_zero, results=[FTy, FTy]) as result_yz:
                    scf.yield_(x, y)
                with else_(result_yz):
                    scf.yield_(nan, nan)
                scf.yield_(result_yz.results)
            with else_(result_xnan2):
                with if_(y_isnan, results=[FTy, FTy]) as result_ynan2:
                    scf.yield_(nan, nan)
                with else_(result_ynan2):
                    # Kahan for asin(iz) where iz=(-y, x), then rotate
                    # asin input: real=-y, imag=x
                    ay_asin = math_dialect.absf(y)  # |re(iz)| = |-y| = |y|
                    ax_asin = math_dialect.absf(x)  # |im(iz)| = |x|

                    opay = arith.addf(c1, ay_asin)
                    omay = arith.subf(c1, ay_asin)
                    ax2 = arith.mulf(ax_asin, ax_asin)
                    half = arith.constant(0.5, FTy)
                    s = math_dialect.sqrt(arith.addf(arith.mulf(opay, opay), ax2))
                    t = math_dialect.sqrt(arith.addf(arith.mulf(omay, omay), ax2))
                    a = arith.mulf(half, arith.addf(s, t))
                    b = arith.divf(ay_asin, a)

                    b_clamped = arith.minimumf(b, c1)
                    asin_real_abs = math_dialect.asin(b_clamped)
                    # asin(iz).real has sign of re(iz) = -y
                    neg_y = arith.negf(y)
                    asin_real = math_dialect.copysign(asin_real_abs, neg_y)

                    a2m1 = arith.subf(arith.mulf(a, a), c1)
                    a2m1_safe = arith.maximumf(a2m1, c0)
                    asin_imag_abs = math_dialect.log(arith.addf(a, math_dialect.sqrt(a2m1_safe)))
                    # asin(iz).imag has sign of im(iz) = x
                    asin_imag = math_dialect.copysign(asin_imag_abs, x)

                    # asinh(z) = -i * asin(iz) = (asin_imag, -asin_real)
                    final_real = asin_imag
                    final_imag = arith.negf(asin_real)
                    scf.yield_(final_real, final_imag)
                scf.yield_(result_ynan2.results)
            scf.yield_(result_xnan2.results)
        scf.yield_(result_yinf2.results)
    return result.results[0], result.results[1]


def get_cmath_intrinsics_module():
    with mlir_mod_ctx() as ctx:
        for fty in [T.f32(), T.f64()]:
            for fn in [
                cmath_exp,
                cmath_sinh,
                cmath_cosh,
                cmath_sin,
                cmath_cos,
                cmath_rect,
                cmath_sqrt,
                cmath_acos,
                cmath_acosh,
                cmath_atan,
                cmath_asin,
                cmath_asinh,
            ]:
                f = fn[fty]
                f.func_attrs["alwaysinline"] = UnitAttr.get()
                f.emit()
    return str(ctx.module)
