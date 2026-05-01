# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba_cuda_mlir.types import (
    int16,
    int32,
    int64,
    float32,
    float64,
    UniTuple,
    Tuple,
)


def abs(x: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_abs.html

    CAPI: int32 __nv_abs(int32 x);
    """


def acos(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acos.html

    CAPI: float64 __nv_acos(float64 x);
    """


def acosf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acosf.html

    CAPI: float32 __nv_acosf(float32 x);
    """


def acosh(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acosh.html

    CAPI: float64 __nv_acosh(float64 x);
    """


def acoshf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acoshf.html

    CAPI: float32 __nv_acoshf(float32 x);
    """


def asin(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asin.html

    CAPI: float64 __nv_asin(float64 x);
    """


def asinf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinf.html

    CAPI: float32 __nv_asinf(float32 x);
    """


def asinh(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinh.html

    CAPI: float64 __nv_asinh(float64 x);
    """


def asinhf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinhf.html

    CAPI: float32 __nv_asinhf(float32 x);
    """


def atan(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan.html

    CAPI: float64 __nv_atan(float64 x);
    """


def atan2(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan2.html

    CAPI: float64 __nv_atan2(float64 x, float64 y);
    """


def atan2f(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan2f.html

    CAPI: float32 __nv_atan2f(float32 x, float32 y);
    """


def atanf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanf.html

    CAPI: float32 __nv_atanf(float32 x);
    """


def atanh(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanh.html

    CAPI: float64 __nv_atanh(float64 x);
    """


def atanhf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanhf.html

    CAPI: float32 __nv_atanhf(float32 x);
    """


def brev(x: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_brev.html

    CAPI: int32 __nv_brev(int32 x);
    """


def brevll(x: int64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_brevll.html

    CAPI: int64 __nv_brevll(int64 x);
    """


def byte_perm(x: int32, y: int32, z: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_byte_perm.html

    CAPI: int32 __nv_byte_perm(int32 x, int32 y, int32 z);
    """


def cbrt(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cbrt.html

    CAPI: float64 __nv_cbrt(float64 x);
    """


def cbrtf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cbrtf.html

    CAPI: float32 __nv_cbrtf(float32 x);
    """


def ceil(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ceil.html

    CAPI: float64 __nv_ceil(float64 x);
    """


def ceilf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ceilf.html

    CAPI: float32 __nv_ceilf(float32 x);
    """


def clz(x: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_clz.html

    CAPI: int32 __nv_clz(int32 x);
    """


def clzll(x: int64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_clzll.html

    CAPI: int32 __nv_clzll(int64 x);
    """


def copysign(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_copysign.html

    CAPI: float64 __nv_copysign(float64 x, float64 y);
    """


def copysignf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_copysignf.html

    CAPI: float32 __nv_copysignf(float32 x, float32 y);
    """


def cos(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cos.html

    CAPI: float64 __nv_cos(float64 x);
    """


def cosf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cosf.html

    CAPI: float32 __nv_cosf(float32 x);
    """


def cosh(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cosh.html

    CAPI: float64 __nv_cosh(float64 x);
    """


def coshf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_coshf.html

    CAPI: float32 __nv_coshf(float32 x);
    """


def cospi(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cospi.html

    CAPI: float64 __nv_cospi(float64 x);
    """


def cospif(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cospif.html

    CAPI: float32 __nv_cospif(float32 x);
    """


def dadd_rd(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rd.html

    CAPI: float64 __nv_dadd_rd(float64 x, float64 y);
    """


def dadd_rn(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rn.html

    CAPI: float64 __nv_dadd_rn(float64 x, float64 y);
    """


def dadd_ru(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_ru.html

    CAPI: float64 __nv_dadd_ru(float64 x, float64 y);
    """


def dadd_rz(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rz.html

    CAPI: float64 __nv_dadd_rz(float64 x, float64 y);
    """


def ddiv_rd(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rd.html

    CAPI: float64 __nv_ddiv_rd(float64 x, float64 y);
    """


def ddiv_rn(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rn.html

    CAPI: float64 __nv_ddiv_rn(float64 x, float64 y);
    """


def ddiv_ru(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_ru.html

    CAPI: float64 __nv_ddiv_ru(float64 x, float64 y);
    """


def ddiv_rz(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rz.html

    CAPI: float64 __nv_ddiv_rz(float64 x, float64 y);
    """


def dmul_rd(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rd.html

    CAPI: float64 __nv_dmul_rd(float64 x, float64 y);
    """


def dmul_rn(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rn.html

    CAPI: float64 __nv_dmul_rn(float64 x, float64 y);
    """


def dmul_ru(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_ru.html

    CAPI: float64 __nv_dmul_ru(float64 x, float64 y);
    """


def dmul_rz(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rz.html

    CAPI: float64 __nv_dmul_rz(float64 x, float64 y);
    """


def double2float_rd(d: float64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rd.html

    CAPI: float32 __nv_double2float_rd(float64 d);
    """


def double2float_rn(d: float64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rn.html

    CAPI: float32 __nv_double2float_rn(float64 d);
    """


def double2float_ru(d: float64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_ru.html

    CAPI: float32 __nv_double2float_ru(float64 d);
    """


def double2float_rz(d: float64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rz.html

    CAPI: float32 __nv_double2float_rz(float64 d);
    """


def double2hiint(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2hiint.html

    CAPI: int32 __nv_double2hiint(float64 d);
    """


def double2int_rd(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rd.html

    CAPI: int32 __nv_double2int_rd(float64 d);
    """


def double2int_rn(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rn.html

    CAPI: int32 __nv_double2int_rn(float64 d);
    """


def double2int_ru(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_ru.html

    CAPI: int32 __nv_double2int_ru(float64 d);
    """


def double2int_rz(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rz.html

    CAPI: int32 __nv_double2int_rz(float64 d);
    """


def double2ll_rd(f: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rd.html

    CAPI: int64 __nv_double2ll_rd(float64 f);
    """


def double2ll_rn(f: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rn.html

    CAPI: int64 __nv_double2ll_rn(float64 f);
    """


def double2ll_ru(f: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_ru.html

    CAPI: int64 __nv_double2ll_ru(float64 f);
    """


def double2ll_rz(f: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rz.html

    CAPI: int64 __nv_double2ll_rz(float64 f);
    """


def double2loint(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2loint.html

    CAPI: int32 __nv_double2loint(float64 d);
    """


def double2uint_rd(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rd.html

    CAPI: int32 __nv_double2uint_rd(float64 d);
    """


def double2uint_rn(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rn.html

    CAPI: int32 __nv_double2uint_rn(float64 d);
    """


def double2uint_ru(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_ru.html

    CAPI: int32 __nv_double2uint_ru(float64 d);
    """


def double2uint_rz(d: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rz.html

    CAPI: int32 __nv_double2uint_rz(float64 d);
    """


def double2ull_rd(f: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rd.html

    CAPI: int64 __nv_double2ull_rd(float64 f);
    """


def double2ull_rn(f: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rn.html

    CAPI: int64 __nv_double2ull_rn(float64 f);
    """


def double2ull_ru(f: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_ru.html

    CAPI: int64 __nv_double2ull_ru(float64 f);
    """


def double2ull_rz(f: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rz.html

    CAPI: int64 __nv_double2ull_rz(float64 f);
    """


def double_as_longlong(x: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double_as_longlong.html

    CAPI: int64 __nv_double_as_longlong(float64 x);
    """


def drcp_rd(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rd.html

    CAPI: float64 __nv_drcp_rd(float64 x);
    """


def drcp_rn(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rn.html

    CAPI: float64 __nv_drcp_rn(float64 x);
    """


def drcp_ru(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_ru.html

    CAPI: float64 __nv_drcp_ru(float64 x);
    """


def drcp_rz(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rz.html

    CAPI: float64 __nv_drcp_rz(float64 x);
    """


def dsqrt_rd(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rd.html

    CAPI: float64 __nv_dsqrt_rd(float64 x);
    """


def dsqrt_rn(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rn.html

    CAPI: float64 __nv_dsqrt_rn(float64 x);
    """


def dsqrt_ru(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_ru.html

    CAPI: float64 __nv_dsqrt_ru(float64 x);
    """


def dsqrt_rz(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rz.html

    CAPI: float64 __nv_dsqrt_rz(float64 x);
    """


def erf(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erf.html

    CAPI: float64 __nv_erf(float64 x);
    """


def erfc(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfc.html

    CAPI: float64 __nv_erfc(float64 x);
    """


def erfcf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcf.html

    CAPI: float32 __nv_erfcf(float32 x);
    """


def erfcinv(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcinv.html

    CAPI: float64 __nv_erfcinv(float64 x);
    """


def erfcinvf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcinvf.html

    CAPI: float32 __nv_erfcinvf(float32 x);
    """


def erfcx(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcx.html

    CAPI: float64 __nv_erfcx(float64 x);
    """


def erfcxf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcxf.html

    CAPI: float32 __nv_erfcxf(float32 x);
    """


def erff(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erff.html

    CAPI: float32 __nv_erff(float32 x);
    """


def erfinv(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfinv.html

    CAPI: float64 __nv_erfinv(float64 x);
    """


def erfinvf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfinvf.html

    CAPI: float32 __nv_erfinvf(float32 x);
    """


def exp(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp.html

    CAPI: float64 __nv_exp(float64 x);
    """


def exp10(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp10.html

    CAPI: float64 __nv_exp10(float64 x);
    """


def exp10f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp10f.html

    CAPI: float32 __nv_exp10f(float32 x);
    """


def exp2(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp2.html

    CAPI: float64 __nv_exp2(float64 x);
    """


def exp2f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp2f.html

    CAPI: float32 __nv_exp2f(float32 x);
    """


def expf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expf.html

    CAPI: float32 __nv_expf(float32 x);
    """


def expm1(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expm1.html

    CAPI: float64 __nv_expm1(float64 x);
    """


def expm1f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expm1f.html

    CAPI: float32 __nv_expm1f(float32 x);
    """


def fabs(f: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fabs.html

    CAPI: float64 __nv_fabs(float64 f);
    """


def fabsf(f: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fabsf.html

    CAPI: float32 __nv_fabsf(float32 f);
    """


def fadd_rd(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rd.html

    CAPI: float32 __nv_fadd_rd(float32 x, float32 y);
    """


def fadd_rn(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rn.html

    CAPI: float32 __nv_fadd_rn(float32 x, float32 y);
    """


def fadd_ru(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_ru.html

    CAPI: float32 __nv_fadd_ru(float32 x, float32 y);
    """


def fadd_rz(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rz.html

    CAPI: float32 __nv_fadd_rz(float32 x, float32 y);
    """


def fast_cosf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_cosf.html

    CAPI: float32 __nv_fast_cosf(float32 x);
    """


def fast_exp10f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_exp10f.html

    CAPI: float32 __nv_fast_exp10f(float32 x);
    """


def fast_expf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_expf.html

    CAPI: float32 __nv_fast_expf(float32 x);
    """


def fast_fdividef(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_fdividef.html

    CAPI: float32 __nv_fast_fdividef(float32 x, float32 y);
    """


def fast_log10f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log10f.html

    CAPI: float32 __nv_fast_log10f(float32 x);
    """


def fast_log2f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log2f.html

    CAPI: float32 __nv_fast_log2f(float32 x);
    """


def fast_logf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_logf.html

    CAPI: float32 __nv_fast_logf(float32 x);
    """


def fast_powf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_powf.html

    CAPI: float32 __nv_fast_powf(float32 x, float32 y);
    """


def fast_sincosf(x: float32) -> UniTuple(float32, 2):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sincosf.html

    CAPI: void __nv_fast_sincosf(float32 x, float32* sptr, float32* cptr);
    """


def fast_sinf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sinf.html

    CAPI: float32 __nv_fast_sinf(float32 x);
    """


def fast_tanf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_tanf.html

    CAPI: float32 __nv_fast_tanf(float32 x);
    """


def fdim(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdim.html

    CAPI: float64 __nv_fdim(float64 x, float64 y);
    """


def fdimf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdimf.html

    CAPI: float32 __nv_fdimf(float32 x, float32 y);
    """


def fdiv_rd(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rd.html

    CAPI: float32 __nv_fdiv_rd(float32 x, float32 y);
    """


def fdiv_rn(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rn.html

    CAPI: float32 __nv_fdiv_rn(float32 x, float32 y);
    """


def fdiv_ru(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_ru.html

    CAPI: float32 __nv_fdiv_ru(float32 x, float32 y);
    """


def fdiv_rz(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rz.html

    CAPI: float32 __nv_fdiv_rz(float32 x, float32 y);
    """


def ffs(x: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ffs.html

    CAPI: int32 __nv_ffs(int32 x);
    """


def ffsll(x: int64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ffsll.html

    CAPI: int32 __nv_ffsll(int64 x);
    """


def finitef(x: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_finitef.html

    CAPI: int32 __nv_finitef(float32 x);
    """


def float2half_rn(f: float32) -> int16:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2half_rn.html

    CAPI: int16 __nv_float2half_rn(float32 f);
    """


def float2int_rd(in_: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rd.html

    CAPI: int32 __nv_float2int_rd(float32 in_);
    """


def float2int_rn(in_: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rn.html

    CAPI: int32 __nv_float2int_rn(float32 in_);
    """


def float2int_ru(in_: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_ru.html

    CAPI: int32 __nv_float2int_ru(float32 in_);
    """


def float2int_rz(in_: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rz.html

    CAPI: int32 __nv_float2int_rz(float32 in_);
    """


def float2ll_rd(f: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rd.html

    CAPI: int64 __nv_float2ll_rd(float32 f);
    """


def float2ll_rn(f: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rn.html

    CAPI: int64 __nv_float2ll_rn(float32 f);
    """


def float2ll_ru(f: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_ru.html

    CAPI: int64 __nv_float2ll_ru(float32 f);
    """


def float2ll_rz(f: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rz.html

    CAPI: int64 __nv_float2ll_rz(float32 f);
    """


def float2uint_rd(in_: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rd.html

    CAPI: int32 __nv_float2uint_rd(float32 in_);
    """


def float2uint_rn(in_: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rn.html

    CAPI: int32 __nv_float2uint_rn(float32 in_);
    """


def float2uint_ru(in_: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_ru.html

    CAPI: int32 __nv_float2uint_ru(float32 in_);
    """


def float2uint_rz(in_: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rz.html

    CAPI: int32 __nv_float2uint_rz(float32 in_);
    """


def float2ull_rd(f: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rd.html

    CAPI: int64 __nv_float2ull_rd(float32 f);
    """


def float2ull_rn(f: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rn.html

    CAPI: int64 __nv_float2ull_rn(float32 f);
    """


def float2ull_ru(f: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_ru.html

    CAPI: int64 __nv_float2ull_ru(float32 f);
    """


def float2ull_rz(f: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rz.html

    CAPI: int64 __nv_float2ull_rz(float32 f);
    """


def float_as_int(x: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float_as_int.html

    CAPI: int32 __nv_float_as_int(float32 x);
    """


def floor(f: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_floor.html

    CAPI: float64 __nv_floor(float64 f);
    """


def floorf(f: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_floorf.html

    CAPI: float32 __nv_floorf(float32 f);
    """


def fma(x: float64, y: float64, z: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma.html

    CAPI: float64 __nv_fma(float64 x, float64 y, float64 z);
    """


def fma_rd(x: float64, y: float64, z: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rd.html

    CAPI: float64 __nv_fma_rd(float64 x, float64 y, float64 z);
    """


def fma_rn(x: float64, y: float64, z: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rn.html

    CAPI: float64 __nv_fma_rn(float64 x, float64 y, float64 z);
    """


def fma_ru(x: float64, y: float64, z: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_ru.html

    CAPI: float64 __nv_fma_ru(float64 x, float64 y, float64 z);
    """


def fma_rz(x: float64, y: float64, z: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rz.html

    CAPI: float64 __nv_fma_rz(float64 x, float64 y, float64 z);
    """


def fmaf(x: float32, y: float32, z: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf.html

    CAPI: float32 __nv_fmaf(float32 x, float32 y, float32 z);
    """


def fmaf_rd(x: float32, y: float32, z: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rd.html

    CAPI: float32 __nv_fmaf_rd(float32 x, float32 y, float32 z);
    """


def fmaf_rn(x: float32, y: float32, z: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rn.html

    CAPI: float32 __nv_fmaf_rn(float32 x, float32 y, float32 z);
    """


def fmaf_ru(x: float32, y: float32, z: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_ru.html

    CAPI: float32 __nv_fmaf_ru(float32 x, float32 y, float32 z);
    """


def fmaf_rz(x: float32, y: float32, z: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rz.html

    CAPI: float32 __nv_fmaf_rz(float32 x, float32 y, float32 z);
    """


def fmax(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmax.html

    CAPI: float64 __nv_fmax(float64 x, float64 y);
    """


def fmaxf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaxf.html

    CAPI: float32 __nv_fmaxf(float32 x, float32 y);
    """


def fmin(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmin.html

    CAPI: float64 __nv_fmin(float64 x, float64 y);
    """


def fminf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fminf.html

    CAPI: float32 __nv_fminf(float32 x, float32 y);
    """


def fmod(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmod.html

    CAPI: float64 __nv_fmod(float64 x, float64 y);
    """


def fmodf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmodf.html

    CAPI: float32 __nv_fmodf(float32 x, float32 y);
    """


def fmul_rd(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rd.html

    CAPI: float32 __nv_fmul_rd(float32 x, float32 y);
    """


def fmul_rn(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rn.html

    CAPI: float32 __nv_fmul_rn(float32 x, float32 y);
    """


def fmul_ru(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_ru.html

    CAPI: float32 __nv_fmul_ru(float32 x, float32 y);
    """


def fmul_rz(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rz.html

    CAPI: float32 __nv_fmul_rz(float32 x, float32 y);
    """


def frcp_rd(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rd.html

    CAPI: float32 __nv_frcp_rd(float32 x);
    """


def frcp_rn(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rn.html

    CAPI: float32 __nv_frcp_rn(float32 x);
    """


def frcp_ru(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_ru.html

    CAPI: float32 __nv_frcp_ru(float32 x);
    """


def frcp_rz(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rz.html

    CAPI: float32 __nv_frcp_rz(float32 x);
    """


def frexp(x: float64) -> Tuple([float64, int32]):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frexp.html

    CAPI: float64 __nv_frexp(float64 x, int32* b);
    """


def frexpf(x: float32) -> Tuple([float32, int32]):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frexpf.html

    CAPI: float32 __nv_frexpf(float32 x, int32* b);
    """


def frsqrt_rn(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frsqrt_rn.html

    CAPI: float32 __nv_frsqrt_rn(float32 x);
    """


def fsqrt_rd(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rd.html

    CAPI: float32 __nv_fsqrt_rd(float32 x);
    """


def fsqrt_rn(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rn.html

    CAPI: float32 __nv_fsqrt_rn(float32 x);
    """


def fsqrt_ru(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_ru.html

    CAPI: float32 __nv_fsqrt_ru(float32 x);
    """


def fsqrt_rz(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rz.html

    CAPI: float32 __nv_fsqrt_rz(float32 x);
    """


def fsub_rd(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rd.html

    CAPI: float32 __nv_fsub_rd(float32 x, float32 y);
    """


def fsub_rn(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rn.html

    CAPI: float32 __nv_fsub_rn(float32 x, float32 y);
    """


def fsub_ru(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_ru.html

    CAPI: float32 __nv_fsub_ru(float32 x, float32 y);
    """


def fsub_rz(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rz.html

    CAPI: float32 __nv_fsub_rz(float32 x, float32 y);
    """


def hadd(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hadd.html

    CAPI: int32 __nv_hadd(int32 x, int32 y);
    """


def half2float(h: int16) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_half2float.html

    CAPI: float32 __nv_half2float(int16 h);
    """


def hiloint2double(x: int32, y: int32) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hiloint2double.html

    CAPI: float64 __nv_hiloint2double(int32 x, int32 y);
    """


def hypot(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hypot.html

    CAPI: float64 __nv_hypot(float64 x, float64 y);
    """


def hypotf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hypotf.html

    CAPI: float32 __nv_hypotf(float32 x, float32 y);
    """


def ilogb(x: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ilogb.html

    CAPI: int32 __nv_ilogb(float64 x);
    """


def ilogbf(x: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ilogbf.html

    CAPI: int32 __nv_ilogbf(float32 x);
    """


def int2double_rn(i: int32) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2double_rn.html

    CAPI: float64 __nv_int2double_rn(int32 i);
    """


def int2float_rd(in_: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rd.html

    CAPI: float32 __nv_int2float_rd(int32 in_);
    """


def int2float_rn(in_: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rn.html

    CAPI: float32 __nv_int2float_rn(int32 in_);
    """


def int2float_ru(in_: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_ru.html

    CAPI: float32 __nv_int2float_ru(int32 in_);
    """


def int2float_rz(in_: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rz.html

    CAPI: float32 __nv_int2float_rz(int32 in_);
    """


def int_as_float(x: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int_as_float.html

    CAPI: float32 __nv_int_as_float(int32 x);
    """


def isfinited(x: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isfinited.html

    CAPI: int32 __nv_isfinited(float64 x);
    """


def isinfd(x: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isinfd.html

    CAPI: int32 __nv_isinfd(float64 x);
    """


def isinff(x: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isinff.html

    CAPI: int32 __nv_isinff(float32 x);
    """


def isnand(x: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isnand.html

    CAPI: int32 __nv_isnand(float64 x);
    """


def isnanf(x: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isnanf.html

    CAPI: int32 __nv_isnanf(float32 x);
    """


def j0(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j0.html

    CAPI: float64 __nv_j0(float64 x);
    """


def j0f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j0f.html

    CAPI: float32 __nv_j0f(float32 x);
    """


def j1(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j1.html

    CAPI: float64 __nv_j1(float64 x);
    """


def j1f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j1f.html

    CAPI: float32 __nv_j1f(float32 x);
    """


def jn(n: int32, x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_jn.html

    CAPI: float64 __nv_jn(int32 n, float64 x);
    """


def jnf(n: int32, x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_jnf.html

    CAPI: float32 __nv_jnf(int32 n, float32 x);
    """


def ldexp(x: float64, y: int32) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ldexp.html

    CAPI: float64 __nv_ldexp(float64 x, int32 y);
    """


def ldexpf(x: float32, y: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ldexpf.html

    CAPI: float32 __nv_ldexpf(float32 x, int32 y);
    """


def lgamma(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_lgamma.html

    CAPI: float64 __nv_lgamma(float64 x);
    """


def lgammaf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_lgammaf.html

    CAPI: float32 __nv_lgammaf(float32 x);
    """


def ll2double_rd(l: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rd.html

    CAPI: float64 __nv_ll2double_rd(int64 l);
    """


def ll2double_rn(l: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rn.html

    CAPI: float64 __nv_ll2double_rn(int64 l);
    """


def ll2double_ru(l: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_ru.html

    CAPI: float64 __nv_ll2double_ru(int64 l);
    """


def ll2double_rz(l: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rz.html

    CAPI: float64 __nv_ll2double_rz(int64 l);
    """


def ll2float_rd(l: int64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rd.html

    CAPI: float32 __nv_ll2float_rd(int64 l);
    """


def ll2float_rn(l: int64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rn.html

    CAPI: float32 __nv_ll2float_rn(int64 l);
    """


def ll2float_ru(l: int64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_ru.html

    CAPI: float32 __nv_ll2float_ru(int64 l);
    """


def ll2float_rz(l: int64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rz.html

    CAPI: float32 __nv_ll2float_rz(int64 l);
    """


def llabs(x: int64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llabs.html

    CAPI: int64 __nv_llabs(int64 x);
    """


def llmax(x: int64, y: int64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llmax.html

    CAPI: int64 __nv_llmax(int64 x, int64 y);
    """


def llmin(x: int64, y: int64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llmin.html

    CAPI: int64 __nv_llmin(int64 x, int64 y);
    """


def llrint(x: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llrint.html

    CAPI: int64 __nv_llrint(float64 x);
    """


def llrintf(x: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llrintf.html

    CAPI: int64 __nv_llrintf(float32 x);
    """


def llround(x: float64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llround.html

    CAPI: int64 __nv_llround(float64 x);
    """


def llroundf(x: float32) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llroundf.html

    CAPI: int64 __nv_llroundf(float32 x);
    """


def log(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log.html

    CAPI: float64 __nv_log(float64 x);
    """


def log10(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log10.html

    CAPI: float64 __nv_log10(float64 x);
    """


def log10f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log10f.html

    CAPI: float32 __nv_log10f(float32 x);
    """


def log1p(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log1p.html

    CAPI: float64 __nv_log1p(float64 x);
    """


def log1pf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log1pf.html

    CAPI: float32 __nv_log1pf(float32 x);
    """


def log2(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log2.html

    CAPI: float64 __nv_log2(float64 x);
    """


def log2f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log2f.html

    CAPI: float32 __nv_log2f(float32 x);
    """


def logb(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logb.html

    CAPI: float64 __nv_logb(float64 x);
    """


def logbf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logbf.html

    CAPI: float32 __nv_logbf(float32 x);
    """


def logf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logf.html

    CAPI: float32 __nv_logf(float32 x);
    """


def longlong_as_double(x: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_longlong_as_double.html

    CAPI: float64 __nv_longlong_as_double(int64 x);
    """


def max(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_max.html

    CAPI: int32 __nv_max(int32 x, int32 y);
    """


def min(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_min.html

    CAPI: int32 __nv_min(int32 x, int32 y);
    """


def modf(x: float64) -> UniTuple(float64, 2):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_modf.html

    CAPI: float64 __nv_modf(float64 x, float64* b);
    """


def modff(x: float32) -> UniTuple(float32, 2):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_modff.html

    CAPI: float32 __nv_modff(float32 x, float32* b);
    """


def mul24(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mul24.html

    CAPI: int32 __nv_mul24(int32 x, int32 y);
    """


def mul64hi(x: int64, y: int64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mul64hi.html

    CAPI: int64 __nv_mul64hi(int64 x, int64 y);
    """


def mulhi(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mulhi.html

    CAPI: int32 __nv_mulhi(int32 x, int32 y);
    """


def nearbyint(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nearbyint.html

    CAPI: float64 __nv_nearbyint(float64 x);
    """


def nearbyintf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nearbyintf.html

    CAPI: float32 __nv_nearbyintf(float32 x);
    """


def nextafter(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nextafter.html

    CAPI: float64 __nv_nextafter(float64 x, float64 y);
    """


def nextafterf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nextafterf.html

    CAPI: float32 __nv_nextafterf(float32 x, float32 y);
    """


def normcdf(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdf.html

    CAPI: float64 __nv_normcdf(float64 x);
    """


def normcdff(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdff.html

    CAPI: float32 __nv_normcdff(float32 x);
    """


def normcdfinv(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdfinv.html

    CAPI: float64 __nv_normcdfinv(float64 x);
    """


def normcdfinvf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdfinvf.html

    CAPI: float32 __nv_normcdfinvf(float32 x);
    """


def popc(x: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_popc.html

    CAPI: int32 __nv_popc(int32 x);
    """


def popcll(x: int64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_popcll.html

    CAPI: int32 __nv_popcll(int64 x);
    """


def pow(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_pow.html

    CAPI: float64 __nv_pow(float64 x, float64 y);
    """


def powf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powf.html

    CAPI: float32 __nv_powf(float32 x, float32 y);
    """


def powi(x: float64, y: int32) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powi.html

    CAPI: float64 __nv_powi(float64 x, int32 y);
    """


def powif(x: float32, y: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powif.html

    CAPI: float32 __nv_powif(float32 x, int32 y);
    """


def rcbrt(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rcbrt.html

    CAPI: float64 __nv_rcbrt(float64 x);
    """


def rcbrtf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rcbrtf.html

    CAPI: float32 __nv_rcbrtf(float32 x);
    """


def remainder(x: float64, y: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remainder.html

    CAPI: float64 __nv_remainder(float64 x, float64 y);
    """


def remainderf(x: float32, y: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remainderf.html

    CAPI: float32 __nv_remainderf(float32 x, float32 y);
    """


def remquo(x: float64, y: float64) -> Tuple([float64, int32]):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remquo.html

    CAPI: float64 __nv_remquo(float64 x, float64 y, int32* c);
    """


def remquof(x: float32, y: float32) -> Tuple([float32, int32]):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remquof.html

    CAPI: float32 __nv_remquof(float32 x, float32 y, int32* quo);
    """


def rhadd(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rhadd.html

    CAPI: int32 __nv_rhadd(int32 x, int32 y);
    """


def rint(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rint.html

    CAPI: float64 __nv_rint(float64 x);
    """


def rintf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rintf.html

    CAPI: float32 __nv_rintf(float32 x);
    """


def round(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_round.html

    CAPI: float64 __nv_round(float64 x);
    """


def roundf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_roundf.html

    CAPI: float32 __nv_roundf(float32 x);
    """


def rsqrt(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rsqrt.html

    CAPI: float64 __nv_rsqrt(float64 x);
    """


def rsqrtf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rsqrtf.html

    CAPI: float32 __nv_rsqrtf(float32 x);
    """


def sad(x: int32, y: int32, z: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sad.html

    CAPI: int32 __nv_sad(int32 x, int32 y, int32 z);
    """


def saturatef(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_saturatef.html

    CAPI: float32 __nv_saturatef(float32 x);
    """


def scalbn(x: float64, y: int32) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_scalbn.html

    CAPI: float64 __nv_scalbn(float64 x, int32 y);
    """


def scalbnf(x: float32, y: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_scalbnf.html

    CAPI: float32 __nv_scalbnf(float32 x, int32 y);
    """


def signbitd(x: float64) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_signbitd.html

    CAPI: int32 __nv_signbitd(float64 x);
    """


def signbitf(x: float32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_signbitf.html

    CAPI: int32 __nv_signbitf(float32 x);
    """


def sin(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sin.html

    CAPI: float64 __nv_sin(float64 x);
    """


def sincos(x: float64) -> UniTuple(float64, 2):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincos.html

    CAPI: void __nv_sincos(float64 x, float64* sptr, float64* cptr);
    """


def sincosf(x: float32) -> UniTuple(float32, 2):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincosf.html

    CAPI: void __nv_sincosf(float32 x, float32* sptr, float32* cptr);
    """


def sincospi(x: float64) -> UniTuple(float64, 2):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincospi.html

    CAPI: void __nv_sincospi(float64 x, float64* sptr, float64* cptr);
    """


def sincospif(x: float32) -> UniTuple(float32, 2):
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincospif.html

    CAPI: void __nv_sincospif(float32 x, float32* sptr, float32* cptr);
    """


def sinf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinf.html

    CAPI: float32 __nv_sinf(float32 x);
    """


def sinh(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinh.html

    CAPI: float64 __nv_sinh(float64 x);
    """


def sinhf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinhf.html

    CAPI: float32 __nv_sinhf(float32 x);
    """


def sinpi(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinpi.html

    CAPI: float64 __nv_sinpi(float64 x);
    """


def sinpif(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinpif.html

    CAPI: float32 __nv_sinpif(float32 x);
    """


def sqrt(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sqrt.html

    CAPI: float64 __nv_sqrt(float64 x);
    """


def sqrtf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sqrtf.html

    CAPI: float32 __nv_sqrtf(float32 x);
    """


def tan(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tan.html

    CAPI: float64 __nv_tan(float64 x);
    """


def tanf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanf.html

    CAPI: float32 __nv_tanf(float32 x);
    """


def tanh(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanh.html

    CAPI: float64 __nv_tanh(float64 x);
    """


def tanhf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanhf.html

    CAPI: float32 __nv_tanhf(float32 x);
    """


def tgamma(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tgamma.html

    CAPI: float64 __nv_tgamma(float64 x);
    """


def tgammaf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tgammaf.html

    CAPI: float32 __nv_tgammaf(float32 x);
    """


def trunc(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_trunc.html

    CAPI: float64 __nv_trunc(float64 x);
    """


def truncf(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_truncf.html

    CAPI: float32 __nv_truncf(float32 x);
    """


def uhadd(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uhadd.html

    CAPI: int32 __nv_uhadd(int32 x, int32 y);
    """


def uint2double_rn(i: int32) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2double_rn.html

    CAPI: float64 __nv_uint2double_rn(int32 i);
    """


def uint2float_rd(in_: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rd.html

    CAPI: float32 __nv_uint2float_rd(int32 in_);
    """


def uint2float_rn(in_: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rn.html

    CAPI: float32 __nv_uint2float_rn(int32 in_);
    """


def uint2float_ru(in_: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_ru.html

    CAPI: float32 __nv_uint2float_ru(int32 in_);
    """


def uint2float_rz(in_: int32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rz.html

    CAPI: float32 __nv_uint2float_rz(int32 in_);
    """


def ull2double_rd(l: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rd.html

    CAPI: float64 __nv_ull2double_rd(int64 l);
    """


def ull2double_rn(l: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rn.html

    CAPI: float64 __nv_ull2double_rn(int64 l);
    """


def ull2double_ru(l: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_ru.html

    CAPI: float64 __nv_ull2double_ru(int64 l);
    """


def ull2double_rz(l: int64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rz.html

    CAPI: float64 __nv_ull2double_rz(int64 l);
    """


def ull2float_rd(l: int64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rd.html

    CAPI: float32 __nv_ull2float_rd(int64 l);
    """


def ull2float_rn(l: int64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rn.html

    CAPI: float32 __nv_ull2float_rn(int64 l);
    """


def ull2float_ru(l: int64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_ru.html

    CAPI: float32 __nv_ull2float_ru(int64 l);
    """


def ull2float_rz(l: int64) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rz.html

    CAPI: float32 __nv_ull2float_rz(int64 l);
    """


def ullmax(x: int64, y: int64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ullmax.html

    CAPI: int64 __nv_ullmax(int64 x, int64 y);
    """


def ullmin(x: int64, y: int64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ullmin.html

    CAPI: int64 __nv_ullmin(int64 x, int64 y);
    """


def umax(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umax.html

    CAPI: int32 __nv_umax(int32 x, int32 y);
    """


def umin(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umin.html

    CAPI: int32 __nv_umin(int32 x, int32 y);
    """


def umul24(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umul24.html

    CAPI: int32 __nv_umul24(int32 x, int32 y);
    """


def umul64hi(x: int64, y: int64) -> int64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umul64hi.html

    CAPI: int64 __nv_umul64hi(int64 x, int64 y);
    """


def umulhi(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umulhi.html

    CAPI: int32 __nv_umulhi(int32 x, int32 y);
    """


def urhadd(x: int32, y: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_urhadd.html

    CAPI: int32 __nv_urhadd(int32 x, int32 y);
    """


def usad(x: int32, y: int32, z: int32) -> int32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_usad.html

    CAPI: int32 __nv_usad(int32 x, int32 y, int32 z);
    """


def y0(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y0.html

    CAPI: float64 __nv_y0(float64 x);
    """


def y0f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y0f.html

    CAPI: float32 __nv_y0f(float32 x);
    """


def y1(x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y1.html

    CAPI: float64 __nv_y1(float64 x);
    """


def y1f(x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y1f.html

    CAPI: float32 __nv_y1f(float32 x);
    """


def yn(n: int32, x: float64) -> float64:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_yn.html

    CAPI: float64 __nv_yn(int32 n, float64 x);
    """


def ynf(n: int32, x: float32) -> float32:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ynf.html

    CAPI: float32 __nv_ynf(int32 n, float32 x);
    """
