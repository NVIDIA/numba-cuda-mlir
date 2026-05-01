# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

logging.basicConfig(level=logging.DEBUG)
import math
from numba_cuda_mlir import cuda
import numpy as np


def test_math_ceil():
    @cuda.jit()
    def math_ceil_kernel(x):
        x[0] = math.ceil(x[0])

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([1.5]))
    math_ceil_kernel[1, 1, 0, 0](x)
    result = x.copy_to_host()[0]
    assert result == 2


def test_complex_ops_1():
    ctx = cuda.current_context()

    @cuda.jit
    def k(x):
        x[0] = 3 + 4j

    x = cuda.to_device(np.array([1.0 + 2.0j], dtype=np.complex128))
    k[1, 1, 0, 0](x)
    result = x.copy_to_host()[0]
    assert result == 3 + 4j, result

    x = cuda.to_device(np.array([1.0 + 2.0j], dtype=np.complex64))
    k[1, 1, 0, 0](x)
    result = x.copy_to_host()[0]
    assert result == 3 + 4j, result


def test_complex_ops_2():
    @cuda.jit
    def k(x):
        e = x[0]
        e = e + 1j
        e = e * 1j
        e = e - 1j
        e = e / 1j
        e = e**2
        x[0] = e

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([1.0 + 2.0j], dtype=np.complex128))
    k[1, 1, 0, 0](x)
    result = x.copy_to_host()[0]
    np.testing.assert_almost_equal(result, -9.0 + 0.0j)


def test_trigonometric_functions():
    """Test sin, cos, tan"""

    @cuda.jit()
    def trig_kernel(x, y, z):
        x[0] = math.sin(x[0])
        y[0] = math.cos(y[0])
        z[0] = math.tan(z[0])

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([math.pi / 2]))
    y = cuda.to_device(np.array([0.0]))
    z = cuda.to_device(np.array([0.0]))

    trig_kernel[1, 1, 0, 0](x, y, z)

    x_result = x.copy_to_host()[0]
    y_result = y.copy_to_host()[0]
    z_result = z.copy_to_host()[0]

    np.testing.assert_almost_equal(x_result, 1.0, decimal=5)
    np.testing.assert_almost_equal(y_result, 1.0, decimal=5)
    np.testing.assert_almost_equal(z_result, 0.0, decimal=5)


def test_exponential_logarithmic():
    """Test exp, log, log2, log10"""

    @cuda.jit()
    def exp_log_kernel(a, b, c, d):
        a[0] = math.exp(a[0])
        b[0] = math.log(b[0])
        c[0] = math.log2(c[0])
        d[0] = math.log10(d[0])

    ctx = cuda.current_context()
    a = cuda.to_device(np.array([1.0]))
    b = cuda.to_device(np.array([math.e]))
    c = cuda.to_device(np.array([8.0]))
    d = cuda.to_device(np.array([100.0]))

    exp_log_kernel[1, 1, 0, 0](a, b, c, d)

    a_result = a.copy_to_host()[0]
    b_result = b.copy_to_host()[0]
    c_result = c.copy_to_host()[0]
    d_result = d.copy_to_host()[0]

    np.testing.assert_almost_equal(a_result, math.e, decimal=5)
    np.testing.assert_almost_equal(b_result, 1.0, decimal=5)
    np.testing.assert_almost_equal(c_result, 3.0, decimal=5)
    np.testing.assert_almost_equal(d_result, 2.0, decimal=5)


def test_sqrt():
    """Test square root"""

    @cuda.jit()
    def sqrt_kernel(x):
        x[0] = math.sqrt(x[0])

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([16.0]))
    sqrt_kernel[1, 1, 0, 0](x)
    result = x.copy_to_host()[0]
    np.testing.assert_almost_equal(result, 4.0, decimal=5)


def test_rounding_functions():
    """Test floor, trunc"""

    @cuda.jit()
    def rounding_kernel(a, b, c):
        a[0] = math.ceil(a[0])
        b[0] = math.floor(b[0])
        c[0] = math.trunc(c[0])

    ctx = cuda.current_context()
    a = cuda.to_device(np.array([2.3]))
    b = cuda.to_device(np.array([2.7]))
    c = cuda.to_device(np.array([-2.7]))

    rounding_kernel[1, 1, 0, 0](a, b, c)

    a_result = a.copy_to_host()[0]
    b_result = b.copy_to_host()[0]
    c_result = c.copy_to_host()[0]

    assert a_result == 3.0
    assert b_result == 2.0
    assert c_result == -2.0


def test_abs_fabs():
    """Test absolute value functions"""

    @cuda.jit()
    def abs_kernel(x):
        x[0] = math.fabs(x[0])

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([-5.5]))
    abs_kernel[1, 1, 0, 0](x)
    result = x.copy_to_host()[0]
    np.testing.assert_almost_equal(result, 5.5, decimal=5)


def test_unary_operators():
    """Test neg and pos operators"""

    @cuda.jit()
    def unary_kernel(x, y):
        x[0] = -x[0]
        y[0] = +y[0]

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([5.0]))
    y = cuda.to_device(np.array([3.0]))

    unary_kernel[1, 1, 0, 0](x, y)

    x_result = x.copy_to_host()[0]
    y_result = y.copy_to_host()[0]

    assert x_result == -5.0
    assert y_result == 3.0


def test_floor_division():
    """Test floor division operator"""

    @cuda.jit()
    def floordiv_kernel(x, y):
        x[0] = x[0] // 3.0
        y[0] = y[0] // 3

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([7.0]))
    y = cuda.to_device(np.array([7], dtype=np.int64))

    floordiv_kernel[1, 1, 0, 0](x, y)

    x_result = x.copy_to_host()[0]
    y_result = y.copy_to_host()[0]

    assert x_result == 2.0
    assert y_result == 2


def test_combined_math_operations():
    """Test multiple math operations in a single kernel"""

    @cuda.jit()
    def combined_kernel(x):
        # Test various operations: sqrt(exp(log(x))) should equal x
        val = x[0]
        val = math.log(val)
        val = math.exp(val)
        val = math.sqrt(val)
        val = val * val
        x[0] = val

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([5.0]))
    combined_kernel[1, 1, 0, 0](x)
    result = x.copy_to_host()[0]
    np.testing.assert_almost_equal(result, 5.0, decimal=5)


def test_integer_operations():
    """Test math operations with integer types"""

    @cuda.jit()
    def int_kernel(x, y):
        x[0] = -x[0]
        y[0] = y[0] // 2

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([5], dtype=np.int32))
    y = cuda.to_device(np.array([7], dtype=np.int32))

    int_kernel[1, 1, 0, 0](x, y)

    x_result = x.copy_to_host()[0]
    y_result = y.copy_to_host()[0]

    assert x_result == -5
    assert y_result == 3


def test_modulo_operator():
    """Test modulo operator for integers and floats"""

    @cuda.jit()
    def mod_kernel(x, y, z):
        x[0] = x[0] % 3
        y[0] = y[0] % 4
        z[0] = z[0] % 2.5

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([10], dtype=np.int64))
    y = cuda.to_device(np.array([17], dtype=np.int32))
    z = cuda.to_device(np.array([7.5], dtype=np.float64))

    mod_kernel[1, 1, 0, 0](x, y, z)

    x_result = x.copy_to_host()[0]
    y_result = y.copy_to_host()[0]
    z_result = z.copy_to_host()[0]

    assert x_result == 1
    assert y_result == 1
    np.testing.assert_almost_equal(z_result, 0.0, decimal=5)


def test_bitwise_shift_operators():
    """Test left shift and right shift operators"""

    @cuda.jit()
    def shift_kernel(a, b, c, d):
        a[0] = a[0] << 2
        b[0] = b[0] >> 1
        c[0] = c[0] << 4
        d[0] = d[0] >> 3

    ctx = cuda.current_context()
    a = cuda.to_device(np.array([5], dtype=np.int32))
    b = cuda.to_device(np.array([20], dtype=np.int32))
    c = cuda.to_device(np.array([3], dtype=np.int64))
    d = cuda.to_device(np.array([64], dtype=np.int64))

    shift_kernel[1, 1, 0, 0](a, b, c, d)

    a_result = a.copy_to_host()[0]
    b_result = b.copy_to_host()[0]
    c_result = c.copy_to_host()[0]
    d_result = d.copy_to_host()[0]

    assert a_result == 20  # 5 << 2
    assert b_result == 10  # 20 >> 1
    assert c_result == 48  # 3 << 4
    assert d_result == 8  # 64 >> 3


def test_bitwise_operators():
    """Test bitwise AND, OR, and XOR operators"""

    @cuda.jit()
    def bitwise_kernel(a, b, c):
        # Test AND
        a[0] = a[0] & 0x0F
        # Test OR
        b[0] = b[0] | 0x0F
        # Test XOR
        c[0] = c[0] ^ 0xFF

    ctx = cuda.current_context()
    a = cuda.to_device(np.array([0x5A], dtype=np.int32))
    b = cuda.to_device(np.array([0x50], dtype=np.int32))
    c = cuda.to_device(np.array([0xAA], dtype=np.int32))

    bitwise_kernel[1, 1, 0, 0](a, b, c)

    a_result = a.copy_to_host()[0]
    b_result = b.copy_to_host()[0]
    c_result = c.copy_to_host()[0]

    assert a_result == 0x0A  # 0x5A & 0x0F
    assert b_result == 0x5F  # 0x50 | 0x0F
    assert c_result == 0x55  # 0xAA ^ 0xFF


def test_combined_bitwise_operations():
    """Test combined bitwise operations"""

    @cuda.jit()
    def combined_bitwise_kernel(x):
        # Extract upper and lower 16 bits, swap, and recombine
        val = x[0]
        upper = (val >> 16) & 0xFFFF
        lower = val & 0xFFFF
        x[0] = (lower << 16) | upper

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([0x12345678], dtype=np.int32))
    combined_bitwise_kernel[1, 1, 0, 0](x)
    result = x.copy_to_host()[0]
    assert result == 0x56781234


import pytest


def test_math_with_integer_inputs():
    """Test that math functions properly convert integer inputs to float."""

    @cuda.jit()
    def int_math_kernel(results):
        # Trig functions with int input
        results[0] = math.sin(0)
        results[1] = math.cos(0)
        results[2] = math.tan(0)
        # Sqrt with int
        results[3] = math.sqrt(4)
        # Exp/log with int
        results[4] = math.exp(0)
        results[5] = math.log(1)
        results[6] = math.log2(8)
        results[7] = math.log10(100)
        # Rounding with int (should return same value as float)
        results[8] = math.floor(5)
        results[9] = math.ceil(5)
        results[10] = math.trunc(5)

    ctx = cuda.current_context()
    results = cuda.to_device(np.zeros(11, dtype=np.float64))
    int_math_kernel[1, 1, 0, 0](results)
    r = results.copy_to_host()

    np.testing.assert_almost_equal(r[0], 0.0, decimal=5)  # sin(0)
    np.testing.assert_almost_equal(r[1], 1.0, decimal=5)  # cos(0)
    np.testing.assert_almost_equal(r[2], 0.0, decimal=5)  # tan(0)
    np.testing.assert_almost_equal(r[3], 2.0, decimal=5)  # sqrt(4)
    np.testing.assert_almost_equal(r[4], 1.0, decimal=5)  # exp(0)
    np.testing.assert_almost_equal(r[5], 0.0, decimal=5)  # log(1)
    np.testing.assert_almost_equal(r[6], 3.0, decimal=5)  # log2(8)
    np.testing.assert_almost_equal(r[7], 2.0, decimal=5)  # log10(100)
    np.testing.assert_almost_equal(r[8], 5.0, decimal=5)  # floor(5)
    np.testing.assert_almost_equal(r[9], 5.0, decimal=5)  # ceil(5)
    np.testing.assert_almost_equal(r[10], 5.0, decimal=5)  # trunc(5)


def test_hyperbolic_functions():
    """Test sinh, cosh, tanh"""

    @cuda.jit()
    def hyperbolic_kernel(a, b, c):
        a[0] = math.sinh(a[0])
        b[0] = math.cosh(b[0])
        c[0] = math.tanh(c[0])

    ctx = cuda.current_context()
    a = cuda.to_device(np.array([0.0]))
    b = cuda.to_device(np.array([0.0]))
    c = cuda.to_device(np.array([0.0]))

    hyperbolic_kernel[1, 1, 0, 0](a, b, c)

    np.testing.assert_almost_equal(a.copy_to_host()[0], 0.0, decimal=5)  # sinh(0)
    np.testing.assert_almost_equal(b.copy_to_host()[0], 1.0, decimal=5)  # cosh(0)
    np.testing.assert_almost_equal(c.copy_to_host()[0], 0.0, decimal=5)  # tanh(0)


def test_inverse_trig_functions():
    """Test asin, acos, atan"""

    @cuda.jit()
    def inv_trig_kernel(a, b, c):
        a[0] = math.asin(a[0])
        b[0] = math.acos(b[0])
        c[0] = math.atan(c[0])

    ctx = cuda.current_context()
    a = cuda.to_device(np.array([0.0]))
    b = cuda.to_device(np.array([1.0]))
    c = cuda.to_device(np.array([0.0]))

    inv_trig_kernel[1, 1, 0, 0](a, b, c)

    np.testing.assert_almost_equal(a.copy_to_host()[0], 0.0, decimal=5)  # asin(0)
    np.testing.assert_almost_equal(b.copy_to_host()[0], 0.0, decimal=5)  # acos(1)
    np.testing.assert_almost_equal(c.copy_to_host()[0], 0.0, decimal=5)  # atan(0)


def test_inverse_hyperbolic_functions():
    """Test asinh, acosh, atanh"""

    @cuda.jit()
    def inv_hyp_kernel(a, b, c):
        a[0] = math.asinh(a[0])
        b[0] = math.acosh(b[0])
        c[0] = math.atanh(c[0])

    ctx = cuda.current_context()
    a = cuda.to_device(np.array([0.0]))
    b = cuda.to_device(np.array([1.0]))
    c = cuda.to_device(np.array([0.0]))

    inv_hyp_kernel[1, 1, 0, 0](a, b, c)

    np.testing.assert_almost_equal(a.copy_to_host()[0], 0.0, decimal=5)  # asinh(0)
    np.testing.assert_almost_equal(b.copy_to_host()[0], 0.0, decimal=5)  # acosh(1)
    np.testing.assert_almost_equal(c.copy_to_host()[0], 0.0, decimal=5)  # atanh(0)


def test_isnan_isinf_isfinite():
    """Test floating-point predicate functions"""

    @cuda.jit()
    def predicates_kernel(results, values):
        results[0] = 1 if math.isnan(values[0]) else 0
        results[1] = 1 if math.isnan(values[1]) else 0
        results[2] = 1 if math.isinf(values[2]) else 0
        results[3] = 1 if math.isinf(values[3]) else 0
        results[4] = 1 if math.isfinite(values[4]) else 0
        results[5] = 1 if math.isfinite(values[5]) else 0

    ctx = cuda.current_context()
    values = cuda.to_device(
        np.array(
            [
                float("nan"),  # isnan -> True
                1.0,  # isnan -> False
                float("inf"),  # isinf -> True
                1.0,  # isinf -> False
                1.0,  # isfinite -> True
                float("inf"),  # isfinite -> False
            ]
        )
    )
    results = cuda.to_device(np.zeros(6, dtype=np.float64))

    predicates_kernel[1, 1, 0, 0](results, values)
    r = results.copy_to_host()

    assert r[0] == 1  # isnan(nan) -> True
    assert r[1] == 0  # isnan(1.0) -> False
    assert r[2] == 1  # isinf(inf) -> True
    assert r[3] == 0  # isinf(1.0) -> False
    assert r[4] == 1  # isfinite(1.0) -> True
    assert r[5] == 0  # isfinite(inf) -> False


def test_predicates_with_integers():
    """Test that isnan/isinf/isfinite work correctly with integer inputs."""

    @cuda.jit()
    def int_predicates_kernel(results):
        # Integers are always finite and never NaN/inf
        results[0] = 1 if math.isnan(5) else 0
        results[1] = 1 if math.isinf(5) else 0
        results[2] = 1 if math.isfinite(5) else 0

    ctx = cuda.current_context()
    results = cuda.to_device(np.zeros(3, dtype=np.float64))
    int_predicates_kernel[1, 1, 0, 0](results)
    r = results.copy_to_host()

    assert r[0] == 0  # isnan(5) -> False
    assert r[1] == 0  # isinf(5) -> False
    assert r[2] == 1  # isfinite(5) -> True


def test_atan2():
    """Test atan2 function"""

    @cuda.jit()
    def atan2_kernel(result, y, x):
        result[0] = math.atan2(y[0], x[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    y = cuda.to_device(np.array([1.0]))
    x = cuda.to_device(np.array([1.0]))

    atan2_kernel[1, 1, 0, 0](result, y, x)

    np.testing.assert_almost_equal(result.copy_to_host()[0], math.pi / 4, decimal=5)


def test_exp2():
    """Test exp2 (2^x) function"""

    @cuda.jit()
    def exp2_kernel(result):
        result[0] = math.exp2(result[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([3.0]))
    exp2_kernel[1, 1, 0, 0](result)

    np.testing.assert_almost_equal(result.copy_to_host()[0], 8.0, decimal=5)


def test_log1p():
    """Test log1p (log(1+x)) function"""

    @cuda.jit()
    def log1p_kernel(result):
        result[0] = math.log1p(result[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    log1p_kernel[1, 1, 0, 0](result)

    np.testing.assert_almost_equal(result.copy_to_host()[0], 0.0, decimal=5)


def test_copysign():
    """Test copysign function"""

    @cuda.jit()
    def copysign_kernel(results, x, y):
        results[0] = math.copysign(x[0], y[0])

    ctx = cuda.current_context()
    results = cuda.to_device(np.array([0.0]))
    x = cuda.to_device(np.array([5.0]))
    y = cuda.to_device(np.array([-1.0]))

    copysign_kernel[1, 1, 0, 0](results, x, y)

    np.testing.assert_almost_equal(results.copy_to_host()[0], -5.0, decimal=5)


def test_hypot():
    """Test hypot function"""

    @cuda.jit()
    def hypot_kernel(result, x, y):
        result[0] = math.hypot(x[0], y[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    x = cuda.to_device(np.array([3.0]))
    y = cuda.to_device(np.array([4.0]))

    hypot_kernel[1, 1, 0, 0](result, x, y)

    np.testing.assert_almost_equal(result.copy_to_host()[0], 5.0, decimal=5)


def test_complex_from_different_float_types():
    """Test creating complex128 from float32 inputs (type conversion)."""

    @cuda.jit()
    def complex_kernel(result, real_f32, imag_f32):
        # Creating complex128 from float32 inputs - needs type conversion
        c = complex(real_f32[0], imag_f32[0])
        result[0] = c

    ctx = cuda.current_context()
    # Input as float32
    real_f32 = cuda.to_device(np.array([3.0], dtype=np.float32))
    imag_f32 = cuda.to_device(np.array([4.0], dtype=np.float32))
    # Output as complex128 (uses float64 element type)
    result = cuda.to_device(np.array([0.0 + 0.0j], dtype=np.complex128))

    complex_kernel[1, 1, 0, 0](result, real_f32, imag_f32)
    r = result.copy_to_host()[0]

    assert r == 3.0 + 4.0j, f"Expected 3+4j, got {r}"


@pytest.mark.parametrize(
    "input_val,expected_mantissa,expected_exp",
    [
        (0.5, 0.5, 0),
        (1.0, 0.5, 1),
        (2.0, 0.5, 2),
        (0.25, 0.5, -1),
        (-0.5, -0.5, 0),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_frexp(input_val, expected_mantissa, expected_exp, dtype):
    """Test math.frexp using LLVM intrinsic."""

    @cuda.jit()
    def frexp_kernel(x, mantissa_out, exp_out):
        m, e = math.frexp(x[0])
        mantissa_out[0] = m
        exp_out[0] = e

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([input_val], dtype=dtype))
    mantissa_out = cuda.to_device(np.zeros(1, dtype=dtype))
    exp_out = cuda.to_device(np.zeros(1, dtype=np.int32))

    frexp_kernel[1, 1, 0, 0](x, mantissa_out, exp_out)

    mantissa = mantissa_out.copy_to_host()[0]
    exp = exp_out.copy_to_host()[0]

    np.testing.assert_almost_equal(mantissa, expected_mantissa, decimal=5)
    assert exp == expected_exp


@pytest.mark.parametrize(
    "mantissa,exp,expected",
    [
        (0.5, 1, 1.0),  # 0.5 * 2^1 = 1.0
        (0.5, 2, 2.0),  # 0.5 * 2^2 = 2.0
        (0.5, 3, 4.0),  # 0.5 * 2^3 = 4.0
        (0.5, -1, 0.25),  # 0.5 * 2^-1 = 0.25
        (0.75, 2, 3.0),  # 0.75 * 2^2 = 3.0
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ldexp(mantissa, exp, expected, dtype):
    """Test math.ldexp using LLVM intrinsic."""

    @cuda.jit()
    def ldexp_kernel(x, e, result_out):
        result_out[0] = math.ldexp(x[0], e[0])

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([mantissa], dtype=dtype))
    e = cuda.to_device(np.array([exp], dtype=np.int32))
    result_out = cuda.to_device(np.zeros(1, dtype=dtype))

    ldexp_kernel[1, 1, 0, 0](x, e, result_out)

    result = result_out.copy_to_host()[0]
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_expm1():
    """Test expm1 (exp(x) - 1) function"""

    @cuda.jit()
    def expm1_kernel(result):
        result[0] = math.expm1(result[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    expm1_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 0.0, decimal=5)

    result = cuda.to_device(np.array([1.0]))
    expm1_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], math.e - 1, decimal=5)


def test_degrees():
    """Test degrees (radians to degrees) function"""

    @cuda.jit()
    def degrees_kernel(result):
        result[0] = math.degrees(result[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([math.pi]))
    degrees_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 180.0, decimal=5)


def test_radians():
    """Test radians (degrees to radians) function"""

    @cuda.jit()
    def radians_kernel(result):
        result[0] = math.radians(result[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([180.0]))
    radians_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], math.pi, decimal=5)


def test_erf():
    """Test erf (error function)"""

    @cuda.jit()
    def erf_kernel(result):
        result[0] = math.erf(result[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    erf_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 0.0, decimal=5)

    result = cuda.to_device(np.array([1.0]))
    erf_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], math.erf(1.0), decimal=5)


def test_erfc():
    """Test erfc (complementary error function)"""

    @cuda.jit()
    def erfc_kernel(result):
        result[0] = math.erfc(result[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    erfc_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 1.0, decimal=5)

    result = cuda.to_device(np.array([1.0]))
    erfc_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], math.erfc(1.0), decimal=5)


def test_gamma():
    """Test gamma function"""

    @cuda.jit()
    def gamma_kernel(result):
        result[0] = math.gamma(result[0])

    ctx = cuda.current_context()
    # gamma(1) = 0! = 1
    result = cuda.to_device(np.array([1.0]))
    gamma_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 1.0, decimal=5)

    # gamma(5) = 4! = 24
    result = cuda.to_device(np.array([5.0]))
    gamma_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 24.0, decimal=4)


def test_lgamma():
    """Test lgamma (log of gamma function)"""

    @cuda.jit()
    def lgamma_kernel(result):
        result[0] = math.lgamma(result[0])

    ctx = cuda.current_context()
    # lgamma(1) = log(0!) = log(1) = 0
    result = cuda.to_device(np.array([1.0]))
    lgamma_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 0.0, decimal=5)

    # lgamma(5) = log(4!) = log(24)
    result = cuda.to_device(np.array([5.0]))
    lgamma_kernel[1, 1, 0, 0](result)
    np.testing.assert_almost_equal(result.copy_to_host()[0], math.log(24.0), decimal=4)


def test_fmod():
    """Test fmod (floating-point remainder with sign of dividend)"""

    @cuda.jit()
    def fmod_kernel(result, x, y):
        result[0] = math.fmod(x[0], y[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    x = cuda.to_device(np.array([7.5]))
    y = cuda.to_device(np.array([2.5]))
    fmod_kernel[1, 1, 0, 0](result, x, y)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 0.0, decimal=5)

    x = cuda.to_device(np.array([-7.5]))
    y = cuda.to_device(np.array([2.5]))
    fmod_kernel[1, 1, 0, 0](result, x, y)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 0.0, decimal=5)


def test_remainder():
    """Test remainder (IEEE 754 remainder)"""

    @cuda.jit()
    def remainder_kernel(result, x, y):
        result[0] = math.remainder(x[0], y[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    x = cuda.to_device(np.array([7.0]))
    y = cuda.to_device(np.array([3.0]))
    remainder_kernel[1, 1, 0, 0](result, x, y)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 1.0, decimal=5)


def test_pow():
    """Test pow function"""

    @cuda.jit()
    def pow_kernel(result, x, y):
        result[0] = math.pow(x[0], y[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    x = cuda.to_device(np.array([2.0]))
    y = cuda.to_device(np.array([3.0]))
    pow_kernel[1, 1, 0, 0](result, x, y)
    np.testing.assert_almost_equal(result.copy_to_host()[0], 8.0, decimal=5)


def test_nextafter():
    """Test nextafter function"""

    @cuda.jit()
    def nextafter_kernel(result, x, y):
        result[0] = math.nextafter(x[0], y[0])

    ctx = cuda.current_context()
    result = cuda.to_device(np.array([0.0]))
    x = cuda.to_device(np.array([1.0]))
    y = cuda.to_device(np.array([2.0]))
    nextafter_kernel[1, 1, 0, 0](result, x, y)
    # nextafter(1.0, 2.0) should be slightly greater than 1.0
    r = result.copy_to_host()[0]
    assert r > 1.0
    assert r < 1.0 + 1e-10


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_modf(dtype):
    """Test modf (returns fractional and integer parts)"""

    @cuda.jit()
    def modf_kernel(frac_out, int_out, x):
        f, i = math.modf(x[0])
        frac_out[0] = f
        int_out[0] = i

    ctx = cuda.current_context()
    x = cuda.to_device(np.array([3.5], dtype=dtype))
    frac_out = cuda.to_device(np.zeros(1, dtype=dtype))
    int_out = cuda.to_device(np.zeros(1, dtype=dtype))

    modf_kernel[1, 1, 0, 0](frac_out, int_out, x)

    frac = frac_out.copy_to_host()[0]
    int_part = int_out.copy_to_host()[0]

    np.testing.assert_almost_equal(frac, 0.5, decimal=5)
    np.testing.assert_almost_equal(int_part, 3.0, decimal=5)


if __name__ == "__main__":
    test_math_ceil()
    test_complex_ops_1()
    test_complex_ops_2()
    test_trigonometric_functions()
    test_exponential_logarithmic()
    test_sqrt()
    test_rounding_functions()
    test_abs_fabs()
    test_unary_operators()
    test_floor_division()
    test_combined_math_operations()
    test_integer_operations()
    test_modulo_operator()
    test_bitwise_shift_operators()
    test_bitwise_operators()
    test_combined_bitwise_operations()
    test_math_with_integer_inputs()
    test_hyperbolic_functions()
    test_inverse_trig_functions()
    test_inverse_hyperbolic_functions()
    test_isnan_isinf_isfinite()
    test_predicates_with_integers()
    test_atan2()
    test_exp2()
    test_log1p()
    test_copysign()
    test_hypot()
    test_complex_from_different_float_types()
    test_expm1()
    test_degrees()
    test_radians()
    test_erf()
    test_erfc()
    test_gamma()
    test_lgamma()
    test_fmod()
    test_remainder()
    test_pow()
    test_nextafter()
