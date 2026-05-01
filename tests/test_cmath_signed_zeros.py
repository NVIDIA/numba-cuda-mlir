# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for IEEE 754 signed zero handling in complex math functions.

These tests verify that the cmath intrinsics correctly handle signed zeros
according to the C99 Annex G specification.
"""

import cmath
import math
import pytest
import numpy as np
from numba_cuda_mlir import cuda


def get_sign(x):
    """Return the sign of a float, distinguishing +0.0 from -0.0."""
    return math.copysign(1.0, x)


def assert_signed_equal(got, expected, msg="", rtol=1e-14):
    """Assert two complex numbers are equal including signed zeros.

    For non-zero values, allows small relative tolerance for floating-point.
    For zeros, checks signs exactly.
    """

    def check_part(g, e, part_name):
        if math.isnan(g) and math.isnan(e):
            return
        if g == 0.0 and e == 0.0:
            assert get_sign(g) == get_sign(
                e
            ), f"{msg}: {part_name} zero signs differ: {g} vs {e}"
        elif math.isinf(g) or math.isinf(e):
            assert g == e, f"{msg}: {part_name} parts differ: {g} != {e}"
        else:
            assert math.isclose(
                g, e, rel_tol=rtol
            ), f"{msg}: {part_name} parts differ: {g} != {e}"

    check_part(got.real, expected.real, "real")
    check_part(got.imag, expected.imag, "imag")


@pytest.fixture
def run_cmath_kernel():
    """Factory for running a cmath function as a CUDA kernel."""

    def _run(func, input_val):
        @cuda.jit
        def kernel(out, inp):
            i = cuda.grid(1)
            if i < out.shape[0]:
                out[i] = func(inp[i])

        inp = np.array([input_val], dtype=np.complex128)
        out = np.empty(1, dtype=np.complex128)
        kernel[1, 1](out, inp)
        return out[0]

    return _run


@pytest.fixture
def run_rect_kernel():
    """Factory for running cmath.rect as a CUDA kernel."""

    def _run(r, phi):
        @cuda.jit
        def kernel(out, r_arr, phi_arr):
            i = cuda.grid(1)
            if i < out.shape[0]:
                out[i] = cmath.rect(r_arr[i], phi_arr[i])

        r_arr = np.array([r], dtype=np.float64)
        phi_arr = np.array([phi], dtype=np.float64)
        out = np.empty(1, dtype=np.complex128)
        kernel[1, 1](out, r_arr, phi_arr)
        return out[0]

    return _run


class TestRectSignedZeros:
    @pytest.mark.parametrize(
        "r,phi",
        [
            (float("inf"), 0.0),
            (float("inf"), -0.0),
            (float("-inf"), 0.0),
            (float("-inf"), -0.0),
        ],
    )
    def test_rect_inf_zero(self, run_rect_kernel, r, phi):
        got = run_rect_kernel(r, phi)
        expected = cmath.rect(r, phi)
        assert_signed_equal(got, expected, f"rect({r}, {phi})")


class TestExpSignedZeros:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(float("-inf"), -1), "exp(-inf-1j)"),
            (complex(float("-inf"), 1), "exp(-inf+1j)"),
            (complex(float("-inf"), 0.5), "exp(-inf+0.5j)"),
            (complex(float("-inf"), -0.5), "exp(-inf-0.5j)"),
        ],
    )
    def test_exp_neg_inf(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.exp, z)
        expected = cmath.exp(z)
        assert_signed_equal(got, expected, desc)


class TestCosSignedZeros:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(0, float("inf")), "cos(infj)"),
            (complex(0, float("-inf")), "cos(-infj)"),
            (complex(-0.0, float("inf")), "cos(-0+infj)"),
        ],
    )
    def test_cos_pure_imag_inf(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.cos, z)
        expected = cmath.cos(z)
        assert_signed_equal(got, expected, desc)


class TestCoshSignedZeros:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(float("-inf"), 0.0), "cosh(-inf+0j)"),
            (complex(float("-inf"), -0.0), "cosh(-inf-0j)"),
            (complex(float("inf"), 0.0), "cosh(inf+0j)"),
            (complex(float("inf"), -0.0), "cosh(inf-0j)"),
        ],
    )
    def test_cosh_inf_zero(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.cosh, z)
        expected = cmath.cosh(z)
        assert_signed_equal(got, expected, desc)


class TestSqrtSignedZeros:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(float("inf"), -1), "sqrt(inf-1j)"),
            (complex(float("inf"), 1), "sqrt(inf+1j)"),
            (complex(float("inf"), -0.0), "sqrt(inf-0j)"),
            (complex(float("inf"), 0.0), "sqrt(inf+0j)"),
            (complex(0, 0), "sqrt(0+0j)"),
            (complex(0, -0.0), "sqrt(0-0j)"),
            (complex(-0.0, 0), "sqrt(-0+0j)"),
            (complex(-0.0, -0.0), "sqrt(-0-0j)"),
        ],
    )
    def test_sqrt_signed_zeros(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.sqrt, z)
        expected = cmath.sqrt(z)
        assert_signed_equal(got, expected, desc)


class TestAcosSignedZeros:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(0, 0), "acos(0+0j)"),
            (complex(0, -0.0), "acos(0-0j)"),
            (complex(-0.0, 0), "acos(-0+0j)"),
            (complex(-0.0, -0.0), "acos(-0-0j)"),
        ],
    )
    def test_acos_zero(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.acos, z)
        expected = cmath.acos(z)
        assert_signed_equal(got, expected, desc)


class TestAcoshSignedZeros:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(1, 0), "acosh(1+0j)"),
            (complex(1, -0.0), "acosh(1-0j)"),
        ],
    )
    def test_acosh_one(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.acosh, z)
        expected = cmath.acosh(z)
        assert_signed_equal(got, expected, desc)


class TestAtanSignedZeros:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(0, 0), "atan(0+0j)"),
            (complex(0, -0.0), "atan(0-0j)"),
            (complex(-0.0, 0), "atan(-0+0j)"),
            (complex(-0.0, -0.0), "atan(-0-0j)"),
            (complex(-0.0, -3.14), "atan(-0-3.14j)"),
            (complex(-0.0, 3.14), "atan(-0+3.14j)"),
            (complex(0.0, -3.14), "atan(0-3.14j)"),
            (complex(0.0, 3.14), "atan(0+3.14j)"),
        ],
    )
    def test_atan_zero(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.atan, z)
        expected = cmath.atan(z)
        assert_signed_equal(got, expected, desc)


class TestAcoshBranchCut:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(-1, 1), "acosh(-1+1j)"),
            (complex(-1, -1), "acosh(-1-1j)"),
            (complex(-2, 1), "acosh(-2+1j)"),
            (complex(-2, -1), "acosh(-2-1j)"),
        ],
    )
    def test_acosh_branch_cut(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.acosh, z)
        expected = cmath.acosh(z)
        # Check that imaginary part has correct sign
        assert math.copysign(1.0, got.imag) == math.copysign(
            1.0, expected.imag
        ), f"{desc}: imag sign wrong: got {got.imag}, expected {expected.imag}"


class TestExpInfInf:
    @pytest.mark.parametrize(
        "z,desc",
        [
            (complex(float("-inf"), float("-inf")), "exp(-inf-infj)"),
            (complex(float("-inf"), float("inf")), "exp(-inf+infj)"),
        ],
    )
    def test_exp_neg_inf_inf(self, run_cmath_kernel, z, desc):
        got = run_cmath_kernel(cmath.exp, z)
        expected = cmath.exp(z)
        assert_signed_equal(got, expected, desc)
